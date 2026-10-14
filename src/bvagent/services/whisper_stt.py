"""Bangla speech-to-text on faster-whisper, as a Pipecat service.

Segmented rather than continuously streaming. Whisper is an encoder-decoder
trained on 30-second windows; feeding it a sliding window and re-decoding on
every chunk costs several times the compute for a partial transcript the LLM
cannot act on anyway. Gating on VAD and decoding once per utterance gives the
same final-transcript latency for a fraction of the work, which on a shared box
is what keeps concurrent calls viable.

The stage this service is charged with in the latency report is exactly
"caller stopped speaking -> final transcript available".
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from bvagent.config import STTConfig


class WhisperSTTService(SegmentedSTTService):
    """Transcribes VAD-delimited segments with faster-whisper.

    Args:
        config: Model, device, precision and decoding settings.
        sample_rate: Input sample rate; Whisper resamples to 16 kHz internally,
            so this only needs to match what the transport delivers.
    """

    def __init__(self, *, config: STTConfig | None = None, sample_rate: int | None = None, **kwargs):
        self._config = config or STTConfig()
        kwargs.setdefault("settings", STTSettings(
            model=self._config.model,
            language=self._config.language,
        ))
        # The turn-stop strategy uses this to size its safety-net timer. Left
        # unset it assumes a cloud STT's 1.0s p99, which would hold every turn
        # open far longer than a local GPU decode actually needs.
        kwargs.setdefault("ttfs_p99_latency", self._config.ttfs_p99_latency)
        super().__init__(
            sample_rate=sample_rate,
            trailing_silence_secs=self._config.trailing_silence_secs,
            **kwargs,
        )
        self._model: Any = None
        self._device, self._compute_type = self._config.resolved()
        # Decoding is CPU/GPU-bound and synchronous; it runs in a worker thread
        # so it cannot stall the event loop that is still moving audio.
        self._lock = asyncio.Lock()
        self._load_secs: float | None = None

    @property
    def wants_wav_segments(self) -> bool:
        """faster-whisper reads a float array, so take the raw PCM buffer and
        skip building (and immediately re-parsing) a WAV container."""
        return False

    # -- lifecycle -------------------------------------------------------

    async def setup(self, setup):
        await super().setup(setup)
        await self._load_model()

    async def _load_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        cfg = self._config
        logger.info(
            f"loading whisper '{cfg.model}' on {self._device}/{self._compute_type}"
        )
        started = time.monotonic()

        def _load():
            kwargs: dict[str, Any] = {
                "device": self._device,
                "compute_type": self._compute_type,
            }
            if self._device == "cpu" and cfg.cpu_threads:
                kwargs["cpu_threads"] = cfg.cpu_threads
            return WhisperModel(cfg.model, **kwargs)

        self._model = await asyncio.to_thread(_load)
        self._load_secs = time.monotonic() - started
        logger.info(f"whisper ready in {self._load_secs:.1f}s")
        await self._warmup()

    async def _warmup(self) -> None:
        """Decode one buffer of silence before any caller is waiting.

        The first call into CTranslate2 allocates kernels and, on GPU, triggers
        lazy CUDA context setup. Without this the first real turn of every
        process pays a one-off cost that would otherwise be reported as STT
        latency and quietly skew p95 on short runs.
        """
        try:
            silence = np.zeros(self.sample_rate // 2, dtype=np.float32)
            await asyncio.to_thread(self._transcribe, silence, 0.5)
            logger.debug("whisper warmup complete")
        except Exception as exc:  # warmup is best-effort
            logger.warning(f"whisper warmup skipped: {exc}")

    # -- transcription ---------------------------------------------------

    def _transcribe(self, audio: np.ndarray, duration: float) -> str:
        cfg = self._config
        # Bound the decode by the audio it is decoding. Without this a
        # repetition loop runs to the model's own token limit, and a two-second
        # utterance can occupy the transcriber for fifteen seconds — long
        # enough that the turn has already been closed by the time the
        # transcript lands, so the reply never happens at all.
        max_new_tokens = max(cfg.min_decode_tokens, int(duration * cfg.max_tokens_per_second))

        segments, _info = self._model.transcribe(
            audio,
            language=cfg.language,
            beam_size=cfg.beam_size,
            no_speech_threshold=cfg.no_speech_threshold,
            repetition_penalty=cfg.repetition_penalty,
            max_new_tokens=max_new_tokens,
            temperature=list(cfg.temperatures),
            # The segment is already VAD-delimited; running Whisper's own VAD
            # over it again only adds latency.
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Transcribe one VAD-delimited segment.

        Args:
            audio: Raw 16-bit little-endian mono PCM at ``self.sample_rate``.
        """
        if self._model is None:
            await self._load_model()

        if len(audio) < 2:
            yield None
            return

        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(samples) / float(self.sample_rate)

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        started = time.monotonic()
        try:
            async with self._lock:
                text = await asyncio.to_thread(self._transcribe, samples, duration)
        except Exception as exc:
            logger.exception("whisper transcription failed")
            yield ErrorFrame(f"STT failed: {exc}")
            return
        finally:
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()

        elapsed = time.monotonic() - started
        if not text:
            logger.debug(f"whisper: empty transcript for {duration:.2f}s segment")
            yield None
            return

        # Real-time factor: decode seconds per audio second. Above 1.0 the
        # transcriber cannot keep up with the caller and queueing begins.
        logger.info(
            f"whisper: {elapsed * 1000:.0f}ms for {duration:.2f}s audio "
            f"(RTF {elapsed / max(duration, 1e-6):.2f}) -> {text!r}"
        )
        yield TranscriptionFrame(
            text,
            "",
            time_now_iso8601(),
            Language(self._config.language),
            result={"duration_secs": duration, "decode_secs": elapsed},
            # This service decodes a whole VAD-delimited segment in one pass,
            # so there is no later revision coming. Saying so lets the turn-stop
            # strategy skip its STT safety-net timer instead of waiting out
            # ttfs_p99_latency on every turn — worth ~500ms of TTFA, and the
            # reason that timer never appears on the critical path here.
            finalized=True,
        )
