"""Bangla text-to-speech on MMS-TTS (VITS), as a Pipecat service.

``facebook/mms-tts-ben`` is end-to-end: text goes in, a waveform comes out, with
no separate vocoder pass. That is what makes its first-byte latency low enough
for a phone call, and it is also its one awkward property — VITS synthesizes a
whole utterance in a single forward pass, so nothing can be emitted until that
pass finishes.

First-byte latency is therefore set by *how much text is handed over at once*,
not by the model's throughput. This service splits long text at clause
boundaries and synthesizes chunk by chunk, so audio starts flowing after the
first clause instead of after the last. The remaining chunks are synthesized
while the first is still being played out, so the split costs nothing as long as
synthesis stays faster than realtime.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from bvagent.config import TTSConfig

# Clause boundaries worth splitting a long sentence at. Bangla marks sentence
# end with the danda (U+0964) rather than a full stop; the comma and the
# dari-comma are mid-sentence pauses a listener already expects, so cutting
# there is inaudible.
_CLAUSE_SPLIT = re.compile(r"(?<=[।॥?!,;:])\s+")


class MMSTTSService(TTSService):
    """Synthesizes Bangla speech with MMS-TTS.

    Args:
        config: Model, device and chunking settings.
        sample_rate: Output rate. Defaults to the model's native rate, which
            avoids a resample on every utterance.
    """

    def __init__(self, *, config: TTSConfig | None = None, sample_rate: int | None = None, **kwargs):
        self._config = config or TTSConfig()
        kwargs.setdefault("settings", TTSSettings(
            model=self._config.model,
            # MMS-TTS ships one speaker per language, so there is no voice to
            # select; the language is fixed by the checkpoint itself.
            voice=None,
            language="ben",
        ))
        super().__init__(
            sample_rate=sample_rate or self._config.sample_rate or None,
            # The base class opens the audio context and emits TTSStartedFrame,
            # which is the moment the profiler charges TTS from.
            push_start_frame=True,
            push_stop_frames=True,
            **kwargs,
        )
        self._model: Any = None
        self._tokenizer: Any = None
        self._device = self._config.resolved_device()
        self._native_rate: int | None = None
        self._lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        return True

    # -- lifecycle -------------------------------------------------------

    async def setup(self, setup):
        await super().setup(setup)
        await self._load_model()

    async def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, VitsModel

        cfg = self._config
        logger.info(f"loading TTS '{cfg.model}' on {self._device}")
        started = time.monotonic()

        def _load():
            tokenizer = AutoTokenizer.from_pretrained(cfg.model)
            model = VitsModel.from_pretrained(cfg.model).eval()
            if self._device != "cpu":
                model = model.to(self._device)
            return tokenizer, model

        self._tokenizer, self._model = await asyncio.to_thread(_load)
        self._native_rate = int(self._model.config.sampling_rate)
        logger.info(
            f"TTS ready in {time.monotonic() - started:.1f}s "
            f"(native {self._native_rate} Hz)"
        )
        await self._warmup()

    async def _warmup(self) -> None:
        """Run one forward pass before a caller is waiting.

        The first pass allocates workspace and, on GPU, builds the CUDA
        context. Measured cold it is roughly 3x a warm pass, which would
        otherwise land on the first turn of every process and distort the
        first-byte numbers the run is there to measure.
        """
        try:
            await asyncio.to_thread(self._synthesize, "হ্যালো")
            logger.debug("TTS warmup complete")
        except Exception as exc:
            logger.warning(f"TTS warmup skipped: {exc}")

    # -- synthesis -------------------------------------------------------

    def _synthesize(self, text: str) -> np.ndarray:
        """One VITS forward pass. Returns float32 mono in [-1, 1]."""
        import torch

        inputs = self._tokenizer(text, return_tensors="pt")
        if self._device != "cpu":
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            waveform = self._model(**inputs).waveform
        return waveform.squeeze().detach().float().cpu().numpy()

    def _chunks(self, text: str) -> list[str]:
        """Split text so the first forward pass is short.

        Splits only at clause boundaries, and only when the text is long enough
        that splitting wins: each chunk is a separate forward pass with its own
        fixed overhead, and VITS prosody is computed per pass, so over-splitting
        both costs time and makes the delivery choppy.
        """
        limit = self._config.max_chars_per_chunk
        text = text.strip()
        if len(text) <= limit:
            return [text] if text else []

        chunks: list[str] = []
        current = ""
        for part in _CLAUSE_SPLIT.split(text):
            if not part:
                continue
            candidate = f"{current} {part}".strip() if current else part
            if current and len(candidate) > limit:
                chunks.append(current)
                current = part
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _to_frames(self, audio: np.ndarray) -> list[bytes]:
        """Convert a waveform to transport-sized 16-bit PCM frames.

        Emitting one huge buffer would hand the whole utterance to the output
        transport at once, which defeats interruption: barge-in can only drop
        audio that has not been queued yet.
        """
        if self._native_rate and self.sample_rate != self._native_rate:
            audio = _resample(audio, self._native_rate, self.sample_rate)
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16).tobytes()
        bytes_per_frame = max(2, int(self.sample_rate * self._config.frame_ms / 1000) * 2)
        return [pcm[i : i + bytes_per_frame] for i in range(0, len(pcm), bytes_per_frame)]

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Synthesize ``text`` and yield it as audio frames.

        Yields the first chunk's audio as soon as that chunk is synthesized;
        the caller hears speech while the rest is still being generated.
        """
        if self._model is None:
            await self._load_model()

        chunks = self._chunks(text)
        if not chunks:
            yield None
            return

        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        first_audio_at: float | None = None
        started = time.monotonic()
        total_samples = 0

        try:
            for index, chunk in enumerate(chunks):
                async with self._lock:
                    audio = await asyncio.to_thread(self._synthesize, chunk)
                total_samples += len(audio)
                for frame_bytes in self._to_frames(audio):
                    if first_audio_at is None:
                        first_audio_at = time.monotonic()
                        await self.stop_ttfb_metrics()
                    yield TTSAudioRawFrame(frame_bytes, self.sample_rate, 1)
                logger.debug(f"TTS chunk {index + 1}/{len(chunks)}: {chunk[:40]!r}")
        except Exception as exc:
            logger.exception("TTS synthesis failed")
            yield ErrorFrame(f"TTS failed: {exc}")
            return
        finally:
            await self.stop_processing_metrics()

        if first_audio_at is not None and self._native_rate:
            spoken = total_samples / self._native_rate
            elapsed = time.monotonic() - started
            logger.info(
                f"TTS: ttfb {(first_audio_at - started) * 1000:.0f}ms, "
                f"{elapsed * 1000:.0f}ms for {spoken:.2f}s audio "
                f"(RTF {elapsed / max(spoken, 1e-6):.2f}, {len(chunks)} chunk(s))"
            )

    def language_to_service_language(self, language: Language) -> str | None:
        return "ben"


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample. Adequate here because the only conversion in practice
    is the model's 16 kHz to a transport rate, and speech has no content near
    Nyquist that would make the difference audible."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.size / src_rate
    dst_len = max(1, int(round(duration * dst_rate)))
    src_positions = np.linspace(0, audio.size - 1, num=audio.size)
    dst_positions = np.linspace(0, audio.size - 1, num=dst_len)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)
