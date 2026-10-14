"""Configuration for the Bangla voice agent.

Every knob that moves latency is here and is settable from the environment, so
a benchmark sweep changes configuration rather than code, and a reported number
can always be traced back to the settings that produced it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

BANGLA = "bn"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_b(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def detect_device() -> tuple[str, str]:
    """Pick a compute device and a matching precision.

    Returns ``(device, compute_type)`` where compute_type is in CTranslate2's
    vocabulary. CPU gets int8, which on a many-core desktop is several times
    faster than float32 and is the difference between Whisper keeping up with
    realtime and not.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


@dataclass
class VADConfig:
    """Voice activity detection.

    ``stop_secs`` is the single most consequential latency knob in the whole
    pipeline: it is silence the caller sits through on every turn, paid before
    any model runs, and it accounts for roughly half of TTFA. It also lengthens
    the segment handed to Whisper, so raising it slows transcription too — in
    the bench sweep, decode went from 388ms at 0.2s to 1393ms at 0.8s.

    Lowering it stops helping below the STT decode time. The turn cannot end
    without a transcript, so once the window is shorter than Whisper takes, the
    turn simply waits for Whisper instead and the knob does nothing: measured at
    0.2/0.2 the configured wait is 400ms but the observed turn-end is 737ms,
    against an STT decode of 736ms. The default sits just above that floor,
    where configuration still sets the wait and the number is predictable.

    Note also what the benchmark cannot see: it replays synthetic speech, which
    never hesitates, so it observes the latency a short window buys and never
    the callers it cuts off mid-clause. That side needs real calls.
    """

    stop_secs: float = field(default_factory=lambda: _env_f("BV_VAD_STOP_SECS", 0.5))
    start_secs: float = field(default_factory=lambda: _env_f("BV_VAD_START_SECS", 0.2))
    confidence: float = field(default_factory=lambda: _env_f("BV_VAD_CONFIDENCE", 0.7))
    min_volume: float = field(default_factory=lambda: _env_f("BV_VAD_MIN_VOLUME", 0.6))
    # Grace period after VAD stop in which the caller may resume before the
    # turn is handed to the LLM. Pipecat's own default turn-stop strategy is a
    # *semantic* end-of-turn model (LocalSmartTurnAnalyzerV3) trained on
    # English; on Bangla it returns INCOMPLETE indefinitely and the turn never
    # ends, so this pipeline uses timeout-based stopping instead and this is
    # the knob that governs it. Added to stop_secs, it is the full endpointing
    # cost the caller sits through.
    turn_stop_timeout: float = field(
        default_factory=lambda: _env_f("BV_VAD_TURN_STOP_TIMEOUT", 0.4)
    )


@dataclass
class STTConfig:
    """Whisper via CTranslate2 (faster-whisper).

    ``model`` accepts a size name or any HF repo id, so a Bangla fine-tune can
    be swapped in and measured against stock large-v3 with the same harness.
    """

    model: str = field(default_factory=lambda: _env("BV_STT_MODEL", "large-v3"))
    language: str = field(default_factory=lambda: _env("BV_STT_LANGUAGE", BANGLA))
    device: str = field(default_factory=lambda: _env("BV_STT_DEVICE", ""))
    compute_type: str = field(default_factory=lambda: _env("BV_STT_COMPUTE_TYPE", ""))
    beam_size: int = field(default_factory=lambda: _env_i("BV_STT_BEAM_SIZE", 1))
    # Whisper hallucinates fluent text on silence; VAD-gated segments plus a
    # no-speech threshold keeps that out of the transcript.
    no_speech_threshold: float = field(
        default_factory=lambda: _env_f("BV_STT_NO_SPEECH_THRESHOLD", 0.6)
    )
    # Padding the end of a VAD-cut segment stops the model dropping the last word.
    trailing_silence_secs: float = field(
        default_factory=lambda: _env_f("BV_STT_TRAILING_SILENCE", 0.3)
    )
    cpu_threads: int = field(default_factory=lambda: _env_i("BV_STT_CPU_THREADS", 0))
    # Whisper's failure mode on accented or noisy Bangla is a repetition loop:
    # it emits the same phrase until it runs out of tokens. Observed decoding
    # 2.1s of audio in 15s (RTF 7.0) and returning
    # "আমি গুন্সান শান্দের" eleven times over. A penalty above 1.0 makes the
    # loop unattractive to the decoder.
    repetition_penalty: float = field(
        default_factory=lambda: _env_f("BV_STT_REPETITION_PENALTY", 1.15)
    )
    # Tokens per second of audio, as a decode-length ceiling: a hard bound on
    # worst-case latency, since no utterance can then cost more than its
    # duration times this. Bengali script is token-hungry — measured at 15-22
    # tokens per second of speech across the test corpus — so the ceiling sits
    # at roughly twice the observed maximum. Tight enough to stop a runaway
    # loop, loose enough that no real utterance is truncated; at 28 it was
    # cutting sentences off mid-word.
    max_tokens_per_second: float = field(
        default_factory=lambda: _env_f("BV_STT_MAX_TOKENS_PER_SEC", 45.0)
    )
    min_decode_tokens: int = field(
        default_factory=lambda: _env_i("BV_STT_MIN_DECODE_TOKENS", 96)
    )
    # Whisper retries the whole window at each temperature when its quality
    # checks fail, and the repetition loop fails them every time — six passes
    # is where the 15s went. Two is enough to recover a genuinely hard segment
    # without letting a pathological one run away.
    temperatures: tuple[float, ...] = (0.0, 0.2)
    # Observed p99 for this service to return a final transcript after VAD
    # stop. The turn-stop strategy sizes its safety-net timer from this, so a
    # cloud-sized default would hold every turn open longer than a local GPU
    # decode needs. Measured, not guessed: large-v3 on an RTX PRO 6000 gives a
    # TTFS p99 of 260ms on short loan-scenario turns and 768ms on the longer
    # booking ones. The default covers both; every bench run reprints the
    # figure measured for the stack it just ran, so it need not be guessed.
    ttfs_p99_latency: float = field(
        default_factory=lambda: _env_f("BV_STT_TTFS_P99", 0.8)
    )

    def resolved(self) -> tuple[str, str]:
        device, compute = detect_device()
        return (self.device or device, self.compute_type or compute)


@dataclass
class TTSConfig:
    """Bangla speech synthesis.

    ``facebook/mms-tts-ben`` is a VITS model: small, single-speaker, offline,
    and end-to-end (no separate vocoder pass), which is why its first-byte
    latency is low enough for a phone call.
    """

    model: str = field(default_factory=lambda: _env("BV_TTS_MODEL", "facebook/mms-tts-ben"))
    device: str = field(default_factory=lambda: _env("BV_TTS_DEVICE", ""))
    sample_rate: int = field(default_factory=lambda: _env_i("BV_TTS_SAMPLE_RATE", 0))
    # VITS synthesizes a whole utterance at once, so first-byte latency is set
    # by how much text is handed over. Splitting long sentences at clause
    # boundaries gets audio flowing sooner on verbose replies.
    max_chars_per_chunk: int = field(
        default_factory=lambda: _env_i("BV_TTS_MAX_CHARS", 160)
    )
    # Size of the audio frames pushed downstream, in milliseconds.
    frame_ms: int = field(default_factory=lambda: _env_i("BV_TTS_FRAME_MS", 40))

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        device, _ = detect_device()
        return device


@dataclass
class LLMConfig:
    """Any OpenAI-compatible endpoint: Ollama, vLLM, or a hosted API.

    Kept behind that interface so the same pipeline and the same profiler can
    compare a local model against a hosted one without touching the code.
    """

    base_url: str = field(
        default_factory=lambda: _env("BV_LLM_BASE_URL", "http://localhost:11434/v1")
    )
    api_key: str = field(default_factory=lambda: _env("BV_LLM_API_KEY", "ollama"))
    model: str = field(default_factory=lambda: _env("BV_LLM_MODEL", "qwen3.8:latest"))
    temperature: float = field(default_factory=lambda: _env_f("BV_LLM_TEMPERATURE", 0.3))
    max_tokens: int = field(default_factory=lambda: _env_i("BV_LLM_MAX_TOKENS", 160))
    # Reasoning traces are latency the caller waits through in silence. Off for
    # voice; a thinking model that ignores this will show up as LLM TTFT.
    disable_thinking: bool = field(
        default_factory=lambda: _env_b("BV_LLM_DISABLE_THINKING", True)
    )


@dataclass
class AudioConfig:
    in_sample_rate: int = field(default_factory=lambda: _env_i("BV_AUDIO_IN_RATE", 16000))
    # MMS-TTS is natively 16 kHz. Matching it end to end removes a resample
    # from the output path, which is both latency and quality nobody gains
    # anything by spending.
    out_sample_rate: int = field(default_factory=lambda: _env_i("BV_AUDIO_OUT_RATE", 16000))


@dataclass
class LiveKitConfig:
    """LiveKit as the media plane.

    Relevant to a deployment already on LiveKit that wants to change what sits
    above it: the room, the SIP trunk and the phone numbers stay put, and only
    the pipeline is replaced. A local ``livekit-server --dev`` ships the
    devkey/secret pair below, so the path is testable without an account.
    """

    url: str = field(default_factory=lambda: _env("BV_LIVEKIT_URL", "ws://localhost:7880"))
    api_key: str = field(default_factory=lambda: _env("BV_LIVEKIT_API_KEY", "devkey"))
    api_secret: str = field(default_factory=lambda: _env("BV_LIVEKIT_API_SECRET", "secret"))
    room: str = field(default_factory=lambda: _env("BV_LIVEKIT_ROOM", "bangla-agent"))
    # The agent's participant identity in the room.
    identity: str = field(default_factory=lambda: _env("BV_LIVEKIT_IDENTITY", "bangla-agent"))
    session_timeout_secs: int = field(
        default_factory=lambda: _env_i("BV_SESSION_TIMEOUT_SECS", 600)
    )


@dataclass
class ProfileConfig:
    enabled: bool = field(default_factory=lambda: _env_b("BV_PROFILE", True))
    jsonl_path: str = field(default_factory=lambda: _env("BV_PROFILE_JSONL", "runs/turns.jsonl"))
    # Print a compact per-turn line as the call happens.
    live: bool = field(default_factory=lambda: _env_b("BV_PROFILE_LIVE", True))


@dataclass
class Settings:
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    livekit: LiveKitConfig = field(default_factory=LiveKitConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    # Which media plane carries the call. The pipeline above it is identical,
    # so switching this is a controlled comparison rather than a rewrite.
    transport: Literal["websocket", "livekit"] = field(
        default_factory=lambda: _env("BV_TRANSPORT", "websocket")  # type: ignore[arg-type]
    )
    # "booking" or "loan": which Verbex-style scenario the agent runs.
    scenario: Literal["booking", "loan"] = field(
        default_factory=lambda: _env("BV_SCENARIO", "loan")  # type: ignore[arg-type]
    )
    host: str = field(default_factory=lambda: _env("BV_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_i("BV_PORT", 8765))

    def stack_description(self) -> dict[str, str]:
        """One line per stage saying what produced its numbers.

        Attached to every trace so a saved run can be interpreted months later
        without guessing which model was in the pipeline.
        """
        stt_device, stt_compute = self.stt.resolved()
        return {
            "vad": f"silero stop_secs={self.vad.stop_secs} start_secs={self.vad.start_secs} "
                   f"+ turn_stop_timeout={self.vad.turn_stop_timeout} (timeout-based turn end)",
            "stt": f"faster-whisper {self.stt.model} [{stt_device}/{stt_compute}] "
                   f"beam={self.stt.beam_size} lang={self.stt.language}",
            "llm": f"{self.llm.model} @ {self.llm.base_url} "
                   f"(max_tokens={self.llm.max_tokens}, thinking="
                   f"{'off' if self.llm.disable_thinking else 'on'})",
            "tts": f"{self.tts.model} [{self.tts.resolved_device()}] "
                   f"chunk<={self.tts.max_chars_per_chunk}chars",
            "audio": f"in {self.audio.in_sample_rate}Hz / out {self.audio.out_sample_rate}Hz",
            "transport": self._transport_description(),
            "scenario": self.scenario,
        }

    def _transport_description(self) -> str:
        if self.transport == "livekit":
            return f"livekit {self.livekit.url} room={self.livekit.room}"
        return f"websocket {self.host}:{self.port} (raw PCM)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return Settings()
