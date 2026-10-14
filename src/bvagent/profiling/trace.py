"""Per-turn latency traces for a realtime voice pipeline.

The unit of measurement is a *turn*: the caller stops speaking, and some time
later the first byte of the bot's reply reaches the transport. That interval is
time-to-first-audio (TTFA), and it is the only latency number a caller can
actually perceive. Everything else here exists to attribute it.

A trace records absolute moments, not durations. Durations are derived, so a
moment that never arrived (no tool call, an interrupted turn) yields ``None``
rather than a misleading zero.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

# Seconds on the *pipeline clock*: monotonic, zeroed when the pipeline starts.
# Pipecat stamps every frame push from this clock, so reading moments off frame
# timestamps rather than calling time.time() in the observer keeps observer
# scheduling jitter out of the measurement and keeps spans immune to NTP steps.
# A run records its wall-clock epoch once, for correlating traces with logs.
Seconds = float


def now() -> Seconds:
    """Fallback for moments with no frame to read a timestamp from."""
    return time.monotonic()


@dataclass
class Moment:
    """One timestamped event in a turn, with whatever produced it."""

    at: Seconds
    kind: str
    source: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnTrace:
    """Everything measured about one user turn.

    Moments are recorded as they are observed. ``first_*`` setters keep the
    earliest value, because a stage's latency is set by its first output — the
    first token, the first audio chunk — not its last.
    """

    turn: int
    started_at: Seconds = field(default_factory=now)

    # --- the timeline ---------------------------------------------------
    # Caller's speech genuinely ended (VAD's own stop event).
    speech_end: Seconds | None = None
    # Pipeline agreed the turn was over. Later than speech_end by the VAD's
    # stop_secs plus any turn-taking strategy.
    user_stopped: Seconds | None = None
    # Final transcript available.
    transcript_at: Seconds | None = None
    # Context handed to the LLM.
    llm_request: Seconds | None = None
    # First token of the reply.
    llm_first_token: Seconds | None = None
    # First text handed to TTS (usually the first sentence boundary).
    tts_text_at: Seconds | None = None
    # First synthesized audio frame out of the TTS service.
    tts_first_audio: Seconds | None = None
    # Transport started emitting the bot's audio.
    bot_speaking: Seconds | None = None

    # --- content --------------------------------------------------------
    transcript: str = ""
    reply_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # --- interruption ---------------------------------------------------
    # Caller began speaking while the bot was speaking.
    barge_in_at: Seconds | None = None
    # Bot's audio actually stopped.
    barge_in_stopped: Seconds | None = None
    interrupted: bool = False

    # The stop_secs the VAD frame reported, so the silence it insisted on can
    # be separated from the decode that ran during it.
    vad_stop_secs: Seconds | None = None

    # --- provenance -----------------------------------------------------
    # Stage -> device/model, so a number is never reported without saying what
    # produced it.
    stack: dict[str, str] = field(default_factory=dict)
    moments: list[Moment] = field(default_factory=list)

    def mark(self, kind: str, source: str = "", at: Seconds | None = None, **detail: Any) -> Seconds:
        at = now() if at is None else at
        self.moments.append(Moment(at=at, kind=kind, source=source, detail=detail))
        return at

    @staticmethod
    def _earliest(current: Seconds | None, candidate: Seconds) -> Seconds:
        """Keep the first observation of a stage's output."""
        return candidate if current is None else min(current, candidate)

    # --- derived durations ----------------------------------------------
    @staticmethod
    def _span(start: Seconds | None, end: Seconds | None) -> Seconds | None:
        if start is None or end is None:
            return None
        span = end - start
        # Frames can be observed out of order across processor boundaries; a
        # negative span is noise, not a measurement.
        return span if span >= 0 else None

    @property
    def turn_end_wait_secs(self) -> Seconds | None:
        """From the caller falling silent to the turn being handed to the LLM.

        This is a *wall-clock* wait, and transcription happens inside it rather
        than after it: the turn-stop strategy starts its timer when VAD reports
        silence, and Whisper decodes the segment while that timer runs. So this
        single span covers both, and is the one that sits on the critical path.

        Charged to configuration — ``VADParams.stop_secs`` plus the strategy's
        ``user_speech_timeout`` — because that is what sets it whenever STT
        finishes first, which on a GPU it almost always does.
        """
        return self._span(self.speech_end, self.user_stopped)

    @property
    def stt_secs(self) -> Seconds | None:
        """Whisper's decode, measured from the moment the caller fell silent.

        Concurrent with :attr:`turn_end_wait_secs`, not sequential with it, so
        it is reported alongside the critical path rather than summed into it.
        Summing the two would double-count and inflate the accounted total past
        the TTFA actually measured.
        """
        return self._span(self.speech_end, self.transcript_at)

    @property
    def stt_ttfs_secs(self) -> Seconds | None:
        """Decode time counted from the VAD stop *frame*, not from the speech.

        This is the quantity Pipecat's turn-stop strategy calls TTFS and sizes
        its safety-net timer from. It is :attr:`stt_secs` minus the silence the
        VAD was still listening through, and it is worth measuring rather than
        guessing: the built-in default assumes a cloud STT, which on a local
        GPU holds every turn open far longer than it needs to be.
        """
        if self.stt_secs is None or self.vad_stop_secs is None:
            return None
        return max(0.0, self.stt_secs - self.vad_stop_secs)

    @property
    def stt_is_binding(self) -> bool | None:
        """Whether the turn was waiting on the transcript, not on the timer.

        The turn-stop strategy needs *both* its timer to expire and a
        transcript to exist. When the transcript is the later of the two, the
        turn-end wait is set by Whisper rather than by ``stop_secs``, and
        lowering ``stop_secs`` further buys nothing at all.

        This is the companion to :attr:`stt_on_critical_path_secs`, which stays
        at zero in exactly this situation and so, on its own, reads as "STT is
        free" at the precise moment STT has become the constraint.
        """
        if self.turn_end_wait_secs is None or self.stt_secs is None:
            return None
        # Within a frame hop of each other: the turn closed as soon as the
        # transcript landed.
        return self.turn_end_wait_secs - self.stt_secs < 0.05

    @property
    def stt_on_critical_path_secs(self) -> Seconds | None:
        """How much of the decode the caller actually waited on.

        Zero while Whisper finishes inside the endpointing window — the usual
        case, and the reason a faster STT model often buys nothing at all. It
        becomes non-zero only once decoding outlasts the wait, which is the
        point at which STT starts costing real latency and is worth tuning.
        """
        if self.transcript_at is None or self.user_stopped is None:
            return None
        return max(0.0, self.transcript_at - self.user_stopped)

    @property
    def llm_queue_secs(self) -> Seconds | None:
        """Turn closing to the LLM actually being asked.

        Context assembly plus whatever the LLM service was doing before it
        could take the request. Normally a millisecond or two — and when it is
        not, it is worth seeing: a cold model load or a busy server shows up
        here as tens of seconds that belong to no model's inference time.
        """
        return self._span(self.user_stopped, self.llm_request)

    @property
    def llm_ttft_secs(self) -> Seconds | None:
        return self._span(self.llm_request, self.llm_first_token)

    @property
    def llm_to_sentence_secs(self) -> Seconds | None:
        """First token to the first chunk TTS was willing to speak.

        Sentence-aggregating TTS cannot start on a partial clause, so this is
        the cost of waiting for a boundary. It is the stage most often mistaken
        for slow TTS.
        """
        return self._span(self.llm_first_token, self.tts_text_at)

    @property
    def tts_ttfb_secs(self) -> Seconds | None:
        return self._span(self.tts_text_at, self.tts_first_audio)

    @property
    def transport_secs(self) -> Seconds | None:
        return self._span(self.tts_first_audio, self.bot_speaking)

    @property
    def tool_secs(self) -> Seconds | None:
        """Wall time spent inside tool handlers on this turn."""
        spans = [
            c["duration_secs"]
            for c in self.tool_calls
            if c.get("duration_secs") is not None
        ]
        return sum(spans) if spans else None

    @property
    def ttfa_secs(self) -> Seconds | None:
        """Time-to-first-audio as the caller experiences it.

        From the moment they stopped talking to the moment the bot's voice
        starts. Includes the endpointing wait, because the caller hears that
        silence too.
        """
        return self._span(self.speech_end, self.bot_speaking)

    @property
    def ttfa_engineered_secs(self) -> Seconds | None:
        """TTFA minus the endpointing wait.

        The part that is answered by making models faster rather than by
        retuning the VAD. Reported alongside TTFA so the two are never
        conflated.
        """
        return self._span(self.user_stopped, self.bot_speaking)

    @property
    def barge_in_secs(self) -> Seconds | None:
        """Caller starts talking over the bot, to the bot's audio stopping."""
        return self._span(self.barge_in_at, self.barge_in_stopped)

    @property
    def complete(self) -> bool:
        """Whether this turn produced a usable TTFA measurement."""
        return self.ttfa_secs is not None

    def stages(self) -> dict[str, Seconds | None]:
        """The critical path: stages that add up to TTFA, in pipeline order.

        STT is deliberately absent. It runs concurrently with the turn-end wait
        and is reported by :meth:`concurrent_stages`; including it here would
        make the parts sum to more than the whole.
        """
        return {
            "turn_end_wait": self.turn_end_wait_secs,
            "llm_queue": self.llm_queue_secs,
            "llm_ttft": self.llm_ttft_secs,
            "llm_to_sentence": self.llm_to_sentence_secs,
            "tts_ttfb": self.tts_ttfb_secs,
            "transport": self.transport_secs,
        }

    def concurrent_stages(self) -> dict[str, Seconds | None]:
        """Work that overlaps the critical path rather than extending it."""
        return {
            "stt": self.stt_secs,
            "stt_on_critical_path": self.stt_on_critical_path_secs,
            "stt_ttfs": self.stt_ttfs_secs,
        }

    @property
    def unattributed_secs(self) -> Seconds | None:
        """TTFA left over after the named stages.

        Frame-hop and scheduling overhead. Reported rather than hidden: if this
        grows, the breakdown has stopped describing the pipeline and the stage
        list needs revisiting.
        """
        total = self.ttfa_secs
        if total is None:
            return None
        named = sum(v for v in self.stages().values() if v is not None)
        return max(0.0, total - named)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["derived"] = {
            **{k: _round(v) for k, v in self.stages().items()},
            **{k: _round(v) for k, v in self.concurrent_stages().items()},
            "tool": _round(self.tool_secs),
            "unattributed": _round(self.unattributed_secs),
            "ttfa": _round(self.ttfa_secs),
            "ttfa_engineered": _round(self.ttfa_engineered_secs),
            "barge_in": _round(self.barge_in_secs),
        }
        return d


def _round(v: Seconds | None, places: int = 4) -> Seconds | None:
    return None if v is None else round(v, places)
