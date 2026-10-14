"""Frame observer that reconstructs a per-turn latency timeline.

Pipecat ships :class:`~pipecat.observers.user_bot_latency_observer.UserBotLatencyObserver`,
which logs a breakdown per turn. This observer exists for the things a realtime
Bangla deployment needs that logging does not give:

* **Barge-in is measured.** How long the bot keeps talking after the caller
  interrupts is a distinct number from response latency, and on a phone call it
  is the one that makes the agent feel deaf.
* **Traces are retained as data**, one object per turn, so a run yields
  percentiles rather than a scrollback. p95 is what a call centre is judged on;
  a mean hides the turn that made the caller hang up.
* **Provenance travels with the numbers** — model, device, precision per stage —
  so a result is never reported without saying what produced it.

It observes; it never pushes frames, so it cannot perturb what it measures.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from bvagent.profiling.trace import TurnTrace


class TurnProfiler(BaseObserver):
    """Builds a :class:`TurnTrace` per user turn from observed frames.

    Args:
        stack: Stage -> description of what runs it (model, device, precision).
            Copied onto every trace so a saved run is self-describing.
        on_turn: Called with each completed trace, for live reporting.
        jsonl_path: If set, append each completed trace as one JSON line.
    """

    def __init__(
        self,
        *,
        stack: dict[str, str] | None = None,
        on_turn: Callable[[TurnTrace], None] | None = None,
        jsonl_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._stack = dict(stack or {})
        self._on_turn = on_turn
        self._jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self._jsonl_path:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        self._turn_index = 0
        self._current: TurnTrace | None = None
        self._traces: list[TurnTrace] = []

        # A turn is finished once its first audio has been reported; later
        # frames belong to the bot finishing its sentence, not to latency.
        self._ttfa_reported = False
        self._bot_speaking = False
        # The turn whose audio is currently playing. A barge-in interrupts
        # *this* turn, which is not the turn the caller's new speech starts.
        self._speaking_trace: TurnTrace | None = None
        # Pipecat broadcasts system frames as a separate object per processor,
        # so identity never repeats and id() cannot dedupe them. Siblings of one
        # broadcast share a pipeline-clock timestamp, so that is the key.
        self._seen: set[tuple[str, int]] = set()
        # Tool calls awaiting a result, keyed by tool_call_id.
        self._pending_tools: dict[str, dict[str, Any]] = {}

    # -- lifecycle -------------------------------------------------------

    @property
    def traces(self) -> list[TurnTrace]:
        """Completed traces, in turn order."""
        return list(self._traces)

    def _begin_turn(self, at: float) -> TurnTrace:
        # The outgoing turn is only finalized here if nothing is still playing
        # it. A turn that is being spoken when the caller cuts in stays open
        # until its audio stops, because the moment its audio stops is the
        # barge-in measurement; closing it now would discard the very number
        # the interruption exists to produce.
        previous = self._current
        if previous is not None and previous is not self._speaking_trace:
            self._finish(previous)

        self._turn_index += 1
        trace = TurnTrace(turn=self._turn_index, started_at=at, stack=dict(self._stack))
        self._current = trace
        self._ttfa_reported = False
        self._pending_tools.clear()
        return trace

    def _finish(self, trace: TurnTrace) -> None:
        """Record one completed trace."""
        if trace is self._current:
            self._current = None
            self._ttfa_reported = False
        # Identity, not equality: TurnTrace is a dataclass, so `in` would walk
        # every field of every recorded turn, and two genuinely distinct turns
        # could compare equal.
        if any(recorded is trace for recorded in self._traces):
            return
        if not trace.complete and not trace.interrupted:
            # A turn with no first audio measured nothing. Dropping it keeps
            # percentiles honest rather than padding them with zeros.
            logger.debug(
                f"turn {trace.turn}: dropped, no first audio measured "
                f"(speech_end={trace.speech_end} user_stopped={trace.user_stopped} "
                f"transcript={trace.transcript_at} tts={trace.tts_first_audio})"
            )
            return
        self._traces.append(trace)
        if self._jsonl_path:
            with self._jsonl_path.open("a") as fh:
                fh.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        if self._on_turn:
            self._on_turn(trace)

    def _finish_turn(self) -> None:
        """Finalize whatever turn is open. Used at end of pipeline."""
        for trace in (self._speaking_trace, self._current):
            if trace is not None:
                self._finish(trace)
        self._speaking_trace = None
        self._current = None

    # -- observation -----------------------------------------------------

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        # Pipeline-clock nanoseconds -> seconds.
        at = data.timestamp / 1e9

        # System frames reach every processor, so dedupe before recording a
        # moment that should appear once per event.
        key = (type(frame).__name__, data.timestamp)
        first_sighting = key not in self._seen
        if first_sighting:
            self._seen.add(key)
            if len(self._seen) > 4096:
                self._seen.clear()
                self._seen.add(key)

        if isinstance(frame, (EndFrame, CancelFrame)):
            self._finish_turn()
            return
        if isinstance(frame, StartFrame):
            return

        self._observe_user_speech(frame, at, first_sighting)
        self._observe_understanding(frame, at, data)
        self._observe_tools(frame, at)
        self._observe_output(frame, at, data)

    # -- the caller ------------------------------------------------------

    def _observe_user_speech(self, frame: Frame, at: float, first: bool) -> None:
        # VAD confirms speech only after it has heard enough of it: a stop
        # event arrives stop_secs *after* the caller actually fell silent, and
        # a start event start_secs after they began. Timing from the frame
        # would credit the pipeline with time the caller already spent waiting,
        # so both are corrected back to when the speech really changed.
        if isinstance(frame, VADUserStartedSpeakingFrame):
            onset = at - getattr(frame, "start_secs", 0.0)

            if self._bot_speaking and self._speaking_trace is not None:
                # Talking over the bot. Recorded against the turn being spoken.
                if self._speaking_trace.barge_in_at is None:
                    self._speaking_trace.barge_in_at = onset
                    self._speaking_trace.interrupted = True
                    self._speaking_trace.mark("barge_in", "vad", onset)

            if not first:
                return
            trace = self._current
            # A turn already underway with no audio yet means the caller
            # resumed after a false endpoint; keep the existing trace.
            if trace is None or trace.bot_speaking is not None:
                trace = self._begin_turn(onset)
            trace.mark("speech_start", "vad", onset)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if not first:
                return
            trace = self._current
            if trace is None or trace.bot_speaking is not None:
                return
            stop_secs = getattr(frame, "stop_secs", 0.0)
            silence_end = at - stop_secs
            trace.vad_stop_secs = stop_secs
            trace.speech_end = TurnTrace._earliest(trace.speech_end, silence_end)
            trace.mark("speech_end", "vad", silence_end, frame_at=at)

        elif isinstance(frame, UserStartedSpeakingFrame):
            if self._current is None and not self._bot_speaking:
                self._begin_turn(at)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            if not first:
                return
            trace = self._current
            if trace is None or trace.bot_speaking is not None:
                return
            trace.user_stopped = TurnTrace._earliest(trace.user_stopped, at)
            if trace.speech_end is None:
                # No separate VAD frame (some transports emit only the
                # aggregated pair); the two moments coincide and the turn-end
                # wait reports as zero rather than as a guess.
                trace.speech_end = trace.user_stopped
            trace.mark("user_stopped", type(frame).__name__, at)

    # -- transcription and reasoning -------------------------------------

    def _observe_understanding(self, frame: Frame, at: float, data: FramePushed) -> None:
        trace = self._current
        if trace is None:
            return

        if isinstance(frame, TranscriptionFrame):
            if trace.transcript_at is None:
                trace.transcript_at = at
                trace.transcript = frame.text
                trace.mark("transcript", data.source.name, at, text=frame.text)
            else:
                # Multi-segment utterance: latency is set by the first
                # transcript, but the full text is what the LLM saw.
                trace.transcript = f"{trace.transcript} {frame.text}".strip()

        elif isinstance(frame, LLMFullResponseStartFrame):
            if trace.llm_request is None:
                trace.llm_request = at
                trace.mark("llm_request", data.source.name, at)

        elif isinstance(frame, LLMTextFrame):
            if trace.llm_first_token is None:
                trace.llm_first_token = at
                trace.mark("llm_first_token", data.source.name, at)
            trace.reply_text += frame.text

    def _observe_tools(self, frame: Frame, at: float) -> None:
        trace = self._current
        if trace is None:
            return

        if isinstance(frame, FunctionCallInProgressFrame):
            if frame.tool_call_id in self._pending_tools:
                return
            record = {
                "name": frame.function_name,
                "tool_call_id": frame.tool_call_id,
                "arguments": _jsonable(frame.arguments),
                "started_at": at,
                "duration_secs": None,
            }
            self._pending_tools[frame.tool_call_id] = record
            trace.tool_calls.append(record)
            trace.mark("tool_start", frame.function_name, at)

        elif isinstance(frame, FunctionCallResultFrame):
            record = self._pending_tools.pop(frame.tool_call_id, None)
            if record is None:
                return
            record["duration_secs"] = round(at - record["started_at"], 4)
            record["result"] = _jsonable(frame.result)
            trace.mark("tool_end", frame.function_name, at)

    # -- the bot ---------------------------------------------------------

    def _observe_output(self, frame: Frame, at: float, data: FramePushed) -> None:
        trace = self._current

        if isinstance(frame, TTSStartedFrame):
            if trace is not None and trace.tts_text_at is None:
                trace.tts_text_at = at
                trace.mark("tts_text", data.source.name, at)

        elif isinstance(frame, TTSAudioRawFrame):
            if trace is not None and trace.tts_first_audio is None:
                trace.tts_first_audio = at
                trace.mark("tts_first_audio", data.source.name, at)

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            if trace is not None and trace.bot_speaking is None:
                trace.bot_speaking = at
                trace.mark("bot_speaking", data.source.name, at)
                self._ttfa_reported = True
            # Whatever is on the wire now is what a barge-in would interrupt.
            self._speaking_trace = trace

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            speaking = self._speaking_trace
            if (
                speaking is not None
                and speaking.barge_in_at is not None
                and speaking.barge_in_stopped is None
            ):
                # The first stop after the interruption is the one that ends
                # it. Matching a later stop would measure the next reply.
                speaking.barge_in_stopped = at
                speaking.mark("barge_in_stopped", data.source.name, at)
            self._speaking_trace = None
            if speaking is not None:
                # Its audio is done, so its measurement is done — including the
                # barge-in span, whose end is this very frame.
                self._finish(speaking)

        elif isinstance(frame, InterruptionFrame):
            # Pipecat broadcasts an interruption whenever a user turn starts,
            # whether or not the bot had the floor — it is how the pipeline
            # flushes stale output. Only the ones that land while the bot is
            # actually speaking are barge-ins; counting the rest would mark
            # every turn interrupted and measure a "barge-in" that is really
            # the length of an uninterrupted reply.
            speaking = self._speaking_trace
            if speaking is None or not self._bot_speaking:
                return
            speaking.interrupted = True
            if speaking.barge_in_at is None:
                # An interruption with no VAD onset recorded (a client-side
                # signal, say); this frame is the best onset available.
                speaking.barge_in_at = at
                speaking.mark("barge_in", type(frame).__name__, at)


def _jsonable(value: Any) -> Any:
    """Coerce tool arguments/results into something json.dumps accepts."""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return repr(value)
