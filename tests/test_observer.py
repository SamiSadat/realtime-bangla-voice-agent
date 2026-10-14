"""Tests that drive the observer with synthetic frames.

Every bug these cover was a real one found while building the pipeline, and
each produced numbers that looked entirely reasonable:

* VAD frames timed at the frame rather than at the speech, understating TTFA by
  the whole ``stop_secs`` window.
* An ``InterruptionFrame`` on every turn start counted as a barge-in, so the
  "barge-in" figure was really the length of an uninterrupted reply.
* The interrupted turn orphaned when the caller's new speech opened a turn,
  losing the barge-in measurement entirely.
"""

from __future__ import annotations

import pytest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection

from bvagent.profiling.observer import TurnProfiler


class FakeProcessor:
    def __init__(self, name: str):
        self.name = name


SOURCE = FakeProcessor("source")
DEST = FakeProcessor("dest")


async def push(profiler: TurnProfiler, frame: Frame, at_secs: float) -> None:
    """Deliver one frame at a pipeline-clock time given in seconds."""
    await profiler.on_push_frame(
        FramePushed(
            source=SOURCE,
            destination=DEST,
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
            timestamp=int(at_secs * 1e9),
        )
    )


async def run_turn(
    profiler: TurnProfiler,
    *,
    start: float = 1.0,
    stop: float = 3.0,
    stop_secs: float = 0.5,
    transcript: float = 3.2,
    user_stopped: float = 3.9,
    first_token: float = 4.2,
    tts_text: float = 4.4,
    first_audio: float = 4.5,
    bot_start: float = 4.52,
    bot_stop: float | None = 9.0,
) -> None:
    """Drive one turn. ``bot_stop=None`` leaves the bot still speaking, which
    is the state a barge-in has to arrive in."""
    await push(profiler, VADUserStartedSpeakingFrame(start_secs=0.2), start)
    await push(profiler, VADUserStoppedSpeakingFrame(stop_secs=stop_secs), stop)
    await push(profiler, TranscriptionFrame("আসসালামু আলাইকুম", "", "", None), transcript)
    await push(profiler, UserStoppedSpeakingFrame(), user_stopped)
    await push(profiler, LLMFullResponseStartFrame(), user_stopped + 0.01)
    await push(profiler, LLMTextFrame("জি"), first_token)
    await push(profiler, TTSStartedFrame(), tts_text)
    await push(profiler, TTSAudioRawFrame(b"\x00\x00", 16000, 1), first_audio)
    await push(profiler, BotStartedSpeakingFrame(), bot_start)
    if bot_stop is not None:
        await push(profiler, BotStoppedSpeakingFrame(), bot_stop)


class TestVADCorrection:
    async def test_speech_end_is_corrected_back_by_stop_secs(self):
        # VAD reports the stop only after hearing stop_secs of silence, so the
        # caller actually stopped that much earlier. Timing from the frame
        # would hand the pipeline free latency it did not earn.
        profiler = TurnProfiler()
        await run_turn(profiler, stop=3.0, stop_secs=0.5)
        trace = profiler.traces[0]
        assert trace.speech_end == pytest.approx(2.5)
        assert trace.ttfa_secs == pytest.approx(4.52 - 2.5)

    async def test_turn_end_wait_covers_vad_and_strategy(self):
        profiler = TurnProfiler()
        await run_turn(profiler, stop=3.0, stop_secs=0.5, user_stopped=3.9)
        assert profiler.traces[0].turn_end_wait_secs == pytest.approx(1.4)


class TestStageCapture:
    async def test_records_a_complete_turn(self):
        profiler = TurnProfiler()
        await run_turn(profiler)
        assert len(profiler.traces) == 1
        trace = profiler.traces[0]
        assert trace.complete
        assert not trace.interrupted
        assert trace.transcript == "আসসালামু আলাইকুম"
        assert trace.llm_ttft_secs is not None
        assert trace.tts_ttfb_secs is not None

    async def test_first_output_of_a_stage_wins(self):
        # Latency is set by the first token and the first audio chunk, so a
        # later one must not overwrite the measurement.
        profiler = TurnProfiler()
        await push(profiler, VADUserStartedSpeakingFrame(start_secs=0.0), 1.0)
        await push(profiler, VADUserStoppedSpeakingFrame(stop_secs=0.0), 2.0)
        await push(profiler, UserStoppedSpeakingFrame(), 2.1)
        await push(profiler, LLMFullResponseStartFrame(), 2.1)
        await push(profiler, LLMTextFrame("ক"), 2.4)
        await push(profiler, LLMTextFrame("খ"), 2.9)
        await push(profiler, TTSStartedFrame(), 3.0)
        await push(profiler, TTSAudioRawFrame(b"\x00\x00", 16000, 1), 3.1)
        await push(profiler, TTSAudioRawFrame(b"\x00\x00", 16000, 1), 3.4)
        await push(profiler, BotStartedSpeakingFrame(), 3.15)
        await push(profiler, BotStoppedSpeakingFrame(), 5.0)
        trace = profiler.traces[0]
        assert trace.llm_first_token == pytest.approx(2.4)
        assert trace.tts_first_audio == pytest.approx(3.1)
        assert trace.reply_text == "কখ"

    async def test_broadcast_duplicates_collapse_to_one_moment(self):
        # Pipecat broadcasts system frames as one object per processor; each
        # arrives separately and they must not each record a moment.
        profiler = TurnProfiler()
        await run_turn(profiler)
        kinds = [m.kind for m in profiler.traces[0].moments]
        assert kinds.count("speech_end") == 1
        assert kinds.count("user_stopped") == 1


class TestBargeIn:
    async def test_interruption_without_bot_audio_is_not_a_barge_in(self):
        # An InterruptionFrame is broadcast on every user turn start. Only the
        # ones landing while the bot holds the floor are barge-ins.
        profiler = TurnProfiler()
        await push(profiler, VADUserStartedSpeakingFrame(start_secs=0.0), 1.0)
        await push(profiler, InterruptionFrame(), 1.01)
        await run_turn(profiler, start=1.0)
        assert not profiler.traces[0].interrupted
        assert profiler.traces[0].barge_in_secs is None

    async def test_talking_over_the_bot_is_measured(self):
        profiler = TurnProfiler()
        # Turn one is still speaking when the caller cuts in.
        await run_turn(profiler, bot_stop=None)
        assert profiler.traces == []  # not finished; still holding the floor

        await push(profiler, VADUserStartedSpeakingFrame(start_secs=0.2), 6.2)
        await push(profiler, InterruptionFrame(), 6.25)
        await push(profiler, BotStoppedSpeakingFrame(), 6.4)

        trace = profiler.traces[0]
        assert trace.interrupted
        # Onset corrected back by start_secs: 6.2 - 0.2 = 6.0.
        assert trace.barge_in_secs == pytest.approx(0.4)

    async def test_interrupted_turn_is_not_orphaned_by_the_next(self):
        # The caller's barge-in opens a new turn. The interrupted turn must
        # stay open until its audio stops, because that stop *is* the
        # measurement.
        profiler = TurnProfiler()
        await run_turn(profiler, bot_stop=None)
        await push(profiler, VADUserStartedSpeakingFrame(start_secs=0.0), 6.0)
        await push(profiler, BotStoppedSpeakingFrame(), 6.3)
        assert len(profiler.traces) == 1
        assert profiler.traces[0].barge_in_secs == pytest.approx(0.3)


class TestDroppedTurns:
    async def test_turn_without_audio_is_not_recorded(self):
        # Recording it as a zero would drag every percentile down.
        profiler = TurnProfiler()
        await push(profiler, VADUserStartedSpeakingFrame(start_secs=0.0), 1.0)
        await push(profiler, VADUserStoppedSpeakingFrame(stop_secs=0.0), 2.0)
        await push(profiler, UserStoppedSpeakingFrame(), 2.1)
        await push(profiler, BotStoppedSpeakingFrame(), 3.0)
        assert profiler.traces == []
