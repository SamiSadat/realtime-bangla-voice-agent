"""A transport that replays recorded audio through the real pipeline.

Benchmarking a voice agent by talking to it does not produce numbers anyone can
compare: the utterance differs every time, and so does the reply length, the
token count, and the synthesis cost. This transport removes the human from the
loop while changing nothing else — the same VAD, STT, LLM, TTS and the same
observer, driven by a fixed script.

Audio is fed at realtime pace on purpose. Pushing a file in as fast as it will
go would make VAD endpointing collapse to nothing and report a TTFA no caller
could ever experience; the wait for silence is a real part of the latency and
has to be paid at real speed.
"""

from __future__ import annotations

import asyncio
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

# Audio is pushed in 20ms frames, matching what a telephony transport delivers.
FRAME_MS = 20


@dataclass
class Utterance:
    """One scripted thing the caller says.

    Args:
        audio: Mono float32 in [-1, 1] at the script's sample rate.
        label: Human-readable tag, carried into logs.
        wait_for_bot: Wait for the bot to finish replying before the next
            utterance. False is what makes a barge-in possible.
        delay_secs: Silence to feed before speaking. After a bot reply this is
            think-time.
        barge_in_after: Speak this many seconds after the bot *starts*
            speaking, instead of after ``delay_secs``. This is what makes a
            barge-in reproducible: timing it from the end of the previous turn
            would land it wherever the reply's length happened to put it, and
            most often after the bot had already stopped. Implies
            ``wait_for_bot=False``.
    """

    audio: np.ndarray
    label: str = ""
    wait_for_bot: bool = True
    delay_secs: float = 0.4
    barge_in_after: float | None = None

    def __post_init__(self) -> None:
        if self.barge_in_after is not None:
            self.wait_for_bot = False


@dataclass
class ReplayScript:
    """A whole scripted call."""

    utterances: list[Utterance]
    sample_rate: int = 16000
    # Silence fed after each utterance so VAD can register the stop. Must
    # exceed VADParams.stop_secs or the turn never ends.
    trailing_silence_secs: float = 1.2
    # Give up on a turn that never produces audio, so a broken run ends.
    turn_timeout_secs: float = 120.0
    # Whether the agent speaks first. When it does, the script must wait for
    # that greeting to finish, or the opening utterance lands on top of it and
    # every following turn is misread as an interruption.
    expect_greeting: bool = True
    # How long the bot must stay quiet before its reply counts as finished.
    # A multi-sentence reply is synthesized as one TTS context per sentence,
    # and the output transport reports the bot stopped whenever its audio
    # queue drains between them. Treating the first of those as end-of-reply
    # makes the script talk into the next sentence, which reads as a barge-in
    # on every turn. Only a sustained silence means the bot is really done.
    reply_settle_secs: float = 0.8


class ReplayInputTransport(BaseInputTransport):
    """Feeds scripted audio into the pipeline at realtime pace."""

    def __init__(self, script: ReplayScript, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._script = script
        self._ready = asyncio.Event()
        # Set while the bot is speaking, so the script can wait for a reply or
        # deliberately talk over it.
        self.bot_speaking = asyncio.Event()
        self.bot_finished = asyncio.Event()
        # Starts clear when the agent greets first, so the opening utterance
        # waits for the greeting instead of talking over it.
        if not script.expect_greeting:
            self.bot_finished.set()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)
        self._ready.set()

    async def wait_ready(self) -> None:
        await self._ready.wait()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # Bot speech state arrives upstream from the output transport.
        if isinstance(frame, BotStartedSpeakingFrame):
            self.bot_speaking.set()
            self.bot_finished.clear()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_speaking.clear()
            self.bot_finished.set()

    async def feed(self, audio: np.ndarray) -> None:
        """Push one buffer as 20ms frames, paced to realtime."""
        rate = self._script.sample_rate
        samples_per_frame = int(rate * FRAME_MS / 1000)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

        # Pace against a fixed origin rather than sleeping a fixed interval per
        # frame, so scheduling jitter cannot accumulate into drift over a call.
        started = time.monotonic()
        for index in range(0, len(pcm), samples_per_frame):
            chunk = pcm[index : index + samples_per_frame]
            await self.push_audio_frame(
                InputAudioRawFrame(
                    audio=chunk.tobytes(),
                    sample_rate=rate,
                    num_channels=1,
                )
            )
            target = started + ((index + samples_per_frame) / rate)
            drift = target - time.monotonic()
            if drift > 0:
                await asyncio.sleep(drift)

    async def feed_silence(self, seconds: float) -> None:
        if seconds > 0:
            await self.feed(np.zeros(int(self._script.sample_rate * seconds), dtype=np.float32))


class ReplayOutputTransport(BaseOutputTransport):
    """Captures the bot's audio instead of playing it, at realtime pace.

    Pipecat's output handler does not sleep between audio frames: it relies on
    ``write_audio_frame`` taking as long as the audio it was handed, which is
    what a sound card or a phone line does naturally. A capture transport that
    returns immediately therefore drains an entire reply in microseconds, which
    breaks the run in two ways — ``BotStoppedSpeakingFrame`` fires almost as
    soon as the bot starts, so every following utterance looks like a barge-in;
    and there is no window left in which a real barge-in could be delivered, so
    interruption cannot be measured at all.

    So this paces. It is the difference between a harness that measures a phone
    call and one that measures how fast a buffer can be copied.
    """

    def __init__(self, params: TransportParams, *, realtime: bool = True, **kwargs):
        super().__init__(params, **kwargs)
        self._captured: list[bytes] = []
        self._sample_rate = params.audio_out_sample_rate or 16000
        self._realtime = realtime
        # Pace against a running deadline rather than sleeping per frame, so
        # scheduling jitter does not accumulate into drift across a long reply.
        self._playhead: float | None = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        self._captured.append(frame.audio)
        self._sample_rate = frame.sample_rate

        if self._realtime:
            # 16-bit mono.
            duration = len(frame.audio) / 2 / max(frame.sample_rate, 1)
            now = time.monotonic()
            # A gap since the last write means playback had drained; restart
            # the deadline from now instead of trying to catch up on silence.
            if self._playhead is None or now > self._playhead:
                self._playhead = now
            self._playhead += duration
            delay = self._playhead - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        return True

    def interrupted(self) -> None:
        """Drop the pacing deadline when queued audio is discarded."""
        self._playhead = None

    def save(self, path: str | Path) -> Path | None:
        """Write everything the bot said to a WAV, for listening to a run.

        A latency report says how fast the agent replied; it cannot say whether
        the Bangla was intelligible. This is how that gets checked.
        """
        if not self._captured:
            return None
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(b"".join(self._captured))
        return p


class ReplayTransport(BaseTransport):
    """Pairs the scripted input with the capturing output."""

    def __init__(
        self,
        script: ReplayScript,
        params: TransportParams | None = None,
        *,
        realtime: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._script = script
        self._params = params or TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=script.sample_rate,
            audio_out_sample_rate=script.sample_rate,
        )
        self._input = ReplayInputTransport(script, self._params, name="replay-in")
        self._output = ReplayOutputTransport(self._params, realtime=realtime, name="replay-out")

    def input(self) -> ReplayInputTransport:
        return self._input

    def output(self) -> ReplayOutputTransport:
        return self._output

    async def _feed_silence_until(self, predicate, timeout: float) -> bool:
        """Feed silence until ``predicate`` holds, or the timeout expires.

        Silence rather than nothing: a real call carries audio continuously,
        and the pipeline's idle timeouts are written on that assumption. Going
        quiet at the transport would trip them and end turns artificially.
        """
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                return False
            await self._input.feed_silence(0.1)
        return True

    async def _await_reply(self, label: str) -> None:
        """Wait for the bot to start its reply and then finish it.

        Waiting only on "the bot is not speaking" is not enough, and getting
        this wrong is what makes a benchmark quietly measure the wrong thing:
        between an utterance ending and the reply starting, the bot is idle and
        still flagged finished from the *previous* turn. A script that takes
        that as its cue speaks again a fraction of a second into the reply,
        interrupting every turn and destroying the very latency it is there to
        measure.
        """
        script = self._script
        started = await self._feed_silence_until(
            self._input.bot_speaking.is_set, script.turn_timeout_secs
        )
        if not started:
            logger.warning(f"replay: no reply to {label!r} within {script.turn_timeout_secs}s")
            return
        deadline = time.monotonic() + script.turn_timeout_secs
        while True:
            finished = await self._feed_silence_until(
                self._input.bot_finished.is_set, script.turn_timeout_secs
            )
            if not finished:
                logger.warning(f"replay: reply to {label!r} never finished")
                return
            # Quiet — but possibly only between two sentences of one reply.
            # Hold for the settle window and see whether it resumes.
            settled = True
            settle_until = time.monotonic() + script.reply_settle_secs
            while time.monotonic() < settle_until:
                await self._input.feed_silence(0.1)
                if self._input.bot_speaking.is_set():
                    settled = False
                    break
            if settled:
                return
            if time.monotonic() >= deadline:
                logger.warning(f"replay: reply to {label!r} never settled")
                return

    async def run_script(self) -> None:
        """Play the script through, honouring waits and barge-ins."""
        await self._input.wait_ready()
        script = self._script

        if script.expect_greeting:
            # The agent speaks first; let it finish before the caller starts.
            await self._await_reply("greeting")

        for index, utterance in enumerate(script.utterances, start=1):
            if utterance.barge_in_after is not None:
                # Cut in a fixed interval into the reply, wherever that reply
                # happens to begin.
                speaking = await self._feed_silence_until(
                    self._input.bot_speaking.is_set, script.turn_timeout_secs
                )
                if not speaking:
                    logger.warning(
                        f"replay: {utterance.label!r} is a barge-in but the bot never "
                        "started speaking; it will measure nothing"
                    )
                else:
                    await self._input.feed_silence(utterance.barge_in_after)
            else:
                await self._input.feed_silence(utterance.delay_secs)

            if not utterance.wait_for_bot and not self._input.bot_speaking.is_set():
                # A barge-in is only a barge-in if the bot is still talking.
                logger.warning(
                    f"replay: {utterance.label!r} was scheduled as a barge-in but the "
                    "bot had already stopped; it will measure nothing"
                )

            logger.info(
                f"replay [{index}/{len(script.utterances)}] {utterance.label!r}"
                + ("" if utterance.wait_for_bot else "  (barge-in)")
            )
            await self._input.feed(utterance.audio)
            # Silence is what ends the turn; without it VAD keeps waiting.
            await self._input.feed_silence(script.trailing_silence_secs)

            if utterance.wait_for_bot:
                await self._await_reply(utterance.label)

        # Let the final reply drain before tearing the pipeline down.
        await self._feed_silence_until(
            self._input.bot_finished.is_set, script.turn_timeout_secs
        )
        await self._input.feed_silence(0.5)


def load_wav(path: str | Path, target_rate: int = 16000) -> np.ndarray:
    """Read a WAV as mono float32 at ``target_rate``."""
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != target_rate:
        duration = audio.size / rate
        dst_len = max(1, int(round(duration * target_rate)))
        audio = np.interp(
            np.linspace(0, audio.size - 1, dst_len),
            np.linspace(0, audio.size - 1, audio.size),
            audio,
        ).astype(np.float32)
    return audio
