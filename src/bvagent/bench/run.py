"""Run a scripted call through the real pipeline and report its latency.

This is the measurement entry point. Nothing here is a simulation: it builds
the same pipeline the live agent uses, with the same observer attached, and
differs only in that the caller is a WAV file rather than a person.

    python -m bvagent.bench.run --scenario loan --calls 3

Each call writes a JSON report and the bot's audio, so a number can always be
checked against what the agent actually said.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.runner import PipelineRunner

from bvagent.agent.backend import Backend
from bvagent.agent.prompts import GREETINGS
from bvagent.bench.corpus import Corpus
from bvagent.bench.replay import ReplayScript, ReplayTransport, Utterance
from bvagent.config import Settings, load_settings
from bvagent.pipeline import build_agent
from bvagent.profiling.report import LatencyReport
from bvagent.profiling.trace import TurnTrace
from bvagent.services.llm import warm_llm


def build_script(settings: Settings, *, with_barge_in: bool, repeats: int) -> ReplayScript:
    """Turn the corpus into a scripted call.

    Args:
        with_barge_in: Append a turn that deliberately talks over the bot, so
            the run measures how fast the agent yields the floor.
        repeats: How many times to run the conversation, for more samples.
    """
    corpus = Corpus(sample_rate=settings.audio.in_sample_rate)
    entries = corpus.scenario(settings.scenario)

    utterances: list[Utterance] = []
    for _ in range(repeats):
        for entry in entries:
            utterances.append(
                Utterance(audio=entry.audio, label=entry.name, wait_for_bot=True, delay_secs=0.4)
            )

    if with_barge_in:
        # The turn before the barge-in must not wait for its reply, or the
        # reply is over before the interruption is fed and there is nothing
        # left to interrupt.
        if utterances:
            utterances[-1] = replace(utterances[-1], wait_for_bot=False)
        barge = corpus.barge_in()
        # Timed from the bot starting to speak, so the interruption lands in
        # the middle of a reply on every run regardless of its length.
        utterances.append(
            Utterance(audio=barge.audio, label="barge_in", barge_in_after=0.8)
        )

    return ReplayScript(utterances=utterances, sample_rate=settings.audio.in_sample_rate)


async def run_call(
    settings: Settings,
    *,
    with_barge_in: bool = True,
    repeats: int = 1,
    out_dir: Path = Path("runs"),
    label: str = "replay",
) -> LatencyReport:
    """Run one scripted call end to end and return its report."""
    # Whisper and MMS warm themselves on setup; the LLM is a separate process
    # that may have evicted the model since the last run, so warm it too. An
    # unwarmed 27B cold load put 121s into a single turn of an earlier run.
    await warm_llm(settings.llm)

    script = build_script(settings, with_barge_in=with_barge_in, repeats=repeats)
    transport = ReplayTransport(script)

    worker, profiler, _context = build_agent(settings, transport, backend=Backend())

    # The greeting is spoken from a fixed string rather than generated, so the
    # call opens at TTS latency instead of LLM+TTS latency. It is also not a
    # response to anything, so it is not a measurable turn and the profiler
    # correctly ignores it.
    @worker.event_handler("on_pipeline_started")
    async def _greet(*_args):
        await worker.queue_frames([TTSSpeakFrame(GREETINGS[settings.scenario])])

    runner = PipelineRunner(handle_sigint=False)

    started = time.monotonic()
    runner_task = asyncio.create_task(runner.run(worker))
    try:
        await transport.run_script()
    finally:
        await worker.queue_frames([EndFrame()])
        try:
            await asyncio.wait_for(runner_task, timeout=60)
        except asyncio.TimeoutError:
            logger.warning("pipeline did not shut down cleanly; cancelling")
            await worker.cancel()
            runner_task.cancel()
    elapsed = time.monotonic() - started

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    audio_path = transport.output().save(out_dir / f"{label}-{stamp}.wav")

    report = LatencyReport.build(
        profiler.traces, label=label, stack=settings.stack_description()
    )
    report.save(out_dir / f"{label}-{stamp}.json")

    logger.info(f"call finished in {elapsed:.1f}s; bot audio -> {audio_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["loan", "booking"], default=None)
    parser.add_argument("--calls", type=int, default=1, help="How many scripted calls to run.")
    parser.add_argument(
        "--repeats", type=int, default=1, help="Times to repeat the conversation within a call."
    )
    parser.add_argument("--no-barge-in", action="store_true", help="Skip the interruption turn.")
    parser.add_argument("--stt-model", default=None, help="Override the Whisper model.")
    parser.add_argument("--llm-model", default=None, help="Override the LLM model.")
    parser.add_argument("--vad-stop-secs", type=float, default=None)
    parser.add_argument("--out", default="runs", help="Directory for reports and audio.")
    parser.add_argument("--label", default=None)
    parser.add_argument("--markdown", action="store_true", help="Also print a markdown table.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final report.")
    args = parser.parse_args(argv)

    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    settings = load_settings()
    if args.scenario:
        settings.scenario = args.scenario
    if args.stt_model:
        settings.stt = replace(settings.stt, model=args.stt_model)
    if args.llm_model:
        settings.llm = replace(settings.llm, model=args.llm_model)
    if args.vad_stop_secs is not None:
        settings.vad = replace(settings.vad, stop_secs=args.vad_stop_secs)

    label = args.label or f"{settings.scenario}-{settings.stt.model}"
    out_dir = Path(args.out)

    traces: list[TurnTrace] = []
    for index in range(args.calls):
        logger.info(f"=== call {index + 1}/{args.calls} ===")
        report = asyncio.run(
            run_call(
                settings,
                with_barge_in=not args.no_barge_in,
                repeats=args.repeats,
                out_dir=out_dir,
                label=f"{label}-call{index + 1}",
            )
        )
        traces.extend(report.traces)

    combined = LatencyReport.build(traces, label=label, stack=settings.stack_description())
    combined.save(out_dir / f"{label}-summary.json")
    print()
    print(combined.to_text())
    if args.markdown:
        print()
        print(combined.to_markdown())
    return 0 if combined.measured else 1


if __name__ == "__main__":
    raise SystemExit(main())
