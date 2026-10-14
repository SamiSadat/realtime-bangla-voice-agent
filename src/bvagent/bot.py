"""Run the agent as a live WebSocket service.

    python -m bvagent.bot --scenario loan

Then open ``client/index.html`` and allow the microphone. The same pipeline and
the same profiler as the benchmark are used here, so a live call produces the
same per-turn traces — the report at the end of a real conversation is directly
comparable to a replayed one.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
from dataclasses import replace
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.runner import PipelineRunner
from bvagent.agent.backend import Backend
from bvagent.agent.prompts import GREETINGS
from bvagent.config import Settings, load_settings
from bvagent.pipeline import build_agent
from bvagent.profiling.report import LatencyReport
from bvagent.transports import build_transport


async def serve(settings: Settings, *, report_path: Path | None = None) -> LatencyReport:
    transport = build_transport(settings)
    worker, profiler, _context = build_agent(settings, transport, backend=Backend())

    async def _greet(label: str) -> None:
        logger.info(f"caller connected ({label})")
        # A fixed greeting: the call opens at TTS latency rather than waiting
        # for the LLM to compose a hello.
        await worker.queue_frames([TTSSpeakFrame(GREETINGS[settings.scenario])])

    async def _hang_up(label: str) -> None:
        logger.info(f"caller gone ({label})")
        await worker.queue_frames([EndFrame()])

    # Both transports raise on_client_connected/disconnected; LiveKit also
    # raises the participant-level events, and registering only the aliases
    # keeps this handler identical across media planes.
    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        await _greet(settings.transport)

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        await _hang_up(settings.transport)

    if settings.transport == "websocket":

        @transport.event_handler("on_session_timeout")
        async def _on_timeout(_transport, _client):
            logger.warning("session timed out")
            await worker.queue_frames([EndFrame()])

    runner = PipelineRunner(handle_sigint=False)
    logger.info(
        f"ready: {settings.stack_description()['transport']} "
        f"(scenario={settings.scenario}, {settings.audio.in_sample_rate} Hz)"
    )

    loop = asyncio.get_running_loop()
    stopping = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stopping.set)
        except NotImplementedError:
            pass

    run_task = asyncio.create_task(runner.run(worker))
    stop_task = asyncio.create_task(stopping.wait())
    try:
        await asyncio.wait({run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if stopping.is_set() and not run_task.done():
            logger.info("shutting down")
            await worker.queue_frames([EndFrame()])
            try:
                await asyncio.wait_for(run_task, timeout=20)
            except asyncio.TimeoutError:
                await worker.cancel()
    finally:
        stop_task.cancel()

    report = LatencyReport.build(
        profiler.traces, label="live", stack=settings.stack_description()
    )
    if report_path:
        report.save(report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["loan", "booking"], default=None)
    parser.add_argument(
        "--transport",
        choices=["websocket", "livekit"],
        default=None,
        help="Media plane. The pipeline above it is identical either way.",
    )
    parser.add_argument("--room", default=None, help="LiveKit room to join.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--stt-model", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--report", default="runs/live-report.json")
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.scenario:
        settings.scenario = args.scenario
    if args.transport:
        settings.transport = args.transport
    if args.room:
        settings.livekit = replace(settings.livekit, room=args.room)
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.stt_model:
        settings.stt = replace(settings.stt, model=args.stt_model)
    if args.llm_model:
        settings.llm = replace(settings.llm, model=args.llm_model)

    report = asyncio.run(serve(settings, report_path=Path(args.report) if args.report else None))
    print()
    print(report.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
