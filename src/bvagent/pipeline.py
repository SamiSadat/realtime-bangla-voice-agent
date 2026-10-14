"""Assembles the voice pipeline and its instrumentation.

Frame order matters and is the reason this lives in one place:

    transport.input()      audio in, VAD runs here
    stt                    VAD-delimited segments -> Bangla text
    context_in             transcript -> LLM context
    llm                    reply tokens + tool calls
    tts                    sentence-aggregated text -> audio
    transport.output()     audio out, and where interruption takes effect
    context_out            what the bot actually said, back into context

The aggregator pair is split around the whole run so that when a caller barges
in, what lands in the conversation history is what the bot *actually said*
before being cut off, not the full reply it had planned. Getting this wrong
makes the agent refer back to sentences the caller never heard.
"""

from __future__ import annotations

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from bvagent.agent.backend import Backend
from bvagent.agent.prompts import SYSTEM_PROMPTS
from bvagent.agent.tools import build_tools
from bvagent.config import Settings
from bvagent.profiling.observer import TurnProfiler
from bvagent.services.llm import build_llm
from bvagent.services.mms_tts import MMSTTSService
from bvagent.services.whisper_stt import WhisperSTTService


def build_vad(settings: Settings) -> SileroVADAnalyzer:
    """Silero VAD, configured from settings.

    ``stop_secs`` is the dominant latency knob in the pipeline and is charged
    to configuration in the report, not to any model.
    """
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=settings.vad.confidence,
            start_secs=settings.vad.start_secs,
            stop_secs=settings.vad.stop_secs,
            min_volume=settings.vad.min_volume,
        )
    )


def build_agent(settings: Settings, transport, *, backend: Backend | None = None):
    """Build the pipeline, its worker, and the profiler watching it.

    Args:
        settings: Resolved configuration.
        transport: Any Pipecat transport exposing ``input()`` and ``output()``.
        backend: Tool backend; a default in-memory one is created if omitted.

    Returns:
        ``(worker, profiler, context)``.
    """
    backend = backend or Backend()

    stt = WhisperSTTService(config=settings.stt)
    llm = build_llm(settings.llm, system_instruction=SYSTEM_PROMPTS[settings.scenario])
    tts = MMSTTSService(config=settings.tts)

    tools = build_tools(backend, settings.scenario)
    context = LLMContext(tools=tools)
    # VAD lives on the user aggregator: it is the component that turns raw
    # speech boundaries into a decision that the caller's turn is over, which
    # is the decision the endpointing stage of the report measures.
    #
    # The stop strategy is set explicitly rather than left to default. Pipecat
    # defaults to LocalSmartTurnAnalyzerV3, a semantic end-of-turn model, and
    # on Bangla speech it never concludes the turn is over: every utterance
    # comes back INCOMPLETE, the aggregator holds the turn open, and the whole
    # call accumulates into a single user message that is only flushed at
    # hangup. Timeout-based stopping has no language model in it and so has no
    # opinion about Bangla.
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=build_vad(settings),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[
                    SpeechTimeoutUserTurnStopStrategy(
                        user_speech_timeout=settings.vad.turn_stop_timeout
                    )
                ],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    profiler = TurnProfiler(
        stack=settings.stack_description(),
        jsonl_path=settings.profile.jsonl_path if settings.profile.enabled else None,
        on_turn=_live_line if settings.profile.live else None,
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=settings.audio.in_sample_rate,
            audio_out_sample_rate=settings.audio.out_sample_rate,
            # Per-service TTFB/processing metrics. The profiler does not depend
            # on them, but they corroborate its stage attribution.
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[profiler],
    )

    logger.info(f"pipeline built for scenario={settings.scenario}")
    return worker, profiler, context


def _live_line(trace) -> None:
    """One line per turn while the call is happening."""
    ttfa = trace.ttfa_secs
    if ttfa is None:
        if trace.interrupted:
            barge = trace.barge_in_secs
            logger.info(
                f"turn {trace.turn}: interrupted"
                + (f", bot stopped in {barge * 1000:.0f}ms" if barge else "")
            )
        return
    stages = trace.stages()

    def ms(key: str) -> str:
        value = stages.get(key)
        return "-" if value is None else f"{value * 1000:.0f}"

    stt = trace.stt_secs
    logger.info(
        f"turn {trace.turn}: TTFA {ttfa * 1000:.0f}ms "
        f"[turn-end {ms('turn_end_wait')} | llm {ms('llm_ttft')} "
        f"| sent {ms('llm_to_sentence')} | tts {ms('tts_ttfb')} | out {ms('transport')}]"
        + (f" (stt {stt * 1000:.0f}ms, concurrent)" if stt is not None else "")
        + (" INTERRUPTED" if trace.interrupted else "")
    )
