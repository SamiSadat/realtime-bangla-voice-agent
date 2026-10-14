"""LLM wiring for the voice agent.

Any OpenAI-compatible endpoint works — Ollama, vLLM, or a hosted API — so the
same pipeline and the same profiler can compare a local model against a remote
one without a code change. That comparison is the point: on a voice call the
question is not which model is smartest but which one starts talking soonest.
"""

from __future__ import annotations

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService

from bvagent.config import LLMConfig


def build_llm(
    config: LLMConfig | None = None, *, system_instruction: str | None = None
) -> OpenAILLMService:
    """Construct the LLM service for a voice pipeline.

    Args:
        config: Endpoint, model and sampling settings.
        system_instruction: The agent's system prompt. Set on the service
            rather than pushed as a context message, so it is not re-sent as
            conversation history on every turn.
    """
    cfg = config or LLMConfig()
    extra: dict[str, object] = {}

    if cfg.disable_thinking:
        # Reasoning tokens are dead air: the caller hears silence while the
        # model thinks, and none of it is ever spoken. On Qwen3 via Ollama a
        # single turn produced ~300 reasoning tokens before the first spoken
        # one. `reasoning_effort` is the switch the OpenAI-compatible endpoint
        # honours; Ollama's own `think` field is ignored on that route, and the
        # `/no_think` prompt suffix is not respected by every build.
        extra["reasoning_effort"] = "none"

    logger.info(f"LLM: {cfg.model} @ {cfg.base_url} (thinking={'off' if cfg.disable_thinking else 'on'})")

    return OpenAILLMService(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        settings=OpenAILLMService.Settings(
            model=cfg.model,
            system_instruction=system_instruction,
            temperature=cfg.temperature,
            # Capped because an over-long reply is latency the caller sits
            # through, and a voice turn that needs more than this has usually
            # gone wrong in a way more tokens will not fix.
            max_tokens=cfg.max_tokens,
            extra=extra,
        ),
    )


async def warm_llm(config: LLMConfig | None = None) -> float | None:
    """Send one throwaway request so the model is resident before the call.

    A server that has evicted the model reloads it on the next request, and for
    a 27B local model that is tens of seconds — paid by whichever caller happens
    to arrive first. It lands between the turn closing and the LLM accepting the
    request, which is why the report names that span rather than folding it into
    inference time.

    Returns how long the warm-up took, or None if the endpoint was unreachable.
    """
    import time

    import httpx

    cfg = config or LLMConfig()
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{cfg.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                json={
                    "model": cfg.model,
                    "messages": [{"role": "user", "content": "হ্যালো"}],
                    "max_tokens": 1,
                    **({"reasoning_effort": "none"} if cfg.disable_thinking else {}),
                },
            )
            response.raise_for_status()
    except Exception as exc:
        logger.warning(f"LLM warm-up failed ({exc}); first turn may pay a cold load")
        return None
    elapsed = time.monotonic() - started
    logger.info(f"LLM warm in {elapsed:.1f}s")
    return elapsed
