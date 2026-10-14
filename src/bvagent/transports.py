"""Media-plane selection: the same pipeline over a different transport.

A voice agent is two planes that are usually discussed as one thing:

* the **media plane** — carrying audio to and from the caller: WebRTC, an SFU,
  SIP trunking, phone numbers, and whatever it takes to hold N calls at once;
* the **control plane** — VAD, STT, LLM, TTS, turn-taking, interruption.

They are separable, and keeping them separable is the point of this module.
Nothing below the transport changes when you switch: ``build_agent()`` takes any
object exposing ``input()`` and ``output()``, so the pipeline, the tools and the
profiler are identical across media planes. A latency comparison between two
transports is then a controlled experiment rather than two different demos.

That matters for a team already running LiveKit who want to change orchestration
without touching telephony: Pipecat speaks LiveKit as a transport, SIP DTMF
included, so the media plane — rooms, trunks, numbers, scaling — stays exactly
where it is and only the pipeline above it is replaced.
"""

from __future__ import annotations

from typing import Protocol

from loguru import logger

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_transport import TransportParams

from bvagent.config import Settings
from bvagent.serializer import RawPCMSerializer


class VoiceTransport(Protocol):
    """What the pipeline needs from a transport, and nothing more."""

    def input(self) -> FrameProcessor: ...
    def output(self) -> FrameProcessor: ...


def build_websocket_transport(settings: Settings):
    """Raw-PCM WebSocket: the browser client and the simplest telephony bridge.

    No SFU and no signalling, so it is the cheapest way to get a real call
    through the pipeline — and the baseline the other transports are measured
    against.
    """
    from pipecat.transports.websocket.server import (
        SingleClientWebsocketServerParams,
        SingleClientWebsocketServerTransport,
    )

    logger.info(f"transport: websocket ws://{settings.host}:{settings.port}")
    return SingleClientWebsocketServerTransport(
        params=SingleClientWebsocketServerParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=settings.audio.in_sample_rate,
            audio_out_sample_rate=settings.audio.out_sample_rate,
            add_wav_header=False,
            serializer=RawPCMSerializer(sample_rate=settings.audio.in_sample_rate),
            session_timeout=settings.livekit.session_timeout_secs,
        ),
        host=settings.host,
        port=settings.port,
    )


def mint_livekit_token(settings: Settings, *, identity: str, name: str | None = None) -> str:
    """Mint a room-join token.

    The agent needs one to join as a participant like any other. ``agent=True``
    marks it as such in LiveKit's own tooling, which is what keeps it out of
    participant counts and dashboards meant for humans.
    """
    from livekit import api

    cfg = settings.livekit
    grants = api.VideoGrants(
        room_join=True,
        room=cfg.room,
        agent=identity == cfg.identity,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    return (
        api.AccessToken(cfg.api_key, cfg.api_secret)
        .with_identity(identity)
        .with_name(name or identity)
        .with_grants(grants)
        .to_jwt()
    )


def build_livekit_transport(settings: Settings):
    """Join a LiveKit room as the agent participant.

    The same pipeline as the WebSocket path, over LiveKit's media plane. For an
    existing LiveKit deployment this is the whole migration: rooms, SIP trunks
    and phone numbers are untouched, and only what sits above the transport
    changes.
    """
    from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

    cfg = settings.livekit
    if not cfg.api_key or not cfg.api_secret:
        raise ValueError(
            "LiveKit transport needs BV_LIVEKIT_API_KEY and BV_LIVEKIT_API_SECRET. "
            "A local `livekit-server --dev` uses devkey/secret."
        )

    token = mint_livekit_token(settings, identity=cfg.identity, name="Bangla Agent")
    logger.info(f"transport: livekit {cfg.url} room={cfg.room} as {cfg.identity!r}")

    return LiveKitTransport(
        url=cfg.url,
        token=token,
        room_name=cfg.room,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=settings.audio.in_sample_rate,
            audio_out_sample_rate=settings.audio.out_sample_rate,
        ),
    )


BUILDERS = {
    "websocket": build_websocket_transport,
    "livekit": build_livekit_transport,
}


def build_transport(settings: Settings):
    """Construct the transport named by ``settings.transport``."""
    builder = BUILDERS.get(settings.transport)
    if builder is None:
        raise ValueError(
            f"unknown transport {settings.transport!r}; expected one of {sorted(BUILDERS)}"
        )
    return builder(settings)
