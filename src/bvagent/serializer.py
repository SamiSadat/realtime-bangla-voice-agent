"""Raw 16-bit PCM over the wire.

Pipecat's default WebSocket serializer is protobuf, which means any client has
to carry a protobuf runtime. A phone gateway does not send protobuf and neither
does a browser's audio graph, so this speaks the same thing both of them do:
little-endian 16-bit mono PCM, and nothing else.

Keeping the wire format this thin also keeps it out of the latency budget — the
transport row in the report stays a measurement of the network rather than of a
serializer.
"""

from __future__ import annotations

from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.serializers.base_serializer import FrameSerializer


class RawPCMSerializer(FrameSerializer):
    """Serializes audio frames as bare PCM byte strings.

    Args:
        sample_rate: Rate the client sends and expects. Both directions use it,
            so the browser and the pipeline agree without negotiation.
    """

    def __init__(self, *, sample_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self._sample_rate = sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if self.should_ignore_frame(frame):
            return None
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        # Everything else is pipeline bookkeeping the client has no use for.
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if not isinstance(data, bytes) or not data:
            return None
        return InputAudioRawFrame(
            audio=data,
            sample_rate=self._sample_rate,
            num_channels=1,
        )
