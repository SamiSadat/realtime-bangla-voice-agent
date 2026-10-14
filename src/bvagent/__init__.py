"""Real-time Bangla voice agent with per-stage latency profiling."""

from bvagent.runtime import ensure_cuda_libraries

# Must run before faster-whisper (CTranslate2) loads, so it happens at package
# import rather than in any one entry point.
ensure_cuda_libraries()

__all__ = ["ensure_cuda_libraries"]
