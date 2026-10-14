"""Process-level setup that has to happen before the model libraries load.

Importing :mod:`bvagent` runs :func:`ensure_cuda_libraries`, so every entry
point — the server, the benchmark, a test — gets the same environment without
anyone having to remember an ``LD_LIBRARY_PATH`` incantation.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path

# CTranslate2 (under faster-whisper) links against the CUDA 12 runtime by
# soname and searches only the system loader path. PyTorch installed from PyPI
# ships those same libraries inside the `nvidia-*` wheels, where the loader
# never looks, so a machine with a working GPU and a pip-installed torch fails
# with "Library libcublas.so.12 is not found" and silently falls back to CPU.
# Loading them RTLD_GLOBAL up front puts the symbols in the process before
# CTranslate2 asks for them.
_CUDA_LIBRARY_GLOBS = (
    "nvidia/cublas/lib/libcublas.so.*",
    "nvidia/cublas/lib/libcublasLt.so.*",
    "nvidia/cudnn/lib/libcudnn.so.*",
    "nvidia/cudnn/lib/libcudnn_ops.so.*",
    "nvidia/cudnn/lib/libcudnn_cnn.so.*",
    "nvidia/cudnn/lib/libcudnn_engines_precompiled.so.*",
    "nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so.*",
    "nvidia/cudnn/lib/libcudnn_heuristic.so.*",
    "nvidia/cudnn/lib/libcudnn_graph.so.*",
    "nvidia/cudnn/lib/libcudnn_adv.so.*",
)

_loaded = False


def site_packages_dirs() -> list[Path]:
    return [Path(p) for p in sys.path if p.endswith("site-packages") and Path(p).is_dir()]


def ensure_cuda_libraries() -> list[str]:
    """Preload the CUDA libraries CTranslate2 needs. Returns what was loaded.

    Best-effort and idempotent: on a CPU-only machine there is nothing to find
    and nothing to do, which is why failures here are silent rather than fatal.
    """
    global _loaded
    if _loaded or os.environ.get("BV_SKIP_CUDA_PRELOAD"):
        return []

    loaded: list[str] = []
    for directory in site_packages_dirs():
        for pattern in _CUDA_LIBRARY_GLOBS:
            for path in sorted(glob.glob(str(directory / pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(path))
                except OSError:
                    # A library that will not load is one CTranslate2 would not
                    # have found either; let it report the real error.
                    continue
    _loaded = True
    return loaded


def describe_device() -> dict[str, str]:
    """What the process will actually run on, for the run's provenance."""
    info = {"device": "cpu", "name": "cpu", "torch": "not installed"}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info["device"] = "cuda"
            info["name"] = properties.name
            info["vram_gb"] = f"{properties.total_memory / 1e9:.0f}"
    except Exception:
        pass
    return info
