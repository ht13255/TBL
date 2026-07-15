"""CPU and optional CuPy GPU array backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import OptionalDependencyError, ValidationError


@dataclass(frozen=True, slots=True)
class ArrayBackend:
    name: str
    module: Any
    is_gpu: bool


def get_backend(name: str = "auto") -> ArrayBackend:
    """Select ``numpy`` or optional ``cupy`` without importing CuPy eagerly."""

    normalized = name.lower()
    if normalized not in {"auto", "numpy", "cupy", "gpu", "cpu"}:
        raise ValidationError(f"unknown backend {name!r}")
    if normalized in {"numpy", "cpu"}:
        return ArrayBackend("numpy", np, False)
    try:
        import cupy as cp
    except ImportError as exc:
        if normalized == "auto":
            return ArrayBackend("numpy", np, False)
        raise OptionalDependencyError(
            "CuPy is not installed; install openphotontwin[gpu] with a CUDA-compatible build"
        ) from exc
    return ArrayBackend("cupy", cp, True)


def propagate_batch(
    transfer_matrix: np.ndarray,
    amplitudes: np.ndarray,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Accelerate a batch of coherent propagations on GPU when available."""

    selected = get_backend(backend)
    xp = selected.module
    matrix = xp.asarray(transfer_matrix, dtype=complex)
    states = xp.asarray(amplitudes, dtype=complex)
    if states.ndim not in (1, 2) or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError("expected a square matrix and one state or a batch of states")
    if states.shape[-1] != matrix.shape[1]:
        raise ValidationError("state width must equal transfer-matrix width")
    result = states @ matrix.T if states.ndim == 2 else matrix @ states
    return selected.module.asnumpy(result) if selected.is_gpu else np.asarray(result)
