"""TBL CPU and optional CuPy GPU array backends."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ._runtime import ensure_cache_environment
from .errors import OptionalDependencyError, SimulationError, ValidationError

_DEFAULT_GPU_MIN_OPERATIONS = 16_000_000


def _ensure_cupy_cache_dir() -> None:
    """Select a writable CuPy cache without overriding user configuration."""

    ensure_cache_environment(
        "CUPY_CACHE_DIR",
        Path.home() / ".cupy" / "kernel_cache",
        "tbl-cupy-cache",
    )


@dataclass(frozen=True, slots=True)
class ArrayBackend:
    """A validated array backend and its synchronization primitive."""

    name: str
    module: Any
    is_gpu: bool

    def synchronize(self) -> None:
        if self.is_gpu:
            self.module.cuda.get_current_stream().synchronize()


@lru_cache(maxsize=1)
def _cupy_status() -> tuple[Any | None, str | None]:
    """Import and execute a real CuPy matrix operation once per process."""

    _ensure_cupy_cache_dir()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="CUDA path could not be detected.*",
                category=UserWarning,
            )
            import cupy as cp
    except ImportError:
        return None, "CuPy is not installed"
    except Exception as exc:
        return None, f"CuPy could not be imported: {exc}"
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None, "CuPy did not find a CUDA device"
        probe = cp.asarray([[1.0]], dtype=cp.float32)
        value = cp.asnumpy(probe @ probe)
        cp.cuda.get_current_stream().synchronize()
        if value.shape != (1, 1) or float(value[0, 0]) != 1.0:
            return None, "CuPy matrix-operation probe returned an invalid result"
    except Exception as exc:  # CuPy exposes runtime failures through several exception types.
        return None, f"CuPy is installed but its CUDA runtime is unusable: {exc}"
    return cp, None


def clear_backend_cache() -> None:
    """Retry optional GPU discovery after the CUDA environment changes."""

    _cupy_status.cache_clear()


def get_backend(name: str = "auto") -> ArrayBackend:
    """Select a usable NumPy or CuPy backend.

    ``auto`` never returns a merely importable GPU backend: a small matrix
    multiplication is executed first, and any driver/runtime failure safely
    falls back to NumPy. Explicit ``cupy`` requests raise a diagnostic error.
    """

    normalized = name.lower()
    if normalized not in {"auto", "numpy", "cupy", "gpu", "cpu"}:
        raise ValidationError(f"unknown backend {name!r}")
    if normalized in {"numpy", "cpu"}:
        return ArrayBackend("numpy", np, False)
    cp, failure = _cupy_status()
    if cp is None:
        if normalized == "auto":
            return ArrayBackend("numpy", np, False)
        raise OptionalDependencyError(
            f"{failure}. Install a CUDA-compatible CuPy wheel and CUDA runtime."
        )
    return ArrayBackend("cupy", cp, True)


def _validate_shapes(matrix: Any, states: Any) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValidationError("expected a square transfer matrix")
    if states.ndim not in (1, 2):
        raise ValidationError("expected one state or a batch of states")
    if states.shape[-1] != matrix.shape[1]:
        raise ValidationError("state width must equal transfer-matrix width")


def _is_cupy_array(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "cupy"


class BatchPropagator:
    """Repeated coherent propagation with a transfer matrix cached per device.

    The ``auto`` policy keeps small products on CPU and uses a validated GPU
    only once the approximate multiply-add count exceeds a transfer- and
    precision-aware threshold based on ``gpu_min_operations``. Device results
    can be retained with ``return_device=True`` to eliminate round trips in a
    larger GPU pipeline.
    """

    def __init__(
        self,
        transfer_matrix: Any,
        *,
        backend: str = "auto",
        gpu_min_operations: int = _DEFAULT_GPU_MIN_OPERATIONS,
    ) -> None:
        if not hasattr(transfer_matrix, "shape"):
            transfer_matrix = np.asarray(transfer_matrix)
        matrix_shape = tuple(transfer_matrix.shape)
        if len(matrix_shape) != 2 or matrix_shape[0] != matrix_shape[1]:
            raise ValidationError("expected a square transfer matrix")
        if gpu_min_operations < 0:
            raise ValidationError("gpu_min_operations cannot be negative")
        matrix_is_gpu = _is_cupy_array(transfer_matrix)
        self._matrix = None if matrix_is_gpu else np.asarray(transfer_matrix)
        self._matrix_shape = matrix_shape
        self._requested_backend = backend
        self._gpu_min_operations = int(gpu_min_operations)
        self._device_matrix: Any | None = transfer_matrix if matrix_is_gpu else None
        self._last_backend = get_backend("numpy")

    @property
    def backend(self) -> ArrayBackend:
        """Backend used for the most recent propagation."""

        return self._last_backend

    def _select_backend(
        self,
        states_shape: tuple[int, ...],
        *,
        states_are_gpu: bool,
        return_device: bool,
        result_dtype: np.dtype[Any],
    ) -> ArrayBackend:
        requested = self._requested_backend.lower()
        if requested != "auto":
            return get_backend(requested)
        if states_are_gpu or self._matrix is None:
            return get_backend("cupy")
        batch_size = int(states_shape[0]) if len(states_shape) == 2 else 1
        operations = batch_size * self._matrix_shape[0] * self._matrix_shape[1]
        threshold = self._gpu_min_operations
        if return_device:
            threshold = max(1, threshold // 4)
        if result_dtype in (np.dtype(np.float64), np.dtype(np.complex128)):
            threshold *= 16
        if operations < threshold:
            return get_backend("numpy")
        return get_backend("auto")

    def propagate(self, amplitudes: Any, *, return_device: bool = False) -> Any:
        states_shape = (
            tuple(amplitudes.shape) if hasattr(amplitudes, "shape") else np.shape(amplitudes)
        )
        if len(states_shape) not in (1, 2):
            raise ValidationError("expected one state or a batch of states")
        if states_shape[-1] != self._matrix_shape[1]:
            raise ValidationError("state width must equal transfer-matrix width")
        if self._matrix is not None:
            matrix_dtype = self._matrix.dtype
        elif self._device_matrix is not None:
            matrix_dtype = self._device_matrix.dtype
        else:  # pragma: no cover - constructor always stores one representation
            raise SimulationError("propagator has no transfer matrix")
        states_dtype = (
            amplitudes.dtype if hasattr(amplitudes, "dtype") else np.asarray(amplitudes).dtype
        )
        result_dtype = np.dtype(np.result_type(matrix_dtype, states_dtype))
        selected = self._select_backend(
            states_shape,
            states_are_gpu=_is_cupy_array(amplitudes),
            return_device=return_device,
            result_dtype=result_dtype,
        )
        xp = selected.module
        states = xp.asarray(amplitudes)
        if selected.is_gpu:
            if self._device_matrix is None:
                self._device_matrix = xp.asarray(self._matrix)
            matrix = self._device_matrix
        else:
            if self._matrix is None:
                self._matrix = get_backend("cupy").module.asnumpy(self._device_matrix)
            matrix = self._matrix
        _validate_shapes(matrix, states)
        result = states @ matrix.T if states.ndim == 2 else matrix @ states
        self._last_backend = selected
        if selected.is_gpu and not return_device:
            return xp.asnumpy(result)
        return result if selected.is_gpu else np.asarray(result)

    __call__ = propagate

    def synchronize(self) -> None:
        self._last_backend.synchronize()


def propagate_batch(
    transfer_matrix: Any,
    amplitudes: Any,
    *,
    backend: str = "auto",
    return_device: bool = False,
    gpu_min_operations: int = _DEFAULT_GPU_MIN_OPERATIONS,
) -> Any:
    """Propagate one state or a batch, preserving input dtypes when possible."""

    if backend.lower() in {"numpy", "cpu"}:
        if _is_cupy_array(transfer_matrix) or _is_cupy_array(amplitudes):
            cp = get_backend("cupy").module
            matrix = cp.asnumpy(transfer_matrix)
            states = cp.asnumpy(amplitudes)
        else:
            matrix = np.asarray(transfer_matrix)
            states = np.asarray(amplitudes)
        _validate_shapes(matrix, states)
        result = states @ matrix.T if states.ndim == 2 else matrix @ states
        return np.asarray(result)
    propagator = BatchPropagator(
        transfer_matrix,
        backend=backend,
        gpu_min_operations=gpu_min_operations,
    )
    return propagator.propagate(amplitudes, return_device=return_device)


def synchronize(backend: str | ArrayBackend = "auto") -> None:
    """Wait for outstanding work on the selected GPU stream, if any."""

    selected = backend if isinstance(backend, ArrayBackend) else get_backend(backend)
    selected.synchronize()
