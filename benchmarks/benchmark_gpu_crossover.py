"""Measure CPU/GPU crossover including transfer and device-resident modes."""

from __future__ import annotations

import gc
import json
import statistics
import time

import numpy as np

import tbl as opt


def timed(function, *, iterations: int, repeats: int = 7) -> dict[str, object]:
    function()
    samples: list[float] = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            for _ in range(iterations):
                function()
            samples.append((time.perf_counter() - started) / iterations)
    finally:
        gc.enable()
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "samples_s": samples,
    }


def main() -> None:
    backend = opt.get_backend("cupy")
    cp = backend.module
    rng = np.random.default_rng(20260716)
    configurations = [
        (64, 256),
        (64, 1024),
        (64, 4096),
        (128, 256),
        (128, 1024),
        (128, 2048),
        (128, 8192),
        (256, 256),
        (256, 1024),
        (256, 4096),
    ]
    results: dict[str, object] = {
        "version": opt.__version__,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "compute_capability": cp.cuda.Device().compute_capability,
        "cases": [],
    }
    cases: list[dict[str, object]] = []
    for dtype in (np.complex64, np.complex128):
        real_dtype = np.float32 if dtype == np.complex64 else np.float64
        for width, batch in configurations:
            matrix = (
                rng.normal(size=(width, width)).astype(real_dtype)
                + 1j * rng.normal(size=(width, width)).astype(real_dtype)
            ).astype(dtype, copy=False)
            states = (
                rng.normal(size=(batch, width)).astype(real_dtype)
                + 1j * rng.normal(size=(batch, width)).astype(real_dtype)
            ).astype(dtype, copy=False)
            propagator = opt.BatchPropagator(matrix, backend="cupy")
            device_states = cp.asarray(states)
            iterations = 10 if width == 64 else 5 if width == 128 else 3

            def cpu(matrix=matrix, states=states) -> np.ndarray:
                return opt.propagate_batch(matrix, states, backend="numpy")

            def gpu_roundtrip(propagator=propagator, states=states) -> np.ndarray:
                return propagator(states, return_device=False)

            def gpu_upload_only(propagator=propagator, states=states):
                result = propagator(states, return_device=True)
                propagator.synchronize()
                return result

            def gpu_resident(propagator=propagator, device_states=device_states):
                result = propagator(device_states, return_device=True)
                propagator.synchronize()
                return result

            expected = cpu()
            measured = gpu_roundtrip()
            tolerance = 2e-4 if dtype == np.complex64 else 2e-11
            if not np.allclose(measured, expected, rtol=tolerance, atol=tolerance):
                raise RuntimeError("CPU/GPU numerical-equivalence check failed")
            case = {
                "dtype": np.dtype(dtype).name,
                "width": width,
                "batch": batch,
                "multiply_count": batch * width * width,
                "cpu": timed(cpu, iterations=iterations),
                "gpu_roundtrip": timed(gpu_roundtrip, iterations=iterations),
                "gpu_upload_only": timed(gpu_upload_only, iterations=iterations),
                "gpu_resident": timed(gpu_resident, iterations=iterations),
            }
            cases.append(case)
            cp.get_default_memory_pool().free_all_blocks()
    results["cases"] = cases
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
