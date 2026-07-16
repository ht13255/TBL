"""Stable microbenchmarks for performance-regression checks.

Run with ``python benchmarks/benchmark_core.py``. Results are JSON so releases
can compare the same workload without a benchmark framework dependency.
"""

from __future__ import annotations

import gc
import json
import statistics
import time

import numpy as np

import tbl as opt


def timed(function, *, repeats=5, warmups=1, iterations=1):
    for _ in range(warmups):
        function()
    samples = []
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


def paired_timed(first, second, *, repeats=7, warmups=2, iterations=10):
    """Interleave two functions so BLAS scheduling affects both fairly."""

    for _ in range(warmups):
        first()
        second()
    samples = [[], []]
    gc.collect()
    gc.disable()
    try:
        for repeat in range(repeats):
            functions = (first, second)
            elapsed = [0.0, 0.0]
            for iteration in range(iterations):
                order = (0, 1) if (repeat + iteration) % 2 == 0 else (1, 0)
                for index in order:
                    started = time.perf_counter()
                    functions[index]()
                    elapsed[index] += time.perf_counter() - started
            for index in (0, 1):
                samples[index].append(elapsed[index] / iterations)
    finally:
        gc.enable()
    medians = [statistics.median(values) for values in samples]
    return {
        "first_median_s": medians[0],
        "second_median_s": medians[1],
        "second_over_first": medians[1] / medians[0],
        "first_samples_s": samples[0],
        "second_samples_s": samples[1],
    }


def random_unitary(rng, modes):
    raw = rng.normal(size=(modes, modes)) + 1j * rng.normal(size=(modes, modes))
    unitary, _ = np.linalg.qr(raw)
    return unitary


def main():
    rng = np.random.default_rng(20260716)
    results = {"version": opt.__version__}

    matrix = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    results["permanent_n9_x20"] = timed(
        lambda: [opt.permanent(matrix) for _ in range(20)], repeats=7, iterations=3
    )

    fock_circuit = opt.LinearOpticalCircuit.from_transfer_matrix(random_unitary(rng, 7))
    fock = opt.FockSimulator(fock_circuit)
    results["fock_5ph_7mode"] = timed(
        lambda: fock.probabilities([1, 1, 1, 1, 1, 0, 0]), repeats=5, iterations=5
    )

    vectors = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    gram = vectors.conj() @ vectors.T
    partial = opt.FockSimulator(
        opt.LinearOpticalCircuit.from_transfer_matrix(random_unitary(rng, 5))
    )
    results["partial_4ph_5mode"] = timed(
        lambda: partial.probabilities([1, 1, 1, 1, 0], overlap_matrix=gram),
        repeats=3,
        iterations=3,
    )

    loop = opt.CoherentTimeBinLoop(
        256,
        1e-9,
        round_trip_bins=3,
        reflectivity=0.45,
        round_trip_transmission=0.96,
        phase=0.17,
    )
    results["coherent_loop_256"] = timed(loop.transfer_matrix, repeats=5, iterations=3)

    component_circuit = opt.LinearOpticalCircuit(64)
    for index in range(210):
        mode = index % 64
        if index % 3 == 0:
            component_circuit.beam_splitter(
                mode, (mode + 1) % 64, reflectivity=0.2 + 0.6 * ((index % 11) / 10)
            )
        elif index % 3 == 1:
            component_circuit.phase(mode, 0.013 * index)
        else:
            component_circuit.loss(0.997, [mode])

    def dense_component_transfer():
        transfer = np.eye(component_circuit.modes, dtype=complex)
        for component in component_circuit.components:
            if isinstance(component, opt.BeamSplitter):
                local = component.unitary(component_circuit.modes)
            elif isinstance(component, opt.PhaseShifter):
                local = np.eye(component_circuit.modes, dtype=complex)
                selected = (
                    range(component_circuit.modes)
                    if component.modes is None
                    else component.modes
                )
                for mode in selected:
                    local[mode, mode] *= np.exp(1j * component.phase)
            else:
                local = np.eye(component_circuit.modes, dtype=complex)
                selected = (
                    range(component_circuit.modes)
                    if component.modes is None
                    else component.modes
                )
                for mode in selected:
                    local[mode, mode] *= np.sqrt(component.transmission)
            transfer = local @ transfer
        return transfer

    optimized_component_transfer = component_circuit.transfer_matrix
    if not np.allclose(dense_component_transfer(), optimized_component_transfer()):
        raise RuntimeError("component transfer optimization changed numerical results")
    results["circuit_64mode_210components"] = paired_timed(
        dense_component_transfer,
        optimized_component_transfer,
        repeats=7,
        iterations=3,
    )

    source = opt.SinglePhotonSource(repetition_rate=20e6, p_single=0.7, p_double=0.01)
    detector = opt.DetectorArray(
        {0: opt.SNSPD(efficiency=0.82, jitter=15e-12, dead_time=30e-9)}
    )
    twin = opt.DigitalTwin(source, [opt.DelayLine(5e-9)], detector)
    results["digital_twin_50k"] = timed(lambda: twin.run(50_000, seed=8), repeats=3)

    batch_matrix = random_unitary(rng, 128)
    batch_states = rng.normal(size=(2048, 128)) + 1j * rng.normal(size=(2048, 128))

    def legacy_cpu_batch():
        matrix = np.asarray(batch_matrix, dtype=complex)
        states = np.asarray(batch_states, dtype=complex)
        return np.asarray(states @ matrix.T)

    def optimized_cpu_batch():
        return opt.propagate_batch(batch_matrix, batch_states, backend="numpy")

    results["batch_cpu_2048x128"] = paired_timed(
        legacy_cpu_batch,
        optimized_cpu_batch,
        repeats=9,
        iterations=10,
    )
    try:
        opt.get_backend("cupy")
    except opt.OptionalDependencyError:
        results["batch_gpu_2048x128"] = {"available": False}
    else:
        gpu_propagator = opt.BatchPropagator(batch_matrix, backend="cupy")

        def propagate_gpu():
            gpu_propagator(batch_states, return_device=True)
            gpu_propagator.synchronize()

        results["batch_gpu_2048x128"] = timed(
            propagate_gpu,
            repeats=7,
            warmups=2,
            iterations=10,
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
