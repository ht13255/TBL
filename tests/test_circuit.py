from itertools import permutations
from math import factorial, prod

import numpy as np
import pytest

import openphotontwin as opt


def test_permanent_known_values():
    assert opt.permanent(np.empty((0, 0))) == 1
    assert opt.permanent(np.eye(3)) == pytest.approx(1)
    assert opt.permanent(np.ones((3, 3))) == pytest.approx(6)


def test_balanced_beam_splitter_hom_bunching():
    circuit = opt.LinearOpticalCircuit(2).beam_splitter(0, 1)
    distribution = opt.FockSimulator(circuit).probabilities([1, 1])
    assert distribution.probabilities[(1, 1)] == pytest.approx(0.0, abs=1e-12)
    assert distribution.probabilities[(2, 0)] == pytest.approx(0.5)
    assert distribution.probabilities[(0, 2)] == pytest.approx(0.5)
    assert distribution.loss_probability == pytest.approx(0.0)


def test_distinguishable_photons_have_classical_coincidences():
    circuit = opt.LinearOpticalCircuit(2).beam_splitter(0, 1)
    distribution = opt.FockSimulator(circuit).probabilities([1, 1], indistinguishability=0)
    assert distribution.probabilities[(1, 1)] == pytest.approx(0.5)
    assert distribution.probabilities[(2, 0)] == pytest.approx(0.25)
    assert distribution.probabilities[(0, 2)] == pytest.approx(0.25)


def test_wavepacket_partial_indistinguishability_matches_hom_helper():
    first = opt.Wavepacket(temporal_width=20e-12)
    second = opt.Wavepacket(arrival_time=20e-12, temporal_width=20e-12)
    expected, _, _ = opt.hom_probabilities(first, second)
    circuit = opt.LinearOpticalCircuit(2).beam_splitter(0, 1)
    distribution = opt.FockSimulator(circuit).probabilities([1, 1], wavepackets=[first, second])
    assert distribution.probabilities[(1, 1)] == pytest.approx(expected)


def _brute_partial_probability(transfer, inputs, outputs, overlap):
    """Independent double-permutation definition used as a falsification check."""

    total = 0.0 + 0.0j
    for sigma in permutations(range(len(inputs))):
        for rho in permutations(range(len(inputs))):
            spatial = np.prod(
                [
                    transfer[outputs[j], inputs[sigma[j]]]
                    * np.conj(transfer[outputs[j], inputs[rho[j]]])
                    for j in range(len(inputs))
                ]
            )
            internal = np.prod([overlap[rho[j], sigma[j]] for j in range(len(inputs))])
            total += spatial * internal
    output_occupation = np.bincount(outputs, minlength=transfer.shape[0])
    output_factor = prod(factorial(int(value)) for value in output_occupation)
    return float((total / output_factor).real)


def test_three_photon_partial_distinguishability_matches_double_sum():
    rng = np.random.default_rng(27)
    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    unitary, _ = np.linalg.qr(raw)
    internal_vectors = np.array(
        [[1, 0, 0], [0.6, 0.8, 0], [0.3, 0.4, np.sqrt(0.75)]], dtype=complex
    )
    overlap = internal_vectors.conj() @ internal_vectors.T
    simulator = opt.FockSimulator(opt.LinearOpticalCircuit.from_transfer_matrix(unitary))
    distribution = simulator.probabilities([1, 1, 1], overlap_matrix=overlap)
    for occupation, probability in distribution.probabilities.items():
        outputs = [mode for mode, count in enumerate(occupation) for _ in range(count)]
        expected = _brute_partial_probability(unitary, [0, 1, 2], outputs, overlap)
        assert probability == pytest.approx(expected, abs=1e-11)
    assert sum(distribution.probabilities.values()) == pytest.approx(1.0)


def test_partial_overlap_validation_and_explicit_mean_fallback():
    circuit = opt.LinearOpticalCircuit(1)
    invalid = np.ones((8, 8), dtype=complex)
    invalid[0, 1] = 2
    invalid[1, 0] = 2
    with pytest.raises(opt.ValidationError):
        opt.FockSimulator(circuit).probabilities([8], overlap_matrix=invalid)

    overlap = np.full((8, 8), np.sqrt(0.8), dtype=complex)
    np.fill_diagonal(overlap, 1)
    with pytest.raises(opt.SimulationError):
        opt.FockSimulator(circuit).probabilities([8], overlap_matrix=overlap)
    result = opt.FockSimulator(circuit).probabilities(
        [8], overlap_matrix=overlap, partial_method="mean"
    )
    assert result.survival_probability == pytest.approx(1.0)


def test_loss_is_reported_instead_of_renormalized():
    circuit = opt.LinearOpticalCircuit(2).loss(0.8, [0])
    distribution = opt.FockSimulator(circuit).probabilities([1, 0])
    assert distribution.probabilities[(1, 0)] == pytest.approx(0.8)
    assert distribution.loss_probability == pytest.approx(0.2)


def test_circuit_propagation_and_dynamic_time_bin():
    circuit = opt.LinearOpticalCircuit(2)
    circuit.add(opt.DynamicBeamSplitter(0, 1, schedule={0: 0.0, 2: 1.0}))
    early = circuit.propagate([1, 0], time_bin=0)
    late = circuit.propagate([1, 0], time_bin=2)
    assert abs(early[0]) == pytest.approx(1)
    assert abs(late[1]) == pytest.approx(1)


def test_spectral_transfer_routes_each_photon_at_its_own_wavelength():
    wavelengths = np.array([1500e-9, 1600e-9])
    identity = np.eye(2, dtype=complex)
    swap = np.array([[0, 1], [1, 0]], dtype=complex)
    circuit = opt.LinearOpticalCircuit.from_spectral_transfer(
        wavelengths, np.stack([identity, swap])
    )
    simulator = opt.FockSimulator(circuit)
    blue = simulator.probabilities([1, 0], wavelengths=[1500e-9])
    red = simulator.probabilities([1, 0], wavelengths=[1600e-9])
    assert blue.probabilities[(1, 0)] == pytest.approx(1)
    assert red.probabilities[(0, 1)] == pytest.approx(1)
    midpoint = circuit.transfer_matrix(wavelength=1550e-9)
    assert midpoint == pytest.approx(0.5 * (identity + swap))
    with pytest.raises(opt.ValidationError):
        circuit.transfer_matrix(wavelength=1400e-9)


def test_vacuum_distribution_remains_normalized():
    result = opt.FockSimulator(opt.LinearOpticalCircuit(2)).probabilities([0, 0])
    assert result.probabilities == {(0, 0): pytest.approx(1.0)}
    assert result.loss_probability == pytest.approx(0.0)


def test_fock_sampling_and_limits():
    distribution = opt.FockSimulator(opt.LinearOpticalCircuit(1)).probabilities([1])
    assert distribution.sample(50, seed=1) == {(1,): 50, "loss": 0}
    with pytest.raises(opt.SimulationError):
        opt.FockSimulator(opt.LinearOpticalCircuit(1), max_photons=2).probabilities([3])


def test_active_matrix_is_rejected():
    with pytest.raises(opt.ValidationError):
        opt.MatrixComponent(np.eye(2) * 1.1)
