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


def test_fock_sampling_and_limits():
    distribution = opt.FockSimulator(opt.LinearOpticalCircuit(1)).probabilities([1])
    assert distribution.sample(50, seed=1) == {(1,): 50, "loss": 0}
    with pytest.raises(opt.SimulationError):
        opt.FockSimulator(opt.LinearOpticalCircuit(1), max_photons=2).probabilities([3])


def test_active_matrix_is_rejected():
    with pytest.raises(opt.ValidationError):
        opt.MatrixComponent(np.eye(2) * 1.1)
