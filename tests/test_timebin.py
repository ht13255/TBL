import numpy as np
import pytest

import tbl as opt


def _slow_loop_transfer(loop, seed):
    reflectivities = loop._reflectivity_trace()
    phases = loop.phase_trace(seed=seed)
    columns = []
    for column in range(loop.time_bins):
        state = np.zeros(loop.time_bins, dtype=complex)
        state[column] = 1
        output, _ = loop._propagate_with_traces(state, reflectivities, phases)
        columns.append(output)
    return np.column_stack(columns)


def test_open_coupler_is_identity_and_closed_coupler_is_delay():
    open_loop = opt.CoherentTimeBinLoop(5, 1e-9, reflectivity=0)
    assert open_loop.transfer_matrix() == pytest.approx(np.eye(5))

    closed_loop = opt.CoherentTimeBinLoop(5, 1e-9, round_trip_bins=1, reflectivity=1)
    expected = np.zeros((5, 5), dtype=complex)
    expected[np.arange(1, 5), np.arange(4)] = -1
    assert closed_loop.transfer_matrix() == pytest.approx(expected)


def test_loop_paths_interfere_in_same_time_bin():
    loop = opt.CoherentTimeBinLoop(5, 1e-9, reflectivity=0.5)
    # y_1 = sqrt(T) x_1 - R x_0, so this pulse pair cancels time bin 1.
    inputs = np.array([1, 1 / np.sqrt(2), 0, 0, 0], dtype=complex)
    output = loop.propagate(inputs)
    assert output[1] == pytest.approx(0.0, abs=1e-12)
    assert np.sum(abs(output) ** 2) <= np.sum(abs(inputs) ** 2) + 1e-12


def test_round_trip_loss_and_truncated_tail_are_subunitary():
    loop = opt.CoherentTimeBinLoop(
        8,
        2e-9,
        round_trip_bins=2,
        reflectivity=0.4,
        round_trip_transmission=0.8,
        coupler_transmission=0.95,
    )
    matrix = loop.transfer_matrix()
    assert np.linalg.svd(matrix, compute_uv=False).max() <= 1 + 1e-12
    distribution = opt.FockSimulator(loop.circuit()).probabilities([1] + [0] * 7)
    assert 0 < distribution.survival_probability < 1

    closed = opt.CoherentTimeBinLoop(4, 1e-9, reflectivity=1)
    detailed = closed.simulate([0, 0, 0, 1])
    assert detailed.output_probability == pytest.approx(0)
    assert detailed.residual_loop_probability == pytest.approx(1)
    assert detailed.physical_loss_probability == pytest.approx(0)


def test_correlated_phase_noise_is_seeded_and_has_expected_lag():
    loop = opt.CoherentTimeBinLoop(
        3000,
        1e-9,
        phase_noise_std=0.2,
        phase_correlation_time=20e-9,
    )
    first = loop.phase_trace(seed=3)
    second = loop.phase_trace(seed=3)
    other = loop.phase_trace(seed=4)
    assert first == pytest.approx(second)
    assert not np.allclose(first, other)
    measured = np.corrcoef(first[:-1], first[1:])[0, 1]
    assert measured == pytest.approx(np.exp(-1 / 20), abs=0.03)


def test_dynamic_coupler_schedule_and_switching_time():
    loop = opt.CoherentTimeBinLoop(
        6,
        1e-9,
        reflectivity={0: 0.0, 2: 1.0},
        switching_time=2e-9,
    )
    matrix = loop.transfer_matrix()
    assert matrix[0, 0] == pytest.approx(1)
    assert abs(matrix[2, 2]) == pytest.approx(1)


def test_vectorized_loop_transfer_matches_basis_by_basis_recurrence():
    loop = opt.CoherentTimeBinLoop(
        17,
        0.8e-9,
        round_trip_bins=3,
        reflectivity={0: 0.1, 4: 0.7, 11: 0.35},
        switching_time=1.6e-9,
        round_trip_transmission=0.91,
        coupler_transmission=0.96,
        phase=lambda index, time: 0.07 * index + 1e7 * time,
        phase_noise_std=0.02,
        phase_correlation_time=4e-9,
    )
    assert loop.transfer_matrix(seed=91) == pytest.approx(
        _slow_loop_transfer(loop, seed=91), abs=2e-13
    )
