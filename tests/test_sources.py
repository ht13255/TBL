from collections import defaultdict

import numpy as np
import pytest

import tbl
from tbl.experiments import _partial_two_mode_distribution


def degenerate_source(**changes):
    parameters = {
        "repetition_rate": 80e6,
        "mean_pairs": 0.2,
        "schmidt_number": 1.0,
        "signal_wavepacket": tbl.Wavepacket(wavelength=1550e-9),
        "idler_wavepacket": tbl.Wavepacket(wavelength=1550e-9),
        "pump_wavelength": 775e-9,
    }
    parameters.update(changes)
    return tbl.SPDCSource(**parameters)


def test_single_mode_pair_distribution_is_geometric_with_exact_tail():
    source = degenerate_source(mean_pairs=0.2, schmidt_number=1.0)
    distribution = source.pair_number_distribution(12)
    n = distribution.pair_numbers
    expected = source.mean_pairs**n / (1 + source.mean_pairs) ** (n + 1)
    expected_tail = (source.mean_pairs / (1 + source.mean_pairs)) ** 13
    assert distribution.probabilities == pytest.approx(expected, rel=1e-13)
    assert distribution.tail_probability == pytest.approx(expected_tail, rel=2e-8)
    assert np.sum(distribution.probabilities) + distribution.tail_probability == pytest.approx(1)
    assert distribution.variance == pytest.approx(0.24)
    assert distribution.g2 == pytest.approx(2.0)


def test_real_schmidt_number_sampling_matches_exact_moments_and_g2():
    source = degenerate_source(mean_pairs=0.4, schmidt_number=2.5)
    counts = source.sample_pair_counts(500_000, np.random.default_rng(2401))
    mean = float(np.mean(counts))
    variance = float(np.var(counts))
    g2 = float(np.mean(counts * (counts - 1)) / mean**2)
    assert mean == pytest.approx(source.mean_pairs, abs=0.004)
    assert variance == pytest.approx(source.pair_variance, abs=0.007)
    assert g2 == pytest.approx(source.unheralded_g2, abs=0.02)


def test_many_schmidt_modes_approach_poisson_pair_statistics():
    source = degenerate_source(mean_pairs=0.8, schmidt_number=1e6)
    counts = source.sample_pair_counts(300_000, np.random.default_rng(3))
    assert np.mean(counts) == pytest.approx(0.8, abs=0.006)
    assert np.var(counts) == pytest.approx(0.8, abs=0.01)
    assert source.unheralded_g2 == pytest.approx(1.000001)


def test_equal_measured_schmidt_weights_match_negative_binomial_model():
    weighted = degenerate_source(
        mean_pairs=0.4,
        schmidt_weights=(0.25, 0.25, 0.25, 0.25),
    )
    effective = degenerate_source(mean_pairs=0.4, schmidt_number=4.0)
    assert weighted.schmidt_number == pytest.approx(4.0)
    assert weighted.pair_number_distribution(14).probabilities == pytest.approx(
        effective.pair_number_distribution(14).probabilities,
        rel=2e-13,
        abs=2e-15,
    )


def test_nonuniform_schmidt_weights_reproduce_exact_moments_and_herald_pgfs():
    weights = (0.7, 0.2, 0.1)
    source = degenerate_source(mean_pairs=0.35, schmidt_weights=weights)
    purity = sum(weight**2 for weight in weights)
    assert source.heralded_spectral_purity == pytest.approx(purity)
    assert source.schmidt_number == pytest.approx(1 / purity)
    counts = source.sample_pair_counts(350_000, np.random.default_rng(811))
    assert np.mean(counts) == pytest.approx(source.mean_pairs, abs=0.004)
    assert np.var(counts) == pytest.approx(source.pair_variance, abs=0.007)
    efficiency = 0.63
    dark = 0.004
    statistics = source.heralded_statistics(efficiency, dark_probability=dark, max_pairs=30)
    no_click = (1 - dark) * np.prod(
        [1 / (1 + source.mean_pairs * weight * efficiency) for weight in weights]
    )
    assert statistics.herald_probability == pytest.approx(1 - no_click, rel=2e-12)


def test_threshold_herald_statistics_match_negative_binomial_generating_function():
    source = degenerate_source(mean_pairs=0.15, schmidt_number=3.2)
    efficiency = 0.72
    statistics = source.heralded_statistics(efficiency)
    expected_herald = 1 - (1 + source.mean_pairs * efficiency / source.schmidt_number) ** (
        -source.schmidt_number
    )
    assert statistics.herald_probability == pytest.approx(expected_herald, rel=1e-12)
    assert np.sum(statistics.conditional_probabilities) == pytest.approx(1)
    assert statistics.conditional_probabilities[0] == 0
    assert statistics.vacuum_fraction == 0
    assert statistics.single_pair_fraction + statistics.multipair_fraction == pytest.approx(1)
    assert statistics.multipair_fraction > 0


def test_threshold_herald_dark_counts_include_false_herald_vacuum():
    source = degenerate_source(mean_pairs=0.0)
    statistics = source.heralded_statistics(0.0, dark_probability=0.02)
    assert statistics.herald_probability == pytest.approx(0.02)
    assert statistics.vacuum_fraction == pytest.approx(1.0)
    assert statistics.single_pair_fraction == 0
    assert statistics.multipair_fraction == 0


def test_pump_energy_conservation_is_checked_at_central_frequencies():
    degenerate_source(pump_wavelength=775e-9)
    with pytest.raises(tbl.ValidationError, match="energy conservation"):
        degenerate_source(pump_wavelength=800e-9, energy_mismatch_tolerance=1e-4)


def test_pair_events_preserve_arm_correlation_and_schmidt_purity():
    source = degenerate_source(mean_pairs=0.35, schmidt_number=4.0)
    events = source.emit_many(2000, np.random.default_rng(91))
    grouped = defaultdict(dict)
    for event in events:
        key = (event.shot, event.metadata["pair_index"])
        grouped[key][event.metadata["arm"]] = event
    assert grouped
    assert all(set(pair) == {"signal", "idler"} for pair in grouped.values())
    for pair in list(grouped.values())[:20]:
        signal, idler = pair["signal"], pair["idler"]
        assert signal.time == idler.time
        assert signal.photon.wavepacket.purity == pytest.approx(0.25)
        assert idler.photon.wavepacket.purity == pytest.approx(0.25)
        assert signal.metadata["pair_count"] == idler.metadata["pair_count"]


def test_arm_collection_loss_is_binomial_without_changing_pair_generation():
    source = degenerate_source(
        mean_pairs=0.3,
        schmidt_number=2.0,
        signal_collection_efficiency=0.4,
        idler_collection_efficiency=0.7,
    )
    shots = 100_000
    events = source.emit_many(shots, np.random.default_rng(17))
    signal = sum(event.metadata["arm"] == "signal" for event in events)
    idler = sum(event.metadata["arm"] == "idler" for event in events)
    assert signal / shots == pytest.approx(0.3 * 0.4, abs=0.004)
    assert idler / shots == pytest.approx(0.3 * 0.7, abs=0.005)


def test_spdc_source_runs_in_digital_twin_with_reproducible_pair_tags():
    source = degenerate_source(mean_pairs=0.1, schmidt_number=2.0)
    detectors = tbl.DetectorArray(
        {
            0: tbl.SNSPD(efficiency=1, jitter=0, dead_time=0, channel=0),
            1: tbl.SNSPD(efficiency=1, jitter=0, dead_time=0, channel=1),
        }
    )
    twin = tbl.DigitalTwin(source, [], detectors)
    first = twin.run(2000, seed=700)
    second = twin.run(2000, seed=700)
    assert first.time_tags == second.time_tags
    assert first.manifest.configuration_sha256 == second.manifest.configuration_sha256
    counts = defaultdict(set)
    for tag in first.time_tags:
        counts[(tag.shot, tag.metadata["pair_index"])].add(tag.channel)
    assert counts and all(channels == {0, 1} for channels in counts.values())


@pytest.mark.parametrize(
    "changes",
    [
        {"mean_pairs": -1},
        {"schmidt_number": 0.9},
        {"signal_mode": 1, "idler_mode": 1},
        {"pump_jitter": np.nan},
        {"max_pairs_per_pulse": 0},
        {"schmidt_weights": (0.8, 0.3)},
        {"schmidt_weights": (0.5, -0.5, 1.0)},
        {"schmidt_weights": (True, False)},
        {"schmidt_weights": ("bad",)},
    ],
)
def test_spdc_source_rejects_nonphysical_parameters(changes):
    with pytest.raises(tbl.ValidationError):
        degenerate_source(**changes)


def test_two_mode_partial_overlap_reproduces_hom_and_classical_limits():
    indistinguishable = _partial_two_mode_distribution(1, 1, 1.0, 0.5)
    distinguishable = _partial_two_mode_distribution(1, 1, 0.0, 0.5)
    assert indistinguishable == pytest.approx([0.5, 0.0, 0.5], abs=1e-15)
    assert distinguishable == pytest.approx([0.25, 0.5, 0.25], abs=1e-15)
    assert np.sum(_partial_two_mode_distribution(3, 4, 0.37, 0.23)) == pytest.approx(1)


def test_two_mode_partial_formula_matches_general_gram_matrix_solver():
    photons_a, photons_b = 2, 2
    overlap = 0.37
    reflectivity = 0.23
    gram = np.ones((photons_a + photons_b, photons_a + photons_b), dtype=complex)
    gram[:photons_a, photons_a:] = np.sqrt(overlap)
    gram[photons_a:, :photons_a] = np.sqrt(overlap)
    exact = tbl.FockSimulator(
        tbl.LinearOpticalCircuit(2).beam_splitter(0, 1, reflectivity=reflectivity)
    ).probabilities([photons_a, photons_b], overlap_matrix=gram)
    specialized = _partial_two_mode_distribution(photons_a, photons_b, overlap, reflectivity)
    for output_a, probability in enumerate(specialized):
        assert probability == pytest.approx(
            exact.probabilities[(output_a, photons_a + photons_b - output_a)],
            abs=2e-12,
        )


def test_heralded_spdc_hom_multipairs_raise_the_dip_floor():
    packet = tbl.Wavepacket(temporal_width=20e-12)
    low = degenerate_source(mean_pairs=1e-4, signal_wavepacket=packet)
    bright = degenerate_source(mean_pairs=0.12, signal_wavepacket=packet)
    delays = [-400e-12, 0.0, 400e-12]
    low_scan = tbl.heralded_spdc_hom_scan(delays, low, low, max_pairs=5, tail_tolerance=1e-12)
    bright_scan = tbl.heralded_spdc_hom_scan(
        delays, bright, bright, max_pairs=12, tail_tolerance=1e-12
    )
    assert low_scan.coincidence_probability[1] < 1e-3
    assert low_scan.visibility > 0.999
    assert bright_scan.coincidence_probability[1] > low_scan.coincidence_probability[1]
    assert bright_scan.visibility < low_scan.visibility
    assert bright_scan.multipair_fractions[0] > low_scan.multipair_fractions[0]
    assert bright_scan.distinguishable_probability == pytest.approx(
        bright_scan.coincidence_probability[0], rel=1e-12
    )


def test_heralded_spdc_hom_applies_collection_threshold_loss_and_dark_clicks():
    source = degenerate_source(
        mean_pairs=0.05,
        signal_collection_efficiency=0.0,
        idler_collection_efficiency=0.7,
    )
    result = tbl.heralded_spdc_hom_scan(
        [-1e-9, 0.0, 1e-9],
        source,
        source,
        herald_detector_efficiencies=(0.8, 0.9),
        signal_detector_efficiencies=(0.6, 0.5),
        signal_dark_probabilities=(0.01, 0.02),
        max_pairs=8,
        shots_per_delay=10_000,
        seed=42,
    )
    assert result.coincidence_probability == pytest.approx([0.0002] * 3)
    assert result.herald_probabilities[0] == pytest.approx(
        1 - (1 + source.mean_pairs * 0.7 * 0.8) ** -1
    )
    repeated = tbl.heralded_spdc_hom_scan(
        [-1e-9, 0.0, 1e-9],
        source,
        source,
        herald_detector_efficiencies=(0.8, 0.9),
        signal_dark_probabilities=(0.01, 0.02),
        max_pairs=8,
        shots_per_delay=10_000,
        seed=42,
    )
    assert repeated.coincidence_counts.tolist() == result.coincidence_counts.tolist()
