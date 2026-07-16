import numpy as np
import pytest

import tbl as opt


def test_wavepacket_normalization_and_identical_overlap():
    packet = opt.Wavepacket(temporal_width=20e-12)
    times = np.linspace(-200e-12, 200e-12, 50_001)
    norm = np.trapezoid(abs(packet.amplitude(times)) ** 2, times)
    assert norm == pytest.approx(1.0, rel=1e-8)
    assert packet.indistinguishability(packet) == pytest.approx(1.0)


def test_wavepacket_overlap_resolves_delay_frequency_and_polarization():
    first = opt.Wavepacket(temporal_width=10e-12)
    delayed = opt.Wavepacket(arrival_time=30e-12, temporal_width=10e-12)
    detuned = opt.Wavepacket(temporal_width=10e-12, wavelength=1549e-9)
    orthogonal = opt.Wavepacket(temporal_width=10e-12, polarization=(0, 1))
    assert 0 < first.indistinguishability(delayed) < 1
    assert 0 < first.indistinguishability(detuned) < 1
    assert first.indistinguishability(orthogonal) == pytest.approx(0.0)


def test_wavepacket_overlap_is_stable_under_large_timestamp_translation():
    width = 20e-12
    delay = 17e-12
    reference = opt.Wavepacket(temporal_width=width).mode_overlap(
        opt.Wavepacket(arrival_time=delay, temporal_width=width)
    )
    offset = 1.0
    translated = opt.Wavepacket(arrival_time=offset, temporal_width=width).mode_overlap(
        opt.Wavepacket(arrival_time=offset + delay, temporal_width=width)
    )
    assert abs(translated) == pytest.approx(abs(reference), rel=2e-6)


def test_chirped_overlap_and_dispersion_obey_gaussian_invariants():
    packet = opt.Wavepacket(temporal_width=18e-12, chirp=0.7)
    assert packet.mode_overlap(packet) == pytest.approx(1.0 + 0.0j)
    dispersed = packet.dispersed(3.2e-22)
    recovered = dispersed.dispersed(-3.2e-22)
    assert dispersed.spectral_width_angular == pytest.approx(
        packet.spectral_width_angular, rel=1e-12
    )
    assert recovered.temporal_width == pytest.approx(packet.temporal_width, rel=1e-12)
    assert recovered.chirp == pytest.approx(packet.chirp, abs=1e-12)


def test_purity_sets_self_hom_indistinguishability_not_mode_norm():
    packet = opt.Wavepacket(purity=0.64)
    assert packet.mode_overlap(packet) == pytest.approx(1.0)
    assert packet.indistinguishability(packet) == pytest.approx(0.64)


def test_invalid_physical_parameters_raise_domain_error():
    with pytest.raises(opt.ValidationError):
        opt.Wavepacket(temporal_width=0)
    with pytest.raises(opt.ValidationError):
        opt.SinglePhotonSource(p_single=0.8, p_double=0.3)
    with pytest.raises(opt.ValidationError):
        opt.TimeBinQubit(alpha=0, beta=0)
    with pytest.raises(opt.ValidationError):
        opt.Wavepacket(arrival_time=np.nan)
    with pytest.raises(opt.ValidationError):
        opt.Wavepacket(wavelength=np.inf)
    with pytest.raises(opt.ValidationError):
        opt.TimeBinQubit(alpha=np.nan, beta=1)


def test_photon_event_rejects_inconsistent_time_and_mode():
    photon = opt.Photon(opt.Wavepacket(arrival_time=1e-9), mode=2)
    with pytest.raises(opt.ValidationError):
        opt.PhotonEvent(photon, 2e-9, 2)
    with pytest.raises(opt.ValidationError):
        opt.PhotonEvent(photon, 1e-9, 1)


def test_source_and_time_bin_sampling_are_reproducible():
    rng = np.random.default_rng(12)
    source = opt.SinglePhotonSource(repetition_rate=10e6, p_single=1.0)
    event = source.emit(3, rng)[0]
    assert event.time == pytest.approx(300e-9)
    assert event.shot == 3

    qubit = opt.TimeBinQubit(alpha=1, beta=1, separation=2e-9)
    samples = [qubit.sample_event(rng).photon.time_bin for _ in range(1000)]
    assert 430 < sum(samples) < 570


def test_correlated_source_reproduces_mean_and_g2_statistics():
    source = opt.CorrelatedPhotonSource(
        mean_photon_number=0.7,
        g2_zero=0.08,
        collection_efficiency=1.0,
    )
    rng = np.random.default_rng(91)
    source.reset(rng)
    counts = np.array([len(source.emit(shot, rng)) for shot in range(80_000)])
    mean = float(np.mean(counts))
    measured_g2 = float(np.mean(counts * (counts - 1)) / mean**2)
    assert mean == pytest.approx(0.7, abs=0.01)
    assert measured_g2 == pytest.approx(0.08, abs=0.01)


def test_correlated_source_applies_initial_blink_state_without_manual_reset():
    source = opt.CorrelatedPhotonSource(
        mean_photon_number=1.0,
        g2_zero=0.0,
        initial_on_probability=0.0,
        blink_off_to_on=0.0,
    )
    assert source.emit(0, np.random.default_rng(17)) == []


def test_spectral_diffusion_has_requested_ou_correlation():
    source = opt.CorrelatedPhotonSource(
        repetition_rate=1e6,
        mean_photon_number=0,
        spectral_diffusion_std=2e9,
        spectral_correlation_time=10e-6,
    )
    rng = np.random.default_rng(92)
    source.reset(rng)
    offsets = []
    for shot in range(10_000):
        source.emit(shot, rng)
        offsets.append(source._frequency_offset)
    measured = np.corrcoef(offsets[:-1], offsets[1:])[0, 1]
    expected = np.exp(-source.period / source.spectral_correlation_time)
    assert measured == pytest.approx(expected, abs=0.02)


def test_time_bin_coherent_events_are_normalized():
    qubit = opt.TimeBinQubit(alpha=1, beta=1j, separation=1e-9)
    events = qubit.events()
    assert sum(abs(event.amplitude) ** 2 for event in events) == pytest.approx(1.0)
    assert events[1].time - events[0].time == pytest.approx(1e-9)
    assert [event.photon.time_bin for event in events] == [0, 1]
    assert all(
        event.photon.wavepacket.arrival_time == pytest.approx(event.time) for event in events
    )
