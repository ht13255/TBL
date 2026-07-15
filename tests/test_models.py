import numpy as np
import pytest

import openphotontwin as opt


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


def test_invalid_physical_parameters_raise_domain_error():
    with pytest.raises(opt.ValidationError):
        opt.Wavepacket(temporal_width=0)
    with pytest.raises(opt.ValidationError):
        opt.SinglePhotonSource(p_single=0.8, p_double=0.3)
    with pytest.raises(opt.ValidationError):
        opt.TimeBinQubit(alpha=0, beta=0)


def test_source_and_time_bin_sampling_are_reproducible():
    rng = np.random.default_rng(12)
    source = opt.SinglePhotonSource(repetition_rate=10e6, p_single=1.0)
    event = source.emit(3, rng)[0]
    assert event.time == pytest.approx(300e-9)
    assert event.shot == 3

    qubit = opt.TimeBinQubit(alpha=1, beta=1, separation=2e-9)
    samples = [qubit.sample_event(rng).photon.time_bin for _ in range(1000)]
    assert 430 < sum(samples) < 570


def test_time_bin_coherent_events_are_normalized():
    qubit = opt.TimeBinQubit(alpha=1, beta=1j, separation=1e-9)
    events = qubit.events()
    assert sum(abs(event.amplitude) ** 2 for event in events) == pytest.approx(1.0)
    assert events[1].time - events[0].time == pytest.approx(1e-9)
