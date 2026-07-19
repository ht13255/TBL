import numpy as np
import pytest

import tbl as opt


def event(time=0.0, mode=0, shot=0):
    photon = opt.Photon(opt.Wavepacket(arrival_time=time), mode=mode)
    return opt.PhotonEvent(photon, time, mode, shot=shot)


def test_delay_loss_phase_and_beam_splitter_components():
    rng = np.random.default_rng(1)
    context = opt.SimulationContext(1e-9)
    events = opt.DelayLine(2e-9).process([event()], rng, context)
    events = opt.PhaseShifter(np.pi / 2).process(events, rng, context)
    events = opt.BeamSplitter(0, 1, reflectivity=1).process(events, rng, context)
    events = opt.LossChannel(1).process(events, rng, context)
    assert events[0].time == pytest.approx(2e-9)
    assert events[0].mode == 1
    assert events[0].photon.mode == 1
    assert events[0].photon.wavepacket.arrival_time == pytest.approx(events[0].time)
    assert abs(events[0].amplitude) == pytest.approx(1)


def test_fiber_loop_roundtrip_schedule_and_length_error():
    rng = np.random.default_rng(2)
    context = opt.SimulationContext(10e-9)
    loop = opt.FiberLoop(
        10e-9,
        transmission=1,
        outcoupling={1: 0.0, 2: 1.0},
        max_roundtrips=3,
        length_error=2.04e-3,
    )
    output = loop.process([event()], rng, context)
    assert len(output) == 1
    assert output[0].roundtrips == 2
    assert output[0].time == pytest.approx(20.02e-9)


def test_fiber_loop_applies_db_loss_dispersion_and_records_provenance():
    rng = np.random.default_rng(21)
    context = opt.SimulationContext(10e-9)
    events = [event(shot=index) for index in range(40_000)]
    loop = opt.FiberLoop(
        10e-9,
        transmission=1,
        outcoupling=1,
        fiber_length=1000,
        attenuation_db_per_km=0.2,
        insertion_loss_db=1.0,
        dispersion_beta2=-21e-27,
        max_roundtrips=1,
    )
    output = loop.process(events, rng, context)
    expected_transmission = 10 ** (-1.2 / 10)
    assert len(output) / len(events) == pytest.approx(expected_transmission, abs=0.006)
    assert output[0].photon.wavepacket.temporal_width > events[0].photon.wavepacket.temporal_width
    assert output[0].metadata["loop_effective_transmission"] == pytest.approx(expected_transmission)
    assert output[0].metadata["loop_gdd_s2"] == pytest.approx(-21e-24)
    assert output[0].photon.mode == output[0].mode
    assert output[0].photon.wavepacket.arrival_time == pytest.approx(output[0].time)


def test_fiber_loop_environment_is_common_to_simultaneous_photons():
    simultaneous = [event(shot=0), event(shot=1)]
    shared = opt.FiberLoop(
        10e-9,
        transmission=1,
        outcoupling=1,
        max_roundtrips=1,
        phase_drift_std=0.4,
        fiber_length=1000,
        pmd_coefficient=2e-12,
        shared_environment=True,
    ).process(simultaneous, np.random.default_rng(222), opt.SimulationContext(1e-9))
    assert shared[0].time == pytest.approx(shared[1].time)
    assert shared[0].amplitude == pytest.approx(shared[1].amplitude)
    assert shared[0].metadata["loop_phase_drift_rad"] == pytest.approx(
        shared[1].metadata["loop_phase_drift_rad"]
    )

    independent = opt.FiberLoop(
        10e-9,
        transmission=1,
        outcoupling=1,
        max_roundtrips=1,
        phase_drift_std=0.4,
        fiber_length=1000,
        pmd_coefficient=2e-12,
        shared_environment=False,
    ).process(simultaneous, np.random.default_rng(222), opt.SimulationContext(1e-9))
    assert independent[0].time != pytest.approx(independent[1].time)
    assert independent[0].amplitude != pytest.approx(independent[1].amplitude)


def test_feedforward_latency_and_eom_switch():
    rng = np.random.default_rng(3)
    context = opt.SimulationContext(1e-9)
    controller = opt.FeedForwardController(latency=5e-9)
    controller.trigger(0, 1)
    switch = opt.EOMSwitch(0, 1, control=controller, extinction_ratio_db=300)
    assert switch.process([event(2e-9)], rng, context)[0].mode == 0
    assert switch.process([event(6e-9)], rng, context)[0].mode == 1


def test_feedforward_rise_time_and_temperature_phase():
    controller = opt.FeedForwardController(latency=5e-9)
    controller.trigger(0, 1)
    assert controller.value_at(5.5e-9, rise_time=1e-9) == pytest.approx(0.5)
    component = opt.TemperaturePhaseDrift(
        length=1e-2,
        temperature=lambda time: 20 + time / 1e-9,
    )
    output = component.process([event(1e-9)], np.random.default_rng(1), opt.SimulationContext(1e-9))
    assert np.angle(output[0].amplitude) != pytest.approx(0.0)


def test_wiener_phase_is_common_at_equal_time_and_correlated():
    component = opt.PhaseDrift(standard_deviation=0.2, reference_time=1e-6)
    events = [event(1e-6, mode=0), event(1e-6, mode=1), event(2e-6, mode=0)]
    output = component.process(events, np.random.default_rng(8), opt.SimulationContext(1e-9))
    assert output[0].amplitude == pytest.approx(output[1].amplitude)
    assert output[2].amplitude != pytest.approx(output[0].amplitude)


def test_dynamic_beam_splitter_switching_ramp():
    component = opt.DynamicBeamSplitter(
        0, 1, reflectivity=0, schedule={0: 0.0, 1: 1.0}, switching_time=1e-9
    )
    context = opt.SimulationContext(2e-9)
    midpoint = event(2.5e-9)
    assert component.reflectivity_at(midpoint, context) == pytest.approx(0.5)


def test_snspd_efficiency_dark_counts_jitter_and_dead_time():
    detector = opt.SNSPD(
        efficiency=1,
        dark_count_rate=0,
        jitter=0,
        dead_time=5e-9,
        channel=2,
    )
    tags = detector.detect(
        [event(1e-9), event(2e-9), event(8e-9)],
        acquisition_start=0,
        acquisition_end=10e-9,
        rng=np.random.default_rng(4),
    )
    assert [(tag.time, tag.channel) for tag in tags] == [(1e-9, 2), (8e-9, 2)]


def test_snspd_pixels_latency_quantization_and_avalanche_metadata():
    detector = opt.SNSPD(
        efficiency=1,
        jitter=0,
        dead_time=0,
        channel=3,
        detection_latency=0.3e-9,
        time_tagger_resolution=1e-9,
    )
    tags = detector.detect(
        [event(1.2e-9)],
        acquisition_start=0,
        acquisition_end=20e-9,
        rng=np.random.default_rng(33),
    )
    assert len(tags) == 1
    assert all(tag.time == pytest.approx(2e-9) for tag in tags)
    assert {tag.metadata["detector_pixel"] for tag in tags} == {0}
    assert all(tag.metadata["true_time_s"] == pytest.approx(1.2e-9) for tag in tags)


def test_multipixel_snspd_has_physical_collision_statistics():
    pairs = 20_000
    detector = opt.SNSPD(
        efficiency=1,
        jitter=0,
        dead_time=10e-9,
        pixel_count=4,
    )
    arrivals = [event(index * 20e-9, shot=index) for index in range(pairs) for _ in range(2)]
    tags = detector.detect(
        arrivals,
        acquisition_start=0,
        acquisition_end=pairs * 20e-9,
        rng=np.random.default_rng(330),
    )
    # Two photons routed uniformly to four pixels produce 2 detections with
    # probability 3/4 and one detection with probability 1/4.
    assert len(tags) / pairs == pytest.approx(1.75, abs=0.015)
    per_shot = np.bincount([tag.shot for tag in tags], minlength=pairs)
    assert set(np.unique(per_shot)).issubset({1, 2})


def test_multipixel_routing_weights_and_afterpulses_stay_on_parent_pixel():
    detector = opt.SNSPD(
        efficiency=1,
        jitter=0,
        dead_time=0,
        pixel_count=4,
        pixel_weights=(0.1, 0.2, 0.3, 0.4),
    )
    arrivals = [event(index * 1e-9, shot=index) for index in range(50_000)]
    tags = detector.detect(
        arrivals,
        acquisition_start=0,
        acquisition_end=50_000e-9,
        rng=np.random.default_rng(331),
    )
    pixels = np.bincount([tag.metadata["detector_pixel"] for tag in tags], minlength=4) / len(tags)
    assert pixels == pytest.approx((0.1, 0.2, 0.3, 0.4), abs=0.006)

    afterpulsing = opt.SNSPD(
        efficiency=1,
        jitter=0,
        dead_time=1e-9,
        pixel_count=4,
        afterpulse_probability=1,
        afterpulse_time_constant=2e-9,
    )
    after_tags = afterpulsing.detect(
        [event(0.0)],
        acquisition_start=0,
        acquisition_end=30e-9,
        rng=np.random.default_rng(332),
    )
    assert len(after_tags) > 1
    assert len({tag.metadata["detector_pixel"] for tag in after_tags}) == 1


def test_snspd_recovery_afterpulse_and_wavelength_efficiency():
    rejecting = opt.SNSPD(
        efficiency=1,
        jitter=0,
        dead_time=0,
        wavelength_efficiency=lambda wavelength: 0.0,
    )
    assert not rejecting.detect(
        [event(1e-9)],
        acquisition_start=0,
        acquisition_end=2e-9,
        rng=np.random.default_rng(1),
    )

    detector = opt.SNSPD(
        efficiency=1,
        jitter=0,
        dead_time=0,
        recovery_time=0,
        afterpulse_probability=0.7,
        afterpulse_time_constant=2e-9,
    )
    tags = detector.detect(
        [event(1e-9)],
        acquisition_start=0,
        acquisition_end=30e-9,
        rng=np.random.default_rng(4),
    )
    assert tags[0].metadata["event_type"] == "photon"
    assert any(tag.metadata["event_type"] == "afterpulse" for tag in tags)


def test_vectorized_snspd_fast_path_matches_generic_detection_statistics():
    arrivals = [event(index * 10e-9, shot=index) for index in range(20_000)]
    fast = opt.SNSPD(efficiency=0.63, jitter=0, dead_time=20e-9)
    generic = opt.SNSPD(
        efficiency=1,
        wavelength_efficiency=lambda wavelength: 0.63,
        jitter=0,
        dead_time=20e-9,
    )
    fast_tags = fast.detect(
        arrivals,
        acquisition_start=0,
        acquisition_end=arrivals[-1].time,
        rng=np.random.default_rng(700),
    )
    generic_tags = generic.detect(
        arrivals,
        acquisition_start=0,
        acquisition_end=arrivals[-1].time,
        rng=np.random.default_rng(701),
    )
    assert len(fast_tags) / len(generic_tags) == pytest.approx(1.0, abs=0.025)
    assert np.min(np.diff([tag.metadata["true_time_s"] for tag in fast_tags])) >= (
        fast.dead_time - 1e-18
    )


def test_digital_twin_end_to_end_and_reproducibility():
    source = opt.SinglePhotonSource(repetition_rate=1e6, p_single=1)
    detectors = opt.DetectorArray({0: opt.SNSPD(efficiency=1, jitter=0, dead_time=0)})
    twin = opt.DigitalTwin(source, [opt.DelayLine(10e-9)], detectors)
    first = twin.run(20, seed=9)
    second = twin.run(20, seed=9)
    assert first.time_tags == second.time_tags
    assert len(first.time_tags) == 20
    assert first.photon_number_distribution == {1: 1.0}


def test_default_acquisition_window_includes_propagation_tail():
    source = opt.SinglePhotonSource(repetition_rate=1e9, p_single=1)
    detector = opt.DetectorArray({0: opt.SNSPD(efficiency=1, jitter=0, dead_time=0)})
    twin = opt.DigitalTwin(source, [opt.DelayLine(20e-9)], detector)
    result = twin.run(1, seed=2)
    assert len(result.time_tags) == 1
    assert result.time_tags[0].time == pytest.approx(20e-9)


def test_detector_substreams_are_order_independent_and_stable_when_array_grows():
    arrivals = [event(index * 10e-9, mode=0, shot=index) for index in range(5000)]
    first = opt.SNSPD(efficiency=0.51, jitter=2e-12, dead_time=0, channel=7)
    unused = opt.SNSPD(efficiency=0.2, jitter=0, dead_time=0, channel=8)

    only = opt.DetectorArray({0: first}).detect(
        arrivals,
        acquisition_start=0,
        acquisition_end=50_000e-9,
        rng=np.random.default_rng(900),
    )
    extended = opt.DetectorArray({1: unused, 0: first}).detect(
        arrivals,
        acquisition_start=0,
        acquisition_end=50_000e-9,
        rng=np.random.default_rng(900),
    )
    reordered = opt.DetectorArray({0: first, 1: unused}).detect(
        arrivals,
        acquisition_start=0,
        acquisition_end=50_000e-9,
        rng=np.random.default_rng(900),
    )
    assert only == extended == reordered


def test_correlated_source_digital_twin_resets_hidden_state_for_seed():
    source = opt.CorrelatedPhotonSource(
        repetition_rate=2e6,
        mean_photon_number=0.6,
        blink_on_to_off=0.05,
        blink_off_to_on=0.2,
        spectral_diffusion_std=1e9,
        spectral_correlation_time=2e-6,
    )
    detectors = opt.DetectorArray({0: opt.SNSPD(efficiency=1, jitter=0, dead_time=0)})
    twin = opt.DigitalTwin(source, [], detectors)
    first = twin.run(1000, seed=18)
    second = twin.run(1000, seed=18)
    assert first.time_tags == second.time_tags


def test_thermo_optic_phase_has_expected_sign():
    assert opt.thermo_optic_phase(1e-2, 1550e-9, 1) > 0
    assert opt.thermo_optic_phase(1e-2, 1550e-9, -1) < 0
