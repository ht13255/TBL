import numpy as np
import pytest

import openphotontwin as opt


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


def test_digital_twin_end_to_end_and_reproducibility():
    source = opt.SinglePhotonSource(repetition_rate=1e6, p_single=1)
    detectors = opt.DetectorArray({0: opt.SNSPD(efficiency=1, jitter=0, dead_time=0)})
    twin = opt.DigitalTwin(source, [opt.DelayLine(10e-9)], detectors)
    first = twin.run(20, seed=9)
    second = twin.run(20, seed=9)
    assert first.time_tags == second.time_tags
    assert len(first.time_tags) == 20
    assert first.photon_number_distribution == {1: 1.0}


def test_thermo_optic_phase_has_expected_sign():
    assert opt.thermo_optic_phase(1e-2, 1550e-9, 1) > 0
    assert opt.thermo_optic_phase(1e-2, 1550e-9, -1) < 0
