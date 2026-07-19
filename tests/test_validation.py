import numpy as np
import pytest

import tbl


@pytest.mark.parametrize(
    "factory",
    [
        lambda: tbl.DelayLine(np.nan),
        lambda: tbl.DelayLine(1.0, jitter=np.inf),
        lambda: tbl.PhaseShifter(np.nan),
        lambda: tbl.BeamSplitter(phase=np.inf),
        lambda: tbl.DynamicBeamSplitter(switching_time=np.nan),
        lambda: tbl.FeedForwardController(default=np.nan),
        lambda: tbl.EOMSwitch(control_latency=np.inf),
        lambda: tbl.PhaseDrift(np.nan),
        lambda: tbl.TemperaturePhaseDrift(length=np.nan),
        lambda: tbl.FiberLoop(np.nan),
        lambda: tbl.SNSPD(jitter=np.nan),
        lambda: tbl.TimeTag(np.nan, 0),
    ],
)
def test_hardware_models_reject_nonfinite_parameters(factory):
    with pytest.raises(tbl.ValidationError):
        factory()


def test_callable_hardware_controls_cannot_inject_nan():
    event = tbl.PhotonEvent(tbl.Photon(mode=0), 0.0, 0)
    switch = tbl.EOMSwitch(0, 1, control=lambda time: np.nan)
    with pytest.raises(tbl.ValidationError, match="non-finite"):
        switch.process([event], np.random.default_rng(1), tbl.SimulationContext())

    drift = tbl.TemperaturePhaseDrift(length=1.0, temperature=lambda time: np.nan)
    with pytest.raises(tbl.ValidationError, match="non-finite"):
        drift.process([event], np.random.default_rng(1), tbl.SimulationContext())


def test_linear_optics_rejects_nonfinite_matrices_and_timing():
    with pytest.raises(tbl.ValidationError, match="finite"):
        tbl.permanent(np.asarray([[np.nan]]))
    with pytest.raises(tbl.ValidationError, match="finite"):
        tbl.LinearOpticalCircuit.from_transfer_matrix(np.asarray([[np.nan]]))
    circuit = tbl.LinearOpticalCircuit(2).beam_splitter(0, 1)
    with pytest.raises(tbl.ValidationError, match="time and wavelength"):
        circuit.transfer_matrix(time=np.nan)


def test_mode_and_schedule_indices_are_validated_at_construction():
    with pytest.raises(tbl.ValidationError, match="modes"):
        tbl.LossChannel(1.0, frozenset({-1}))
    with pytest.raises(tbl.ValidationError, match="schedule keys"):
        tbl.DynamicBeamSplitter(schedule={0.5: 0.2})
    with pytest.raises(tbl.ValidationError, match="outcoupling keys"):
        tbl.FiberLoop(1e-9, outcoupling={-1: 0.5})
    with pytest.raises(tbl.ValidationError, match="detector modes"):
        tbl.DetectorArray({-1: tbl.SNSPD()})


@pytest.mark.parametrize(
    "changes",
    [
        {"bin_width": np.nan},
        {"switching_time": np.inf},
        {"phase_noise_std": np.nan},
        {"reflectivity": {0.5: 0.2}},
        {"phase": {0: np.nan}},
    ],
)
def test_coherent_loop_rejects_nonfinite_or_malformed_schedules(changes):
    parameters = {"time_bins": 4, "bin_width": 1e-9}
    parameters.update(changes)
    with pytest.raises(tbl.ValidationError):
        tbl.CoherentTimeBinLoop(**parameters)


def test_coherent_loop_rejects_nonfinite_input_amplitudes():
    loop = tbl.CoherentTimeBinLoop(3, 1e-9)
    with pytest.raises(tbl.ValidationError, match="finite"):
        loop.simulate([1, np.nan, 0])
