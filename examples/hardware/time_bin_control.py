"""Use a time-bin qubit source factory with delayed electronic switching."""

import tbl

qubit = tbl.TimeBinQubit(alpha=1, beta=1j, separation=10e-9)


def source(shot, rng):
    event = qubit.sample_event(rng, shot=shot)
    return [event.copy(time=event.time + shot * 100e-9)]


controller = tbl.FeedForwardController(latency=4e-9)
controller.trigger(0.0, 1.0)
switch = tbl.EOMSwitch(0, 1, controller, control_latency=1e-9, extinction_ratio_db=30)
detectors = tbl.DetectorArray(
    {
        0: tbl.SNSPD(channel=0, efficiency=0.9),
        1: tbl.SNSPD(channel=1, efficiency=0.9),
    }
)
result = tbl.DigitalTwin(source, [switch], detectors, time_bin_width=10e-9).run(
    1000, seed=9, acquisition_end=100e-6
)
print(result.tags_dataframe().groupby("channel").size())
