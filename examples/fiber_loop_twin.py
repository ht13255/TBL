"""Run a time-resolved fiber-loop experiment and save synthetic time tags."""

import openphotontwin as opt

source = opt.SinglePhotonSource(
    repetition_rate=5e6,
    p_single=0.85,
    p_double=0.005,
    wavepacket=opt.Wavepacket(temporal_width=15e-12),
    emission_jitter=3e-12,
)
loop = opt.FiberLoop(
    round_trip_time=25e-9,
    transmission=0.96,
    outcoupling={1: 0.0, 2: 0.15, 3: 0.5, 4: 1.0},
    phase_per_roundtrip=0.1,
    phase_drift_std=0.02,
    max_roundtrips=4,
    length_error=0.5e-3,
)
detectors = opt.DetectorArray(
    {0: opt.SNSPD(efficiency=0.91, dark_count_rate=5.0, jitter=18e-12, channel=0)}
)
twin = opt.DigitalTwin(source, [loop], detectors, time_bin_width=25e-9)
result = twin.run(25_000, seed=2026)
result.save_time_tags("fiber_loop_time_tags.csv")
print(result.photon_number_distribution)
