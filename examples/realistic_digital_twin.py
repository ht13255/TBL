"""Telecom-band loop twin with correlated source and avalanche-level SNSPDs."""

import numpy as np

import openphotontwin as opt

source = opt.CorrelatedPhotonSource(
    repetition_rate=10e6,
    mean_photon_number=0.65,
    g2_zero=0.015,
    collection_efficiency=0.72,
    wavepacket=opt.Wavepacket(
        temporal_width=22e-12,
        wavelength=1550e-9,
        purity=0.94,
        chirp=0.15,
    ),
    emission_jitter=8e-12,
    blink_on_to_off=0.002,
    blink_off_to_on=0.03,
    spectral_diffusion_std=2 * np.pi * 150e6,
    spectral_correlation_time=50e-6,
)

loop_length = 20.0
group_velocity = 2.04e8
loop = opt.FiberLoop(
    round_trip_time=loop_length / group_velocity,
    transmission=0.97,
    outcoupling={1: 0.0, 2: 0.15, 3: 0.45, 4: 1.0},
    phase_drift_std=0.015,
    max_roundtrips=4,
    group_velocity=group_velocity,
    fiber_length=loop_length,
    attenuation_db_per_km=0.2,
    insertion_loss_db=1.2,
    dispersion_beta2=-21.7e-27,
    pmd_coefficient=0.1e-12 / np.sqrt(1000),
)

detector = opt.SNSPD(
    efficiency=0.82,
    dark_count_rate=20,
    jitter=18e-12,
    dead_time=40e-9,
    recovery_time=15e-9,
    jitter_tail_probability=0.03,
    jitter_tail_time=80e-12,
    afterpulse_probability=0.002,
    afterpulse_time_constant=100e-9,
    time_tagger_resolution=1e-12,
)

twin = opt.DigitalTwin(source, [loop], opt.DetectorArray({0: detector}))
result = twin.run(10_000, seed=20260716)
print(result.photon_number_distribution)
print(result.tags_dataframe()["event_type"].value_counts())
