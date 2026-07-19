"""Simulate a realistic heralded SPDC source and report multipair noise."""

import tbl

source = tbl.SPDCSource(
    repetition_rate=80e6,
    mean_pairs=0.08,
    schmidt_number=2.4,
    signal_wavepacket=tbl.Wavepacket(wavelength=1550e-9, purity=0.98),
    idler_wavepacket=tbl.Wavepacket(wavelength=1550e-9, purity=0.98),
    signal_collection_efficiency=0.72,
    idler_collection_efficiency=0.68,
    pump_wavelength=775e-9,
    pump_jitter=2e-12,
    relative_time_jitter=3e-12,
)

heralded = source.heralded_statistics(herald_efficiency=0.85)
print("herald probability:", heralded.herald_probability)
print("single-pair fraction:", heralded.single_pair_fraction)
print("multipair fraction:", heralded.multipair_fraction)
print("spectral purity:", source.heralded_spectral_purity)

detectors = tbl.DetectorArray(
    {
        0: tbl.SNSPD(efficiency=0.82, jitter=18e-12, dead_time=30e-9, channel=0),
        1: tbl.SNSPD(efficiency=0.86, jitter=16e-12, dead_time=30e-9, channel=1),
    }
)
result = tbl.DigitalTwin(source, [], detectors).run(100_000, seed=2027)
print(result.tags_dataframe().groupby("channel").size())
