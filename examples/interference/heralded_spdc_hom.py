"""Predict a measured SPDC HOM dip including herald and detector effects."""

import numpy as np

import tbl

packet = tbl.Wavepacket(temporal_width=25e-12, wavelength=1550e-9, purity=0.98)
source_a = tbl.SPDCSource(
    mean_pairs=0.08,
    schmidt_number=1.3,
    signal_wavepacket=packet,
    idler_wavepacket=packet,
    signal_collection_efficiency=0.74,
    idler_collection_efficiency=0.71,
)
source_b = tbl.SPDCSource(
    mean_pairs=0.10,
    schmidt_number=1.4,
    signal_wavepacket=packet,
    idler_wavepacket=packet,
    signal_collection_efficiency=0.72,
    idler_collection_efficiency=0.69,
)

result = tbl.heralded_spdc_hom_scan(
    np.linspace(-150e-12, 150e-12, 101),
    source_a,
    source_b,
    herald_detector_efficiencies=(0.84, 0.86),
    herald_dark_probabilities=(2e-6, 2e-6),
    signal_detector_efficiencies=(0.82, 0.80),
    signal_dark_probabilities=(1e-6, 1e-6),
    max_pairs=8,
    tail_tolerance=1e-9,
    shots_per_delay=100_000,
    seed=2027,
)

print("herald probabilities:", result.herald_probabilities)
print("conditional multipair fractions:", result.multipair_fractions)
print("distinguishable coincidence probability:", result.distinguishable_probability)
print("observed HOM visibility:", result.visibility)
print("minimum coincidence probability:", result.coincidence_probability.min())
