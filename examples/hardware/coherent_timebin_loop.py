"""Interference and Fock statistics in an expanded-mode coherent loop."""

import numpy as np

import tbl

loop = tbl.CoherentTimeBinLoop(
    time_bins=10,
    bin_width=5e-9,
    round_trip_bins=2,
    reflectivity={0: 1.0, 2: 0.5, 8: 1.0},
    round_trip_transmission=0.96,
    coupler_transmission=0.98,
    phase=0.15,
    switching_time=300e-12,
    phase_noise_std=0.01,
    phase_correlation_time=500e-9,
)

input_amplitudes = np.zeros(10, dtype=complex)
input_amplitudes[[0, 2]] = [1 / np.sqrt(2), 1j / np.sqrt(2)]
detailed = loop.simulate(input_amplitudes, seed=7)
print("output:", np.round(detailed.output_amplitudes, 5))
print("output probability:", detailed.output_probability)
print("residual loop probability:", detailed.residual_loop_probability)
print("physical loss probability:", detailed.physical_loss_probability)

occupation = [1, 0, 1] + [0] * 7
distribution = tbl.FockSimulator(loop.circuit(seed=7)).probabilities(occupation)
print("two-photon survival:", distribution.survival_probability)
