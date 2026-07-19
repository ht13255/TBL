"""Model a lifetime-limited quantum-dot HOM dip with spectral detuning."""

import numpy as np

import tbl

lifetime = 450e-12
first = tbl.Wavepacket.exponential(
    lifetime,
    wavelength=930e-9,
    purity=0.96,
    label="QD pulse A",
)

# Construct the second wavelength from an experimentally meaningful angular
# frequency detuning. The 1/tau scale halves the ideal zero-delay overlap.
detuning = 1 / lifetime
second_frequency = first.angular_frequency + detuning
second_wavelength = first.angular_frequency * first.wavelength / second_frequency
second = tbl.Wavepacket.exponential(
    lifetime,
    wavelength=second_wavelength,
    purity=0.94,
    label="QD pulse B",
)

delays = np.linspace(-4 * lifetime, 4 * lifetime, 161)
scan = tbl.hom_scan(delays, first, second)
zero_index = int(np.argmin(abs(delays)))

print("zero-delay indistinguishability:", first.indistinguishability(second))
print("zero-delay coincidence probability:", scan.coincidence_probability[zero_index])
print("far-delay coincidence probability:", scan.coincidence_probability[-1])
print("spectral standard deviation (rad/s):", first.spectral_width_angular)
