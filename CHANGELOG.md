# Changelog

## 1.1.0 — 2026-07-16

- Replaced mean-overlap multiphoton interpolation with an exact complex
  Gram-matrix permutation/permanent calculation, with an explicit scalable
  approximation only when requested.
- Added chirped Gaussian packets, analytic dispersion, spectral transfer-matrix
  sweeps, and per-photon wavelength-dependent circuit propagation.
- Added a correlated pulsed source with measured-style mean photon number,
  g2(0), collection loss, Markov blinking, and OU spectral diffusion.
- Added dB/km fiber attenuation, insertion loss, beta2 dispersion, PMD, and
  properly time-correlated Wiener phase noise.
- Rebuilt SNSPD simulation around chronological avalanches, recovery, multiple
  pixels, wavelength response, jitter tails, afterpulsing, latency, and time
  quantization.
- Added Poisson HOM estimation, fixed measured background, Fisher and bootstrap
  uncertainty, Clopper-Pearson loss intervals, and accidental-coincidence
  estimation.
- Added `CoherentTimeBinLoop`, an expanded-mode coherent recurrence that retains
  interference between injections and round trips and distinguishes physical
  loss from loop energy beyond the simulated window.
- Added a physics/validity document, realistic hardware examples, and
  independent limiting/statistical/falsification tests.

## 1.0.0 — 2026-07-15

- Initial importable release with Gaussian photons, passive Fock circuits,
  event-domain time-bin/fiber-loop hardware, SNSPD time tags, HOM experiments,
  calibration, optional external adapters, and GPU batch propagation.

