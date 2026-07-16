# Changelog

## 2.0.0 — 2026-07-16

- Renamed the canonical distribution and import namespace to `TBL` / `tbl`;
  retained `openphotontwin` as a deprecation-marked compatibility shim.
- Made Gaussian wavepacket overlap translation-stable for long acquisitions,
  eliminating catastrophic cancellation between absolute picosecond-scale
  timestamps.
- Enforced event/photon mode and arrival-time consistency through delay lines,
  switches, beam splitters, time-bin qubits, and dispersive fiber loops.
- Corrected intrinsic HOM visibility in the presence of detector inefficiency
  and accidental background, and centered coincidence histograms exactly at
  zero delay with the requested bin spacing.
- Applied initial correlated-source blinking without requiring a manual reset,
  rejected fractional Fock occupations and non-finite physical parameters,
  and added repeated-input partial-distinguishability falsification tests.
- Replaced dense matrix construction for beam splitters, phase shifters, and
  loss elements with row-wise composition; a 64-mode, 210-component reference
  circuit measured about 14x faster with equivalent output.

## 1.2.0 — 2026-07-16

- Replaced scalar Ryser evaluation with chunked, batched Glynn permanents and
  cached sign tables, while retaining independent Ryser equivalence tests.
- Batched all Fock output submatrices and exact partial-distinguishability
  permutation terms, eliminating repeated Python-level permanent calls.
- Vectorized coherent time-bin loop construction across all input basis states
  and removed a redundant cubic SVD from the analytically passive recurrence.
- Added vectorized pulsed-source emission, fast immutable wavepacket/event
  copies, chronological detector fast paths, and single-pass detector routing.
- Added a validated CuPy runtime probe, safe automatic NumPy fallback,
  dtype-preserving propagation, a reusable `BatchPropagator` with cached device
  matrices, device-resident outputs, explicit synchronization, and a CPU/GPU
  workload threshold.
- Added stable core benchmarks and numerical/performance regression coverage.
- Cached repeated Fock output layouts and added a statistically equivalent
  vectorized SNSPD fast path for the common single-pixel, no-afterpulse model.
- Added a writable temporary CuPy kernel-cache fallback for read-only or
  sandboxed home directories, while preserving an explicit user cache path.
- Measured real GTX 1060 crossover behavior and made automatic GPU selection
  transfer-, return-location-, and precision-aware; GPU extras now install the
  matching user-space CUDA Toolkit libraries.

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
