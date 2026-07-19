# Changelog

## 2.2.0 — 2026-07-17

- Added a pulsed SPDC/SFWM pair source with real-valued effective Schmidt mode
  count, exact multimode-thermal pair statistics, spectral-purity propagation,
  pump energy-conservation checks, correlated timing, and independent arm loss.
- Added normalized nonuniform Schmidt-weight spectra with exact convolution of
  per-mode thermal statistics, direct sampling, measured spectral purity, and
  general product-form herald probabilities.
- Added analytic threshold-herald conditioning with single-pair and multipair
  fractions, conditional mean, explicit infinite-tail control, and exact pair
  mean/variance/g2 observables.
- Added heralded SPDC HOM observables with exact multipair Fock output
  statistics, false-herald vacuum, collection loss, threshold-detector
  saturation, signal/herald dark clicks, and independently checked partial
  distinguishability limits.
- Replaced optimistic “most recovered pixel” routing with calibrated stochastic
  pixel weights, physical same-pixel collisions, and parent-pixel afterpulsing.
- Added mode/channel-keyed detector RNG substreams so array order or an unused
  detector cannot perturb an existing channel's stochastic realization.
- Made loop phase noise and phenomenological PMD common to photons traversing
  in the same time bin, preventing artificial loss of simultaneous-photon
  indistinguishability; retained an explicit independent-ensemble mode.
- Added FFT/initial-positive-sequence autocorrelation diagnostics, effective
  sample size, batch means, and adjusted Wilson intervals so correlated blinking
  or drift data are not reported with invalid iid confidence intervals.
- Added causal exponential spontaneous-emission wavepackets, analytic
  exponential/exponential and Gaussian/exponential overlaps, and schema-2
  research bundles while retaining schema-1 Gaussian read compatibility.
- Vectorized exact Gaussian, exponential, and mixed-profile delay scans while
  preserving scalar closed-form results; a 4,001-point exponential HOM scan
  measured roughly three orders of magnitude faster than the former per-point
  object loop.
- Added a publication-oriented validation report mapping physical claims to
  independent analytic limits, numerical falsification tests, stochastic
  tolerances, package checks, and explicitly excluded validity regimes.
- Standardized the entire Python tree with Ruff formatting and stricter
  maintainability rules, clarified small control-flow constructs, grouped
  examples by workflow, and moved immutable benchmark data under
  `benchmarks/results/`.

## 2.1.0 — 2026-07-16

- Added a versioned run manifest with configuration and result SHA-256 hashes,
  SI units, RNG details, acquisition bounds, and environment/dependency versions.
- Added lossless, pickle-free compressed research bundles with full event,
  wavepacket, detector, nested metadata, round-trip, and integrity restoration.
- Added in-memory mutation detection, malformed bundle-shape validation, and
  finite acquisition/time-bin contracts.
- Isolated source, component, and detector random streams with a versioned
  SeedSequence policy, and recorded entropy even for initially unseeded runs.
- Added per-shot detector probability estimates with exact Clopper-Pearson
  confidence intervals for finite Monte Carlo acquisitions.
- Closed non-finite-value paths across hardware timing, phases, controls,
  detector parameters, S-parameter matrices, permanents, and acquisition input.
- Protected canonical time-tag CSV/DataFrame columns from collisions with
  user-supplied metadata keys.
- Separated always-on configuration provenance from optional in-memory payload
  sealing, while making research-bundle saves seal automatically; this avoids
  imposing event-wise hashing cost on exploratory Monte Carlo sweeps.
- Versioned and accelerated payload sealing with a platform-independent binary
  numerical encoding and canonical repeated-metadata cache.
- Added a typed-package marker, clean mypy gate, Python 3.10-3.13 CI matrix,
  wheel-content audit, dependency monitoring, contribution standards, citation
  metadata, and a publication-oriented reproducibility guide.
- Tightened component-union, detector-routing, and CPU/GPU state boundaries
  exposed by static analysis without changing numerical behavior.

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
