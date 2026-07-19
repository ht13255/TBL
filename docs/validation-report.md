# TBL 2.2.0 verification and validation report

This document records the evidence supporting scientific use of TBL 2.2.0.
It is a software verification report, not a claim that every optical device is
captured microscopically. Governing equations and parameter conventions are in
[`physics-model.md`](physics-model.md); reproducible result handling is in
[`reproducibility.md`](reproducibility.md).

## Verified release state

The final wheel was installed into a clean Python 3.12 environment and tested
from `site-packages`, rather than imported from the source tree.

```text
TBL version:               2.2.0
Tests:                     167 passed
Statement coverage:        87%
Ruff:                      passed
mypy:                      passed (16 source files)
pip dependency check:      passed
Wheel contents:            23 entries, audited
Source distribution:       73 entries, audited
Python CI matrix:           3.10, 3.11, 3.12, 3.13
```

The exact local verification commands are:

```bash
python -m pytest --cov=tbl --cov-report=term-missing
python -m ruff check .
python -m mypy
python -m pip check
python benchmarks/benchmark_core.py
```

All eleven shipped examples were also run from an otherwise empty working
directory using the installed wheel.

## Claim-to-evidence map

| Scientific claim | Independent or limiting evidence | Primary tests |
|---|---|---|
| Gaussian fields are normalized and overlap is stable at long laboratory timestamps | Numerical quadrature; translation by one second while retaining picosecond relative delay | `test_models.py` |
| Causal exponential fields have the expected delay and Lorentzian-detuning overlap | Midpoint quadrature; closed forms \(e^{-|\Delta t|/\tau}\) and \([1+(\Delta\omega\tau)^2]^{-1}\) | `test_models.py` |
| Mixed Gaussian/exponential overlap is correct | Direct complex time-domain quadrature compared with the analytic `erfcx` expression | `test_models.py` |
| Vectorized delay scans preserve physics | Pointwise comparison against scalar closed forms for unequal Gaussian, unequal exponential, and both mixed-profile orders | `test_models.py` |
| Passive Fock probabilities are correct | Batched Glynn permanents checked against an independent Ryser implementation; HOM and distinguishable limits | `test_circuit.py` |
| General partial distinguishability retains the full Gram matrix | Independent double-permutation formula, including repeated input modes and complex overlaps | `test_circuit.py` |
| Spectral transfer matrices act at each photon's wavelength | Calibrated interpolation limits and per-photon routing checks | `test_circuit.py`, `test_adapters_backends.py` |
| Equal-mode SPDC/SFWM counts follow the real-\(K\) negative-binomial law | Exact PMF and tail; 500,000-sample mean, variance, and \(g^{(2)}\) checks | `test_sources.py` |
| Nonuniform Schmidt weights reproduce multimode thermal statistics | Equal-weight convolution agrees with negative binomial; sampled moments and product-form herald PGF for a nonuniform spectrum | `test_sources.py` |
| Herald conditioning includes multipairs and false-herald vacuum | Exact threshold generating function, conditional normalization, dark-only vacuum limit | `test_sources.py` |
| Heralded SPDC HOM includes exact two-mode multiphoton interference under its stated overlap closure | Quantum/classical two-photon limits and a four-photon result compared with the general Gram-matrix solver to \(2\times10^{-12}\) absolute tolerance | `test_sources.py` |
| Loop energy accounting distinguishes loss from residual circulation | Transfer-matrix passivity and explicit output/residual/loss conservation | `test_timebin.py` |
| Shared loop environments do not decorrelate simultaneous photons artificially | Same-bin phase and PMD realizations compared event by event | `test_hardware.py` |
| Multi-pixel SNSPD routing and recovery are statistically physical | Analytic occupancy mean, weighted routing frequencies, collision loss, and parent-pixel afterpulse checks | `test_hardware.py` |
| Correlated Monte Carlo uncertainty does not assume iid shots | iid Bernoulli and persistent Markov sequences compared through integrated autocorrelation time and effective sample size | `test_calibration.py` |
| Research bundles are lossless and tamper-evident | Full round trip, mutation detection, malformed-array rejection, exponential-profile schema 2, and schema 1 backward reading | `test_research.py` |

Statistical tests use fixed seeds and tolerances chosen from their sampling
variance. They test distributions rather than requiring identical random-number
sequences across NumPy releases, except where deterministic stream hierarchy is
itself the contract.

## Numerical and performance evidence

The release benchmark stores every timing sample and its runtime descriptor in
[`../benchmarks/results/results_2_2_0.json`](../benchmarks/results/results_2_2_0.json). On the
recorded Windows/Python 3.12 system:

- one million real-\(K\) SPDC pair counts take below 0.1 s;
- a 101-point heralded multipair HOM scan takes below 0.1 s;
- a 4,001-point exponential HOM scan takes below 0.2 ms;
- a 50,000-shot event digital twin takes below 1 s;
- cached CuPy propagation on the tested GTX 1060 remains operational.

Wall-clock timings are hardware- and scheduler-dependent. The raw samples, not
rounded prose values, are the authoritative performance record.

## Reproducibility controls

Every `DigitalTwin.run` records the complete canonicalized configuration,
dependency versions, acquisition bounds, SI units, root RNG entropy, and the
`seedsequence-hierarchy-v2` stream policy. Optional sealing hashes all physical
event, wavepacket, tag, and metadata fields. Pickle is never loaded. Bundle
schema 2 records temporal profiles; schema 1 Gaussian bundles remain readable
and retain their original hash policy.

For a paper, archive the wheel, input data, configuration manifest, sealed
result bundle, analysis script, and the SHA-256 values of all artifacts. Report
the TBL version, Python/dependency versions, random seed, shot count, truncation
tolerance, confidence method, and every fitted or measured hardware parameter.

## Validity boundaries that must be disclosed

- `DigitalTwin` samples event paths and must not be used for coherent path
  recombination. Use `FockSimulator` or `CoherentTimeBinLoop` for that regime.
- The SPDC Schmidt-weight model captures populations and pair statistics, but
  not Schmidt-mode shapes, phases, or a continuous joint spectral amplitude.
- Multipair HOM uses one uniform cross-source overlap. It is exact for two pure
  internal modes and a documented closure for mixed multimode states.
- Fiber PMD is a calibrated scalar arrival-time perturbation, not a
  polarization-resolved principal-state propagation model.
- Detector afterpulsing, recovery, jitter tails, switching curves, phase drift,
  and spectral diffusion are phenomenological and should be fitted to the
  specific experiment.
- The optical network is passive and linear. Kerr, Raman, Brillouin,
  master-equation baths, and detector microscopic electrothermal dynamics are
  outside scope.
- Exact Fock and general partial-distinguishability calculations scale
  exponentially or factorially; reported truncation limits are part of the
  model definition, not merely implementation details.

Within these declared regimes, the tests above support use of TBL 2.2.0 to
generate figures, uncertainty estimates, simulation datasets, and calibrated
comparisons for a research publication. Claims outside them require an
additional model and independent validation.
