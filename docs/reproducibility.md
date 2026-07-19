# Reproducible research with TBL

Every `DigitalTwin.run` creates a versioned `RunManifest`. It records the seed,
actual SeedSequence entropy, PCG bit generator, acquisition window, SI units,
TBL/Python/dependency versions,
a canonical configuration snapshot, and separate SHA-256 fingerprints for the
configuration and result payload.

```python
result = twin.run(100_000, seed=2026, integrity=True)

print(result.manifest.configuration_sha256)
print(result.manifest.result_sha256)
assert result.verify_integrity()

result.save_bundle("experiment-2026-07-16.npz")
restored = tbl.load_simulation_result("experiment-2026-07-16.npz")
assert restored.verify_integrity()
```

The `tbl.simulation-result/2` bundle contains the complete propagated events,
photon wavepackets, complex amplitudes, detector tags, nested metadata, units,
and manifest. It uses compressed NumPy arrays and canonical JSON only;
`allow_pickle=False` is enforced on load. Modified payload or provenance data
is rejected by checksum verification. The manifest records the versioned
`event-binary-v2` result-hash policy, including the temporal profile, so future
readers never have to infer which canonicalization produced a digest. TBL 2.2
continues to read schema-1 / binary-v1 Gaussian bundles.

`integrity=True` seals an in-memory result immediately. The default run records
configuration provenance without paying the event-payload hashing cost, which
is preferable for large exploratory sweeps. Calling `result.seal()` returns a
sealed copy, and `save_bundle` always seals the archived manifest automatically.

## Reproduction protocol

For a publication or hardware comparison, archive:

1. the `.npz` result bundle;
2. the exact TBL wheel and its SHA-256;
3. input measurement files such as time tags or S-parameters;
4. the analysis script or notebook;
5. calibration data and instrument settings not represented by the model.

The `seedsequence-hierarchy-v2` policy gives the source, each component, and
the detector stage independent deterministic streams. Each detector then gets
a child stream keyed by `(mode, channel)`. An unrelated component, dictionary
order change, or unused detector therefore cannot perturb an existing channel.

Use the manifest's configuration hash to establish that two jobs used the
same model. Use its result hash to establish that their complete outputs are
identical. A fixed seed is reproducible only together with the recorded NumPy
version: random-stream implementation details can change between dependency
releases. If `seed=None` was used, rerun with the recorded `rng_entropy` value
to reproduce the generated stream.

## Portable and non-portable configurations

Dataclass-based TBL sources, components, and detectors are captured as portable
configuration. User callables are identified by module and qualified name.
Lambdas and functions defined in `__main__` are marked non-portable because
their implementation cannot be reconstructed from a name. Archive their source
code and closure parameters separately, or replace them with an importable,
version-controlled callable.

The manifest is provenance, not an automatic hardware calibration certificate.
Phenomenological parameters still require independently documented measurements
and uncertainty estimates.

For detector click probabilities, use `SimulationResult.detection_probability`
instead of reporting only a point estimate. It returns the observed Bernoulli
probability, standard error, and exact Clopper-Pearson finite-sample interval.
That interval assumes independent shots. For blinking, drift, feedback, or any
correlated source, use `SimulationResult.detection_convergence`; it estimates
integrated autocorrelation time by FFT with an initial-positive-sequence
truncation and reports the effective sample size, batch means, Monte Carlo
standard error, and an autocorrelation-adjusted Wilson interval.
