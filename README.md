# TBL

**TBL** is a Python simulation toolkit and hardware digital twin for time-bin and fiber-loop quantum photonics. It combines small-Fock-state interference models with realistic timing, loss, switching, detector, and time-tag effects.

The package is designed to run without a GUI and can be imported directly from Python.

> Python 3.10+ · NumPy/SciPy · MIT License

Current release: **2.0.0**

## Highlights

- Chirped Gaussian photons with analytic delay, detuning, polarization, purity, and dispersion
- Exact full-Gram-matrix partial distinguishability, including multiphoton overlap phases
- Wavelength-resolved measured transfer matrices and per-photon S-parameter interpolation
- Beam splitters, dynamic beam splitters, phase shifts, delay lines, loss, and phase drift
- Time-bin qubits and fiber loops with dB/km loss, insertion loss, beta2 dispersion, PMD, and Wiener phase noise
- Sources with measured g2(0), collection loss, Markov blinking, and OU spectral diffusion
- EOM insertion loss, drive noise, extinction ratio, rise time, and FPGA feed-forward latency
- Chronological SNSPD avalanches with recovery, pixels, wavelength response, jitter tails, afterpulses, and tagger resolution
- HOM-dip scans, coincidence histograms, photon-number distributions, and experiment comparison
- CSV time tags, Poisson HOM fitting, bootstrap/Fisher uncertainty, loss intervals, and loss localization
- Optional adapters for Perceval, Strawberry Fields, and GDSFactory/SAX-style S-parameters
- NumPy CPU backend with optional cached, device-resident CuPy batch propagation

## Download and installation

Clone the public repository:

```bash
git clone https://github.com/ht13255/TBL.git
cd TBL
```

Create an isolated environment and install TBL in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a normal installation, use `python -m pip install .`. You can also download the source archive from GitHub using **Code → Download ZIP**.

The core dependencies are NumPy, SciPy, pandas, and Matplotlib. For CUDA 12,
install `python -m pip install ".[gpu]"`; CUDA 11 users can install
`python -m pip install ".[gpu-cuda11]"`. These extras include CuPy and its
user-space CUDA runtime libraries; a compatible NVIDIA driver is still
required. Optional external circuit packages are only needed for their
respective adapters.

## Migrating to TBL 2.x

The canonical package is now `tbl`:

```python
import tbl
```

Existing `import openphotontwin` applications still work through a thin
compatibility shim and emit a `DeprecationWarning`. New code, documentation,
type annotations, and package metadata should use `tbl`. The compatibility
alias `OpenPhotonTwinError` remains available, while `TBLError` is the
canonical base exception.

## Repeated CPU/GPU propagation

`BatchPropagator` keeps a transfer matrix on the selected device across calls.
With `backend="auto"`, small products remain on CPU while larger products use a
GPU only after a real CuPy matrix-operation probe succeeds. The crossover is
adjusted for dtype and whether the result remains on-device; double precision
stays on CPU much longer on consumer GPUs. An installed but broken CUDA runtime
therefore falls back safely to NumPy.

```python
import numpy as np
import tbl

matrix = np.eye(128, dtype=np.complex64)
propagator = tbl.BatchPropagator(matrix, backend="auto")

host_result = propagator(np.ones((2048, 128), dtype=np.complex64))

# In a GPU-only pipeline, create a GPU propagator and retain its output there.
gpu = tbl.get_backend("cupy").module
gpu_propagator = tbl.BatchPropagator(matrix, backend="cupy")
device_result = gpu_propagator(gpu.ones((2048, 128), dtype=gpu.complex64), return_device=True)
gpu_propagator.synchronize()
```

Use `return_device=True` between GPU stages to avoid an unnecessary device-to-host
copy. Explicit `backend="cupy"` requests raise `OptionalDependencyError` with the
runtime failure; `backend="auto"` falls back to CPU.

## Quick start

```python
import numpy as np
import tbl

photon_a = tbl.Wavepacket(temporal_width=20e-12, wavelength=1550e-9)
photon_b = tbl.Wavepacket(temporal_width=20e-12, wavelength=1550e-9)

result = tbl.hom_scan(
    np.linspace(-100e-12, 100e-12, 101),
    photon_a,
    photon_b,
    shots_per_delay=10_000,
    detector_efficiency=0.92,
    seed=7,
)

print(f"HOM visibility: {result.visibility:.3f}")
result.as_dataframe().to_csv("hom.csv", index=False)
```

For scripts running on servers or CI without a display, configure Matplotlib
before importing `matplotlib.pyplot`:

```python
import tbl

tbl.configure_matplotlib("Agg")
```

## Exact Fock simulation

```python
import tbl

circuit = tbl.LinearOpticalCircuit(2).beam_splitter(
    0, 1, reflectivity=0.5
)
distribution = tbl.FockSimulator(circuit).probabilities([1, 1])

print(distribution.probabilities)
print(distribution.loss_probability)
```

For a lossless 50:50 beam splitter, the two-photon input demonstrates Hong-Ou-Mandel suppression of the `(1, 1)` output. In lossy circuits, the missing probability is reported as `loss_probability`.

For three or more partially distinguishable photons, pass either one
`Wavepacket` per photon or a measured positive-semidefinite complex Gram matrix:

```python
distribution = tbl.FockSimulator(circuit).probabilities(
    [1, 1, 1], overlap_matrix=measured_internal_gram
)
```

This uses the exact permutation/permanent expression rather than averaging all
pairwise visibilities into one number.

## Coherent time-bin loop interference

Use `CoherentTimeBinLoop` when amplitudes from different injections or round
trips meet in the same time bin. The expanded transfer matrix plugs directly
into `FockSimulator`:

```python
loop = tbl.CoherentTimeBinLoop(
    time_bins=12,
    bin_width=10e-9,
    round_trip_bins=2,
    reflectivity={0: 1.0, 2: 0.5, 10: 1.0},
    round_trip_transmission=0.94,
    phase_noise_std=0.01,
    phase_correlation_time=1e-6,
)
fock = tbl.FockSimulator(loop.circuit(seed=4)).probabilities(
    [1, 0, 1] + [0] * 9
)
detailed = loop.simulate([1] + [0] * 11)
print(detailed.output_probability)
print(detailed.residual_loop_probability)
print(detailed.physical_loss_probability)
```

`simulate` distinguishes true optical loss from energy still circulating after
the final simulated bin. Event-domain `FiberLoop` remains faster for time-tag
Monte Carlo when coherent recombination is not required.

## Fiber-loop digital twin

```python
import tbl

source = tbl.CorrelatedPhotonSource(
    repetition_rate=10e6,
    mean_photon_number=0.8,
    g2_zero=0.01,
    collection_efficiency=0.75,
    emission_jitter=5e-12,
    spectral_diffusion_std=2 * 3.14159 * 100e6,
    spectral_correlation_time=20e-6,
)
loop = tbl.FiberLoop(
    round_trip_time=20e-9,
    transmission=0.94,
    outcoupling={1: 0.0, 2: 0.2, 3: 1.0},
    phase_drift_std=0.01,
    max_roundtrips=4,
    fiber_length=4.08,
    attenuation_db_per_km=0.2,
    insertion_loss_db=0.8,
    dispersion_beta2=-21.7e-27,
)
detectors = tbl.DetectorArray({
    0: tbl.SNSPD(efficiency=0.9, jitter=20e-12, dead_time=40e-9, channel=0)
})

twin = tbl.DigitalTwin(source, [loop], detectors, time_bin_width=20e-9)
run = twin.run(10_000, seed=42)
run.save_time_tags("time_tags.csv")
```

## Experimental data and calibration

TBL accepts CSV time tags with at least `time` and `channel` columns. The optional `shot` column can preserve acquisition grouping.

```python
import tbl

tags = tbl.load_time_tags("time_tags.csv")
histogram = tbl.coincidence_histogram(
    tags, 0, 1, bin_width=20e-12, max_delay=2e-9
)
profile = tbl.locate_loss({
    "source": 100_000,
    "after_switch": 93_000,
    "after_loop": 61_000,
    "detector_input": 58_000,
})
print(profile.dominant_segment)
```

Use `fit_hom_dip`, `bootstrap_hom_fit`, `estimate_loss`,
`estimate_accidental_coincidences`, `compare_to_ideal`, and `auto_calibrate` to
analyze measured data. Supply independently measured accidental background to
`fit_hom_dip(..., background=...)`; background and intrinsic visibility cannot
both be identified from a HOM curve alone.

## Scope and assumptions

- `FockSimulator` models coherent multi-photon probabilities in passive linear networks. Exact partial distinguishability additionally scales factorially and defaults to at most seven photons.
- `DigitalTwin` provides event-level Monte Carlo for hardware timing, noise, loss, and time-tag generation.
- Event-domain beam splitters sample paths. Use `CoherentTimeBinLoop` and `FockSimulator` when paths coherently recombine.
- Absolute loss location cannot be identified from only one input/output measurement pair; ordered intermediate measurements are required.
- Stochastic hardware models are calibrated phenomenological models, not microscopic device simulations.

All times are in seconds, lengths in meters, and frequencies in SI units. Randomized APIs accept `seed` for reproducibility.

The governing equations, parameter conventions, validity regimes, and known
limitations are documented in [`docs/physics-model.md`](docs/physics-model.md).

## Development and testing

Install development dependencies, then run:

```bash
python -m pytest
python -m ruff check .
```

Runnable examples are available in [`examples/`](examples). The canonical implementation is in [`src/tbl/`](src/tbl/), with tests in [`tests/`](tests).
Release changes are listed in [`CHANGELOG.md`](CHANGELOG.md).

## License

TBL is released under the [MIT License](LICENSE).
