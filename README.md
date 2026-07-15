# TBL

**TBL** is a Python simulation toolkit and hardware digital twin for time-bin and fiber-loop quantum photonics. It combines small-Fock-state interference models with realistic timing, loss, switching, detector, and time-tag effects.

The package is designed to run without a GUI and can be imported directly from Python.

> Python 3.10+ · NumPy/SciPy · MIT License

## Highlights

- Gaussian single-photon wavepackets with temporal, spectral, polarization, and purity overlap
- Exact permanent-based passive linear-optical Fock simulation for small photon numbers
- Beam splitters, dynamic beam splitters, phase shifts, delay lines, loss, and phase drift
- Time-bin qubits, fiber loops, length errors, and dynamic out-coupling
- EOM switching, finite extinction ratio, electronic control, and FPGA feed-forward latency
- SNSPD efficiency, timing jitter, dead time, dark counts, and time-tag generation
- HOM-dip scans, coincidence histograms, photon-number distributions, and experiment comparison
- CSV time-tag loading, HOM fitting, indistinguishability/loss estimation, and loss localization
- Optional adapters for Perceval, Strawberry Fields, and GDSFactory/SAX-style S-parameters
- NumPy CPU backend with optional CuPy batch propagation

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

The core dependencies are NumPy, SciPy, pandas, and Matplotlib. Install CuPy separately when using the GPU backend, matching your CUDA version. Optional external circuit packages are only needed for their respective adapters.

## Quick start

```python
import numpy as np
import openphotontwin as tbl

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

## Exact Fock simulation

```python
import openphotontwin as tbl

circuit = tbl.LinearOpticalCircuit(2).beam_splitter(
    0, 1, reflectivity=0.5
)
distribution = tbl.FockSimulator(circuit).probabilities([1, 1])

print(distribution.probabilities)
print(distribution.loss_probability)
```

For a lossless 50:50 beam splitter, the two-photon input demonstrates Hong-Ou-Mandel suppression of the `(1, 1)` output. In lossy circuits, the missing probability is reported as `loss_probability`.

## Fiber-loop digital twin

```python
import openphotontwin as tbl

source = tbl.SinglePhotonSource(
    repetition_rate=10e6,
    p_single=0.8,
    p_double=0.01,
    emission_jitter=5e-12,
)
loop = tbl.FiberLoop(
    round_trip_time=20e-9,
    transmission=0.94,
    outcoupling={1: 0.0, 2: 0.2, 3: 1.0},
    phase_drift_std=0.01,
    max_roundtrips=4,
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
import openphotontwin as tbl

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

Use `fit_hom_dip`, `estimate_loss`, `compare_to_ideal`, and `auto_calibrate` to analyze measured data against an ideal model.

## Scope and assumptions

- `FockSimulator` models coherent multi-photon probabilities in passive linear networks. Permanent calculations become expensive as photon number grows.
- `DigitalTwin` provides event-level Monte Carlo for hardware timing, noise, loss, and time-tag generation.
- `FiberLoop` models per-round-trip transmission, Gaussian phase drift, and controlled out-coupling.
- Absolute loss location cannot be identified from only one input/output measurement pair; ordered intermediate measurements are required.
- HOM Gaussian fitting assumes transform-limited Gaussian wavepackets.

All times are in seconds, lengths in meters, and frequencies in SI units. Randomized APIs accept `seed` for reproducibility.

## Development and testing

Install development dependencies, then run:

```bash
python -m pytest
python -m ruff check .
```

Runnable examples are available in [`examples/`](examples). The implementation is in [`src/openphotontwin/`](src/openphotontwin/), with tests in [`tests/`](tests).

## License

TBL is released under the [MIT License](LICENSE).
