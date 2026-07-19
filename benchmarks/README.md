# Core performance benchmarks

Run from an installed checkout with:

```bash
python benchmarks/benchmark_core.py
```

The benchmark uses fixed random seeds, warmups, garbage-collection control,
and median wall time. TBL 2.0.0 was measured on Windows with Python 3.12,
NumPy 2.5.1, and an NVIDIA GTX 1060 3 GB.

## TBL 2.2.0 source, correlation, and wavepacket models

The 2.2.0 run used Windows 10, Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0,
an Intel family-6 model-94 CPU, and the same GTX 1060. The JSON now embeds
this environment descriptor rather than relying only on surrounding prose.

| Workload | TBL 2.2.0 median | Note |
|---|---:|---|
| 1,000,000 multimode-thermal SPDC pair counts | 85.4 ms | real-\(K\) Gamma-Poisson sampling |
| SPDC event objects for 100,000 pump pulses | 115.9 ms | signal/idler loss and metadata included |
| 101-point heralded SPDC HOM scan | 85.8 ms | multipairs, losses, and threshold detectors |
| Correlation-adjusted uncertainty, 200,000 shots | 42.4 ms | FFT autocovariance and Geyer truncation |
| 20,000 scalar mixed-profile overlaps | 216.7 ms | Gaussian/exponential complex `erfcx` integral |
| 20,000-point mixed-profile delay scan | 5.911 ms | vectorized exact complex integral |
| 4,001-point exponential HOM scan | 0.175 ms | about 1,000x faster than the former scalar scan loop |
| 50,000-shot digital twin | 0.710 s | configuration provenance included |
| GPU batch, 2048x128 | 4.271 ms | cached device-resident execution |

The vectorized HOM path was checked point-by-point against scalar closed-form
overlaps for unequal Gaussian, unequal exponential, and both mixed-profile
orders. Raw timings and all samples are stored in
`results/results_2_2_0.json`.

## TBL 2.1.0 research architecture

| Workload | TBL 2.1.0 median | Comparison |
|---|---:|---:|
| 20 permanents, 9x9 | 1.191 ms | 83.2x faster than 1.1.0 |
| 5 photons / 7 modes Fock distribution | 1.048 ms | 278.1x faster than 1.1.0 |
| 4 photons / 5 modes, exact partial overlap | 3.051 ms | 93.4x faster than 1.1.0 |
| 256-bin coherent loop matrix | 4.818 ms | 56.1x faster than 1.1.0 |
| 64-mode / 210-component transfer matrix | 2.690 ms | 15.3x faster than dense composition |
| 50,000-shot digital twin with configuration provenance | 0.520 s | 3.4x faster than 1.1.0 |
| Seal an existing 50,000-shot result | 1.241 s | full event/tag/metadata SHA-256 |
| CPU batch, 2048x128 | 5.200 ms | 7.2% faster than the legacy copy path |
| GPU batch, 2048x128 | 4.268 ms | cached device-resident GTX 1060 execution |

Configuration provenance is always recorded. Full payload sealing is measured
separately because exploratory sweeps do not need to hash every event; an
`integrity=True` run or `save_bundle` performs that explicit work. The binary-v1
sealing path is roughly three times faster than the first canonical-JSON
prototype. Raw samples are stored in `results/results_2_1_0.json`.

## TBL 2.0.0

| Workload | TBL 2.0.0 median | Comparison |
|---|---:|---:|
| 20 permanents, 9x9 | 0.976 ms | 101.5x faster than 1.1.0 |
| 5 photons / 7 modes Fock distribution | 1.194 ms | 244.1x faster than 1.1.0 |
| 4 photons / 5 modes, exact partial overlap | 3.517 ms | 81.0x faster than 1.1.0 |
| 256-bin coherent loop matrix | 3.982 ms | 67.8x faster than 1.1.0 |
| 64-mode / 210-component transfer matrix | 2.321 ms | 13.9x faster than dense composition |
| 50,000-shot event digital twin | 0.446 s | 4.0x faster than 1.1.0 |
| CPU batch, 2048x128 | 5.950 ms | 8.9% faster than the legacy copy path |
| GPU batch, 2048x128 | 4.269 ms | cached device-resident GTX 1060 execution |

TBL 2.0 fixes event-time/wavepacket consistency, stable overlap calculations
at large absolute timestamps, intrinsic HOM visibility with detector and
accidental effects, exact histogram bin centers, and stricter finite/Fock
validation. The event digital twin therefore does more physically required
work than 1.2.0; its timing should not be compared as though the two versions
implemented identical semantics. It remains about four times faster than the
original 1.1.0 baseline.

Beam splitters, phase shifters, and loss channels now update transfer-matrix
rows directly. The benchmark also computes the old dense component matrices
and asserts numerical equivalence before timing the 13.9x speedup. Raw samples
are stored in `results/results_2_0_0.json`.

## Historical TBL 1.2.0 results

The 1.2.0 comparison below used the same Windows system with an OpenBLAS
Haswell build (up to 24 threads), NumPy 2.5.1, and the GTX 1060.

| Workload | 1.1.0 baseline | 1.2.0 | Speedup |
|---|---:|---:|---:|
| 20 permanents, 9x9 | 99.03 ms | 1.005 ms | 98.5x |
| 5 photons / 7 modes Fock distribution | 291.35 ms | 1.240 ms | 234.9x |
| 4 photons / 5 modes, exact partial overlap | 285.10 ms | 3.333 ms | 85.5x |
| 256-bin coherent loop matrix | 270.14 ms | 4.289 ms | 63.0x |
| 50,000-shot event digital twin | 1.781 s | 0.307 s | 5.81x |
| CPU batch, 2048x128 | paired legacy path | 2.2% faster | 1.02x |

The BLAS-dominated CPU batch samples vary noticeably with Windows scheduling.
The benchmark therefore alternates the legacy and optimized operation on every
call; the final installed-wheel run measured the new dtype-preserving path 2.2% faster
while also avoiding forced-complex copies for float32 inputs.

CuPy 14.1.1 was tested with the pip-distributed CUDA 12.9 runtime on the GTX
1060. For complex64, cached GPU propagation became faster than CPU at roughly
16 million multiplies with a host round trip, or 4 million when the result
remained on-device. At 128 modes x 2048 states, device-resident execution was
18.8x faster and a full host round trip was 1.49x faster. Pascal-class
complex128 only reached host-round-trip parity around 256 million multiplies,
which is why `auto` applies a 16x higher double-precision threshold. Detailed
measurements are in `results/results_gpu_1_2_0.json`.
