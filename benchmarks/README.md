# Core performance benchmarks

Run from an installed checkout with:

```bash
python benchmarks/benchmark_core.py
```

The benchmark uses fixed random seeds, warmups, garbage-collection control,
and median wall time. TBL 2.0.0 was measured on Windows with Python 3.12,
NumPy 2.5.1, and an NVIDIA GTX 1060 3 GB.

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
are stored in `results_2_0_0.json`.

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
measurements are in `results_gpu_1_2_0.json`.
