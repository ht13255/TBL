# Benchmark result archive

This directory contains immutable JSON measurements captured for released TBL
versions. Each file stores raw samples; newer files also include the complete
runtime descriptor. Benchmark programs live one directory above.

Run the current suite without modifying the archive:

```bash
python benchmarks/benchmark_core.py
```

To deliberately record a new release result:

```bash
python benchmarks/benchmark_core.py --output benchmarks/results/results_X_Y_Z.json
```

Do not compare wall times without checking the recorded Python, NumPy, SciPy,
CPU, GPU, and operating-system details.
