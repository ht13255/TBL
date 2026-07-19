# TBL examples

Every example imports the installed `tbl` package and can be run from any
working directory. Install TBL first, then invoke a script by path.

## Interference

- `interference/hom_scan.py` — Gaussian single-photon HOM scan.
- `interference/quantum_dot_hom.py` — causal exponential quantum-dot photons.
- `interference/heralded_spdc_hom.py` — multipair heralded SPDC HOM experiment.
- `interference/exact_partial_distinguishability.py` — full Gram-matrix Fock calculation.

## Hardware digital twins

- `hardware/coherent_timebin_loop.py` — coherent expanded-mode loop.
- `hardware/fiber_loop_twin.py` — event-domain fiber-loop time tags.
- `hardware/time_bin_control.py` — scheduled switching and time-bin routing.
- `hardware/realistic_digital_twin.py` — source, hardware, and SNSPD chain.

## Sources

- `sources/spdc_heralding.py` — multimode-thermal pair and herald statistics.

## Research workflows

- `research/calibrate_experiment.py` — HOM, loss, and model calibration.
- `research/research_bundle.py` — sealed, pickle-free result persistence.

For example:

```bash
python examples/interference/heralded_spdc_hom.py
python examples/hardware/fiber_loop_twin.py
```

Scripts that write CSV or NPZ output place it in the current working directory,
not alongside the source file.
