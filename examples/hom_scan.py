"""Generate a shot-sampled HOM dip and save CSV/PNG outputs."""

from pathlib import Path

import matplotlib
import numpy as np

import openphotontwin as opt

matplotlib.use("Agg")

output = Path("output")
output.mkdir(exist_ok=True)

first = opt.Wavepacket(temporal_width=25e-12)
second = opt.Wavepacket(temporal_width=25e-12, purity=0.97)
scan = opt.hom_scan(
    np.linspace(-150e-12, 150e-12, 121),
    first,
    second,
    shots_per_delay=20_000,
    detector_efficiency=0.92,
    seed=123,
)
scan.as_dataframe().to_csv(output / "hom_scan.csv", index=False)
scan.plot().figure.savefig(output / "hom_scan.png", dpi=180, bbox_inches="tight")
print(f"HOM visibility: {scan.visibility:.4f}")
