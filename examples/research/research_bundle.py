"""Create, verify, save, and reload a reproducible TBL research result."""

from pathlib import Path

import tbl

source = tbl.SinglePhotonSource(repetition_rate=20e6, p_single=0.75, p_double=0.01)
detectors = tbl.DetectorArray({0: tbl.SNSPD(efficiency=0.88, jitter=15e-12, dead_time=30e-9)})
twin = tbl.DigitalTwin(source, [tbl.DelayLine(5e-9)], detectors)

result = twin.run(10_000, seed=2026, integrity=True)
assert result.verify_integrity()

destination = result.save_bundle(Path("output") / "research-run.npz")
restored = tbl.load_simulation_result(destination)
assert restored.verify_integrity()

print("configuration:", restored.manifest.configuration_sha256)
print("result:", restored.manifest.result_sha256)
