"""Fit synthetic HOM data, infer loss, and print a flat calibration report."""

import numpy as np

import openphotontwin as opt

rng = np.random.default_rng(4)
delays = np.linspace(-100e-12, 100e-12, 81)
ideal = 1500 * (1 - 0.94 * np.exp(-((delays - 4e-12) ** 2) / (2 * (28e-12) ** 2))) + 12
measured = rng.poisson(ideal)

result = opt.auto_calibrate(
    delays=delays,
    coincidences=measured,
    ideal_coincidences=ideal,
    input_count=1_000_000,
    output_count=640_000,
    detector_efficiency=0.8,
    passes=5,
)
print(opt.calibration_report(result))
