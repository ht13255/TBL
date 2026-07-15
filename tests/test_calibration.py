import numpy as np
import pandas as pd
import pytest

import openphotontwin as opt


def test_hom_scan_visibility_and_sampled_counts():
    packet = opt.Wavepacket(temporal_width=20e-12)
    scan = opt.hom_scan(
        np.linspace(-200e-12, 200e-12, 101), packet, packet, shots_per_delay=1000, seed=2
    )
    assert scan.visibility == pytest.approx(1.0, abs=1e-6)
    assert scan.coincidence_counts.shape == (101,)
    assert scan.coincidence_probability[50] == pytest.approx(0.0)


def test_hom_fit_recovers_synthetic_parameters():
    rng = np.random.default_rng(3)
    delays = np.linspace(-100e-12, 100e-12, 101)
    expected = 20 + 1000 * (1 - 0.91 * np.exp(-((delays - 5e-12) ** 2) / (2 * (25e-12) ** 2)))
    measured = rng.poisson(expected)
    fit = opt.fit_hom_dip(delays, measured)
    assert fit.visibility == pytest.approx(0.91, abs=0.05)
    assert fit.center == pytest.approx(5e-12, abs=3e-12)
    assert fit.width == pytest.approx(25e-12, abs=4e-12)
    assert fit.r_squared > 0.98


def test_loss_estimation_and_validation():
    loss = opt.estimate_loss(1000, 405, detector_efficiency=0.9, passes=2)
    assert loss.total_transmission == pytest.approx(0.45)
    assert loss.per_pass_transmission == pytest.approx(np.sqrt(0.45))
    with pytest.raises(opt.ValidationError):
        opt.estimate_loss(100, 101)


def test_loss_localization_finds_dominant_interval():
    profile = opt.locate_loss(
        {"source": 1000, "switch": 900, "loop": 450, "detector": 405},
        detector_efficiencies={"source": 1, "switch": 1, "loop": 1, "detector": 0.9},
    )
    assert profile.dominant_segment == "switch->loop"
    assert profile.cumulative_transmission == pytest.approx(0.45)
    assert profile.segments["switch->loop"].total_loss == pytest.approx(0.5)


def test_model_comparison_and_auto_calibration():
    ideal = np.array([10, 20, 30, 40, 50], dtype=float)
    measured = ideal + np.array([0, 1, -1, 1, -1])
    comparison = opt.compare_to_ideal(measured, ideal)
    assert comparison.rmse == pytest.approx(np.sqrt(0.8))
    result = opt.auto_calibrate(
        delays=np.arange(5),
        coincidences=[10, 8, 5, 8, 10],
        ideal_coincidences=[10, 8, 5, 8, 10],
        input_count=100,
        output_count=80,
    )
    report = opt.calibration_report(result)
    assert "hom_visibility" in report
    assert report["total_transmission"] == pytest.approx(0.8)


def test_time_tag_csv_and_coincidence_histogram(tmp_path):
    path = tmp_path / "tags.csv"
    pd.DataFrame({"t_ps": [0, 10, 100, 111], "det": [0, 1, 0, 1], "shot": [0, 0, 1, 1]}).to_csv(
        path, index=False
    )
    tags = opt.load_time_tags(path, time_column="t_ps", channel_column="det", time_scale=1e-12)
    histogram = opt.coincidence_histogram(tags, 0, 1, bin_width=5e-12, max_delay=20e-12)
    assert histogram.counts.sum() == 2
    assert tags["time"].max() == pytest.approx(111e-12)
