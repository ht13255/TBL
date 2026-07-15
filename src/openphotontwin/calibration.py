"""Experimental time-tag analysis, parameter estimation, and model comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .detectors import TimeTag
from .errors import SimulationError, ValidationError


@dataclass(frozen=True, slots=True)
class CoincidenceHistogram:
    bin_centers: np.ndarray
    counts: np.ndarray
    bin_width: float
    channel_pair: tuple[int, int]

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.step(self.bin_centers, self.counts, where="mid")
        ax.set(xlabel="Relative delay (s)", ylabel="Coincidences", title="Coincidence histogram")
        return ax


@dataclass(frozen=True, slots=True)
class HOMFit:
    baseline: float
    visibility: float
    center: float
    width: float
    background: float
    covariance: np.ndarray
    fitted: np.ndarray
    r_squared: float

    @property
    def indistinguishability(self) -> float:
        return self.visibility


@dataclass(frozen=True, slots=True)
class LossEstimate:
    total_transmission: float
    per_pass_transmission: float
    total_loss: float
    loss_db: float


@dataclass(frozen=True, slots=True)
class LossProfile:
    """Loss estimates between ordered experimental checkpoints."""

    segments: Mapping[str, LossEstimate]
    dominant_segment: str | None
    cumulative_transmission: float


@dataclass(frozen=True, slots=True)
class ModelComparison:
    rmse: float
    mae: float
    reduced_chi_squared: float
    correlation: float
    residuals: np.ndarray


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    hom: HOMFit | None = None
    loss: LossEstimate | None = None
    comparison: ModelComparison | None = None


def load_time_tags(
    path: str | Path,
    *,
    time_column: str = "time",
    channel_column: str = "channel",
    shot_column: str = "shot",
    time_scale: float = 1.0,
) -> pd.DataFrame:
    """Load and normalize CSV time-tag data.

    Extra columns are preserved. ``time_scale`` converts the file's unit to
    seconds (for example, use ``1e-12`` for picoseconds).
    """

    if time_scale <= 0:
        raise ValidationError("time_scale must be positive")
    frame = pd.read_csv(path)
    required = {time_column, channel_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValidationError(f"missing required CSV columns: {sorted(missing)}")
    rename = {time_column: "time", channel_column: "channel"}
    if shot_column in frame.columns:
        rename[shot_column] = "shot"
    frame = frame.rename(columns=rename)
    frame["time"] = pd.to_numeric(frame["time"], errors="raise") * time_scale
    frame["channel"] = pd.to_numeric(frame["channel"], errors="raise").astype(int)
    if "shot" not in frame:
        frame["shot"] = -1
    frame["shot"] = pd.to_numeric(frame["shot"], errors="raise").astype(int)
    return frame.sort_values("time", kind="stable").reset_index(drop=True)


def tags_to_dataframe(tags: Sequence[TimeTag]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [tag.time for tag in tags],
            "channel": [tag.channel for tag in tags],
            "shot": [tag.shot for tag in tags],
            "dark_count": [tag.dark_count for tag in tags],
        }
    )


def _as_frame(tags: pd.DataFrame | Sequence[TimeTag]) -> pd.DataFrame:
    if isinstance(tags, pd.DataFrame):
        if not {"time", "channel"}.issubset(tags.columns):
            raise ValidationError("time-tag data needs time and channel columns")
        return tags
    return tags_to_dataframe(tags)


def coincidence_histogram(
    tags: pd.DataFrame | Sequence[TimeTag],
    channel_a: int,
    channel_b: int,
    *,
    bin_width: float,
    max_delay: float,
) -> CoincidenceHistogram:
    """Histogram all cross-channel delays within a symmetric time window."""

    if bin_width <= 0 or max_delay <= 0:
        raise ValidationError("bin_width and max_delay must be positive")
    frame = _as_frame(tags)
    first = np.sort(frame.loc[frame["channel"] == channel_a, "time"].to_numpy(float))
    second = np.sort(frame.loc[frame["channel"] == channel_b, "time"].to_numpy(float))
    edge_count = int(np.ceil(2 * max_delay / bin_width))
    edges = np.linspace(-max_delay, max_delay, edge_count + 1)
    counts = np.zeros(edge_count, dtype=np.int64)
    for timestamp in first:
        low = np.searchsorted(second, timestamp - max_delay, side="left")
        high = np.searchsorted(second, timestamp + max_delay, side="right")
        if high > low:
            local, _ = np.histogram(second[low:high] - timestamp, bins=edges)
            counts += local
    return CoincidenceHistogram(
        (edges[:-1] + edges[1:]) / 2, counts, bin_width, (channel_a, channel_b)
    )


def _hom_model(
    delay: np.ndarray,
    baseline: float,
    visibility: float,
    center: float,
    width: float,
    background: float,
) -> np.ndarray:
    return background + baseline * (
        1 - visibility * np.exp(-((delay - center) ** 2) / (2 * width**2))
    )


def fit_hom_dip(delays: Sequence[float], coincidences: Sequence[float]) -> HOMFit:
    """Fit a Gaussian HOM dip using bounded nonlinear least squares."""

    x = np.asarray(delays, dtype=float)
    y = np.asarray(coincidences, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 5:
        raise ValidationError("HOM fitting needs at least five paired samples")
    if np.any(y < 0) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValidationError("HOM samples must be finite and non-negative")
    span = float(np.ptp(x))
    if span <= 0:
        raise ValidationError("HOM delays must span a non-zero interval")
    # Work in a dimensionless delay coordinate. Optimizers otherwise see
    # 1e-12-scale centre/width derivatives next to 1e3-scale count derivatives,
    # which makes the covariance ill-conditioned and can produce false minima.
    origin = float(np.mean(x))
    scaled_x = (x - origin) / span
    edge = max(1, x.size // 5)
    baseline_guess = max(float(np.mean(np.r_[y[:edge], y[-edge:]])), np.finfo(float).eps)
    background_guess = max(0.0, float(np.min(y)) * 0.05)
    visibility_guess = float(np.clip(1 - np.min(y) / baseline_guess, 0.01, 0.99))
    center_guess = float(scaled_x[np.argmin(y)])
    p0 = [baseline_guess, visibility_guess, center_guess, 1 / 6, background_guess]
    lower = [0.0, 0.0, float(np.min(scaled_x) - 1), 1 / 10_000, 0.0]
    upper = [
        max(float(np.max(y)) * 10, 1.0),
        1.0,
        float(np.max(scaled_x) + 1),
        10.0,
        max(float(np.max(y)) * 2, 1.0),
    ]
    sigma = np.sqrt(np.maximum(y, 1.0))
    try:
        parameters, covariance = curve_fit(
            _hom_model,
            scaled_x,
            y,
            p0=p0,
            bounds=(lower, upper),
            sigma=sigma,
            absolute_sigma=True,
            maxfev=100_000,
        )
    except (RuntimeError, ValueError) as exc:
        raise SimulationError(f"HOM fit did not converge: {exc}") from exc
    fitted = _hom_model(scaled_x, *parameters)
    residuals = y - fitted
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - float(np.sum(residuals**2)) / denominator if denominator else 1.0
    center = float(parameters[2] * span + origin)
    width = abs(float(parameters[3] * span))
    jacobian = np.diag([1.0, 1.0, span, span, 1.0])
    covariance = jacobian @ covariance @ jacobian
    return HOMFit(
        float(parameters[0]),
        float(parameters[1]),
        center,
        width,
        float(parameters[4]),
        covariance,
        fitted,
        r_squared,
    )


def estimate_loss(
    input_count: float,
    output_count: float,
    *,
    detector_efficiency: float = 1.0,
    passes: int = 1,
) -> LossEstimate:
    """Estimate optical transmission after correcting detector efficiency."""

    if input_count <= 0 or output_count < 0 or passes < 1:
        raise ValidationError("counts must be physical and passes at least one")
    if not 0 < detector_efficiency <= 1:
        raise ValidationError("detector_efficiency must be in (0, 1]")
    transmission = output_count / (input_count * detector_efficiency)
    if transmission > 1 + 1e-9:
        raise ValidationError("corrected output exceeds input; check counts or detector efficiency")
    transmission = float(np.clip(transmission, 0.0, 1.0))
    per_pass = transmission ** (1 / passes)
    loss_db = float(-10 * np.log10(transmission)) if transmission > 0 else float("inf")
    return LossEstimate(transmission, per_pass, 1 - transmission, loss_db)


def locate_loss(
    checkpoint_counts: Mapping[str, float],
    *,
    detector_efficiencies: Mapping[str, float] | float = 1.0,
) -> LossProfile:
    """Locate the dominant loss interval from ordered checkpoint counts.

    Checkpoints can be physical taps (``source``, ``after_switch``,
    ``after_loop``) or round-trip time-bin counts. Absolute loss position is
    not identifiable from a single input/output measurement; at least two
    checkpoints are therefore required. Each count is corrected by the
    detector efficiency at that checkpoint before adjacent ratios are taken.
    """

    items = list(checkpoint_counts.items())
    if len(items) < 2:
        raise ValidationError("loss localization requires at least two ordered checkpoints")
    corrected: list[tuple[str, float]] = []
    for name, count in items:
        efficiency = (
            detector_efficiencies.get(name)
            if isinstance(detector_efficiencies, Mapping)
            else detector_efficiencies
        )
        if efficiency is None or not 0 < efficiency <= 1:
            raise ValidationError(f"invalid or missing detector efficiency for {name!r}")
        if count < 0:
            raise ValidationError("checkpoint counts cannot be negative")
        corrected.append((name, float(count) / efficiency))
    if corrected[0][1] <= 0:
        raise ValidationError("the first checkpoint count must be positive")
    segments: dict[str, LossEstimate] = {}
    for (left_name, left), (right_name, right) in zip(corrected, corrected[1:], strict=False):
        if right > left * (1 + 1e-9):
            raise ValidationError(
                f"corrected count increases from {left_name!r} to {right_name!r}; "
                "check normalization"
            )
        segment_name = f"{left_name}->{right_name}"
        segments[segment_name] = estimate_loss(left, right)
    dominant = max(segments, key=lambda name: segments[name].total_loss) if segments else None
    cumulative = corrected[-1][1] / corrected[0][1]
    return LossProfile(segments, dominant, cumulative)


def compare_to_ideal(
    experimental: Sequence[float],
    ideal: Sequence[float],
    *,
    uncertainty: Sequence[float] | None = None,
    fitted_parameters: int = 0,
) -> ModelComparison:
    measured = np.asarray(experimental, dtype=float)
    expected = np.asarray(ideal, dtype=float)
    if measured.shape != expected.shape or measured.ndim != 1 or measured.size == 0:
        raise ValidationError("experimental and ideal arrays must be equal non-empty vectors")
    residuals = measured - expected
    if uncertainty is None:
        sigma = np.sqrt(np.maximum(abs(expected), 1.0))
    else:
        sigma = np.asarray(uncertainty, dtype=float)
        if sigma.shape != measured.shape or np.any(sigma <= 0):
            raise ValidationError("uncertainties must be positive and match data")
    degrees = max(1, measured.size - fitted_parameters)
    chi_squared = float(np.sum((residuals / sigma) ** 2) / degrees)
    correlation = float(np.corrcoef(measured, expected)[0, 1]) if measured.size > 1 else 1.0
    return ModelComparison(
        float(np.sqrt(np.mean(residuals**2))),
        float(np.mean(abs(residuals))),
        chi_squared,
        correlation,
        residuals,
    )


def auto_calibrate(
    *,
    delays: Sequence[float] | None = None,
    coincidences: Sequence[float] | None = None,
    ideal_coincidences: Sequence[float] | None = None,
    input_count: float | None = None,
    output_count: float | None = None,
    detector_efficiency: float = 1.0,
    passes: int = 1,
) -> CalibrationResult:
    """Run all estimators supported by the supplied experimental data."""

    hom = None
    loss = None
    comparison = None
    if delays is not None or coincidences is not None:
        if delays is None or coincidences is None:
            raise ValidationError("delays and coincidences must be supplied together")
        hom = fit_hom_dip(delays, coincidences)
    if input_count is not None or output_count is not None:
        if input_count is None or output_count is None:
            raise ValidationError("input_count and output_count must be supplied together")
        loss = estimate_loss(
            input_count,
            output_count,
            detector_efficiency=detector_efficiency,
            passes=passes,
        )
    if ideal_coincidences is not None:
        if coincidences is None:
            raise ValidationError("coincidences are required for ideal comparison")
        comparison = compare_to_ideal(coincidences, ideal_coincidences)
    return CalibrationResult(hom, loss, comparison)


def calibration_report(result: CalibrationResult) -> Mapping[str, float]:
    """Flatten a calibration result into serialization-friendly metrics."""

    report: dict[str, float] = {}
    if result.hom:
        report.update(
            hom_visibility=result.hom.visibility,
            indistinguishability=result.hom.indistinguishability,
            hom_center=result.hom.center,
            hom_width=result.hom.width,
            hom_r_squared=result.hom.r_squared,
        )
    if result.loss:
        report.update(
            total_transmission=result.loss.total_transmission,
            per_pass_transmission=result.loss.per_pass_transmission,
            total_loss=result.loss.total_loss,
            loss_db=result.loss.loss_db,
        )
    if result.comparison:
        report.update(
            rmse=result.comparison.rmse,
            mae=result.comparison.mae,
            reduced_chi_squared=result.comparison.reduced_chi_squared,
            correlation=result.comparison.correlation,
        )
    return report
