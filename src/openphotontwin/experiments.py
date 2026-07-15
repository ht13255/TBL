"""Reusable quantum-optics experiments and observable calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .errors import ValidationError
from .models import Wavepacket, _probability


@dataclass(frozen=True, slots=True)
class HOMResult:
    delays: np.ndarray
    coincidence_probability: np.ndarray
    coincidence_counts: np.ndarray | None
    shots_per_delay: int | None
    visibility: float
    indistinguishability: np.ndarray

    def as_dataframe(self):
        """Return a pandas DataFrame without making pandas an import-time cost."""

        import pandas as pd

        data: dict[str, object] = {
            "delay": self.delays,
            "coincidence_probability": self.coincidence_probability,
            "indistinguishability": self.indistinguishability,
        }
        if self.coincidence_counts is not None:
            data["coincidence_counts"] = self.coincidence_counts
        return pd.DataFrame(data)

    def plot(self, ax=None):
        """Plot the HOM dip and return the Matplotlib axes."""

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.delays, self.coincidence_probability, label="probability")
        if self.coincidence_counts is not None and self.shots_per_delay:
            ax.scatter(
                self.delays,
                self.coincidence_counts / self.shots_per_delay,
                s=14,
                alpha=0.7,
                label="sampled",
            )
        ax.set(xlabel="Delay (s)", ylabel="Coincidence probability", title="HOM interference")
        ax.legend()
        return ax


def hom_probabilities(
    first: Wavepacket,
    second: Wavepacket,
    *,
    reflectivity: float = 0.5,
) -> tuple[float, float, float]:
    """Return probabilities ``(coincidence, two-at-a, two-at-b)``."""

    reflectivity = _probability("reflectivity", reflectivity)
    transmission = 1 - reflectivity
    overlap = first.indistinguishability(second)
    coincidence = reflectivity**2 + transmission**2 - 2 * reflectivity * transmission * overlap
    bunching = reflectivity * transmission * (1 + overlap)
    return float(coincidence), float(bunching), float(bunching)


def hom_scan(
    delays: Sequence[float],
    first: Wavepacket,
    second: Wavepacket,
    *,
    reflectivity: float = 0.5,
    shots_per_delay: int | None = None,
    detector_efficiency: float = 1.0,
    background_probability: float = 0.0,
    seed: int | None = None,
) -> HOMResult:
    """Run an analytical or shot-sampled Hong-Ou-Mandel delay scan."""

    delay_array = np.asarray(delays, dtype=float)
    if delay_array.ndim != 1 or delay_array.size == 0:
        raise ValidationError("delays must be a non-empty one-dimensional sequence")
    efficiency = _probability("detector_efficiency", detector_efficiency)
    background = _probability("background_probability", background_probability)
    if shots_per_delay is not None and shots_per_delay < 1:
        raise ValidationError("shots_per_delay must be positive")
    probabilities = []
    overlaps = []
    for delay in delay_array:
        shifted = second.shifted(float(delay))
        overlap = first.indistinguishability(shifted)
        coincidence, _, _ = hom_probabilities(first, shifted, reflectivity=reflectivity)
        observed = coincidence * efficiency**2
        observed = observed + (1 - observed) * background
        probabilities.append(observed)
        overlaps.append(overlap)
    probability_array = np.asarray(probabilities)
    counts = None
    if shots_per_delay is not None:
        counts = np.random.default_rng(seed).binomial(shots_per_delay, probability_array)
    reflectivity = _probability("reflectivity", reflectivity)
    distinguishable_baseline = reflectivity**2 + (1 - reflectivity) ** 2
    minimum = float(np.min(probability_array - background)) / max(
        efficiency**2, np.finfo(float).eps
    )
    visibility = 1 - minimum / distinguishable_baseline if distinguishable_baseline else 0.0
    return HOMResult(
        delay_array,
        probability_array,
        counts,
        shots_per_delay,
        float(np.clip(visibility, 0.0, 1.0)),
        np.asarray(overlaps),
    )
