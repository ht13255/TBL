"""Reusable TBL quantum-optics experiments and observable calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb, factorial, sqrt

import numpy as np

from ._runtime import ensure_matplotlib_cache
from .errors import ValidationError
from .models import Wavepacket, _probability
from .sources import HeraldedStatistics, SPDCSource


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

        ensure_matplotlib_cache()
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


@dataclass(frozen=True, slots=True)
class HeraldedHOMResult:
    """Threshold-detector HOM scan for two independently heralded pair sources."""

    delays: np.ndarray
    coincidence_probability: np.ndarray
    coincidence_counts: np.ndarray | None
    shots_per_delay: int | None
    distinguishable_probability: float
    visibility: float
    indistinguishability: np.ndarray
    herald_probabilities: tuple[float, float]
    multipair_fractions: tuple[float, float]
    omitted_pair_tail_probabilities: tuple[float, float]

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
    if not np.all(np.isfinite(delay_array)):
        raise ValidationError("delays must be finite")
    efficiency = _probability("detector_efficiency", detector_efficiency)
    background = _probability("background_probability", background_probability)
    if shots_per_delay is not None and (
        isinstance(shots_per_delay, bool)
        or not isinstance(shots_per_delay, (int, np.integer))
        or shots_per_delay < 1
    ):
        raise ValidationError("shots_per_delay must be a positive integer")
    reflectivity = _probability("reflectivity", reflectivity)
    transmission = 1 - reflectivity
    overlaps = first.indistinguishability_scan(second, delay_array)
    intrinsic_probabilities = (
        reflectivity**2 + transmission**2 - 2 * reflectivity * transmission * overlaps
    )
    probability_array = intrinsic_probabilities * efficiency**2
    probability_array += (1 - probability_array) * background
    counts = None
    if shots_per_delay is not None:
        counts = np.random.default_rng(seed).binomial(shots_per_delay, probability_array)
    distinguishable_baseline = reflectivity**2 + transmission**2
    minimum = float(np.min(intrinsic_probabilities))
    visibility = 1 - minimum / distinguishable_baseline if distinguishable_baseline else 0.0
    return HOMResult(
        delay_array,
        probability_array,
        counts,
        shots_per_delay,
        float(np.clip(visibility, 0.0, 1.0)),
        overlaps,
    )


def _indistinguishable_two_mode_distribution(
    photons_a: int, photons_b: int, reflectivity: float
) -> np.ndarray:
    """Exact output-a occupation distribution for a lossless beam splitter."""

    transmission = 1 - reflectivity
    total = photons_a + photons_b
    probabilities = np.zeros(total + 1)
    for output_a in range(total + 1):
        amplitude = 0.0j
        lower = max(0, output_a - photons_b)
        upper = min(photons_a, output_a)
        for from_a in range(lower, upper + 1):
            from_b = output_a - from_a
            amplitude += (
                comb(photons_a, from_a)
                * comb(photons_b, from_b)
                * sqrt(transmission) ** from_a
                * (1j * sqrt(reflectivity)) ** (photons_a - from_a)
                * (1j * sqrt(reflectivity)) ** from_b
                * sqrt(transmission) ** (photons_b - from_b)
            )
        amplitude *= sqrt(
            factorial(output_a)
            * factorial(total - output_a)
            / (factorial(photons_a) * factorial(photons_b))
        )
        probabilities[output_a] = abs(amplitude) ** 2
    return probabilities


def _partial_two_mode_distribution(
    photons_a: int,
    photons_b: int,
    indistinguishability: float,
    reflectivity: float,
) -> np.ndarray:
    """Two pure internal modes with squared overlap ``indistinguishability``."""

    transmission = 1 - reflectivity
    total = photons_a + photons_b
    probabilities = np.zeros(total + 1)
    for shared_b in range(photons_b + 1):
        orthogonal_b = photons_b - shared_b
        internal_weight = (
            comb(photons_b, shared_b)
            * indistinguishability**shared_b
            * (1 - indistinguishability) ** orthogonal_b
        )
        shared_distribution = _indistinguishable_two_mode_distribution(
            photons_a, shared_b, reflectivity
        )
        orthogonal_distribution = np.asarray(
            [
                comb(orthogonal_b, output_a)
                * reflectivity**output_a
                * transmission ** (orthogonal_b - output_a)
                for output_a in range(orthogonal_b + 1)
            ]
        )
        probabilities += internal_weight * np.convolve(shared_distribution, orthogonal_distribution)
    probabilities = np.maximum(probabilities, 0.0)
    return probabilities / np.sum(probabilities)


def _heralded_survivor_distribution(
    source: SPDCSource,
    herald_detector_efficiency: float,
    herald_dark_probability: float,
    max_pairs: int,
    tail_tolerance: float,
) -> tuple[np.ndarray, HeraldedStatistics]:
    total_herald_efficiency = source.idler_collection_efficiency * herald_detector_efficiency
    statistics = source.heralded_statistics(
        total_herald_efficiency,
        dark_probability=herald_dark_probability,
        max_pairs=max_pairs,
        tail_tolerance=tail_tolerance,
    )
    survivors = np.zeros(max_pairs + 1)
    efficiency = source.signal_collection_efficiency
    for pairs, pair_weight in zip(
        statistics.pair_numbers, statistics.conditional_probabilities, strict=True
    ):
        for survived in range(int(pairs) + 1):
            survivors[survived] += (
                pair_weight
                * comb(int(pairs), survived)
                * (efficiency**survived * (1 - efficiency) ** (int(pairs) - survived))
            )
    return survivors, statistics


def heralded_spdc_hom_scan(
    delays: Sequence[float],
    first: SPDCSource,
    second: SPDCSource,
    *,
    herald_detector_efficiencies: tuple[float, float] = (1.0, 1.0),
    herald_dark_probabilities: tuple[float, float] = (0.0, 0.0),
    signal_detector_efficiencies: tuple[float, float] = (1.0, 1.0),
    signal_dark_probabilities: tuple[float, float] = (0.0, 0.0),
    reflectivity: float = 0.5,
    max_pairs: int = 8,
    tail_tolerance: float = 1e-10,
    shots_per_delay: int | None = None,
    seed: int | None = None,
) -> HeraldedHOMResult:
    """HOM scan including SPDC multipairs, loss, and threshold detection.

    Each idler collection channel followed by its herald detector conditions the
    pair distribution. Signal-arm collection loss is then applied before an
    ideal lossless beam splitter. Output detectors are non-number-resolving.

    The effective Schmidt purity and wavepacket overlap set one uniform
    cross-source internal-mode overlap. This is exact for two pure internal
    modes and is an explicitly phenomenological closure for mixed multimode
    SPDC states; a measured joint spectral amplitude contains more information
    than an effective Schmidt number.
    """

    delay_array = np.asarray(delays, dtype=float)
    if delay_array.ndim != 1 or delay_array.size == 0 or not np.all(np.isfinite(delay_array)):
        raise ValidationError("delays must be a non-empty finite one-dimensional sequence")
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, (int, np.integer)) or max_pairs < 1:
        raise ValidationError("max_pairs must be a positive integer")
    if not np.isfinite(tail_tolerance) or tail_tolerance < 0:
        raise ValidationError("tail_tolerance must be finite and non-negative")
    if shots_per_delay is not None and (
        isinstance(shots_per_delay, bool)
        or not isinstance(shots_per_delay, (int, np.integer))
        or shots_per_delay < 1
    ):
        raise ValidationError("shots_per_delay must be a positive integer")
    reflectivity = _probability("reflectivity", reflectivity)
    if len(herald_detector_efficiencies) != 2:
        raise ValidationError("herald_detector_efficiencies must contain two values")
    if len(signal_detector_efficiencies) != 2:
        raise ValidationError("signal_detector_efficiencies must contain two values")
    if len(herald_dark_probabilities) != 2:
        raise ValidationError("herald_dark_probabilities must contain two values")
    if len(signal_dark_probabilities) != 2:
        raise ValidationError("signal_dark_probabilities must contain two values")
    herald_efficiencies = tuple(
        _probability("herald_detector_efficiency", value) for value in herald_detector_efficiencies
    )
    detector_efficiencies = tuple(
        _probability("signal_detector_efficiency", value) for value in signal_detector_efficiencies
    )
    herald_dark = tuple(
        _probability("herald_dark_probability", value) for value in herald_dark_probabilities
    )
    dark_probabilities = tuple(
        _probability("signal_dark_probability", value) for value in signal_dark_probabilities
    )
    if any(
        efficiency == 0 and dark == 0
        for efficiency, dark in zip(herald_efficiencies, herald_dark, strict=True)
    ):
        raise ValidationError("each herald channel needs non-zero efficiency or dark probability")

    first_survivors, first_statistics = _heralded_survivor_distribution(
        first,
        herald_efficiencies[0],
        herald_dark[0],
        int(max_pairs),
        tail_tolerance,
    )
    second_survivors, second_statistics = _heralded_survivor_distribution(
        second,
        herald_efficiencies[1],
        herald_dark[1],
        int(max_pairs),
        tail_tolerance,
    )
    overlaps = first.signal_wavepacket.indistinguishability_scan(
        second.signal_wavepacket, delay_array
    ) / sqrt(first.schmidt_number * second.schmidt_number)
    overlaps = np.clip(overlaps, 0.0, 1.0)

    def coincidence(overlap_values: np.ndarray) -> np.ndarray:
        probability = np.zeros_like(overlap_values, dtype=float)
        for photons_a, weight_a in enumerate(first_survivors):
            if weight_a == 0:
                continue
            for photons_b, weight_b in enumerate(second_survivors):
                if weight_b == 0:
                    continue
                for shared_b in range(photons_b + 1):
                    orthogonal_b = photons_b - shared_b
                    shared_output = _indistinguishable_two_mode_distribution(
                        photons_a, shared_b, reflectivity
                    )
                    orthogonal_output = np.asarray(
                        [
                            comb(orthogonal_b, output_a)
                            * reflectivity**output_a
                            * (1 - reflectivity) ** (orthogonal_b - output_a)
                            for output_a in range(orthogonal_b + 1)
                        ]
                    )
                    output = np.convolve(shared_output, orthogonal_output)
                    total = photons_a + photons_b
                    sector_click = 0.0
                    for output_a, output_weight in enumerate(output):
                        click_a = (
                            1
                            - (1 - dark_probabilities[0])
                            * (1 - detector_efficiencies[0]) ** output_a
                        )
                        click_b = 1 - (1 - dark_probabilities[1]) * (
                            1 - detector_efficiencies[1]
                        ) ** (total - output_a)
                        sector_click += output_weight * click_a * click_b
                    internal_weight = (
                        comb(photons_b, shared_b)
                        * overlap_values**shared_b
                        * (1 - overlap_values) ** orthogonal_b
                    )
                    probability += weight_a * weight_b * sector_click * internal_weight
        return np.clip(probability, 0.0, 1.0)

    distinguishable = float(coincidence(np.asarray([0.0]))[0])
    probabilities = coincidence(overlaps)
    counts = None
    if shots_per_delay is not None:
        counts = np.random.default_rng(seed).binomial(shots_per_delay, probabilities)
    minimum = float(np.min(probabilities))
    visibility = 1 - minimum / distinguishable if distinguishable > 0 else 0.0
    return HeraldedHOMResult(
        delay_array,
        probabilities,
        counts,
        shots_per_delay,
        distinguishable,
        float(np.clip(visibility, 0.0, 1.0)),
        overlaps,
        (
            first_statistics.herald_probability,
            second_statistics.herald_probability,
        ),
        (
            first_statistics.multipair_fraction,
            second_statistics.multipair_fraction,
        ),
        (
            first_statistics.omitted_tail_probability,
            second_statistics.omitted_tail_probability,
        ),
    )
