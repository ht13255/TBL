"""Experiment-oriented photon-pair source models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln

from .errors import SimulationError, ValidationError
from .models import PhotonEvent, Wavepacket, _probability, _source_photon_event


@dataclass(frozen=True, slots=True)
class PairNumberDistribution:
    """Finite view of an infinite pair-number distribution."""

    pair_numbers: np.ndarray
    probabilities: np.ndarray
    tail_probability: float
    mean_pairs: float
    variance: float
    g2: float


@dataclass(frozen=True, slots=True)
class HeraldedStatistics:
    """Pair-number statistics conditioned on a threshold herald click."""

    pair_numbers: np.ndarray
    conditional_probabilities: np.ndarray
    herald_probability: float
    vacuum_fraction: float
    single_pair_fraction: float
    multipair_fraction: float
    conditional_mean_pairs: float
    omitted_tail_probability: float


@dataclass(frozen=True, slots=True)
class SPDCSource:
    """Pulsed multimode thermal photon-pair source.

    This model also applies to spontaneous four-wave mixing when ``mean_pairs``
    and ``schmidt_number`` describe the measured pair statistics. For ``K``
    equally occupied Schmidt modes, the pair count is negative binomial with
    ``Var(n)=mu+mu**2/K`` and marginal ``g2=1+1/K``. The reduced single-photon
    spectral purity is ``1/K`` before additional packet impurity. Optional
    normalized ``schmidt_weights`` replace the equal-mode approximation with
    independently squeezed thermal modes of means ``mu * weight``.
    """

    repetition_rate: float = 80e6
    mean_pairs: float = 0.05
    schmidt_number: float = 1.0
    schmidt_weights: tuple[float, ...] | None = None
    signal_wavepacket: Wavepacket = field(default_factory=Wavepacket)
    idler_wavepacket: Wavepacket = field(default_factory=Wavepacket)
    signal_mode: int = 0
    idler_mode: int = 1
    signal_collection_efficiency: float = 1.0
    idler_collection_efficiency: float = 1.0
    pump_wavelength: float | None = None
    energy_mismatch_tolerance: float = 1e-3
    pump_jitter: float = 0.0
    relative_time_jitter: float = 0.0
    max_pairs_per_pulse: int = 100

    def __post_init__(self) -> None:
        if self.schmidt_weights is not None:
            if any(isinstance(value, (bool, np.bool_)) for value in self.schmidt_weights):
                raise ValidationError("schmidt_weights cannot contain booleans")
            try:
                weights = np.asarray(self.schmidt_weights, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValidationError("schmidt_weights must be real numbers") from exc
            if (
                weights.ndim != 1
                or weights.size == 0
                or not np.all(np.isfinite(weights))
                or np.any(weights < 0)
                or not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1e-10)
            ):
                raise ValidationError(
                    "schmidt_weights must be a non-negative finite vector summing to one"
                )
            weights = weights[weights > 0]
            object.__setattr__(self, "schmidt_weights", tuple(float(value) for value in weights))
            object.__setattr__(self, "schmidt_number", float(1 / np.sum(weights**2)))
        finite = (
            self.repetition_rate,
            self.mean_pairs,
            self.schmidt_number,
            self.energy_mismatch_tolerance,
            self.pump_jitter,
            self.relative_time_jitter,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValidationError("SPDC source parameters must be finite")
        if self.repetition_rate <= 0 or self.mean_pairs < 0:
            raise ValidationError("repetition_rate must be positive and mean_pairs non-negative")
        if self.schmidt_number < 1:
            raise ValidationError("schmidt_number must be at least one")
        if self.signal_mode < 0 or self.idler_mode < 0 or self.signal_mode == self.idler_mode:
            raise ValidationError("signal and idler modes must be distinct and non-negative")
        _probability("signal_collection_efficiency", self.signal_collection_efficiency)
        _probability("idler_collection_efficiency", self.idler_collection_efficiency)
        if self.energy_mismatch_tolerance < 0:
            raise ValidationError("energy_mismatch_tolerance cannot be negative")
        if self.pump_jitter < 0 or self.relative_time_jitter < 0:
            raise ValidationError("source timing jitter cannot be negative")
        if (
            isinstance(self.max_pairs_per_pulse, bool)
            or not isinstance(self.max_pairs_per_pulse, (int, np.integer))
            or self.max_pairs_per_pulse < 1
        ):
            raise ValidationError("max_pairs_per_pulse must be a positive integer")
        if self.pump_wavelength is not None:
            if not np.isfinite(self.pump_wavelength) or self.pump_wavelength <= 0:
                raise ValidationError("pump_wavelength must be positive and finite")
            pump_inverse = 1 / self.pump_wavelength
            generated_inverse = (
                1 / self.signal_wavepacket.wavelength + 1 / self.idler_wavepacket.wavelength
            )
            relative_mismatch = abs(generated_inverse - pump_inverse) / pump_inverse
            if relative_mismatch > self.energy_mismatch_tolerance:
                raise ValidationError(
                    "signal and idler central frequencies violate pump energy conservation: "
                    f"relative mismatch {relative_mismatch:.6g} exceeds "
                    f"{self.energy_mismatch_tolerance:.6g}"
                )

    @property
    def period(self) -> float:
        return 1 / self.repetition_rate

    @property
    def pair_variance(self) -> float:
        return self.mean_pairs + self.mean_pairs**2 * self.heralded_spectral_purity

    @property
    def unheralded_g2(self) -> float:
        return 1 + self.heralded_spectral_purity

    @property
    def heralded_spectral_purity(self) -> float:
        if self.schmidt_weights is None:
            return 1 / self.schmidt_number
        return float(np.sum(np.asarray(self.schmidt_weights) ** 2))

    def pair_number_distribution(self, max_pairs: int = 20) -> PairNumberDistribution:
        """Return probabilities from zero through ``max_pairs`` and omitted tail."""

        if isinstance(max_pairs, bool) or not isinstance(max_pairs, (int, np.integer)):
            raise ValidationError("max_pairs must be an integer")
        if max_pairs < 0:
            raise ValidationError("max_pairs cannot be negative")
        pair_numbers = np.arange(max_pairs + 1, dtype=np.int64)
        if self.mean_pairs == 0:
            probabilities = np.zeros(max_pairs + 1)
            probabilities[0] = 1.0
        elif self.schmidt_weights is not None:
            probabilities = np.asarray([1.0])
            for weight in self.schmidt_weights:
                mode_mean = self.mean_pairs * weight
                ratio = mode_mean / (1 + mode_mean)
                mode_probabilities = (1 - ratio) * ratio**pair_numbers
                probabilities = np.convolve(probabilities, mode_probabilities)[: max_pairs + 1]
        else:
            ratio = self.mean_pairs / self.schmidt_number
            log_probabilities = (
                gammaln(pair_numbers + self.schmidt_number)
                - gammaln(self.schmidt_number)
                - gammaln(pair_numbers + 1)
                + pair_numbers * np.log(ratio)
                - (pair_numbers + self.schmidt_number) * np.log1p(ratio)
            )
            probabilities = np.exp(log_probabilities)
        tail = max(0.0, 1 - float(np.sum(probabilities)))
        return PairNumberDistribution(
            pair_numbers,
            probabilities,
            tail,
            self.mean_pairs,
            self.pair_variance,
            self.unheralded_g2,
        )

    def heralded_statistics(
        self,
        herald_efficiency: float,
        *,
        dark_probability: float = 0.0,
        max_pairs: int = 100,
        tail_tolerance: float = 1e-12,
    ) -> HeraldedStatistics:
        """Condition the source on a non-number-resolving idler click."""

        efficiency = _probability("herald_efficiency", herald_efficiency)
        dark_probability = _probability("dark_probability", dark_probability)
        if efficiency == 0 and dark_probability == 0:
            raise ValidationError("herald_efficiency or dark_probability must be non-zero")
        if not np.isfinite(tail_tolerance) or tail_tolerance < 0:
            raise ValidationError("tail_tolerance must be finite and non-negative")
        distribution = self.pair_number_distribution(max_pairs)
        if distribution.tail_probability > tail_tolerance:
            raise SimulationError(
                "heralded-statistics truncation is too small: omitted probability "
                f"{distribution.tail_probability:.3g} exceeds {tail_tolerance:.3g}"
            )
        click_given_pairs = (
            1 - (1 - dark_probability) * (1 - efficiency) ** distribution.pair_numbers
        )
        joint = distribution.probabilities * click_given_pairs
        herald_probability = float(np.sum(joint))
        if herald_probability == 0:
            raise SimulationError("source has zero herald probability")
        conditional = joint / herald_probability
        vacuum = float(conditional[0])
        single = float(conditional[1]) if len(conditional) > 1 else 0.0
        multipair = float(np.sum(conditional[2:]))
        conditional_mean = float(np.dot(distribution.pair_numbers, conditional))
        return HeraldedStatistics(
            distribution.pair_numbers,
            conditional,
            herald_probability,
            vacuum,
            single,
            multipair,
            conditional_mean,
            distribution.tail_probability,
        )

    def sample_pair_counts(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Sample the exact real-``K`` negative-binomial distribution."""

        if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots < 0:
            raise ValidationError("shots must be a non-negative integer")
        if self.mean_pairs == 0:
            return np.zeros(shots, dtype=np.int64)
        if self.schmidt_weights is not None:
            counts = np.zeros(shots, dtype=np.int64)
            for weight in self.schmidt_weights:
                success_probability = 1 / (1 + self.mean_pairs * weight)
                counts += rng.geometric(success_probability, size=shots) - 1
        else:
            poisson_rates = rng.gamma(
                shape=self.schmidt_number,
                scale=self.mean_pairs / self.schmidt_number,
                size=shots,
            )
            counts = rng.poisson(poisson_rates).astype(np.int64, copy=False)
        sampled_maximum = int(counts.max(initial=0))
        if sampled_maximum > self.max_pairs_per_pulse:
            raise SimulationError(
                f"sampled {sampled_maximum} pairs in one pulse, above "
                f"max_pairs_per_pulse={self.max_pairs_per_pulse}"
            )
        return counts

    def emit(
        self, shot: int, rng: np.random.Generator, *, mode: int | None = None
    ) -> list[PhotonEvent]:
        """Emit signal/idler events for one pump pulse."""

        if mode is not None:
            raise ValidationError("SPDCSource fixes separate signal_mode and idler_mode")
        if isinstance(shot, bool) or not isinstance(shot, (int, np.integer)) or shot < 0:
            raise ValidationError("shot must be a non-negative integer")
        return self._events_from_counts(self.sample_pair_counts(1, rng), rng, shot_offset=int(shot))

    def emit_many(
        self, shots: int, rng: np.random.Generator, *, mode: int | None = None
    ) -> list[PhotonEvent]:
        """Vectorized pump statistics with event-level signal/idler output."""

        if mode is not None:
            raise ValidationError("SPDCSource fixes separate signal_mode and idler_mode")
        return self._events_from_counts(self.sample_pair_counts(shots, rng), rng)

    def _events_from_counts(
        self,
        counts: np.ndarray,
        rng: np.random.Generator,
        *,
        shot_offset: int = 0,
    ) -> list[PhotonEvent]:
        total_pairs = int(np.sum(counts))
        if total_pairs == 0:
            return []
        local_shots = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        absolute_shots = local_shots + shot_offset
        starts = np.repeat(np.cumsum(counts) - counts, counts)
        pair_indices = np.arange(total_pairs, dtype=np.int64) - starts
        pair_counts = counts[local_shots]
        common_jitter = (
            rng.normal(0.0, self.pump_jitter, total_pairs)
            if self.pump_jitter
            else np.zeros(total_pairs)
        )
        relative_jitter = (
            rng.normal(0.0, self.relative_time_jitter, total_pairs)
            if self.relative_time_jitter
            else np.zeros(total_pairs)
        )
        keep_signal = rng.random(total_pairs) < self.signal_collection_efficiency
        keep_idler = rng.random(total_pairs) < self.idler_collection_efficiency
        spectral_purity = self.heralded_spectral_purity
        signal_purity = self.signal_wavepacket.purity * spectral_purity
        idler_purity = self.idler_wavepacket.purity * spectral_purity
        events: list[PhotonEvent] = []
        for index in range(total_pairs):
            shot = int(absolute_shots[index])
            pair_index = int(pair_indices[index])
            base_time = shot * self.period + float(common_jitter[index])
            common_metadata = {
                "source_model": "spdc",
                "pair_index": pair_index,
                "pair_count": int(pair_counts[index]),
                "schmidt_number": self.schmidt_number,
                "schmidt_weights": self.schmidt_weights,
                "pump_wavelength_m": self.pump_wavelength,
            }
            if keep_signal[index]:
                signal_packet = self.signal_wavepacket._copy_with(
                    arrival_time=base_time,
                    purity=signal_purity,
                )
                metadata = {**common_metadata, "arm": "signal"}
                events.append(
                    _source_photon_event(
                        signal_packet,
                        self.signal_mode,
                        shot,
                        pair_index,
                        metadata=metadata,
                    )
                )
            if keep_idler[index]:
                idler_packet = self.idler_wavepacket._copy_with(
                    arrival_time=base_time + float(relative_jitter[index]),
                    purity=idler_purity,
                )
                metadata = {**common_metadata, "arm": "idler"}
                events.append(
                    _source_photon_event(
                        idler_packet,
                        self.idler_mode,
                        shot,
                        pair_index,
                        metadata=metadata,
                    )
                )
        return events
