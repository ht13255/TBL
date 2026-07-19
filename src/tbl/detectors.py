"""TBL SNSPD and time-tag detector models."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .errors import ValidationError
from .models import PhotonEvent, _probability


@dataclass(frozen=True, order=True, slots=True)
class TimeTag:
    """A detector timestamp in SI units."""

    time: float
    channel: int
    shot: int = -1
    dark_count: bool = False
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.time):
            raise ValidationError("time-tag time must be finite")
        if self.channel < 0 or self.shot < -1:
            raise ValidationError("time-tag channel and shot are invalid")


@dataclass(frozen=True, slots=True)
class SNSPD:
    """Superconducting nanowire single-photon detector.

    ``dark_count_rate`` is in counts per second; jitter and dead time are in
    seconds. Avalanches are processed in physical arrival-time order. Dead time
    and exponential efficiency recovery act on true avalanche time; readout
    latency, Gaussian jitter, an optional positive jitter tail, and time-tagger
    quantization are applied only afterward.

    ``pixel_count`` models independent nanowire pixels. The legacy
    ``number_resolving=True`` setting selects four pixels when no explicit
    pixel count is supplied.
    """

    efficiency: float = 0.9
    dark_count_rate: float = 0.0
    jitter: float = 20e-12
    dead_time: float = 50e-9
    channel: int = 0
    number_resolving: bool = False
    recovery_time: float = 0.0
    jitter_tail_probability: float = 0.0
    jitter_tail_time: float = 0.0
    afterpulse_probability: float = 0.0
    afterpulse_time_constant: float = 0.0
    detection_latency: float = 0.0
    time_tagger_resolution: float = 0.0
    pixel_count: int = 1
    pixel_weights: tuple[float, ...] | None = None
    wavelength_efficiency: Callable[[float], float] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        _probability("efficiency", self.efficiency)
        finite = (
            self.dark_count_rate,
            self.jitter,
            self.dead_time,
            self.recovery_time,
            self.jitter_tail_probability,
            self.jitter_tail_time,
            self.afterpulse_probability,
            self.afterpulse_time_constant,
            self.detection_latency,
            self.time_tagger_resolution,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValidationError("detector rates, probabilities, and times must be finite")
        if (
            self.dark_count_rate < 0
            or self.jitter < 0
            or self.dead_time < 0
            or self.recovery_time < 0
            or self.jitter_tail_time < 0
            or self.afterpulse_time_constant < 0
            or self.detection_latency < 0
            or self.time_tagger_resolution < 0
        ):
            raise ValidationError("detector rates and timing widths must be non-negative")
        _probability("jitter_tail_probability", self.jitter_tail_probability)
        _probability("afterpulse_probability", self.afterpulse_probability)
        if self.jitter_tail_probability and self.jitter_tail_time == 0:
            raise ValidationError(
                "jitter_tail_time must be positive when its probability is nonzero"
            )
        if self.afterpulse_probability and self.afterpulse_time_constant == 0:
            raise ValidationError(
                "afterpulse_time_constant must be positive when afterpulsing is enabled"
            )
        if self.channel < 0:
            raise ValidationError("detector channel must be non-negative")
        if self.pixel_count < 1:
            raise ValidationError("pixel_count must be at least one")
        if self.number_resolving and self.pixel_count == 1:
            object.__setattr__(self, "pixel_count", 4)
        if self.pixel_weights is None:
            weights = np.full(self.pixel_count, 1 / self.pixel_count)
        else:
            weights = np.asarray(self.pixel_weights, dtype=float)
            if (
                weights.shape != (self.pixel_count,)
                or not np.all(np.isfinite(weights))
                or np.any(weights < 0)
                or float(np.sum(weights)) <= 0
            ):
                raise ValidationError(
                    "pixel_weights must be finite non-negative weights matching pixel_count"
                )
            weights = weights / np.sum(weights)
        object.__setattr__(self, "pixel_weights", tuple(float(value) for value in weights))
        if self.wavelength_efficiency is not None and not callable(self.wavelength_efficiency):
            raise ValidationError("wavelength_efficiency must be callable")

    def _base_efficiency(self, event: PhotonEvent) -> float:
        if self.wavelength_efficiency is None:
            return self.efficiency
        value = self.wavelength_efficiency(event.photon.wavepacket.wavelength)
        return _probability("wavelength-dependent detector efficiency", value)

    def _recovery(self, elapsed: float) -> float:
        if elapsed < self.dead_time:
            return 0.0
        if self.recovery_time == 0 or np.isinf(elapsed):
            return 1.0
        return float(1 - np.exp(-(elapsed - self.dead_time) / self.recovery_time))

    def _readout_time(self, true_time: float, rng: np.random.Generator) -> float:
        timestamp = true_time + self.detection_latency
        if self.jitter:
            timestamp += float(rng.normal(0.0, self.jitter))
        if self.jitter_tail_probability and rng.random() < self.jitter_tail_probability:
            timestamp += float(rng.exponential(self.jitter_tail_time))
        if self.time_tagger_resolution:
            timestamp = (
                np.rint(timestamp / self.time_tagger_resolution) * self.time_tagger_resolution
            )
        return float(timestamp)

    def detect(
        self,
        arrivals: Sequence[PhotonEvent],
        *,
        acquisition_start: float,
        acquisition_end: float,
        rng: np.random.Generator,
    ) -> list[TimeTag]:
        if not np.isfinite(acquisition_start) or not np.isfinite(acquisition_end):
            raise ValidationError("acquisition bounds must be finite")
        if acquisition_end < acquisition_start:
            raise ValidationError("acquisition_end must not precede acquisition_start")
        queue: list[tuple[float, int, str, PhotonEvent | None, int | None]] = [
            (event.time, sequence, "photon", event, None)
            for sequence, event in enumerate(arrivals)
            if acquisition_start <= event.time <= acquisition_end
        ]
        sequence = len(arrivals)
        duration = acquisition_end - acquisition_start
        dark_count = int(rng.poisson(self.dark_count_rate * duration))
        if dark_count:
            for true_time in rng.uniform(acquisition_start, acquisition_end, dark_count):
                queue.append((float(true_time), sequence, "dark", None, None))
                sequence += 1
        use_heap = self.afterpulse_probability > 0
        if use_heap:
            heapq.heapify(queue)
        else:
            queue.sort()
        accepted_records: list[tuple[float, str, PhotonEvent | None, int, float, float]] = []
        simple_avalanche_model = (
            not use_heap
            and self.pixel_count == 1
            and self.recovery_time == 0
            and self.wavelength_efficiency is None
        )
        if simple_avalanche_model:
            efficiency_draws = rng.random(len(queue))
            last_avalanche_time = -np.inf
            for draw, (true_time, _, origin, event, _) in zip(efficiency_draws, queue, strict=True):
                base_efficiency = self.efficiency if event is not None else 1.0
                if draw >= base_efficiency or true_time - last_avalanche_time < self.dead_time:
                    continue
                last_avalanche_time = true_time
                accepted_records.append((true_time, origin, event, 0, 1.0, base_efficiency))
        else:
            single_pixel = self.pixel_count == 1
            last_single_avalanche = -np.inf
            last_avalanche = [-np.inf] * self.pixel_count
            queue_index = 0
            while queue_index < len(queue):
                if use_heap:
                    true_time, _, origin, event, forced_pixel = heapq.heappop(queue)
                else:
                    true_time, _, origin, event, forced_pixel = queue[queue_index]
                    queue_index += 1
                if single_pixel:
                    pixel = 0
                    elapsed = true_time - last_single_avalanche
                else:
                    pixel = (
                        forced_pixel
                        if forced_pixel is not None
                        else int(rng.choice(self.pixel_count, p=self.pixel_weights))
                    )
                    elapsed = true_time - last_avalanche[pixel]
                recovery = self._recovery(elapsed)
                if recovery == 0:
                    continue
                base_efficiency = self._base_efficiency(event) if event is not None else 1.0
                effective_efficiency = base_efficiency * recovery
                if rng.random() >= effective_efficiency:
                    continue
                if single_pixel:
                    last_single_avalanche = true_time
                else:
                    last_avalanche[pixel] = true_time
                accepted_records.append(
                    (true_time, origin, event, pixel, recovery, effective_efficiency)
                )
                if self.afterpulse_probability and rng.random() < self.afterpulse_probability:
                    afterpulse_time = (
                        true_time
                        + self.dead_time
                        + float(rng.exponential(self.afterpulse_time_constant))
                    )
                    if afterpulse_time <= acquisition_end:
                        heapq.heappush(
                            queue,
                            (afterpulse_time, sequence, "afterpulse", None, pixel),
                        )
                        sequence += 1
        if not accepted_records:
            return []
        true_times = np.fromiter(
            (record[0] for record in accepted_records),
            dtype=float,
            count=len(accepted_records),
        )
        timestamps = true_times + self.detection_latency
        if self.jitter:
            timestamps += rng.normal(0.0, self.jitter, len(timestamps))
        if self.jitter_tail_probability:
            has_tail = rng.random(len(timestamps)) < self.jitter_tail_probability
            tail_count = int(np.count_nonzero(has_tail))
            if tail_count:
                timestamps[has_tail] += rng.exponential(self.jitter_tail_time, tail_count)
        if self.time_tagger_resolution:
            timestamps = (
                np.rint(timestamps / self.time_tagger_resolution) * self.time_tagger_resolution
            )
        accepted: list[TimeTag] = []
        for timestamp, record in zip(timestamps, accepted_records, strict=True):
            true_time, origin, event, pixel, recovery, effective_efficiency = record
            metadata = dict(event.metadata) if event is not None else {}
            metadata.update(
                event_type=origin,
                true_time_s=true_time,
                detector_pixel=pixel,
                detector_recovery=recovery,
                effective_efficiency=effective_efficiency,
            )
            accepted.append(
                TimeTag(
                    float(timestamp),
                    self.channel,
                    event.shot if event is not None else -1,
                    origin != "photon",
                    metadata,
                )
            )
        if self.jitter or self.jitter_tail_probability:
            accepted.sort(key=lambda tag: tag.time)
        return accepted


@dataclass(slots=True)
class DetectorArray:
    """Map propagating optical modes to independent SNSPD channels."""

    detectors: Mapping[int, SNSPD]

    def __post_init__(self) -> None:
        if not self.detectors:
            raise ValidationError("a detector array cannot be empty")
        if any(mode < 0 for mode in self.detectors):
            raise ValidationError("detector modes must be non-negative")
        channels = [detector.channel for detector in self.detectors.values()]
        if len(channels) != len(set(channels)):
            raise ValidationError("detector channels must be unique")

    def detect(
        self,
        events: Sequence[PhotonEvent],
        *,
        acquisition_start: float,
        acquisition_end: float,
        rng: np.random.Generator,
    ) -> list[TimeTag]:
        arrivals_by_mode: dict[int, list[PhotonEvent]] = {mode: [] for mode in self.detectors}
        for event in events:
            bucket = arrivals_by_mode.get(event.mode)
            if bucket is not None:
                bucket.append(event)
        tags: list[TimeTag] = []
        root_entropy = int(rng.bit_generator.random_raw())
        for mode, detector in sorted(self.detectors.items()):
            detector_rng = np.random.default_rng(
                np.random.SeedSequence(
                    root_entropy,
                    spawn_key=(int(mode), int(detector.channel)),
                )
            )
            tags.extend(
                detector.detect(
                    arrivals_by_mode[mode],
                    acquisition_start=acquisition_start,
                    acquisition_end=acquisition_end,
                    rng=detector_rng,
                )
            )
        return sorted(tags)
