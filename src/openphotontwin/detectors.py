"""SNSPD and time-tag detector models."""

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
    wavelength_efficiency: Callable[[float], float] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        _probability("efficiency", self.efficiency)
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
        if acquisition_end < acquisition_start:
            raise ValidationError("acquisition_end must not precede acquisition_start")
        queue: list[tuple[float, int, str, PhotonEvent | None]] = []
        sequence = 0
        for event in arrivals:
            if not acquisition_start <= event.time <= acquisition_end:
                continue
            heapq.heappush(queue, (event.time, sequence, "photon", event))
            sequence += 1
        duration = acquisition_end - acquisition_start
        dark_count = int(rng.poisson(self.dark_count_rate * duration))
        if dark_count:
            for true_time in rng.uniform(acquisition_start, acquisition_end, dark_count):
                heapq.heappush(queue, (float(true_time), sequence, "dark", None))
                sequence += 1
        accepted: list[TimeTag] = []
        last_avalanche = np.full(self.pixel_count, -np.inf)
        while queue:
            true_time, _, origin, event = heapq.heappop(queue)
            elapsed_by_pixel = true_time - last_avalanche
            pixel = int(np.argmax(elapsed_by_pixel))
            recovery = self._recovery(float(elapsed_by_pixel[pixel]))
            if recovery == 0:
                continue
            base_efficiency = self._base_efficiency(event) if event is not None else 1.0
            effective_efficiency = base_efficiency * recovery
            if rng.random() >= effective_efficiency:
                continue
            last_avalanche[pixel] = true_time
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
                    self._readout_time(true_time, rng),
                    self.channel,
                    event.shot if event is not None else -1,
                    origin != "photon",
                    metadata,
                )
            )
            if self.afterpulse_probability and rng.random() < self.afterpulse_probability:
                afterpulse_time = (
                    true_time
                    + self.dead_time
                    + float(rng.exponential(self.afterpulse_time_constant))
                )
                if afterpulse_time <= acquisition_end:
                    heapq.heappush(queue, (afterpulse_time, sequence, "afterpulse", None))
                    sequence += 1
        return sorted(accepted)


@dataclass(slots=True)
class DetectorArray:
    """Map propagating optical modes to independent SNSPD channels."""

    detectors: Mapping[int, SNSPD]

    def __post_init__(self) -> None:
        if not self.detectors:
            raise ValidationError("a detector array cannot be empty")
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
        tags: list[TimeTag] = []
        for mode, detector in self.detectors.items():
            arrivals = [event for event in events if event.mode == mode]
            tags.extend(
                detector.detect(
                    arrivals,
                    acquisition_start=acquisition_start,
                    acquisition_end=acquisition_end,
                    rng=rng,
                )
            )
        return sorted(tags)
