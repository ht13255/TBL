"""SNSPD and time-tag detector models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    seconds. Dead time is applied after jitter, matching a time-tagger view.
    """

    efficiency: float = 0.9
    dark_count_rate: float = 0.0
    jitter: float = 20e-12
    dead_time: float = 50e-9
    channel: int = 0
    number_resolving: bool = False

    def __post_init__(self) -> None:
        _probability("efficiency", self.efficiency)
        if self.dark_count_rate < 0 or self.jitter < 0 or self.dead_time < 0:
            raise ValidationError("detector rates and timing widths must be non-negative")
        if self.channel < 0:
            raise ValidationError("detector channel must be non-negative")

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
        candidates: list[TimeTag] = []
        for event in arrivals:
            if not acquisition_start <= event.time <= acquisition_end:
                continue
            if rng.random() >= self.efficiency:
                continue
            timestamp = event.time + (rng.normal(0.0, self.jitter) if self.jitter else 0.0)
            candidates.append(
                TimeTag(timestamp, self.channel, event.shot, False, dict(event.metadata))
            )
        duration = acquisition_end - acquisition_start
        dark_count = int(rng.poisson(self.dark_count_rate * duration))
        if dark_count:
            for timestamp in rng.uniform(acquisition_start, acquisition_end, dark_count):
                candidates.append(TimeTag(float(timestamp), self.channel, -1, True))
        candidates.sort()
        accepted: list[TimeTag] = []
        last_time = -np.inf
        multiplicity: dict[tuple[int, int], int] = {}
        for candidate in candidates:
            if candidate.time - last_time < self.dead_time:
                if not self.number_resolving:
                    continue
                key = (candidate.shot, round(candidate.time / max(self.jitter, 1e-15)))
                multiplicity[key] = multiplicity.get(key, 1) + 1
                metadata = dict(candidate.metadata)
                metadata["multiplicity"] = multiplicity[key]
                accepted.append(
                    TimeTag(
                        candidate.time,
                        candidate.channel,
                        candidate.shot,
                        candidate.dark_count,
                        metadata,
                    )
                )
                continue
            accepted.append(candidate)
            last_time = candidate.time
        return accepted


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
