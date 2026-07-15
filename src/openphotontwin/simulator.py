"""Time-resolved hardware digital twin orchestration."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .components import EventComponent, SimulationContext
from .detectors import DetectorArray, TimeTag
from .errors import ValidationError
from .models import PhotonEvent, SinglePhotonSource


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Raw propagated events, detector tags, and common summaries."""

    events: tuple[PhotonEvent, ...]
    time_tags: tuple[TimeTag, ...]
    shots: int
    seed: int | None

    @property
    def arrival_times(self) -> np.ndarray:
        return np.asarray([tag.time for tag in self.time_tags], dtype=float)

    @property
    def photon_number_distribution(self) -> dict[int, float]:
        counts = Counter(tag.shot for tag in self.time_tags if tag.shot >= 0 and not tag.dark_count)
        distribution = Counter(counts.get(shot, 0) for shot in range(self.shots))
        return {number: count / self.shots for number, count in sorted(distribution.items())}

    def tags_dataframe(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "time": tag.time,
                    "channel": tag.channel,
                    "shot": tag.shot,
                    "dark_count": tag.dark_count,
                    **tag.metadata,
                }
                for tag in self.time_tags
            ]
        )

    def save_time_tags(self, path: str) -> None:
        self.tags_dataframe().to_csv(path, index=False)


EventFactory = Callable[[int, np.random.Generator], Sequence[PhotonEvent]]


@dataclass(slots=True)
class DigitalTwin:
    """Shot-based time-resolved digital twin.

    This engine models source statistics, physical propagation time, loops,
    switching, loss, electronic latency, and detector timestamps. Use
    :class:`~openphotontwin.circuit.FockSimulator` for exact coherent
    multi-photon probabilities and the HOM helpers for two-photon scans.
    """

    source: SinglePhotonSource | EventFactory
    components: list[EventComponent]
    detectors: DetectorArray
    time_bin_width: float = 1e-9

    def run(
        self,
        shots: int,
        *,
        seed: int | None = None,
        acquisition_start: float = 0.0,
        acquisition_end: float | None = None,
    ) -> SimulationResult:
        if shots < 1:
            raise ValidationError("shots must be positive")
        rng = np.random.default_rng(seed)
        reset = getattr(self.source, "reset", None)
        if callable(reset):
            reset(rng)
        context = SimulationContext(self.time_bin_width, acquisition_start)
        events: list[PhotonEvent] = []
        for shot in range(shots):
            if hasattr(self.source, "emit"):
                events.extend(self.source.emit(shot, rng))
            else:
                events.extend(self.source(shot, rng))
        for component in self.components:
            events = component.process(events, rng, context)
        events.sort(key=lambda event: event.time)
        if acquisition_end is None:
            period = getattr(self.source, "period", None)
            if period is not None:
                nominal_end = shots * float(period)
                propagated_end = (
                    max(event.time for event in events) + self.time_bin_width
                    if events
                    else acquisition_start
                )
                acquisition_end = max(nominal_end, propagated_end)
            elif events:
                acquisition_end = max(event.time for event in events) + self.time_bin_width
            else:
                acquisition_end = acquisition_start + shots * self.time_bin_width
        tags = self.detectors.detect(
            events,
            acquisition_start=acquisition_start,
            acquisition_end=acquisition_end,
            rng=rng,
        )
        return SimulationResult(tuple(events), tuple(tags), shots, seed)
