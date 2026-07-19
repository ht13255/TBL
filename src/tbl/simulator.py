"""TBL time-resolved hardware digital twin orchestration."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .calibration import (
    BinomialEstimate,
    CorrelatedBernoulliEstimate,
    estimate_binomial,
    estimate_correlated_bernoulli,
)
from .components import EventComponent, SimulationContext
from .detectors import DetectorArray, TimeTag
from .errors import SimulationError, ValidationError
from .models import PhotonEvent, SinglePhotonSource
from .research import (
    RunManifest,
    build_run_manifest,
    result_sha256,
    save_simulation_result,
    snapshot_configuration,
)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Raw propagated events, detector tags, and common summaries."""

    events: tuple[PhotonEvent, ...]
    time_tags: tuple[TimeTag, ...]
    shots: int
    seed: int | None
    manifest: RunManifest | None = None

    def __post_init__(self) -> None:
        if self.shots < 1:
            raise ValidationError("simulation results must contain at least one shot")
        if self.manifest is not None and (
            self.manifest.shots != self.shots or self.manifest.seed != self.seed
        ):
            raise ValidationError("simulation result and run manifest do not agree")

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
                    **tag.metadata,
                    "time": tag.time,
                    "channel": tag.channel,
                    "shot": tag.shot,
                    "dark_count": tag.dark_count,
                }
                for tag in self.time_tags
            ]
        )

    def save_time_tags(self, path: str | Path) -> None:
        self.tags_dataframe().to_csv(path, index=False)

    def save_bundle(self, path: str | Path) -> Path:
        """Persist events, tags, metadata, units, and provenance without pickle."""

        return save_simulation_result(self, path)

    def verify_integrity(self) -> bool:
        """Verify that provenance and in-memory result data are unchanged."""

        if self.manifest is None:
            raise ValidationError("simulation result has no run manifest")
        self.manifest.verify()
        if self.manifest.result_sha256 is None:
            raise ValidationError(
                "simulation result is not sealed; call seal() or run(..., integrity=True)"
            )
        if (
            result_sha256(
                self.events,
                self.time_tags,
                policy=self.manifest.result_hash_policy,
            )
            != self.manifest.result_sha256
        ):
            raise SimulationError("simulation result checksum mismatch")
        return True

    def seal(self) -> SimulationResult:
        """Return a result carrying a checksum of its complete current payload."""

        if self.manifest is None:
            raise ValidationError("simulation result has no run manifest")
        self.manifest.verify()
        measured = result_sha256(
            self.events,
            self.time_tags,
            policy=self.manifest.result_hash_policy,
        )
        if self.manifest.result_sha256 is not None:
            if measured != self.manifest.result_sha256:
                raise SimulationError("simulation result checksum mismatch")
            return self
        manifest = replace(self.manifest, result_sha256=measured)
        return SimulationResult(self.events, self.time_tags, self.shots, self.seed, manifest)

    def detection_probability(
        self,
        channel: int | None = None,
        *,
        confidence_level: float = 0.95,
        include_dark_counts: bool = False,
    ) -> BinomialEstimate:
        """Estimate the per-shot click probability with an exact interval."""

        if channel is not None and channel < 0:
            raise ValidationError("channel must be non-negative")
        successful_shots = {
            tag.shot
            for tag in self.time_tags
            if 0 <= tag.shot < self.shots
            and (channel is None or tag.channel == channel)
            and (include_dark_counts or not tag.dark_count)
        }
        return estimate_binomial(
            len(successful_shots), self.shots, confidence_level=confidence_level
        )

    def detection_convergence(
        self,
        channel: int | None = None,
        *,
        confidence_level: float = 0.95,
        include_dark_counts: bool = False,
        max_lag: int | None = None,
        batches: int = 20,
    ) -> CorrelatedBernoulliEstimate:
        """Estimate click uncertainty without assuming independent shots."""

        if channel is not None and channel < 0:
            raise ValidationError("channel must be non-negative")
        outcomes = np.zeros(self.shots, dtype=bool)
        for tag in self.time_tags:
            if (
                0 <= tag.shot < self.shots
                and (channel is None or tag.channel == channel)
                and (include_dark_counts or not tag.dark_count)
            ):
                outcomes[tag.shot] = True
        return estimate_correlated_bernoulli(
            outcomes,
            confidence_level=confidence_level,
            max_lag=max_lag,
            batches=batches,
        )


EventFactory = Callable[[int, np.random.Generator], Sequence[PhotonEvent]]


@dataclass(slots=True)
class DigitalTwin:
    """Shot-based time-resolved digital twin.

    This engine models source statistics, physical propagation time, loops,
    switching, loss, electronic latency, and detector timestamps. Use
    :class:`~tbl.circuit.FockSimulator` for exact coherent
    multi-photon probabilities and the HOM helpers for two-photon scans.
    """

    source: SinglePhotonSource | EventFactory
    components: list[EventComponent]
    detectors: DetectorArray
    time_bin_width: float = 1e-9

    def __post_init__(self) -> None:
        if not np.isfinite(self.time_bin_width) or self.time_bin_width <= 0:
            raise ValidationError("time_bin_width must be positive and finite")

    def run(
        self,
        shots: int,
        *,
        seed: int | None = None,
        integrity: bool = False,
        acquisition_start: float = 0.0,
        acquisition_end: float | None = None,
    ) -> SimulationResult:
        if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots < 1:
            raise ValidationError("shots must be positive")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0
        ):
            raise ValidationError("seed must be a non-negative integer or None")
        if not np.isfinite(acquisition_start):
            raise ValidationError("acquisition_start must be finite")
        if acquisition_end is not None and (
            not np.isfinite(acquisition_end) or acquisition_end < acquisition_start
        ):
            raise ValidationError(
                "acquisition_end must be finite and not precede acquisition_start"
            )
        root_sequence = np.random.SeedSequence(seed)
        raw_entropy = root_sequence.entropy
        if not isinstance(raw_entropy, (int, np.integer)):
            raise SimulationError("integer seed produced non-integer SeedSequence entropy")
        entropy = int(raw_entropy)

        def stage_rng(*spawn_key: int) -> np.random.Generator:
            sequence = np.random.SeedSequence(entropy, spawn_key=spawn_key)
            return np.random.default_rng(sequence)

        source_rng = stage_rng(0)
        configuration = snapshot_configuration(
            self.source,
            self.components,
            self.detectors,
            time_bin_width=self.time_bin_width,
        )
        reset = getattr(self.source, "reset", None)
        if callable(reset):
            reset(source_rng)
        context = SimulationContext(self.time_bin_width, acquisition_start)
        events: list[PhotonEvent] = []
        emit_many = getattr(self.source, "emit_many", None)
        if callable(emit_many):
            events.extend(emit_many(shots, source_rng))
        else:
            for shot in range(shots):
                if hasattr(self.source, "emit"):
                    events.extend(self.source.emit(shot, source_rng))
                else:
                    events.extend(self.source(shot, source_rng))
        for index, component in enumerate(self.components):
            events = component.process(events, stage_rng(1, index), context)
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
            rng=stage_rng(2),
        )
        event_tuple = tuple(events)
        tag_tuple = tuple(tags)
        manifest = build_run_manifest(
            configuration,
            rng_algorithm=type(source_rng.bit_generator).__name__,
            rng_stream_policy="seedsequence-hierarchy-v2",
            rng_entropy=entropy,
            seed=seed,
            shots=shots,
            acquisition_start=acquisition_start,
            acquisition_end=float(acquisition_end),
            result_digest=(result_sha256(event_tuple, tag_tuple) if integrity else None),
        )
        return SimulationResult(event_tuple, tag_tuple, shots, seed, manifest)
