"""Time-domain hardware components for event-resolved simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import pi, sqrt
from typing import Protocol

import numpy as np

from .errors import ValidationError
from .models import PhotonEvent, _probability


@dataclass(slots=True)
class SimulationContext:
    """Shared timing configuration passed through an event pipeline."""

    time_bin_width: float = 1e-9
    start_time: float = 0.0

    def __post_init__(self) -> None:
        if self.time_bin_width <= 0:
            raise ValidationError("time_bin_width must be positive")

    def bin_at(self, time: float) -> int:
        return int(np.floor((time - self.start_time) / self.time_bin_width + 1e-12))


class EventComponent(Protocol):
    """Protocol implemented by hardware blocks in a ``DigitalTwin``."""

    def process(
        self,
        events: Sequence[PhotonEvent],
        rng: np.random.Generator,
        context: SimulationContext,
    ) -> list[PhotonEvent]: ...


@dataclass(frozen=True, slots=True)
class LossChannel:
    """Independent photon-loss channel."""

    transmission: float
    modes: frozenset[int] | None = None

    def __post_init__(self) -> None:
        _probability("transmission", self.transmission)

    def process(
        self,
        events: Sequence[PhotonEvent],
        rng: np.random.Generator,
        context: SimulationContext,
    ) -> list[PhotonEvent]:
        del context
        return [
            event
            for event in events
            if self.modes is not None
            and event.mode not in self.modes
            or rng.random() < self.transmission
        ]


@dataclass(frozen=True, slots=True)
class DelayLine:
    """Deterministic propagation delay, optionally with Gaussian jitter."""

    delay: float
    jitter: float = 0.0
    modes: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if self.delay < 0 or self.jitter < 0:
            raise ValidationError("delay and jitter must be non-negative")

    def process(
        self,
        events: Sequence[PhotonEvent],
        rng: np.random.Generator,
        context: SimulationContext,
    ) -> list[PhotonEvent]:
        del context
        result: list[PhotonEvent] = []
        for event in events:
            if self.modes is not None and event.mode not in self.modes:
                result.append(event.copy())
                continue
            jitter = rng.normal(0.0, self.jitter) if self.jitter else 0.0
            result.append(event.copy(time=event.time + self.delay + jitter))
        return result


@dataclass(frozen=True, slots=True)
class PhaseShifter:
    """Static phase shift on selected modes."""

    phase: float
    modes: frozenset[int] | None = None

    def process(
        self,
        events: Sequence[PhotonEvent],
        rng: np.random.Generator,
        context: SimulationContext,
    ) -> list[PhotonEvent]:
        del rng, context
        factor = np.exp(1j * self.phase)
        return [
            event.copy(amplitude=event.amplitude * factor)
            if self.modes is None or event.mode in self.modes
            else event.copy()
            for event in events
        ]


@dataclass(frozen=True, slots=True)
class BeamSplitter:
    """Two-port beam splitter.

    Event propagation samples one output path. Coherent multi-photon circuit
    simulation uses the exact matrix returned by :meth:`unitary`.
    """

    mode_a: int = 0
    mode_b: int = 1
    reflectivity: float = 0.5
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.mode_a == self.mode_b or min(self.mode_a, self.mode_b) < 0:
            raise ValidationError("beam-splitter modes must be distinct and non-negative")
        _probability("reflectivity", self.reflectivity)

    def unitary(self, modes: int, reflectivity: float | None = None) -> np.ndarray:
        if modes <= max(self.mode_a, self.mode_b):
            raise ValidationError("matrix has too few modes for this beam splitter")
        r_prob = (
            self.reflectivity
            if reflectivity is None
            else _probability("reflectivity", reflectivity)
        )
        t, r = sqrt(1 - r_prob), sqrt(r_prob)
        matrix = np.eye(modes, dtype=complex)
        a, b = self.mode_a, self.mode_b
        matrix[a, a] = matrix[b, b] = t
        matrix[a, b] = 1j * r * np.exp(1j * self.phase)
        matrix[b, a] = 1j * r * np.exp(-1j * self.phase)
        return matrix

    def reflectivity_at(self, event: PhotonEvent, context: SimulationContext) -> float:
        del event, context
        return self.reflectivity

    def process(
        self,
        events: Sequence[PhotonEvent],
        rng: np.random.Generator,
        context: SimulationContext,
    ) -> list[PhotonEvent]:
        result: list[PhotonEvent] = []
        for event in events:
            if event.mode not in (self.mode_a, self.mode_b):
                result.append(event.copy())
                continue
            reflectivity = self.reflectivity_at(event, context)
            reflected = rng.random() < reflectivity
            if not reflected:
                result.append(event.copy())
                continue
            target = self.mode_b if event.mode == self.mode_a else self.mode_a
            sign = 1.0 if event.mode == self.mode_a else -1.0
            phase = 1j * np.exp(sign * 1j * self.phase)
            result.append(event.copy(mode=target, amplitude=event.amplitude * phase))
        return result


ReflectivitySchedule = float | Mapping[int, float] | Callable[[int, float], float]


@dataclass(frozen=True, slots=True)
class DynamicBeamSplitter(BeamSplitter):
    """Beam splitter controlled by time bin or a callable schedule."""

    schedule: ReflectivitySchedule = 0.5
    switching_time: float = 0.0

    def __post_init__(self) -> None:
        BeamSplitter.__post_init__(self)
        if self.switching_time < 0:
            raise ValidationError("switching_time cannot be negative")
        if isinstance(self.schedule, (int, float)):
            _probability("schedule", float(self.schedule))
        elif isinstance(self.schedule, Mapping):
            for value in self.schedule.values():
                _probability("scheduled reflectivity", value)

    def _scheduled(self, bin_index: int, time: float) -> float:
        if callable(self.schedule):
            return _probability("scheduled reflectivity", self.schedule(bin_index, time))
        if isinstance(self.schedule, Mapping):
            if not self.schedule:
                return self.reflectivity
            eligible = [key for key in self.schedule if key <= bin_index]
            key = max(eligible) if eligible else min(self.schedule)
            return float(self.schedule[key])
        return float(self.schedule)

    def reflectivity_at(self, event: PhotonEvent, context: SimulationContext) -> float:
        bin_index = context.bin_at(event.time)
        target = self._scheduled(bin_index, event.time)
        if self.switching_time == 0 or not isinstance(self.schedule, Mapping):
            return target
        changes = sorted(self.schedule)
        previous = self.reflectivity
        for change in changes:
            change_time = context.start_time + change * context.time_bin_width
            if event.time < change_time:
                break
            new = float(self.schedule[change])
            elapsed = event.time - change_time
            if elapsed < self.switching_time:
                fraction = elapsed / self.switching_time
                return previous + fraction * (new - previous)
            previous = new
        return target


@dataclass(slots=True)
class FeedForwardController:
    """Timestamped electronic feed-forward signal with finite latency."""

    latency: float = 0.0
    jitter: float = 0.0
    default: float = 0.0
    _commands: list[tuple[float, float]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.latency < 0 or self.jitter < 0:
            raise ValidationError("feed-forward latency and jitter must be non-negative")

    def trigger(self, time: float, value: float, rng: np.random.Generator | None = None) -> float:
        noise = 0.0
        if self.jitter and rng is not None:
            noise = float(rng.normal(0.0, self.jitter))
        effective = time + self.latency + noise
        self._commands.append((effective, float(value)))
        self._commands.sort(key=lambda item: item[0])
        return effective

    def value_at(self, time: float, rise_time: float = 0.0) -> float:
        """Return the control value, linearly interpolating finite transitions."""

        if rise_time < 0:
            raise ValidationError("rise_time cannot be negative")
        previous = self.default
        for effective, value in self._commands:
            if time < effective:
                break
            if rise_time and time < effective + rise_time:
                fraction = (time - effective) / rise_time
                return previous + fraction * (value - previous)
            previous = value
        return previous

    def clear(self) -> None:
        self._commands.clear()


@dataclass(frozen=True, slots=True)
class EOMSwitch:
    """Electro-optic 2x2 switch with rise time and finite extinction ratio."""

    mode_a: int = 0
    mode_b: int = 1
    control: FeedForwardController | Callable[[float], float] | float = 0.0
    rise_time: float = 0.0
    control_latency: float = 0.0
    extinction_ratio_db: float = 40.0

    def __post_init__(self) -> None:
        if self.mode_a == self.mode_b or min(self.mode_a, self.mode_b) < 0:
            raise ValidationError("switch modes must be distinct and non-negative")
        if self.rise_time < 0 or self.control_latency < 0 or self.extinction_ratio_db < 0:
            raise ValidationError("switch timing and extinction ratio must be non-negative")

    def _control_value(self, time: float) -> float:
        query_time = time - self.control_latency
        if isinstance(self.control, FeedForwardController):
            return self.control.value_at(query_time, self.rise_time)
        if callable(self.control):
            return float(self.control(query_time))
        return float(self.control)

    def process(
        self,
        events: Sequence[PhotonEvent],
        rng: np.random.Generator,
        context: SimulationContext,
    ) -> list[PhotonEvent]:
        del context
        leakage = 10 ** (-self.extinction_ratio_db / 10)
        result: list[PhotonEvent] = []
        for event in events:
            if event.mode not in (self.mode_a, self.mode_b):
                result.append(event.copy())
                continue
            control = float(np.clip(self._control_value(event.time), 0.0, 1.0))
            switch_probability = leakage + (1 - 2 * leakage) * control
            if rng.random() < switch_probability:
                target = self.mode_b if event.mode == self.mode_a else self.mode_a
                result.append(event.copy(mode=target))
            else:
                result.append(event.copy())
        return result


@dataclass(frozen=True, slots=True)
class PhaseDrift:
    """Wiener-like phase drift applied independently to each event."""

    standard_deviation: float
    reference_time: float = 1.0
    modes: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if self.standard_deviation < 0 or self.reference_time <= 0:
            raise ValidationError("invalid phase-drift scale")

    def process(
        self,
        events: Sequence[PhotonEvent],
        rng: np.random.Generator,
        context: SimulationContext,
    ) -> list[PhotonEvent]:
        result: list[PhotonEvent] = []
        for event in events:
            if self.modes is not None and event.mode not in self.modes:
                result.append(event.copy())
                continue
            elapsed = max(0.0, event.time - context.start_time)
            sigma = self.standard_deviation * sqrt(elapsed / self.reference_time)
            phase = rng.normal(0.0, sigma) if sigma else 0.0
            result.append(event.copy(amplitude=event.amplitude * np.exp(1j * phase)))
        return result


@dataclass(frozen=True, slots=True)
class TemperaturePhaseDrift:
    """Apply thermo-optic phase from a static or time-dependent temperature."""

    length: float
    wavelength: float = 1550e-9
    temperature: float | Callable[[float], float] = 20.0
    reference_temperature: float = 20.0
    thermo_optic_coefficient: float = 1.86e-4
    modes: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if self.length < 0 or self.wavelength <= 0:
            raise ValidationError("length must be non-negative and wavelength positive")

    def process(
        self,
        events: Sequence[PhotonEvent],
        rng: np.random.Generator,
        context: SimulationContext,
    ) -> list[PhotonEvent]:
        del rng, context
        result: list[PhotonEvent] = []
        for event in events:
            if self.modes is not None and event.mode not in self.modes:
                result.append(event.copy())
                continue
            temperature = (
                self.temperature(event.time) if callable(self.temperature) else self.temperature
            )
            phase = thermo_optic_phase(
                self.length,
                self.wavelength,
                temperature - self.reference_temperature,
                self.thermo_optic_coefficient,
            )
            result.append(event.copy(amplitude=event.amplitude * np.exp(1j * phase)))
        return result


@dataclass(frozen=True, slots=True)
class FiberLoop:
    """Recirculating fiber loop with loss, drift, delay, and dynamic out-coupling."""

    round_trip_time: float
    transmission: float = 0.95
    outcoupling: float | Mapping[int, float] | Callable[[int, float], float] = 0.5
    phase_per_roundtrip: float = 0.0
    phase_drift_std: float = 0.0
    max_roundtrips: int = 32
    input_mode: int = 0
    output_mode: int = 0
    length_error: float = 0.0
    group_velocity: float = 2.04e8

    def __post_init__(self) -> None:
        if self.round_trip_time <= 0 or self.max_roundtrips < 1:
            raise ValidationError("round_trip_time must be positive and max_roundtrips >= 1")
        _probability("transmission", self.transmission)
        if isinstance(self.outcoupling, (int, float)):
            _probability("outcoupling", float(self.outcoupling))
        if self.phase_drift_std < 0 or self.group_velocity <= 0:
            raise ValidationError("invalid fiber-loop drift or group velocity")

    def _outcoupling(self, roundtrip: int, time: float) -> float:
        if callable(self.outcoupling):
            return _probability("outcoupling", self.outcoupling(roundtrip, time))
        if isinstance(self.outcoupling, Mapping):
            eligible = [key for key in self.outcoupling if key <= roundtrip]
            if not eligible:
                return 0.0
            return _probability("outcoupling", self.outcoupling[max(eligible)])
        return float(self.outcoupling)

    def process(
        self,
        events: Sequence[PhotonEvent],
        rng: np.random.Generator,
        context: SimulationContext,
    ) -> list[PhotonEvent]:
        del context
        result: list[PhotonEvent] = []
        corrected_delay = self.round_trip_time + self.length_error / self.group_velocity
        if corrected_delay <= 0:
            raise ValidationError("length_error makes the effective round-trip time non-positive")
        for event in events:
            if event.mode != self.input_mode:
                result.append(event.copy())
                continue
            current = event.copy()
            for roundtrip in range(1, self.max_roundtrips + 1):
                if rng.random() >= self.transmission:
                    break
                drift = rng.normal(0.0, self.phase_drift_std) if self.phase_drift_std else 0.0
                current = current.copy(
                    time=current.time + corrected_delay,
                    amplitude=current.amplitude * np.exp(1j * (self.phase_per_roundtrip + drift)),
                    roundtrips=roundtrip,
                )
                if roundtrip == self.max_roundtrips or rng.random() < self._outcoupling(
                    roundtrip, current.time
                ):
                    result.append(current.copy(mode=self.output_mode))
                    break
        return result


def thermo_optic_phase(
    length: float,
    wavelength: float,
    delta_temperature: float,
    thermo_optic_coefficient: float = 1.86e-4,
) -> float:
    """Return phase drift caused by a temperature change in a waveguide/fiber."""

    if length < 0 or wavelength <= 0:
        raise ValidationError("length must be non-negative and wavelength positive")
    return 2 * pi * length * thermo_optic_coefficient * delta_temperature / wavelength
