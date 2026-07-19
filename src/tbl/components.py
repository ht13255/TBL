"""TBL time-domain hardware components for event-resolved simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
        if not np.isfinite(self.time_bin_width) or self.time_bin_width <= 0:
            raise ValidationError("time_bin_width must be positive and finite")
        if not np.isfinite(self.start_time):
            raise ValidationError("start_time must be finite")

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
        if self.modes is not None and any(mode < 0 for mode in self.modes):
            raise ValidationError("loss-channel modes must be non-negative")

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
            if (self.modes is not None and event.mode not in self.modes)
            or rng.random() < self.transmission
        ]


@dataclass(frozen=True, slots=True)
class DelayLine:
    """Deterministic propagation delay, optionally with Gaussian jitter."""

    delay: float
    jitter: float = 0.0
    modes: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.delay) or not np.isfinite(self.jitter):
            raise ValidationError("delay and jitter must be finite")
        if self.delay < 0 or self.jitter < 0:
            raise ValidationError("delay and jitter must be non-negative")
        if self.modes is not None and any(mode < 0 for mode in self.modes):
            raise ValidationError("delay-line modes must be non-negative")

    def process(
        self,
        events: Sequence[PhotonEvent],
        rng: np.random.Generator,
        context: SimulationContext,
    ) -> list[PhotonEvent]:
        del context
        if self.modes is None:
            if self.jitter:
                delays = self.delay + rng.normal(0.0, self.jitter, len(events))
                return [
                    event.shifted(float(delay)) for event, delay in zip(events, delays, strict=True)
                ]
            return [event.shifted(self.delay) for event in events]
        result: list[PhotonEvent] = []
        for event in events:
            if event.mode not in self.modes:
                result.append(event.copy())
                continue
            jitter = rng.normal(0.0, self.jitter) if self.jitter else 0.0
            result.append(event.shifted(self.delay + jitter))
        return result


@dataclass(frozen=True, slots=True)
class PhaseShifter:
    """Static phase shift on selected modes."""

    phase: float
    modes: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.phase):
            raise ValidationError("phase must be finite")
        if self.modes is not None and any(mode < 0 for mode in self.modes):
            raise ValidationError("phase-shifter modes must be non-negative")

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
        if not np.isfinite(self.phase):
            raise ValidationError("beam-splitter phase must be finite")

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
        if not np.isfinite(self.switching_time) or self.switching_time < 0:
            raise ValidationError("switching_time must be finite and non-negative")
        if isinstance(self.schedule, (int, float)):
            _probability("schedule", float(self.schedule))
        elif isinstance(self.schedule, Mapping):
            if any(
                isinstance(key, bool) or not isinstance(key, (int, np.integer))
                for key in self.schedule
            ):
                raise ValidationError("schedule keys must be integer time bins")
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
        if not all(np.isfinite(value) for value in (self.latency, self.jitter, self.default)):
            raise ValidationError("feed-forward parameters must be finite")
        if self.latency < 0 or self.jitter < 0:
            raise ValidationError("feed-forward latency and jitter must be non-negative")

    def trigger(self, time: float, value: float, rng: np.random.Generator | None = None) -> float:
        if not np.isfinite(time) or not np.isfinite(value):
            raise ValidationError("feed-forward trigger time and value must be finite")
        noise = 0.0
        if self.jitter and rng is not None:
            noise = float(rng.normal(0.0, self.jitter))
        effective = time + self.latency + noise
        self._commands.append((effective, float(value)))
        self._commands.sort(key=lambda item: item[0])
        return effective

    def value_at(self, time: float, rise_time: float = 0.0) -> float:
        """Return the control value, linearly interpolating finite transitions."""

        if not np.isfinite(time) or not np.isfinite(rise_time) or rise_time < 0:
            raise ValidationError("time and rise_time must be finite, with rise_time non-negative")
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
    transmission: float = 1.0
    drive_noise_std: float = 0.0

    def __post_init__(self) -> None:
        if self.mode_a == self.mode_b or min(self.mode_a, self.mode_b) < 0:
            raise ValidationError("switch modes must be distinct and non-negative")
        finite = (
            self.rise_time,
            self.control_latency,
            self.extinction_ratio_db,
            self.transmission,
            self.drive_noise_std,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValidationError("switch parameters must be finite")
        if self.rise_time < 0 or self.control_latency < 0 or self.extinction_ratio_db < 0:
            raise ValidationError("switch timing and extinction ratio must be non-negative")
        _probability("transmission", self.transmission)
        if self.drive_noise_std < 0:
            raise ValidationError("drive_noise_std cannot be negative")

    def _control_value(self, time: float) -> float:
        query_time = time - self.control_latency
        if isinstance(self.control, FeedForwardController):
            value = self.control.value_at(query_time, self.rise_time)
        elif callable(self.control):
            value = float(self.control(query_time))
        else:
            value = float(self.control)
        if not np.isfinite(value):
            raise ValidationError("switch control returned a non-finite value")
        return value

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
            if rng.random() >= self.transmission:
                continue
            noise = rng.normal(0.0, self.drive_noise_std) if self.drive_noise_std else 0.0
            control = float(np.clip(self._control_value(event.time) + noise, 0.0, 1.0))
            switch_probability = leakage + (1 - 2 * leakage) * control
            if rng.random() < switch_probability:
                target = self.mode_b if event.mode == self.mode_a else self.mode_a
                result.append(event.copy(mode=target))
            else:
                result.append(event.copy())
        return result


@dataclass(frozen=True, slots=True)
class PhaseDrift:
    """Time-correlated Wiener phase drift.

    ``standard_deviation`` is the RMS phase accumulated over
    ``reference_time``. Events at the same time see exactly the same phase.
    Set ``independent_by_mode`` when distinct paths have independent baths.
    """

    standard_deviation: float
    reference_time: float = 1.0
    modes: frozenset[int] | None = None
    independent_by_mode: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.standard_deviation) or not np.isfinite(self.reference_time):
            raise ValidationError("phase-drift scale must be finite")
        if self.standard_deviation < 0 or self.reference_time <= 0:
            raise ValidationError("invalid phase-drift scale")

    def process(
        self,
        events: Sequence[PhotonEvent],
        rng: np.random.Generator,
        context: SimulationContext,
    ) -> list[PhotonEvent]:
        result = [event.copy() for event in events]
        states: dict[int, tuple[float, float]] = {}
        selected = sorted(
            (
                (index, event)
                for index, event in enumerate(events)
                if self.modes is None or event.mode in self.modes
            ),
            key=lambda item: item[1].time,
        )
        for index, event in selected:
            key = event.mode if self.independent_by_mode else 0
            last_time, phase = states.get(key, (context.start_time, 0.0))
            elapsed = max(0.0, event.time - last_time)
            sigma = self.standard_deviation * sqrt(elapsed / self.reference_time)
            if sigma:
                phase += float(rng.normal(0.0, sigma))
            states[key] = (max(last_time, event.time), phase)
            result[index] = event.copy(amplitude=event.amplitude * np.exp(1j * phase))
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
        parameters = (
            self.length,
            self.wavelength,
            self.reference_temperature,
            self.thermo_optic_coefficient,
        )
        if not all(np.isfinite(value) for value in parameters):
            raise ValidationError("thermo-optic parameters must be finite")
        if not callable(self.temperature) and not np.isfinite(self.temperature):
            raise ValidationError("temperature must be finite")
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
            if not np.isfinite(temperature):
                raise ValidationError("temperature callable returned a non-finite value")
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
    fiber_length: float | None = None
    attenuation_db_per_km: float = 0.0
    insertion_loss_db: float = 0.0
    dispersion_beta2: float = 0.0
    pmd_coefficient: float = 0.0
    shared_environment: bool = True

    def __post_init__(self) -> None:
        finite = (
            self.round_trip_time,
            self.transmission,
            self.phase_per_roundtrip,
            self.phase_drift_std,
            self.length_error,
            self.group_velocity,
            self.attenuation_db_per_km,
            self.insertion_loss_db,
            self.dispersion_beta2,
            self.pmd_coefficient,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValidationError("fiber-loop parameters must be finite")
        if self.fiber_length is not None and not np.isfinite(self.fiber_length):
            raise ValidationError("fiber_length must be finite")
        if self.round_trip_time <= 0 or self.max_roundtrips < 1:
            raise ValidationError("round_trip_time must be positive and max_roundtrips >= 1")
        _probability("transmission", self.transmission)
        if isinstance(self.outcoupling, (int, float)):
            _probability("outcoupling", float(self.outcoupling))
        elif isinstance(self.outcoupling, Mapping):
            if any(
                isinstance(key, bool) or not isinstance(key, (int, np.integer)) or key < 0
                for key in self.outcoupling
            ):
                raise ValidationError("outcoupling keys must be non-negative round trips")
            for value in self.outcoupling.values():
                _probability("outcoupling", value)
        if self.phase_drift_std < 0 or self.group_velocity <= 0:
            raise ValidationError("invalid fiber-loop drift or group velocity")
        if self.fiber_length is not None and self.fiber_length <= 0:
            raise ValidationError("fiber_length must be positive when supplied")
        if self.attenuation_db_per_km < 0 or self.insertion_loss_db < 0:
            raise ValidationError("fiber attenuation and insertion loss cannot be negative")
        if not np.isfinite(self.dispersion_beta2) or self.pmd_coefficient < 0:
            raise ValidationError("invalid dispersion or PMD coefficient")

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
        result: list[PhotonEvent] = []
        phase_by_bin: dict[int, float] = {}
        pmd_by_bin: dict[int, float] = {}

        def environmental_noise(time: float, length: float) -> tuple[float, float]:
            if not self.shared_environment:
                drift = (
                    float(rng.normal(0.0, self.phase_drift_std)) if self.phase_drift_std else 0.0
                )
                pmd = (
                    float(rng.normal(0.0, self.pmd_coefficient * sqrt(length)))
                    if self.pmd_coefficient
                    else 0.0
                )
                return drift, pmd
            time_bin = context.bin_at(time)
            if time_bin not in phase_by_bin:
                phase_by_bin[time_bin] = (
                    float(rng.normal(0.0, self.phase_drift_std)) if self.phase_drift_std else 0.0
                )
                pmd_by_bin[time_bin] = (
                    float(rng.normal(0.0, self.pmd_coefficient * sqrt(length)))
                    if self.pmd_coefficient
                    else 0.0
                )
            return phase_by_bin[time_bin], pmd_by_bin[time_bin]

        corrected_delay = self.round_trip_time + self.length_error / self.group_velocity
        if corrected_delay <= 0:
            raise ValidationError("length_error makes the effective round-trip time non-positive")
        length = (
            self.fiber_length
            if self.fiber_length is not None
            else self.group_velocity * self.round_trip_time
        )
        distributed_loss_db = self.attenuation_db_per_km * length / 1000
        effective_transmission = self.transmission * 10 ** (
            -(distributed_loss_db + self.insertion_loss_db) / 10
        )
        for event in events:
            if event.mode != self.input_mode:
                result.append(event.copy())
                continue
            current = event.copy()
            for roundtrip in range(1, self.max_roundtrips + 1):
                if rng.random() >= effective_transmission:
                    break
                traversal_time = current.time + corrected_delay
                drift, pmd_delay = environmental_noise(traversal_time, length)
                packet = current.photon.wavepacket
                if self.dispersion_beta2:
                    packet = packet.dispersed(self.dispersion_beta2 * length)
                new_time = traversal_time + pmd_delay
                packet = packet._copy_with(arrival_time=new_time)
                photon = replace(current.photon, wavepacket=packet)
                metadata = dict(current.metadata)
                metadata.update(
                    loop_effective_transmission=effective_transmission,
                    loop_phase_drift_rad=drift,
                    loop_pmd_delay_s=pmd_delay,
                    loop_gdd_s2=self.dispersion_beta2 * length,
                )
                current = current.copy(
                    photon=photon,
                    time=new_time,
                    amplitude=current.amplitude * np.exp(1j * (self.phase_per_roundtrip + drift)),
                    roundtrips=roundtrip,
                    metadata=metadata,
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

    if not all(
        np.isfinite(value)
        for value in (length, wavelength, delta_temperature, thermo_optic_coefficient)
    ):
        raise ValidationError("thermo-optic inputs must be finite")
    if length < 0 or wavelength <= 0:
        raise ValidationError("length must be non-negative and wavelength positive")
    return 2 * pi * length * thermo_optic_coefficient * delta_temperature / wavelength
