"""TBL physical data models used by circuit and event simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt
from typing import Any

import numpy as np

from .errors import ValidationError

_C = 299_792_458.0
_SETATTR = object.__setattr__
_PHOTON_EVENT_FIELDS = frozenset(
    {"photon", "time", "mode", "amplitude", "shot", "roundtrips", "metadata"}
)


def _probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValidationError(f"{name} must be in [0, 1], got {value}")
    return value


@dataclass(frozen=True, slots=True)
class Wavepacket:
    """Gaussian single-photon wavepacket with optional quadratic chirp.

    Times are seconds, wavelength is metres, and ``temporal_width`` is the
    standard deviation of the complex field envelope. Polarization is a Jones
    vector in the H/V basis and is normalized on construction. ``chirp`` is
    dimensionless in the convention
    ``exp(-(1-i*chirp)*(t-t0)**2/(4*temporal_width**2))``.

    ``purity`` is a phenomenological internal-state purity. The maximum HOM
    indistinguishability of otherwise identical packets is
    ``sqrt(purity_1 * purity_2)``.
    """

    arrival_time: float = 0.0
    temporal_width: float = 20e-12
    wavelength: float = 1550e-9
    polarization: tuple[complex, complex] = (1.0 + 0j, 0.0 + 0j)
    purity: float = 1.0
    chirp: float = 0.0
    label: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.arrival_time):
            raise ValidationError("arrival_time must be finite")
        if not np.isfinite(self.temporal_width) or self.temporal_width <= 0:
            raise ValidationError("temporal_width must be positive")
        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValidationError("wavelength must be positive")
        _probability("purity", self.purity)
        if not np.isfinite(self.chirp):
            raise ValidationError("chirp must be finite")
        p = np.asarray(self.polarization, dtype=complex)
        if p.shape != (2,) or not np.all(np.isfinite(p)) or np.linalg.norm(p) == 0:
            raise ValidationError("polarization must be a non-zero two-element Jones vector")
        p = p / np.linalg.norm(p)
        _SETATTR(self, "polarization", (complex(p[0]), complex(p[1])))

    @property
    def angular_frequency(self) -> float:
        return 2 * pi * _C / self.wavelength

    def shifted(self, delay: float) -> Wavepacket:
        return self._copy_with(arrival_time=self.arrival_time + float(delay))

    def _copy_with(
        self,
        *,
        arrival_time: float | None = None,
        temporal_width: float | None = None,
        wavelength: float | None = None,
        chirp: float | None = None,
    ) -> Wavepacket:
        """Fast immutable copy after callers have established physical validity."""

        packet = object.__new__(Wavepacket)
        _SETATTR(
            packet,
            "arrival_time",
            self.arrival_time if arrival_time is None else float(arrival_time),
        )
        _SETATTR(
            packet,
            "temporal_width",
            self.temporal_width if temporal_width is None else float(temporal_width),
        )
        _SETATTR(
            packet,
            "wavelength",
            self.wavelength if wavelength is None else float(wavelength),
        )
        _SETATTR(packet, "polarization", self.polarization)
        _SETATTR(packet, "purity", self.purity)
        _SETATTR(packet, "chirp", self.chirp if chirp is None else float(chirp))
        _SETATTR(packet, "label", self.label)
        return packet

    @property
    def spectral_width_angular(self) -> float:
        """Intensity standard deviation in angular frequency (rad/s)."""

        return sqrt(1 + self.chirp**2) / (2 * self.temporal_width)

    def mode_overlap(self, other: Wavepacket) -> complex:
        """Return normalized pure temporal-polarization mode overlap.

        This analytic complex Gaussian integral includes unequal temporal
        widths, arrival times, carrier frequencies, Jones vectors, and chirp.
        Purity is excluded and applied by :meth:`overlap`.
        """

        s1, s2 = self.temporal_width, other.temporal_width
        delay = other.arrival_time - self.arrival_time
        w1, w2 = self.angular_frequency, other.angular_frequency
        a1 = (1 - 1j * self.chirp) / (4 * s1**2)
        a2 = (1 - 1j * other.chirp) / (4 * s2**2)
        quadratic = np.conj(a1) + a2
        # Integrate in the coordinate u=t-self.arrival_time. Using absolute
        # laboratory timestamps here causes catastrophic cancellation once a
        # long acquisition time is many orders above the picosecond envelope.
        linear = 2 * a2 * delay + 1j * (w1 - w2)
        constant = -a2 * delay**2 + 1j * w2 * delay
        normalization = (2 * pi * s1**2) ** (-0.25) * (2 * pi * s2**2) ** (-0.25)
        temporal = (
            normalization * np.sqrt(pi / quadratic) * np.exp(linear**2 / (4 * quadratic) + constant)
        )
        polarization = np.vdot(np.asarray(self.polarization), np.asarray(other.polarization))
        return complex(temporal * polarization)

    def overlap(self, other: Wavepacket) -> complex:
        """Return effective internal-mode overlap amplitude.

        Its squared magnitude is the pair indistinguishability. The fourth-root
        purity amplitude makes that observable scale as
        ``sqrt(purity_1 * purity_2)``.
        """

        purity_amplitude = (self.purity * other.purity) ** 0.25
        return self.mode_overlap(other) * purity_amplitude

    def indistinguishability(self, other: Wavepacket) -> float:
        return float(np.clip(abs(self.overlap(other)) ** 2, 0.0, 1.0))

    def amplitude(self, time: np.ndarray | float) -> np.ndarray:
        """Evaluate the normalized temporal field amplitude."""

        t = np.asarray(time, dtype=float)
        relative = t - self.arrival_time
        norm = (2 * pi * self.temporal_width**2) ** (-0.25)
        envelope = np.exp(-(1 - 1j * self.chirp) * relative**2 / (4 * self.temporal_width**2))
        return norm * envelope * np.exp(-1j * self.angular_frequency * relative)

    def dispersed(self, group_delay_dispersion: float) -> Wavepacket:
        """Propagate through second-order group-delay dispersion in s².

        The spectral field acquires ``exp(i*GDD*detuning**2/2)``. The analytic
        update conserves pulse energy and spectral width.
        """

        if not np.isfinite(group_delay_dispersion):
            raise ValidationError("group_delay_dispersion must be finite")
        spectral_quadratic = self.temporal_width**2 / (1 - 1j * self.chirp)
        spectral_quadratic -= 0.5j * group_delay_dispersion
        temporal_quadratic = 1 / (4 * spectral_quadratic)
        if temporal_quadratic.real <= 0:
            raise ValidationError("dispersion produced a non-normalizable wavepacket")
        width = sqrt(1 / (4 * temporal_quadratic.real))
        chirp = -temporal_quadratic.imag / temporal_quadratic.real
        return self._copy_with(temporal_width=width, chirp=float(chirp))


@dataclass(frozen=True, slots=True)
class Photon:
    """A photon carried by an event or injected into a linear-optical mode."""

    wavepacket: Wavepacket = field(default_factory=Wavepacket)
    mode: int = 0
    time_bin: int = 0
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.mode < 0:
            raise ValidationError("mode must be non-negative")


def _copy_photon(
    photon: Photon,
    *,
    wavepacket: Wavepacket | None = None,
    mode: int | None = None,
) -> Photon:
    copied = object.__new__(Photon)
    _SETATTR(
        copied, "wavepacket", photon.wavepacket if wavepacket is None else wavepacket
    )
    _SETATTR(copied, "mode", photon.mode if mode is None else mode)
    _SETATTR(copied, "time_bin", photon.time_bin)
    _SETATTR(copied, "metadata", photon.metadata)
    return copied


@dataclass(slots=True)
class PhotonEvent:
    """Mutable time-domain propagation event."""

    photon: Photon
    time: float
    mode: int
    amplitude: complex = 1.0 + 0j
    shot: int = 0
    roundtrips: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.time):
            raise ValidationError("event time must be finite")
        if self.mode < 0 or self.photon.mode != self.mode:
            raise ValidationError("event and photon modes must match and be non-negative")
        if self.photon.wavepacket.arrival_time != self.time:
            raise ValidationError("event time must equal wavepacket arrival_time")
        if self.roundtrips < 0:
            raise ValidationError("roundtrips cannot be negative")
        if not np.isfinite(self.amplitude.real) or not np.isfinite(self.amplitude.imag):
            raise ValidationError("event amplitude must be finite")

    def copy(self, **changes: Any) -> PhotonEvent:
        for name in changes:
            if name not in _PHOTON_EVENT_FIELDS:
                raise TypeError(f"PhotonEvent.copy() got an unexpected field {name!r}")
        event = object.__new__(PhotonEvent)
        event.time = float(changes.get("time", self.time))
        if not np.isfinite(event.time):
            raise ValidationError("event time must be finite")
        event.mode = int(changes.get("mode", self.mode))
        if event.mode < 0:
            raise ValidationError("mode must be non-negative")
        photon = changes.get("photon", self.photon)
        packet = photon.wavepacket
        if packet.arrival_time != event.time:
            packet = packet._copy_with(arrival_time=event.time)
        if packet is not photon.wavepacket or photon.mode != event.mode:
            photon = _copy_photon(photon, wavepacket=packet, mode=event.mode)
        event.photon = photon
        event.amplitude = changes.get("amplitude", self.amplitude)
        event.shot = changes.get("shot", self.shot)
        event.roundtrips = changes.get("roundtrips", self.roundtrips)
        if event.roundtrips < 0:
            raise ValidationError("roundtrips cannot be negative")
        if not np.isfinite(event.amplitude.real) or not np.isfinite(event.amplitude.imag):
            raise ValidationError("event amplitude must be finite")
        event.metadata = changes["metadata"] if "metadata" in changes else dict(self.metadata)
        return event

    def shifted(self, delay: float) -> PhotonEvent:
        """Fast physically consistent time shift used by propagation components."""

        target_time = self.time + float(delay)
        if not np.isfinite(target_time):
            raise ValidationError("shifted event time must be finite")
        packet = self.photon.wavepacket._copy_with(arrival_time=target_time)
        photon = _copy_photon(self.photon, wavepacket=packet, mode=self.mode)
        event = object.__new__(PhotonEvent)
        event.photon = photon
        event.time = target_time
        event.mode = self.mode
        event.amplitude = self.amplitude
        event.shot = self.shot
        event.roundtrips = self.roundtrips
        event.metadata = dict(self.metadata)
        return event


def _source_photon_event(
    packet: Wavepacket, mode: int, shot: int, source_index: int
) -> PhotonEvent:
    """Construct a validated source event without repeated dataclass dispatch."""

    photon = object.__new__(Photon)
    _SETATTR(photon, "wavepacket", packet)
    _SETATTR(photon, "mode", mode)
    _SETATTR(photon, "time_bin", 0)
    _SETATTR(photon, "metadata", {"source_index": source_index})
    event = object.__new__(PhotonEvent)
    event.photon = photon
    event.time = packet.arrival_time
    event.mode = mode
    event.amplitude = 1.0 + 0j
    event.shot = shot
    event.roundtrips = 0
    event.metadata = {}
    return event


@dataclass(frozen=True, slots=True)
class TimeBinQubit:
    """Single-photon qubit ``alpha|early> + beta|late>``."""

    alpha: complex = 1.0 + 0j
    beta: complex = 0.0 + 0j
    separation: float = 1e-9
    wavepacket: Wavepacket = field(default_factory=Wavepacket)

    def __post_init__(self) -> None:
        if not np.isfinite(self.separation) or self.separation <= 0:
            raise ValidationError("separation must be positive")
        if not all(
            np.isfinite(value)
            for value in (self.alpha.real, self.alpha.imag, self.beta.real, self.beta.imag)
        ):
            raise ValidationError("alpha and beta must be finite")
        norm = sqrt(abs(self.alpha) ** 2 + abs(self.beta) ** 2)
        if norm == 0:
            raise ValidationError("alpha and beta cannot both be zero")
        _SETATTR(self, "alpha", self.alpha / norm)
        _SETATTR(self, "beta", self.beta / norm)

    def events(self, *, mode: int = 0, shot: int = 0) -> list[PhotonEvent]:
        early_photon = Photon(self.wavepacket, mode=mode, time_bin=0)
        late_packet = self.wavepacket.shifted(self.separation)
        late_photon = Photon(late_packet, mode=mode, time_bin=1)
        return [
            PhotonEvent(early_photon, self.wavepacket.arrival_time, mode, self.alpha, shot, 0),
            PhotonEvent(
                late_photon, late_packet.arrival_time, mode, self.beta, shot, 0
            ),
        ]

    def sample_event(
        self, rng: np.random.Generator, *, mode: int = 0, shot: int = 0
    ) -> PhotonEvent:
        """Sample one time-bin detection path for event-domain Monte Carlo."""

        late = rng.random() >= abs(self.alpha) ** 2
        delay = self.separation if late else 0.0
        amplitude = self.beta if late else self.alpha
        packet = self.wavepacket.shifted(delay)
        photon = Photon(packet, mode=mode, time_bin=int(late))
        return PhotonEvent(
            photon,
            packet.arrival_time,
            mode,
            amplitude / max(abs(amplitude), np.finfo(float).eps),
            shot,
        )


@dataclass(frozen=True, slots=True)
class SinglePhotonSource:
    """Pulsed source with vacuum, one-photon, and two-photon emission."""

    repetition_rate: float = 80e6
    p_single: float = 1.0
    p_double: float = 0.0
    wavepacket: Wavepacket = field(default_factory=Wavepacket)
    emission_jitter: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.repetition_rate) or self.repetition_rate <= 0:
            raise ValidationError("repetition_rate must be positive")
        _probability("p_single", self.p_single)
        _probability("p_double", self.p_double)
        if self.p_single + self.p_double > 1:
            raise ValidationError("p_single + p_double cannot exceed 1")
        if not np.isfinite(self.emission_jitter) or self.emission_jitter < 0:
            raise ValidationError("emission_jitter cannot be negative")

    @property
    def period(self) -> float:
        return 1.0 / self.repetition_rate

    def emit(self, shot: int, rng: np.random.Generator, *, mode: int = 0) -> list[PhotonEvent]:
        if mode < 0:
            raise ValidationError("mode must be non-negative")
        draw = rng.random()
        count = 2 if draw < self.p_double else 1 if draw < self.p_double + self.p_single else 0
        base_time = shot * self.period
        events: list[PhotonEvent] = []
        for index in range(count):
            jitter = rng.normal(0.0, self.emission_jitter) if self.emission_jitter else 0.0
            packet = self.wavepacket.shifted(base_time + jitter - self.wavepacket.arrival_time)
            events.append(_source_photon_event(packet, mode, shot, index))
        return events

    def emit_many(
        self, shots: int, rng: np.random.Generator, *, mode: int = 0
    ) -> list[PhotonEvent]:
        """Vectorize pulse statistics while preserving event-level objects."""

        if shots < 0:
            raise ValidationError("shots cannot be negative")
        if mode < 0:
            raise ValidationError("mode must be non-negative")
        draws = rng.random(shots)
        counts = np.where(
            draws < self.p_double,
            2,
            np.where(draws < self.p_double + self.p_single, 1, 0),
        )
        shot_indices = np.repeat(np.arange(shots, dtype=np.int64), counts)
        if self.emission_jitter:
            jitters = rng.normal(0.0, self.emission_jitter, len(shot_indices))
        else:
            jitters = np.zeros(len(shot_indices))
        events: list[PhotonEvent] = []
        source_indices = np.zeros(len(shot_indices), dtype=np.int8)
        double_pulses = counts == 2
        if np.any(double_pulses):
            source_indices[np.cumsum(counts)[double_pulses] - 1] = 1
        for shot, source_index, jitter in zip(
            shot_indices, source_indices, jitters, strict=True
        ):
            arrival = int(shot) * self.period + float(jitter)
            packet = self.wavepacket._copy_with(arrival_time=arrival)
            events.append(
                _source_photon_event(packet, mode, int(shot), int(source_index))
            )
        return events


@dataclass(slots=True)
class CorrelatedPhotonSource:
    """Pulsed source with measured-style g²(0), blinking, and spectral diffusion.

    ``mean_photon_number`` and ``g2_zero`` define a truncated 0/1/2 photon
    distribution through ``p2=g2*mean**2/2`` and ``p1=mean-2*p2``. Blinking is
    a two-state Markov chain per excitation pulse. Spectral diffusion is a
    stationary Ornstein-Uhlenbeck process in angular frequency (rad/s).

    This model is useful for quantum-dot and heralded-source digital twins
    when independent, identical pulses are not a defensible approximation.
    """

    repetition_rate: float = 80e6
    mean_photon_number: float = 0.8
    g2_zero: float = 0.01
    collection_efficiency: float = 1.0
    wavepacket: Wavepacket = field(default_factory=Wavepacket)
    emission_jitter: float = 0.0
    blink_on_to_off: float = 0.0
    blink_off_to_on: float = 1.0
    initial_on_probability: float = 1.0
    spectral_diffusion_std: float = 0.0
    spectral_correlation_time: float = 0.0
    _is_on: bool = field(default=True, init=False, repr=False)
    _frequency_offset: float = field(default=0.0, init=False, repr=False)
    _last_shot: int = field(default=-1, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        finite_parameters = (
            self.repetition_rate,
            self.mean_photon_number,
            self.g2_zero,
            self.emission_jitter,
            self.spectral_diffusion_std,
            self.spectral_correlation_time,
        )
        if not all(np.isfinite(value) for value in finite_parameters):
            raise ValidationError("source rates and noise parameters must be finite")
        if self.repetition_rate <= 0:
            raise ValidationError("repetition_rate must be positive")
        if self.mean_photon_number < 0 or self.g2_zero < 0:
            raise ValidationError("mean_photon_number and g2_zero must be non-negative")
        _probability("collection_efficiency", self.collection_efficiency)
        _probability("blink_on_to_off", self.blink_on_to_off)
        _probability("blink_off_to_on", self.blink_off_to_on)
        _probability("initial_on_probability", self.initial_on_probability)
        if self.emission_jitter < 0 or self.spectral_diffusion_std < 0:
            raise ValidationError("jitter and spectral diffusion must be non-negative")
        if self.spectral_correlation_time < 0:
            raise ValidationError("spectral_correlation_time cannot be negative")
        p0, p1, p2 = self.photon_probabilities
        if min(p0, p1, p2) < -1e-12:
            raise ValidationError(
                "mean_photon_number and g2_zero do not define a physical 0/1/2 distribution"
            )

    @property
    def period(self) -> float:
        return 1.0 / self.repetition_rate

    @property
    def photon_probabilities(self) -> tuple[float, float, float]:
        p2 = 0.5 * self.g2_zero * self.mean_photon_number**2
        p1 = self.mean_photon_number - 2 * p2
        p0 = 1 - p1 - p2
        return float(p0), float(p1), float(p2)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset correlated hidden state before an independent acquisition."""

        self._is_on = True if rng is None else bool(rng.random() < self.initial_on_probability)
        self._frequency_offset = 0.0
        self._last_shot = -1
        self._initialized = True

    def _advance_hidden_state(self, shot: int, rng: np.random.Generator) -> None:
        if not self._initialized:
            self._is_on = bool(rng.random() < self.initial_on_probability)
            self._initialized = True
        if self._last_shot >= 0 and shot != self._last_shot + 1:
            raise ValidationError("CorrelatedPhotonSource shots must be emitted in order")
        if self._last_shot >= 0:
            if self._is_on:
                self._is_on = rng.random() >= self.blink_on_to_off
            else:
                self._is_on = rng.random() < self.blink_off_to_on
        if self.spectral_diffusion_std:
            if self._last_shot < 0 or self.spectral_correlation_time == 0:
                self._frequency_offset = float(rng.normal(0.0, self.spectral_diffusion_std))
            else:
                correlation = np.exp(-self.period / self.spectral_correlation_time)
                innovation = self.spectral_diffusion_std * sqrt(1 - correlation**2)
                self._frequency_offset = float(
                    correlation * self._frequency_offset + rng.normal(0.0, innovation)
                )
        self._last_shot = shot

    def emit(self, shot: int, rng: np.random.Generator, *, mode: int = 0) -> list[PhotonEvent]:
        self._advance_hidden_state(shot, rng)
        if not self._is_on:
            return []
        p0, p1, _ = self.photon_probabilities
        draw = rng.random()
        count = 0 if draw < p0 else 1 if draw < p0 + p1 else 2
        base_frequency = self.wavepacket.angular_frequency
        shifted_frequency = base_frequency + self._frequency_offset
        if shifted_frequency <= 0:
            raise ValidationError("spectral diffusion produced a non-positive carrier frequency")
        wavelength = 2 * pi * _C / shifted_frequency
        base_time = shot * self.period
        events: list[PhotonEvent] = []
        for index in range(count):
            if rng.random() >= self.collection_efficiency:
                continue
            jitter = rng.normal(0.0, self.emission_jitter) if self.emission_jitter else 0.0
            packet = self.wavepacket._copy_with(
                arrival_time=base_time + jitter, wavelength=wavelength
            )
            metadata = {
                "source_index": index,
                "source_model": "correlated",
                "spectral_offset_rad_s": self._frequency_offset,
                "source_on": self._is_on,
            }
            photon = Photon(packet, mode=mode, metadata=metadata)
            events.append(
                PhotonEvent(photon, packet.arrival_time, mode, shot=shot, metadata=metadata)
            )
        return events

    def __call__(self, shot: int, rng: np.random.Generator) -> list[PhotonEvent]:
        return self.emit(shot, rng)
