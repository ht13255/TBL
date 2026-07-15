"""Physical data models used by both circuit and event simulations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import exp, pi, sqrt
from typing import Any

import numpy as np

from .errors import ValidationError

_C = 299_792_458.0


def _probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValidationError(f"{name} must be in [0, 1], got {value}")
    return value


@dataclass(frozen=True, slots=True)
class Wavepacket:
    """Transform-limited Gaussian single-photon wavepacket.

    Times are seconds, wavelength is metres, and ``temporal_width`` is the
    standard deviation of the complex field envelope. Polarization is a Jones
    vector in the H/V basis and is normalized on construction.
    """

    arrival_time: float = 0.0
    temporal_width: float = 20e-12
    wavelength: float = 1550e-9
    polarization: tuple[complex, complex] = (1.0 + 0j, 0.0 + 0j)
    purity: float = 1.0
    label: str | None = None

    def __post_init__(self) -> None:
        if self.temporal_width <= 0:
            raise ValidationError("temporal_width must be positive")
        if self.wavelength <= 0:
            raise ValidationError("wavelength must be positive")
        _probability("purity", self.purity)
        p = np.asarray(self.polarization, dtype=complex)
        if p.shape != (2,) or np.linalg.norm(p) == 0:
            raise ValidationError("polarization must be a non-zero two-element Jones vector")
        p = p / np.linalg.norm(p)
        object.__setattr__(self, "polarization", (complex(p[0]), complex(p[1])))

    @property
    def angular_frequency(self) -> float:
        return 2 * pi * _C / self.wavelength

    def shifted(self, delay: float) -> Wavepacket:
        return replace(self, arrival_time=self.arrival_time + float(delay))

    def overlap(self, other: Wavepacket) -> complex:
        """Return the complex mode-overlap amplitude with another Gaussian.

        The expression includes temporal width, delay, centre-frequency,
        polarization, and mixed-state purity. ``abs(overlap)**2`` is the mode
        indistinguishability used for two-photon interference.
        """

        s1, s2 = self.temporal_width, other.temporal_width
        denom = s1 * s1 + s2 * s2
        dt = other.arrival_time - self.arrival_time
        dw = other.angular_frequency - self.angular_frequency
        temporal = sqrt(2 * s1 * s2 / denom)
        temporal *= exp(-(dt * dt) / (4 * denom) - (dw * dw * s1 * s1 * s2 * s2) / denom)
        temporal *= np.exp(-0.5j * dw * (self.arrival_time + other.arrival_time))
        pol = np.vdot(np.asarray(self.polarization), np.asarray(other.polarization))
        purity = sqrt(self.purity * other.purity)
        return complex(temporal * pol * purity)

    def indistinguishability(self, other: Wavepacket) -> float:
        return float(np.clip(abs(self.overlap(other)) ** 2, 0.0, 1.0))

    def amplitude(self, time: np.ndarray | float) -> np.ndarray:
        """Evaluate the normalized temporal field amplitude."""

        t = np.asarray(time, dtype=float)
        norm = (2 * pi * self.temporal_width**2) ** (-0.25)
        envelope = np.exp(-((t - self.arrival_time) ** 2) / (4 * self.temporal_width**2))
        return norm * envelope * np.exp(-1j * self.angular_frequency * t)


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

    def copy(self, **changes: Any) -> PhotonEvent:
        values = {
            "photon": self.photon,
            "time": self.time,
            "mode": self.mode,
            "amplitude": self.amplitude,
            "shot": self.shot,
            "roundtrips": self.roundtrips,
            "metadata": dict(self.metadata),
        }
        values.update(changes)
        return PhotonEvent(**values)


@dataclass(frozen=True, slots=True)
class TimeBinQubit:
    """Single-photon qubit ``alpha|early> + beta|late>``."""

    alpha: complex = 1.0 + 0j
    beta: complex = 0.0 + 0j
    separation: float = 1e-9
    wavepacket: Wavepacket = field(default_factory=Wavepacket)

    def __post_init__(self) -> None:
        if self.separation <= 0:
            raise ValidationError("separation must be positive")
        norm = sqrt(abs(self.alpha) ** 2 + abs(self.beta) ** 2)
        if norm == 0:
            raise ValidationError("alpha and beta cannot both be zero")
        object.__setattr__(self, "alpha", self.alpha / norm)
        object.__setattr__(self, "beta", self.beta / norm)

    def events(self, *, mode: int = 0, shot: int = 0) -> list[PhotonEvent]:
        photon = Photon(self.wavepacket, mode=mode)
        return [
            PhotonEvent(photon, self.wavepacket.arrival_time, mode, self.alpha, shot, 0),
            PhotonEvent(
                photon, self.wavepacket.arrival_time + self.separation, mode, self.beta, shot, 0
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
        if self.repetition_rate <= 0:
            raise ValidationError("repetition_rate must be positive")
        _probability("p_single", self.p_single)
        _probability("p_double", self.p_double)
        if self.p_single + self.p_double > 1:
            raise ValidationError("p_single + p_double cannot exceed 1")
        if self.emission_jitter < 0:
            raise ValidationError("emission_jitter cannot be negative")

    @property
    def period(self) -> float:
        return 1.0 / self.repetition_rate

    def emit(self, shot: int, rng: np.random.Generator, *, mode: int = 0) -> list[PhotonEvent]:
        draw = rng.random()
        count = 2 if draw < self.p_double else 1 if draw < self.p_double + self.p_single else 0
        base_time = shot * self.period
        events: list[PhotonEvent] = []
        for index in range(count):
            jitter = rng.normal(0.0, self.emission_jitter) if self.emission_jitter else 0.0
            packet = self.wavepacket.shifted(base_time + jitter - self.wavepacket.arrival_time)
            photon = Photon(packet, mode=mode, metadata={"source_index": index})
            events.append(PhotonEvent(photon, packet.arrival_time, mode, shot=shot))
        return events
