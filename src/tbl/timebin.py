"""TBL coherent time-bin transfer matrices for recirculating fiber loops."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np

from .circuit import LinearOpticalCircuit
from .errors import ValidationError
from .models import _probability

ScalarSchedule = float | Mapping[int, float] | Callable[[int, float], float]


@dataclass(frozen=True, slots=True)
class CoherentLoopResult:
    output_amplitudes: np.ndarray
    residual_loop_amplitudes: np.ndarray
    input_probability: float
    output_probability: float
    residual_loop_probability: float
    physical_loss_probability: float


@dataclass(frozen=True, slots=True)
class CoherentTimeBinLoop:
    """Expanded-mode coherent model of a dynamically coupled fiber loop.

    Each external time bin is one optical mode. At bin ``n``, a unitary
    beam-splitter couples the external input to the loop field returning from
    ``n-round_trip_bins``. The surviving loop field is delayed and re-injected,
    so amplitudes from different injections and round trips interfere before
    any measurement is sampled.

    The returned external transfer matrix is generally subunitary because
    round-trip/coupler loss and loop energy remaining after the final simulated
    bin belong to unobserved environment modes.
    """

    time_bins: int
    bin_width: float
    round_trip_bins: int = 1
    reflectivity: ScalarSchedule = 0.5
    round_trip_transmission: float = 1.0
    coupler_transmission: float = 1.0
    phase: ScalarSchedule = 0.0
    switching_time: float = 0.0
    phase_noise_std: float = 0.0
    phase_correlation_time: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.time_bins, bool)
            or isinstance(self.round_trip_bins, bool)
            or not isinstance(self.time_bins, (int, np.integer))
            or not isinstance(self.round_trip_bins, (int, np.integer))
            or self.time_bins < 1
            or self.round_trip_bins < 1
        ):
            raise ValidationError("time_bins and round_trip_bins must be positive integers")
        finite = (
            self.bin_width,
            self.round_trip_transmission,
            self.coupler_transmission,
            self.switching_time,
            self.phase_noise_std,
            self.phase_correlation_time,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValidationError("coherent-loop scalar parameters must be finite")
        if self.bin_width <= 0:
            raise ValidationError("bin_width must be positive")
        _probability("round_trip_transmission", self.round_trip_transmission)
        _probability("coupler_transmission", self.coupler_transmission)
        if self.switching_time < 0 or self.phase_noise_std < 0:
            raise ValidationError("switching time and phase noise must be non-negative")
        if self.phase_correlation_time < 0:
            raise ValidationError("phase_correlation_time cannot be negative")
        if isinstance(self.reflectivity, (int, float)):
            _probability("reflectivity", float(self.reflectivity))
        elif isinstance(self.reflectivity, Mapping):
            if any(
                isinstance(key, bool) or not isinstance(key, (int, np.integer))
                for key in self.reflectivity
            ):
                raise ValidationError("reflectivity schedule keys must be integer time bins")
            for value in self.reflectivity.values():
                _probability("scheduled reflectivity", value)
        if isinstance(self.phase, (int, float)) and not np.isfinite(self.phase):
            raise ValidationError("phase must be finite")
        if isinstance(self.phase, Mapping) and (
            any(
                isinstance(key, bool) or not isinstance(key, (int, np.integer))
                for key in self.phase
            )
            or not all(np.isfinite(value) for value in self.phase.values())
        ):
            raise ValidationError("phase schedule keys and values must be finite integer bins")

    @staticmethod
    def _piecewise(schedule: ScalarSchedule, index: int, time: float) -> float:
        if callable(schedule):
            return float(schedule(index, time))
        if isinstance(schedule, Mapping):
            if not schedule:
                return 0.0
            eligible = [change for change in schedule if change <= index]
            key = max(eligible) if eligible else min(schedule)
            return float(schedule[key])
        return float(schedule)

    def _reflectivity_trace(self) -> np.ndarray:
        values = np.empty(self.time_bins, dtype=float)
        for index in range(self.time_bins):
            time = index * self.bin_width
            target = self._piecewise(self.reflectivity, index, time)
            if self.switching_time and isinstance(self.reflectivity, Mapping):
                previous = target
                for change in sorted(self.reflectivity):
                    if change > index:
                        break
                    new = float(self.reflectivity[change])
                    elapsed = time - change * self.bin_width
                    if elapsed < self.switching_time:
                        fraction = elapsed / self.switching_time
                        target = previous + fraction * (new - previous)
                        break
                    previous = new
            values[index] = _probability("scheduled reflectivity", target)
        return values

    def phase_trace(self, *, seed: int | None = None) -> np.ndarray:
        """Return deterministic schedule plus a sampled correlated phase trace."""

        deterministic = np.asarray(
            [
                self._piecewise(self.phase, index, index * self.bin_width)
                for index in range(self.time_bins)
            ],
            dtype=float,
        )
        if self.phase_noise_std == 0:
            if not np.all(np.isfinite(deterministic)):
                raise ValidationError("phase schedule must be finite")
            return deterministic
        rng = np.random.default_rng(seed)
        noise = np.empty(self.time_bins, dtype=float)
        if self.phase_correlation_time == 0:
            noise[:] = rng.normal(0.0, self.phase_noise_std, self.time_bins)
        else:
            correlation = np.exp(-self.bin_width / self.phase_correlation_time)
            innovation = self.phase_noise_std * sqrt(1 - correlation**2)
            noise[0] = rng.normal(0.0, self.phase_noise_std)
            for index in range(1, self.time_bins):
                noise[index] = correlation * noise[index - 1] + rng.normal(0.0, innovation)
        trace = deterministic + noise
        if not np.all(np.isfinite(trace)):
            raise ValidationError("phase schedule must be finite")
        return trace

    def _propagate_with_traces(
        self,
        state: np.ndarray,
        reflectivities: np.ndarray,
        phases: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        loop_arrivals = np.zeros(self.time_bins + self.round_trip_bins, dtype=complex)
        output = np.zeros(self.time_bins, dtype=complex)
        external_attenuation = sqrt(self.coupler_transmission)
        loop_attenuation = sqrt(self.coupler_transmission * self.round_trip_transmission)
        for index in range(self.time_bins):
            reflection = sqrt(reflectivities[index])
            transmission = sqrt(1 - reflectivities[index])
            loop_input = loop_arrivals[index]
            output[index] = external_attenuation * (
                transmission * state[index] + 1j * reflection * loop_input
            )
            loop_output = loop_attenuation * (
                1j * reflection * state[index] + transmission * loop_input
            )
            loop_arrivals[index + self.round_trip_bins] += np.exp(1j * phases[index]) * loop_output
        return output, loop_arrivals[self.time_bins :]

    def transfer_matrix(self, *, seed: int | None = None) -> np.ndarray:
        """Build the external time-bin transfer matrix for one noise realization."""

        reflectivities = self._reflectivity_trace()
        phases = self.phase_trace(seed=seed)
        transfer = np.zeros((self.time_bins, self.time_bins), dtype=complex)
        loop_arrivals = np.zeros(
            (self.time_bins + self.round_trip_bins, self.time_bins), dtype=complex
        )
        external_attenuation = sqrt(self.coupler_transmission)
        loop_attenuation = sqrt(self.coupler_transmission * self.round_trip_transmission)
        for index in range(self.time_bins):
            reflection = sqrt(reflectivities[index])
            transmission = sqrt(1 - reflectivities[index])
            loop_input = loop_arrivals[index]
            transfer[index] = external_attenuation * 1j * reflection * loop_input
            transfer[index, index] += external_attenuation * transmission
            loop_output = loop_attenuation * transmission * loop_input
            loop_output[index] += loop_attenuation * 1j * reflection
            loop_arrivals[index + self.round_trip_bins] += np.exp(1j * phases[index]) * loop_output
        # Each recurrence step is a lossy two-port unitary, so passivity follows
        # analytically from the validated transmissions and reflectivities.
        # Avoiding an O(N^3) SVD keeps matrix construction itself O(N^2).
        return transfer

    def propagate(self, amplitudes: Sequence[complex], *, seed: int | None = None) -> np.ndarray:
        state = np.asarray(amplitudes, dtype=complex)
        if state.shape != (self.time_bins,) or not np.all(np.isfinite(state)):
            raise ValidationError("amplitudes must be finite with one value per time bin")
        return self.transfer_matrix(seed=seed) @ state

    def simulate(
        self, amplitudes: Sequence[complex], *, seed: int | None = None
    ) -> CoherentLoopResult:
        """Propagate one coherent state and separate tail energy from true loss."""

        state = np.asarray(amplitudes, dtype=complex)
        if state.shape != (self.time_bins,) or not np.all(np.isfinite(state)):
            raise ValidationError("amplitudes must be finite with one value per time bin")
        output, residual = self._propagate_with_traces(
            state, self._reflectivity_trace(), self.phase_trace(seed=seed)
        )
        input_probability = float(np.vdot(state, state).real)
        output_probability = float(np.vdot(output, output).real)
        residual_probability = float(np.vdot(residual, residual).real)
        physical_loss = input_probability - output_probability - residual_probability
        if physical_loss < -1e-9:
            raise ValidationError("coherent loop energy bookkeeping violated passivity")
        return CoherentLoopResult(
            output,
            residual,
            input_probability,
            output_probability,
            residual_probability,
            max(0.0, physical_loss),
        )

    def circuit(self, *, seed: int | None = None) -> LinearOpticalCircuit:
        """Return a circuit directly consumable by ``FockSimulator``."""

        return LinearOpticalCircuit.from_transfer_matrix(
            self.transfer_matrix(seed=seed), name="coherent_time_bin_loop"
        )
