"""Exact small-Fock-space linear-optical circuit simulation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from math import factorial, prod, sqrt

import numpy as np

from .components import BeamSplitter, DynamicBeamSplitter, LossChannel, PhaseShifter
from .errors import SimulationError, ValidationError
from .models import Wavepacket


def permanent(matrix: np.ndarray) -> complex:
    """Compute a matrix permanent using Ryser's formula.

    This implementation is intended for exact, small photon-number problems.
    Runtime scales as :math:`O(n 2^n)`.
    """

    a = np.asarray(matrix)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValidationError("permanent requires a square matrix")
    n = a.shape[0]
    if n == 0:
        return 1.0 + 0j
    total: complex = 0.0 + 0j
    for mask in range(1, 1 << n):
        bits = mask.bit_count()
        row_sums = np.zeros(n, dtype=np.result_type(a.dtype, complex))
        for column in range(n):
            if mask & (1 << column):
                row_sums += a[:, column]
        total += (-1) ** (n - bits) * np.prod(row_sums)
    return complex(total)


def _occupations(total: int, modes: int) -> Iterator[tuple[int, ...]]:
    if modes == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _occupations(total - first, modes - 1):
            yield (first, *rest)


def _repeated_indices(occupation: Sequence[int]) -> list[int]:
    return [index for index, count in enumerate(occupation) for _ in range(int(count))]


@dataclass(frozen=True, slots=True)
class MatrixComponent:
    """Arbitrary linear transfer matrix, including measured S-parameters."""

    matrix: np.ndarray
    name: str = "matrix"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=complex)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValidationError("component transfer matrix must be square")
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular.size and singular.max() > 1 + 1e-9:
            raise ValidationError("passive transfer matrix cannot have singular value above one")
        object.__setattr__(self, "matrix", matrix)


LinearComponent = BeamSplitter | DynamicBeamSplitter | PhaseShifter | LossChannel | MatrixComponent


@dataclass(slots=True)
class LinearOpticalCircuit:
    """Ordered passive linear-optical network."""

    modes: int
    components: list[LinearComponent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.modes < 1:
            raise ValidationError("a circuit needs at least one mode")

    def add(self, component: LinearComponent) -> LinearOpticalCircuit:
        if (
            isinstance(component, (BeamSplitter, DynamicBeamSplitter))
            and max(component.mode_a, component.mode_b) >= self.modes
        ):
            raise ValidationError("component addresses a mode outside this circuit")
        if isinstance(component, MatrixComponent) and component.matrix.shape != (
            self.modes,
            self.modes,
        ):
            raise ValidationError("matrix component size must equal circuit mode count")
        self.components.append(component)
        return self

    def beam_splitter(
        self, mode_a: int, mode_b: int, reflectivity: float = 0.5, phase: float = 0.0
    ) -> LinearOpticalCircuit:
        return self.add(BeamSplitter(mode_a, mode_b, reflectivity, phase))

    def phase(self, mode: int, phase: float) -> LinearOpticalCircuit:
        return self.add(PhaseShifter(phase, frozenset({mode})))

    def loss(self, transmission: float, modes: Sequence[int] | None = None) -> LinearOpticalCircuit:
        selected = None if modes is None else frozenset(modes)
        return self.add(LossChannel(transmission, selected))

    @classmethod
    def from_transfer_matrix(
        cls, matrix: np.ndarray, name: str = "imported"
    ) -> LinearOpticalCircuit:
        matrix = np.asarray(matrix, dtype=complex)
        circuit = cls(matrix.shape[0])
        return circuit.add(MatrixComponent(matrix, name))

    def transfer_matrix(self, *, time_bin: int = 0, time: float = 0.0) -> np.ndarray:
        transfer = np.eye(self.modes, dtype=complex)
        for component in self.components:
            if isinstance(component, DynamicBeamSplitter):
                reflectivity = component._scheduled(time_bin, time)
                local = component.unitary(self.modes, reflectivity)
            elif isinstance(component, BeamSplitter):
                local = component.unitary(self.modes)
            elif isinstance(component, PhaseShifter):
                local = np.eye(self.modes, dtype=complex)
                selected = range(self.modes) if component.modes is None else component.modes
                for mode in selected:
                    if mode >= self.modes:
                        raise ValidationError("phase shifter addresses a mode outside the circuit")
                    local[mode, mode] *= np.exp(1j * component.phase)
            elif isinstance(component, LossChannel):
                local = np.eye(self.modes, dtype=complex)
                selected = range(self.modes) if component.modes is None else component.modes
                for mode in selected:
                    if mode >= self.modes:
                        raise ValidationError("loss channel addresses a mode outside the circuit")
                    local[mode, mode] *= sqrt(component.transmission)
            else:
                local = component.matrix
            transfer = local @ transfer
        return transfer

    def propagate(self, amplitudes: Sequence[complex], **timing: float | int) -> np.ndarray:
        state = np.asarray(amplitudes, dtype=complex)
        if state.shape != (self.modes,):
            raise ValidationError("amplitudes must contain one value per circuit mode")
        return self.transfer_matrix(**timing) @ state


@dataclass(frozen=True, slots=True)
class FockDistribution:
    """Output occupation probabilities plus probability lost to the environment."""

    probabilities: dict[tuple[int, ...], float]
    loss_probability: float

    @property
    def survival_probability(self) -> float:
        return float(sum(self.probabilities.values()))

    def normalized(self) -> dict[tuple[int, ...], float]:
        survival = self.survival_probability
        if survival == 0:
            return {key: 0.0 for key in self.probabilities}
        return {key: value / survival for key, value in self.probabilities.items()}

    def sample(
        self, shots: int, *, seed: int | None = None, include_loss: bool = True
    ) -> dict[tuple[int, ...] | str, int]:
        if shots < 0:
            raise ValidationError("shots cannot be negative")
        labels: list[tuple[int, ...] | str] = list(self.probabilities)
        weights = list(self.probabilities.values())
        if include_loss:
            labels.append("loss")
            weights.append(self.loss_probability)
        total = sum(weights)
        if total <= 0:
            raise SimulationError("distribution has no probability mass")
        weights = np.asarray(weights, dtype=float) / total
        counts = np.random.default_rng(seed).multinomial(shots, weights)
        return {label: int(count) for label, count in zip(labels, counts, strict=True)}


@dataclass(slots=True)
class FockSimulator:
    """Exact bosonic output probabilities for small passive circuits.

    Partial distinguishability is represented by interpolation between exact
    indistinguishable-boson and fully distinguishable probabilities. For two
    photons this is exact; for larger states it is a documented mean-overlap
    approximation.
    """

    circuit: LinearOpticalCircuit
    max_photons: int = 10

    def _mean_overlap(self, packets: Sequence[Wavepacket] | None, photon_count: int) -> float:
        if packets is None:
            return 1.0
        if len(packets) != photon_count:
            raise ValidationError("one wavepacket is required for every input photon")
        if photon_count < 2:
            return 1.0
        overlaps = [
            packets[i].indistinguishability(packets[j])
            for i in range(photon_count)
            for j in range(i + 1, photon_count)
        ]
        return float(np.mean(overlaps))

    def probabilities(
        self,
        input_occupation: Sequence[int],
        *,
        wavepackets: Sequence[Wavepacket] | None = None,
        indistinguishability: float | None = None,
        time_bin: int = 0,
        time: float = 0.0,
    ) -> FockDistribution:
        occupation = tuple(int(value) for value in input_occupation)
        if len(occupation) != self.circuit.modes or any(value < 0 for value in occupation):
            raise ValidationError("input occupation must be non-negative and match circuit modes")
        photon_count = sum(occupation)
        if photon_count > self.max_photons:
            raise SimulationError(
                f"{photon_count} photons exceed configured exact limit {self.max_photons}"
            )
        if indistinguishability is None:
            overlap = self._mean_overlap(wavepackets, photon_count)
        else:
            overlap = float(indistinguishability)
            if not 0 <= overlap <= 1:
                raise ValidationError("indistinguishability must be in [0, 1]")
        transfer = self.circuit.transfer_matrix(time_bin=time_bin, time=time)
        input_indices = _repeated_indices(occupation)
        input_factor = prod(factorial(value) for value in occupation)
        probabilities: dict[tuple[int, ...], float] = {}
        for output in _occupations(photon_count, self.circuit.modes):
            output_indices = _repeated_indices(output)
            submatrix = transfer[np.ix_(output_indices, input_indices)]
            output_factor = prod(factorial(value) for value in output)
            bosonic_amplitude = permanent(submatrix) / sqrt(input_factor * output_factor)
            bosonic_probability = abs(bosonic_amplitude) ** 2
            classical_probability = permanent(abs(submatrix) ** 2).real / output_factor
            value = overlap * bosonic_probability + (1 - overlap) * classical_probability
            probabilities[output] = max(0.0, float(value))
        survival = sum(probabilities.values())
        if survival > 1 + 1e-8:
            # Numerical error can make passive, high-order calculations slightly exceed one.
            probabilities = {key: value / survival for key, value in probabilities.items()}
            survival = 1.0
        return FockDistribution(probabilities, max(0.0, 1.0 - survival))
