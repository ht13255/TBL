"""Exact small-Fock-space linear-optical circuit simulation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import permutations
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


@dataclass(frozen=True, slots=True)
class SpectralMatrixComponent:
    """Measured passive transfer matrices sampled over wavelength.

    Complex matrix elements are linearly interpolated. The sampled wavelength
    grid is in metres. Extrapolation is disabled by default because measured
    S-parameters do not justify behavior outside their calibration band.
    """

    wavelengths: np.ndarray
    matrices: np.ndarray
    name: str = "spectral_matrix"
    extrapolate: bool = False

    def __post_init__(self) -> None:
        wavelengths = np.asarray(self.wavelengths, dtype=float)
        matrices = np.asarray(self.matrices, dtype=complex)
        if wavelengths.ndim != 1 or wavelengths.size < 2:
            raise ValidationError("spectral component needs at least two wavelengths")
        if np.any(wavelengths <= 0) or np.any(np.diff(wavelengths) <= 0):
            raise ValidationError("wavelength grid must be positive and strictly increasing")
        if (
            matrices.ndim != 3
            or matrices.shape[0] != wavelengths.size
            or matrices.shape[1] != matrices.shape[2]
        ):
            raise ValidationError("spectral matrices must have shape (wavelength, mode, mode)")
        singular_values = np.linalg.svd(matrices, compute_uv=False)
        if singular_values.size and singular_values.max() > 1 + 1e-9:
            raise ValidationError("all spectral transfer matrices must be passive")
        object.__setattr__(self, "wavelengths", wavelengths)
        object.__setattr__(self, "matrices", matrices)

    @property
    def modes(self) -> int:
        return int(self.matrices.shape[1])

    @property
    def reference_wavelength(self) -> float:
        return float(self.wavelengths[len(self.wavelengths) // 2])

    def matrix_at(self, wavelength: float) -> np.ndarray:
        if wavelength <= 0:
            raise ValidationError("wavelength must be positive")
        below = wavelength < self.wavelengths[0]
        above = wavelength > self.wavelengths[-1]
        if (below or above) and not self.extrapolate:
            raise ValidationError(
                f"wavelength {wavelength:.9g} m is outside calibrated range "
                f"[{self.wavelengths[0]:.9g}, {self.wavelengths[-1]:.9g}] m"
            )
        if below:
            left, right = 0, 1
        elif above:
            left, right = len(self.wavelengths) - 2, len(self.wavelengths) - 1
        else:
            right = int(np.searchsorted(self.wavelengths, wavelength, side="right"))
            right = min(max(right, 1), len(self.wavelengths) - 1)
            left = right - 1
        fraction = (wavelength - self.wavelengths[left]) / (
            self.wavelengths[right] - self.wavelengths[left]
        )
        matrix = (1 - fraction) * self.matrices[left] + fraction * self.matrices[right]
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular.size and singular.max() > 1 + 1e-8:
            raise SimulationError("spectral extrapolation produced an active transfer matrix")
        return matrix


LinearComponent = (
    BeamSplitter
    | DynamicBeamSplitter
    | PhaseShifter
    | LossChannel
    | MatrixComponent
    | SpectralMatrixComponent
)


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
        if isinstance(component, SpectralMatrixComponent) and component.modes != self.modes:
            raise ValidationError("spectral component size must equal circuit mode count")
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

    @classmethod
    def from_spectral_transfer(
        cls,
        wavelengths: Sequence[float],
        matrices: np.ndarray,
        *,
        name: str = "spectral_import",
        extrapolate: bool = False,
    ) -> LinearOpticalCircuit:
        component = SpectralMatrixComponent(
            np.asarray(wavelengths), np.asarray(matrices), name, extrapolate
        )
        return cls(component.modes).add(component)

    @property
    def has_spectral_response(self) -> bool:
        return any(isinstance(item, SpectralMatrixComponent) for item in self.components)

    def transfer_matrix(
        self,
        *,
        time_bin: int = 0,
        time: float = 0.0,
        wavelength: float | None = None,
    ) -> np.ndarray:
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
            elif isinstance(component, SpectralMatrixComponent):
                selected_wavelength = (
                    component.reference_wavelength if wavelength is None else wavelength
                )
                local = component.matrix_at(selected_wavelength)
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

    General partial distinguishability is evaluated from the full Hermitian
    internal-mode overlap matrix using the permutation/permanent formula. This
    retains all pairwise overlaps and multiphoton phases. ``partial_method``
    can explicitly request the older mean-overlap approximation for large
    exploratory calculations.
    """

    circuit: LinearOpticalCircuit
    max_photons: int = 10
    max_exact_partial_photons: int = 7

    def __post_init__(self) -> None:
        if self.max_photons < 0 or self.max_exact_partial_photons < 0:
            raise ValidationError("photon limits must be non-negative")

    @staticmethod
    def _validate_overlap_matrix(matrix: np.ndarray, photon_count: int) -> np.ndarray:
        overlap = np.asarray(matrix, dtype=complex)
        if overlap.shape != (photon_count, photon_count):
            raise ValidationError("overlap_matrix must have one row and column per photon")
        if not np.allclose(overlap, overlap.conj().T, atol=1e-10):
            raise ValidationError("overlap_matrix must be Hermitian")
        if not np.allclose(np.diag(overlap), 1.0, atol=1e-10):
            raise ValidationError("overlap_matrix diagonal must be one")
        eigenvalues = np.linalg.eigvalsh(overlap)
        if eigenvalues.min(initial=0.0) < -1e-9:
            raise ValidationError("overlap_matrix must be positive semidefinite")
        if np.max(abs(overlap), initial=0.0) > 1 + 1e-9:
            raise ValidationError("internal-mode overlap magnitude cannot exceed one")
        return overlap

    def _internal_overlap_matrix(
        self,
        photon_count: int,
        wavepackets: Sequence[Wavepacket] | None,
        indistinguishability: float | None,
        overlap_matrix: np.ndarray | None,
    ) -> np.ndarray:
        supplied = sum(
            value is not None for value in (wavepackets, indistinguishability, overlap_matrix)
        )
        if supplied > 1:
            raise ValidationError(
                "supply only one of wavepackets, indistinguishability, or overlap_matrix"
            )
        if overlap_matrix is not None:
            return self._validate_overlap_matrix(overlap_matrix, photon_count)
        if wavepackets is not None:
            if len(wavepackets) != photon_count:
                raise ValidationError("one wavepacket is required for every input photon")
            matrix = np.eye(photon_count, dtype=complex)
            for first in range(photon_count):
                for second in range(first + 1, photon_count):
                    value = wavepackets[first].overlap(wavepackets[second])
                    matrix[first, second] = value
                    matrix[second, first] = np.conj(value)
            return self._validate_overlap_matrix(matrix, photon_count)
        if indistinguishability is not None:
            value = float(indistinguishability)
            if not 0 <= value <= 1:
                raise ValidationError("indistinguishability must be in [0, 1]")
            amplitude = sqrt(value)
            matrix = np.full((photon_count, photon_count), amplitude, dtype=complex)
            np.fill_diagonal(matrix, 1.0)
            return matrix
        return np.ones((photon_count, photon_count), dtype=complex)

    @staticmethod
    def _exact_partial_probability(
        submatrix: np.ndarray,
        internal_overlap: np.ndarray,
        input_indices: Sequence[int],
        output_factor: int,
    ) -> float:
        photon_count = len(input_indices)
        same_spatial_input = np.equal.outer(input_indices, input_indices)
        input_gram = internal_overlap * same_spatial_input
        input_normalization = permanent(input_gram)
        if abs(input_normalization.imag) > 1e-8 or input_normalization.real <= 0:
            raise SimulationError("input-state Gram normalization is not physical")
        total = 0.0 + 0.0j
        for permutation in permutations(range(photon_count)):
            permutation_array = np.asarray(permutation)
            internal_weight = np.prod(internal_overlap[np.arange(photon_count), permutation_array])
            interference_matrix = submatrix * np.conj(submatrix[:, permutation_array])
            total += internal_weight * permanent(interference_matrix)
        probability = total / (output_factor * input_normalization.real)
        if abs(probability.imag) > 1e-8:
            raise SimulationError("partial-distinguishability probability became complex")
        if probability.real < -1e-8:
            raise SimulationError("partial-distinguishability probability became negative")
        return max(0.0, float(probability.real))

    def probabilities(
        self,
        input_occupation: Sequence[int],
        *,
        wavepackets: Sequence[Wavepacket] | None = None,
        indistinguishability: float | None = None,
        overlap_matrix: np.ndarray | None = None,
        wavelengths: Sequence[float] | None = None,
        partial_method: str = "exact",
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
        if partial_method not in {"exact", "mean"}:
            raise ValidationError("partial_method must be 'exact' or 'mean'")
        internal_overlap = self._internal_overlap_matrix(
            photon_count, wavepackets, indistinguishability, overlap_matrix
        )
        fully_indistinguishable = np.allclose(internal_overlap, 1.0, atol=1e-12)
        fully_distinguishable = np.allclose(internal_overlap, np.eye(photon_count), atol=1e-12)
        if (
            partial_method == "exact"
            and not fully_indistinguishable
            and not fully_distinguishable
            and photon_count > self.max_exact_partial_photons
        ):
            raise SimulationError(
                "exact partial distinguishability scales factorially; "
                f"{photon_count} photons exceed limit {self.max_exact_partial_photons}. "
                "Increase max_exact_partial_photons or explicitly use partial_method='mean'."
            )
        pair_indistinguishability = 1.0
        if photon_count > 1:
            pair_indistinguishability = float(
                np.mean(
                    [
                        abs(internal_overlap[i, j]) ** 2
                        for i in range(photon_count)
                        for j in range(i + 1, photon_count)
                    ]
                )
            )
        if wavelengths is not None:
            photon_wavelengths = [float(value) for value in wavelengths]
            if len(photon_wavelengths) != photon_count:
                raise ValidationError("wavelengths must contain one value per input photon")
        elif wavepackets is not None:
            photon_wavelengths = [packet.wavelength for packet in wavepackets]
        else:
            photon_wavelengths = [None] * photon_count
        if self.circuit.has_spectral_response:
            photon_transfers = [
                self.circuit.transfer_matrix(
                    time_bin=time_bin, time=time, wavelength=photon_wavelength
                )
                for photon_wavelength in photon_wavelengths
            ]
        else:
            common_transfer = self.circuit.transfer_matrix(time_bin=time_bin, time=time)
            photon_transfers = [common_transfer] * photon_count
        input_indices = _repeated_indices(occupation)
        input_factor = prod(factorial(value) for value in occupation)
        probabilities: dict[tuple[int, ...], float] = {}
        for output in _occupations(photon_count, self.circuit.modes):
            output_indices = _repeated_indices(output)
            if photon_count:
                submatrix = np.column_stack(
                    [
                        photon_transfers[photon][output_indices, input_indices[photon]]
                        for photon in range(photon_count)
                    ]
                )
            else:
                submatrix = np.empty((0, 0), dtype=complex)
            output_factor = prod(factorial(value) for value in output)
            bosonic_amplitude = permanent(submatrix) / sqrt(input_factor * output_factor)
            bosonic_probability = abs(bosonic_amplitude) ** 2
            classical_probability = permanent(abs(submatrix) ** 2).real / output_factor
            if fully_indistinguishable:
                value = bosonic_probability
            elif fully_distinguishable:
                value = classical_probability
            elif partial_method == "mean":
                value = (
                    pair_indistinguishability * bosonic_probability
                    + (1 - pair_indistinguishability) * classical_probability
                )
            else:
                value = self._exact_partial_probability(
                    submatrix, internal_overlap, input_indices, output_factor
                )
            probabilities[output] = max(0.0, float(value))
        survival = sum(probabilities.values())
        if survival > 1 + 1e-7:
            raise SimulationError(
                f"output probabilities sum to {survival:.12g}; physical normalization failed"
            )
        if survival > 1:
            probabilities = {key: value / survival for key, value in probabilities.items()}
            survival = 1.0
        return FockDistribution(probabilities, max(0.0, 1.0 - survival))
