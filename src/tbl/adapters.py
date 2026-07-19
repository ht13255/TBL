"""TBL import adapters for external photonic design frameworks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .circuit import LinearOpticalCircuit
from .components import BeamSplitter, LossChannel, PhaseShifter
from .errors import OptionalDependencyError, ValidationError


def _numeric(value: Any) -> float:
    for attribute in ("x", "value"):
        if hasattr(value, attribute):
            value = getattr(value, attribute)
            break
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"cannot resolve symbolic parameter {value!r}") from exc


def from_perceval(circuit: Any) -> LinearOpticalCircuit:
    """Import a Perceval circuit through its public unitary computation API."""

    if not hasattr(circuit, "compute_unitary"):
        raise ValidationError("object does not look like a Perceval circuit")
    try:
        matrix = circuit.compute_unitary(use_symbolic=False)
    except TypeError:
        matrix = circuit.compute_unitary()
    if hasattr(matrix, "to_numpy"):
        matrix = matrix.to_numpy()
    try:
        return LinearOpticalCircuit.from_transfer_matrix(
            np.asarray(matrix, dtype=complex), "perceval"
        )
    except (TypeError, ValueError) as exc:
        raise OptionalDependencyError("could not convert the Perceval unitary to NumPy") from exc


def from_strawberry_fields(program: Any) -> LinearOpticalCircuit:
    """Import common passive operations from a Strawberry Fields ``Program``."""

    commands = getattr(program, "circuit", None)
    modes = getattr(program, "num_subsystems", None)
    if commands is None or modes is None:
        raise ValidationError("object does not look like a Strawberry Fields Program")
    circuit = LinearOpticalCircuit(int(modes))
    for command in commands:
        operation = command.op
        name = operation.__class__.__name__.lower()
        registers = [int(reg.ind) for reg in command.reg]
        parameters = list(getattr(operation, "p", []))
        if name == "bsgate":
            theta = _numeric(parameters[0])
            phase = _numeric(parameters[1]) if len(parameters) > 1 else 0.0
            circuit.add(BeamSplitter(registers[0], registers[1], np.sin(theta) ** 2, phase))
        elif name in {"rgate", "phaseshift"}:
            circuit.add(PhaseShifter(_numeric(parameters[0]), frozenset({registers[0]})))
        elif name == "losschannel":
            circuit.add(LossChannel(_numeric(parameters[0]), frozenset({registers[0]})))
        else:
            raise ValidationError(
                f"unsupported Strawberry Fields operation: {operation.__class__.__name__}"
            )
    return circuit


def from_sparameters(
    sparameters: Mapping[tuple[str, str], complex | Sequence[complex] | np.ndarray],
    *,
    ports: Sequence[str] | None = None,
    frequency_index: int = 0,
    normalize_passive: bool = False,
    name: str = "sparameters",
) -> LinearOpticalCircuit:
    """Convert a GDSFactory/SAX-style S-parameter mapping into a circuit."""

    if not sparameters:
        raise ValidationError("S-parameter mapping cannot be empty")
    if ports is None:
        ports = sorted({port for pair in sparameters for port in pair})
    indices = {port: index for index, port in enumerate(ports)}
    matrix = np.zeros((len(ports), len(ports)), dtype=complex)
    for (output, input_), value in sparameters.items():
        if output not in indices or input_ not in indices:
            continue
        array = np.asarray(value)
        selected = array.item() if array.ndim == 0 else array[frequency_index]
        matrix[indices[output], indices[input_]] = complex(selected)
    largest = float(np.max(np.linalg.svd(matrix, compute_uv=False))) if matrix.size else 0.0
    if largest > 1 + 1e-9:
        if not normalize_passive:
            raise ValidationError(
                "S-parameters imply gain; pass normalize_passive=True "
                "to project onto a passive matrix"
            )
        matrix /= largest
    return LinearOpticalCircuit.from_transfer_matrix(matrix, name)


def from_sparameter_sweep(
    wavelengths: Sequence[float],
    sparameters: Mapping[tuple[str, str], complex | Sequence[complex] | np.ndarray],
    *,
    ports: Sequence[str] | None = None,
    normalize_passive: bool = False,
    extrapolate: bool = False,
    name: str = "sparameter_sweep",
) -> LinearOpticalCircuit:
    """Import wavelength-resolved GDSFactory/SAX S-parameters.

    Every non-scalar S-parameter value must have one sample per wavelength.
    Scalars are broadcast, which is useful for ports known to be flat over the
    measured band.
    """

    wavelength_array = np.asarray(wavelengths, dtype=float)
    if wavelength_array.ndim != 1 or wavelength_array.size < 2:
        raise ValidationError("wavelength sweep needs at least two samples")
    if not sparameters:
        raise ValidationError("S-parameter mapping cannot be empty")
    if ports is None:
        ports = sorted({port for pair in sparameters for port in pair})
    indices = {port: index for index, port in enumerate(ports)}
    matrices = np.zeros((wavelength_array.size, len(ports), len(ports)), dtype=complex)
    for (output, input_), value in sparameters.items():
        if output not in indices or input_ not in indices:
            continue
        array = np.asarray(value, dtype=complex)
        if array.ndim == 0:
            array = np.full(wavelength_array.size, array.item(), dtype=complex)
        if array.shape != wavelength_array.shape:
            raise ValidationError(
                f"S-parameter {(output, input_)!r} does not match wavelength grid"
            )
        matrices[:, indices[output], indices[input_]] = array
    for index, matrix in enumerate(matrices):
        largest = float(np.max(np.linalg.svd(matrix, compute_uv=False)))
        if largest > 1 + 1e-9:
            if not normalize_passive:
                raise ValidationError(f"S-parameters imply gain at wavelength index {index}")
            matrices[index] /= largest
    return LinearOpticalCircuit.from_spectral_transfer(
        wavelength_array.tolist(), matrices, name=name, extrapolate=extrapolate
    )


def from_sax_model(
    model: Callable[..., Mapping[tuple[str, str], complex | Sequence[complex]]],
    *,
    settings: Mapping[str, Any] | None = None,
    **conversion: Any,
) -> LinearOpticalCircuit:
    """Evaluate a SAX-compatible model and import its S-parameters."""

    if not callable(model):
        raise ValidationError("SAX model must be callable")
    sparameters = model(**dict(settings or {}))
    return from_sparameters(sparameters, name=getattr(model, "__name__", "sax"), **conversion)
