import numpy as np
import pytest

import openphotontwin as opt


def test_sparameter_import_and_batch_cpu_backend():
    scale = 1 / np.sqrt(2)
    circuit = opt.from_sparameters(
        {
            ("o1", "o1"): scale,
            ("o2", "o1"): 1j * scale,
            ("o1", "o2"): 1j * scale,
            ("o2", "o2"): scale,
        },
        ports=["o1", "o2"],
    )
    states = np.array([[1, 0], [0, 1]], dtype=complex)
    output = opt.propagate_batch(circuit.transfer_matrix(), states, backend="numpy")
    assert output.shape == (2, 2)
    assert np.sum(abs(output) ** 2, axis=1) == pytest.approx([1, 1])


def test_sax_callable_and_nonpassive_validation():
    def model(coupling=0.5):
        return {("a", "a"): coupling, ("b", "a"): np.sqrt(1 - coupling**2)}

    circuit = opt.from_sax_model(model, settings={"coupling": 0.6})
    assert circuit.modes == 2
    with pytest.raises(opt.ValidationError):
        opt.from_sparameters({("a", "a"): 2})


def test_perceval_duck_typed_import():
    class FakePerceval:
        def compute_unitary(self, use_symbolic=False):
            return np.eye(2)

    circuit = opt.from_perceval(FakePerceval())
    assert np.allclose(circuit.transfer_matrix(), np.eye(2))


def test_strawberry_fields_duck_typed_import():
    class Reg:
        def __init__(self, index):
            self.ind = index

    class BSgate:
        def __init__(self):
            self.p = [np.pi / 4, 0.0]

    class Rgate:
        def __init__(self):
            self.p = [0.2]

    class Command:
        def __init__(self, op, regs):
            self.op = op
            self.reg = [Reg(index) for index in regs]

    class Program:
        num_subsystems = 2
        circuit = [Command(BSgate(), [0, 1]), Command(Rgate(), [0])]

    circuit = opt.from_strawberry_fields(Program())
    assert circuit.transfer_matrix().shape == (2, 2)


def test_missing_gpu_dependency_has_clear_error_or_gpu_backend():
    try:
        backend = opt.get_backend("cupy")
    except opt.OptionalDependencyError as exc:
        assert "CuPy" in str(exc)
    else:
        assert backend.is_gpu
