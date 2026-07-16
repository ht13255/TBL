import numpy as np
import pytest

import tbl as opt
import tbl._runtime as runtime_module
import tbl.backends as backend_module


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


def test_sparameter_sweep_preserves_wavelength_dependence():
    wavelengths = np.array([1540e-9, 1560e-9])
    circuit = opt.from_sparameter_sweep(
        wavelengths,
        {
            ("a", "a"): [1, 0],
            ("b", "a"): [0, 1],
            ("a", "b"): [0, 1],
            ("b", "b"): [1, 0],
        },
        ports=["a", "b"],
    )
    assert circuit.transfer_matrix(wavelength=1540e-9) == pytest.approx(np.eye(2))
    assert circuit.transfer_matrix(wavelength=1560e-9) == pytest.approx(np.array([[0, 1], [1, 0]]))


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


def test_auto_backend_falls_back_when_cupy_runtime_is_unusable(monkeypatch):
    monkeypatch.setattr(backend_module, "_cupy_status", lambda: (None, "mock runtime failure"))
    assert backend_module.get_backend("auto").name == "numpy"
    with pytest.raises(opt.OptionalDependencyError, match="mock runtime failure"):
        backend_module.get_backend("cupy")


def test_cupy_cache_uses_temp_fallback_when_home_is_read_only(monkeypatch, tmp_path):
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    original_mkdir = runtime_module.Path.mkdir

    def selective_mkdir(path, *args, **kwargs):
        if ".cupy" in path.parts:
            raise PermissionError("mock read-only home")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(runtime_module.Path, "mkdir", selective_mkdir)
    monkeypatch.setattr(runtime_module.tempfile, "gettempdir", lambda: str(tmp_path))
    backend_module._ensure_cupy_cache_dir()
    expected = tmp_path / "tbl-cupy-cache"
    assert runtime_module.os.environ["CUPY_CACHE_DIR"] == str(expected)
    assert expected.is_dir()


def test_batch_propagator_caches_device_matrix_and_can_keep_device_results(monkeypatch):
    class FakeStream:
        synchronizations = 0

        def synchronize(self):
            self.synchronizations += 1

    stream = FakeStream()

    class FakeCuda:
        @staticmethod
        def get_current_stream():
            return stream

    class FakeCupy:
        cuda = FakeCuda()
        asarray_calls = 0
        asnumpy_calls = 0

        @classmethod
        def asarray(cls, value):
            cls.asarray_calls += 1
            return np.asarray(value)

        @classmethod
        def asnumpy(cls, value):
            cls.asnumpy_calls += 1
            return np.asarray(value)

    fake_gpu = opt.ArrayBackend("cupy", FakeCupy, True)

    def fake_get_backend(name="auto"):
        return opt.ArrayBackend("numpy", np, False) if name in {"numpy", "cpu"} else fake_gpu

    monkeypatch.setattr(backend_module, "get_backend", fake_get_backend)
    matrix = np.array([[1, 2], [3, 4]], dtype=np.float32)
    propagator = opt.BatchPropagator(matrix, backend="cupy")
    host = propagator(np.ones((3, 2), dtype=np.float32))
    device = propagator(np.ones((2, 2), dtype=np.float32), return_device=True)
    assert host.dtype == np.float32
    assert device.dtype == np.float32
    assert FakeCupy.asarray_calls == 3  # two state uploads, one cached matrix upload
    assert FakeCupy.asnumpy_calls == 1
    propagator.synchronize()
    assert stream.synchronizations == 1


def test_cpu_batch_propagation_preserves_supported_dtype_and_small_auto_stays_cpu():
    matrix = np.eye(3, dtype=np.float32)
    states = np.ones((4, 3), dtype=np.float32)
    propagator = opt.BatchPropagator(matrix, backend="auto", gpu_min_operations=10_000)
    result = propagator(states)
    assert result.dtype == np.float32
    assert propagator.backend.name == "numpy"


def test_auto_backend_threshold_is_transfer_and_precision_aware(monkeypatch):
    class FakeStream:
        def synchronize(self):
            pass

    class FakeCuda:
        @staticmethod
        def get_current_stream():
            return FakeStream()

    class FakeCupy:
        cuda = FakeCuda()
        asarray = staticmethod(np.asarray)
        asnumpy = staticmethod(np.asarray)

    gpu = opt.ArrayBackend("cupy", FakeCupy, True)

    def fake_get_backend(name="auto"):
        return opt.ArrayBackend("numpy", np, False) if name in {"numpy", "cpu"} else gpu

    monkeypatch.setattr(backend_module, "get_backend", fake_get_backend)
    matrix64 = np.eye(128, dtype=np.complex64)
    small64 = np.ones((256, 128), dtype=np.complex64)
    large64 = np.ones((1024, 128), dtype=np.complex64)

    small = opt.BatchPropagator(matrix64, backend="auto")
    small(small64)
    assert small.backend.name == "numpy"
    small(small64, return_device=True)
    assert small.backend.name == "cupy"

    large = opt.BatchPropagator(matrix64, backend="auto")
    large(large64)
    assert large.backend.name == "cupy"

    matrix128 = np.eye(128, dtype=np.complex128)
    double_precision = opt.BatchPropagator(matrix128, backend="auto")
    double_precision(large64.astype(np.complex128))
    assert double_precision.backend.name == "numpy"


def test_real_gpu_batch_matches_cpu_when_available():
    try:
        opt.get_backend("cupy")
    except opt.OptionalDependencyError:
        pytest.skip("a usable CUDA runtime is not available")
    rng = np.random.default_rng(816)
    matrix = (
        rng.normal(size=(128, 128)).astype(np.float32)
        + 1j * rng.normal(size=(128, 128)).astype(np.float32)
    ).astype(np.complex64)
    states = (
        rng.normal(size=(1024, 128)).astype(np.float32)
        + 1j * rng.normal(size=(1024, 128)).astype(np.float32)
    ).astype(np.complex64)
    expected = opt.propagate_batch(matrix, states, backend="numpy")
    propagator = opt.BatchPropagator(matrix, backend="auto")
    measured = propagator(states)
    assert propagator.backend.name == "cupy"
    assert measured == pytest.approx(expected, rel=2e-4, abs=2e-4)
