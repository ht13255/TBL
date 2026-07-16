import importlib
import sys

import pytest

import tbl


def test_tbl_is_canonical_namespace_and_legacy_import_remains_compatible():
    sys.modules.pop("openphotontwin", None)
    with pytest.warns(DeprecationWarning, match="now named 'tbl'"):
        legacy = importlib.import_module("openphotontwin")
    legacy_circuit = importlib.import_module("openphotontwin.circuit")
    assert tbl.__version__ == "2.0.0"
    assert legacy.__version__ == tbl.__version__
    assert legacy.Wavepacket is tbl.Wavepacket
    assert legacy.circuit is legacy_circuit
    assert legacy_circuit.FockSimulator is tbl.FockSimulator
    assert tbl.ValidationError.__module__ == "tbl.errors"


def test_tbl_error_is_canonical_base_exception():
    assert issubclass(tbl.ValidationError, tbl.TBLError)
    assert tbl.OpenPhotonTwinError is tbl.TBLError
