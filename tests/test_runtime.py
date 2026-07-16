import tbl._runtime as runtime


def test_matplotlib_cache_uses_stable_temp_location(monkeypatch, tmp_path):
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    runtime.ensure_matplotlib_cache()
    expected = tmp_path / "tbl-matplotlib-cache"
    assert runtime.os.environ["MPLCONFIGDIR"] == str(expected)
    assert expected.is_dir()
