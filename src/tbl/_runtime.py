"""Small runtime-environment helpers shared by optional dependencies."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def ensure_cache_environment(
    variable: str,
    preferred: Path,
    fallback_name: str,
) -> None:
    """Point a library cache at temp only when its normal location is unwritable."""

    if os.environ.get(variable):
        return
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".tbl-write-probe-", dir=preferred):
            pass
    except OSError:
        fallback = Path(tempfile.gettempdir()) / fallback_name
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(fallback)


def ensure_matplotlib_cache() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache = Path(tempfile.gettempdir()) / "tbl-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)


def configure_matplotlib(backend: str | None = None):
    """Prepare a writable cache and optionally select a Matplotlib backend."""

    ensure_matplotlib_cache()
    import matplotlib

    if backend is not None:
        matplotlib.use(backend)
    return matplotlib
