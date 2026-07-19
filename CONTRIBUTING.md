# Contributing to TBL

TBL accepts changes that preserve physical meaning, reproducibility, and the
public research-data contract. Open an issue before a large model or API
change so its assumptions and validation evidence can be agreed first.

## Local quality gates

Install the development environment and run every gate before submitting:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m mypy
python -m pytest --cov=tbl --cov-report=term --cov-fail-under=80
python -m build
```

New physics must include all of the following:

- equations, SI units, assumptions, and validity limits in the physics docs;
- an analytic limit or independent reference implementation in tests;
- conservation, normalization, positivity, or statistical checks as relevant;
- seeded stochastic tests with tolerances justified by sampling uncertainty;
- a changelog entry when behavior or the public API changes.

Never silently clip invalid physical inputs unless the documented estimator
requires it. Prefer `ValidationError` for bad parameters and `SimulationError`
when valid inputs lead to a numerically or physically unusable result.

Research-bundle schema changes require a new schema identifier and backward
read tests. Existing schema meaning must not be changed in place.
