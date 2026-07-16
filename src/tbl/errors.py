"""Domain-specific exceptions for TBL."""


class TBLError(Exception):
    """Base exception for the package."""


# Backward-compatible public alias retained for TBL 1.x applications.
OpenPhotonTwinError = TBLError


class ValidationError(TBLError, ValueError):
    """Raised when a physical or numerical parameter is invalid."""


class OptionalDependencyError(TBLError, ImportError):
    """Raised when an optional integration is requested but unavailable."""


class SimulationError(TBLError, RuntimeError):
    """Raised when a simulation cannot produce a meaningful result."""
