"""Domain-specific exceptions."""


class OpenPhotonTwinError(Exception):
    """Base exception for the package."""


class ValidationError(OpenPhotonTwinError, ValueError):
    """Raised when a physical or numerical parameter is invalid."""


class OptionalDependencyError(OpenPhotonTwinError, ImportError):
    """Raised when an optional integration is requested but unavailable."""


class SimulationError(OpenPhotonTwinError, RuntimeError):
    """Raised when a simulation cannot produce a meaningful result."""
