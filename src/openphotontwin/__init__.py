"""OpenPhotonTwin public API.

The package combines exact small-Fock-space linear optics with a time-resolved
Monte Carlo hardware digital twin for time-bin and fiber-loop experiments.
"""

from .adapters import from_perceval, from_sax_model, from_sparameters, from_strawberry_fields
from .backends import ArrayBackend, get_backend, propagate_batch
from .calibration import (
    CalibrationResult,
    CoincidenceHistogram,
    HOMFit,
    LossEstimate,
    LossProfile,
    ModelComparison,
    auto_calibrate,
    calibration_report,
    coincidence_histogram,
    compare_to_ideal,
    estimate_loss,
    fit_hom_dip,
    load_time_tags,
    locate_loss,
    tags_to_dataframe,
)
from .circuit import (
    FockDistribution,
    FockSimulator,
    LinearOpticalCircuit,
    MatrixComponent,
    permanent,
)
from .components import (
    BeamSplitter,
    DelayLine,
    DynamicBeamSplitter,
    EOMSwitch,
    FeedForwardController,
    FiberLoop,
    LossChannel,
    PhaseDrift,
    PhaseShifter,
    SimulationContext,
    TemperaturePhaseDrift,
    thermo_optic_phase,
)
from .detectors import SNSPD, DetectorArray, TimeTag
from .errors import OpenPhotonTwinError, OptionalDependencyError, SimulationError, ValidationError
from .experiments import HOMResult, hom_probabilities, hom_scan
from .models import Photon, PhotonEvent, SinglePhotonSource, TimeBinQubit, Wavepacket
from .simulator import DigitalTwin, SimulationResult

__version__ = "1.0.0"

__all__ = [
    "ArrayBackend",
    "BeamSplitter",
    "CalibrationResult",
    "CoincidenceHistogram",
    "DelayLine",
    "DetectorArray",
    "DigitalTwin",
    "DynamicBeamSplitter",
    "EOMSwitch",
    "FeedForwardController",
    "FiberLoop",
    "FockDistribution",
    "FockSimulator",
    "HOMFit",
    "HOMResult",
    "LinearOpticalCircuit",
    "LossChannel",
    "LossEstimate",
    "LossProfile",
    "MatrixComponent",
    "ModelComparison",
    "OpenPhotonTwinError",
    "OptionalDependencyError",
    "PhaseDrift",
    "PhaseShifter",
    "Photon",
    "PhotonEvent",
    "SNSPD",
    "SimulationContext",
    "TemperaturePhaseDrift",
    "SimulationError",
    "SimulationResult",
    "SinglePhotonSource",
    "TimeBinQubit",
    "TimeTag",
    "ValidationError",
    "Wavepacket",
    "auto_calibrate",
    "calibration_report",
    "coincidence_histogram",
    "compare_to_ideal",
    "estimate_loss",
    "fit_hom_dip",
    "from_perceval",
    "from_sax_model",
    "from_sparameters",
    "from_strawberry_fields",
    "get_backend",
    "hom_probabilities",
    "hom_scan",
    "load_time_tags",
    "locate_loss",
    "permanent",
    "propagate_batch",
    "tags_to_dataframe",
    "thermo_optic_phase",
]
