"""Versioned provenance and lossless research-result bundles.

The bundle format deliberately uses NumPy arrays plus canonical JSON and never
loads Python pickles.  It is therefore inspectable, checksum-verifiable, and
safe to exchange between research environments.
"""

from __future__ import annotations

import hashlib
import json
import platform
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._version import __version__
from .detectors import DetectorArray, TimeTag
from .errors import SimulationError, ValidationError
from .models import Photon, PhotonEvent, Wavepacket

if TYPE_CHECKING:
    from .simulator import SimulationResult

_MANIFEST_SCHEMA = "tbl.run-manifest/1"
_BUNDLE_SCHEMA = "tbl.simulation-result/2"
_LEGACY_BUNDLE_SCHEMA = "tbl.simulation-result/1"
_RESULT_HASH_POLICY = "event-binary-v2"
_LEGACY_RESULT_HASH_POLICY = "event-binary-v1"
_EVENT_STRUCT = struct.Struct("<dqddqqq" + "d" * 9)
_TAG_STRUCT = struct.Struct("<dqq?")
_UNITS = {
    "time": "s",
    "wavelength": "m",
    "angular_frequency": "rad s^-1",
    "amplitude": "dimensionless",
}


def _qualified_name(value: object) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _json_safe(value: Any) -> Any:
    """Convert supported scientific values to canonical JSON data."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValidationError("research metadata cannot contain non-finite floats")
        return value
    if isinstance(value, complex):
        return {"__complex__": [float(value.real), float(value.imag)]}
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": _json_safe(value.tolist()),
            }
        }
    if is_dataclass(value) and not isinstance(value, type):
        parameters = {}
        for descriptor in fields(value):
            if descriptor.init or descriptor.name == "_commands":
                parameters[descriptor.name] = _json_safe(getattr(value, descriptor.name))
        return {"__type__": _qualified_name(value), "parameters": parameters}
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        items = [_json_safe(item) for item in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=_canonical_json)
        return items
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        portable = (
            module not in {"__main__", None}
            and "<lambda>" not in qualname
            and "<locals>" not in qualname
        )
        return {"__callable__": f"{module}:{qualname}", "portable": portable}
    if hasattr(value, "__dict__"):
        qualname = _qualified_name(value)
        parameters = {
            name: _json_safe(item) for name, item in vars(value).items() if not name.startswith("_")
        }
        portable = "__main__." not in qualname and "<locals>" not in qualname
        return {"__type__": qualname, "parameters": parameters, "portable": portable}
    raise ValidationError(
        f"unsupported research metadata type {_qualified_name(value)}; use JSON-compatible values"
    )


def _restore_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_json(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"__complex__"}:
        real, imag = value["__complex__"]
        return complex(real, imag)
    if set(value) == {"__ndarray__"}:
        descriptor = value["__ndarray__"]
        restored = np.asarray(_restore_json(descriptor["data"]), dtype=descriptor["dtype"])
        return restored.reshape(descriptor["shape"])
    return {key: _restore_json(item) for key, item in value.items()}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze_json(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _update_json_blob(digest: Any, value: Any, cache: dict[Any, bytes]) -> None:
    safe = _json_safe(value)
    key = _freeze_json(safe)
    payload = cache.get(key)
    if payload is None:
        payload = json.dumps(
            safe,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        cache[key] = payload
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _is_portable(value: Any) -> bool:
    if isinstance(value, dict):
        if "__callable__" in value and not value.get("portable", False):
            return False
        if "__type__" in value and value.get("portable") is False:
            return False
        return all(_is_portable(item) for item in value.values())
    if isinstance(value, list):
        return all(_is_portable(item) for item in value)
    return True


def snapshot_configuration(
    source: object,
    components: Sequence[object],
    detectors: DetectorArray,
    *,
    time_bin_width: float,
) -> dict[str, Any]:
    """Return a canonical, JSON-compatible snapshot before a run mutates state."""

    snapshot = _json_safe(
        {
            "source": source,
            "components": list(components),
            "detectors": detectors,
            "time_bin_width_s": time_bin_width,
        }
    )
    if not isinstance(snapshot, dict):  # pragma: no cover - defensive invariant
        raise SimulationError("configuration snapshot was not a mapping")
    return snapshot


def result_sha256(
    events: Sequence[PhotonEvent],
    time_tags: Sequence[TimeTag],
    *,
    policy: str = _RESULT_HASH_POLICY,
) -> str:
    """Hash physical fields using a versioned platform-independent policy."""

    if policy not in {_RESULT_HASH_POLICY, _LEGACY_RESULT_HASH_POLICY}:
        raise ValidationError(f"unsupported result hash policy {policy!r}")

    digest = hashlib.sha256()
    digest.update((policy + "\n").encode("ascii"))
    metadata_cache: dict[Any, bytes] = {}
    for event in events:
        packet = event.photon.wavepacket
        digest.update(
            _EVENT_STRUCT.pack(
                event.time,
                event.mode,
                event.amplitude.real,
                event.amplitude.imag,
                event.shot,
                event.roundtrips,
                event.photon.time_bin,
                packet.arrival_time,
                packet.temporal_width,
                packet.wavelength,
                packet.polarization[0].real,
                packet.polarization[0].imag,
                packet.polarization[1].real,
                packet.polarization[1].imag,
                packet.purity,
                packet.chirp,
            )
        )
        if policy == _RESULT_HASH_POLICY:
            _update_json_blob(digest, packet.profile, metadata_cache)
        _update_json_blob(digest, packet.label, metadata_cache)
        _update_json_blob(digest, event.photon.metadata, metadata_cache)
        _update_json_blob(digest, event.metadata, metadata_cache)
    for tag in time_tags:
        digest.update(_TAG_STRUCT.pack(tag.time, tag.channel, tag.shot, tag.dark_count))
        _update_json_blob(digest, tag.metadata, metadata_cache)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "scipy", "pandas", "matplotlib", "cupy-cuda12x"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Self-describing provenance captured for every digital-twin run."""

    schema: str
    created_utc: str
    tbl_version: str
    python_version: str
    platform: str
    dependencies: dict[str, str]
    rng_algorithm: str
    rng_stream_policy: str
    rng_entropy: int
    result_hash_policy: str
    seed: int | None
    shots: int
    acquisition_start_s: float
    acquisition_end_s: float
    configuration: dict[str, Any]
    configuration_sha256: str
    result_sha256: str | None
    portable_configuration: bool
    units: dict[str, str]

    def __post_init__(self) -> None:
        if self.schema != _MANIFEST_SCHEMA:
            raise ValidationError(f"unsupported run-manifest schema {self.schema!r}")
        if self.result_hash_policy not in {
            _RESULT_HASH_POLICY,
            _LEGACY_RESULT_HASH_POLICY,
        }:
            raise ValidationError(f"unsupported result hash policy {self.result_hash_policy!r}")
        if self.shots < 1:
            raise ValidationError("run manifest shots must be positive")
        if (
            not np.isfinite(self.acquisition_start_s)
            or not np.isfinite(self.acquisition_end_s)
            or self.acquisition_end_s < self.acquisition_start_s
        ):
            raise ValidationError("run manifest acquisition window is invalid")
        digests = [("configuration_sha256", self.configuration_sha256)]
        if self.result_sha256 is not None:
            digests.append(("result_sha256", self.result_sha256))
        for name, value in digests:
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValidationError(f"{name} is not a lowercase SHA-256 digest")
        if not isinstance(self.configuration, dict):
            raise ValidationError("run manifest configuration must be a mapping")
        try:
            datetime.fromisoformat(self.created_utc)
        except ValueError as exc:
            raise ValidationError("run manifest created_utc is not ISO-8601") from exc

    def to_dict(self) -> dict[str, Any]:
        return {field.name: _json_safe(getattr(self, field.name)) for field in fields(self)}

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False
        )

    def verify(self) -> bool:
        """Verify schema and the canonical configuration fingerprint."""

        if self.schema != _MANIFEST_SCHEMA:
            raise ValidationError(f"unsupported run-manifest schema {self.schema!r}")
        measured = _sha256_json(self.configuration)
        if measured != self.configuration_sha256:
            raise SimulationError(
                "run-manifest configuration checksum mismatch; provenance was modified"
            )
        return True

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunManifest:
        # Configuration values intentionally remain in their canonical JSON
        # representation (including explicit complex/array markers).
        restored = dict(value)
        if restored.get("schema") != _MANIFEST_SCHEMA:
            raise ValidationError(f"unsupported run-manifest schema {restored.get('schema')!r}")
        try:
            return cls(**{field.name: restored[field.name] for field in fields(cls)})
        except (KeyError, TypeError) as exc:
            raise ValidationError("run manifest is missing required fields") from exc


def build_run_manifest(
    configuration: dict[str, Any],
    *,
    rng_algorithm: str,
    rng_stream_policy: str,
    rng_entropy: int,
    seed: int | None,
    shots: int,
    acquisition_start: float,
    acquisition_end: float,
    result_digest: str | None = None,
) -> RunManifest:
    return RunManifest(
        schema=_MANIFEST_SCHEMA,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tbl_version=__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        dependencies=_dependency_versions(),
        rng_algorithm=rng_algorithm,
        rng_stream_policy=rng_stream_policy,
        rng_entropy=rng_entropy,
        result_hash_policy=_RESULT_HASH_POLICY,
        seed=seed,
        shots=shots,
        acquisition_start_s=acquisition_start,
        acquisition_end_s=acquisition_end,
        configuration=configuration,
        configuration_sha256=_sha256_json(configuration),
        result_sha256=result_digest,
        portable_configuration=_is_portable(configuration),
        units=dict(_UNITS),
    )


def _metadata_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([_canonical_json(value) for value in values], dtype=np.str_)


def save_simulation_result(result: SimulationResult, path: str | Path) -> Path:
    """Save a complete, checksummed result as a pickle-free compressed bundle."""

    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValidationError("research result bundles must use the .npz extension")
    if result.manifest is None:
        raise ValidationError("a SimulationResult without a run manifest cannot be bundled")
    manifest = result.manifest
    manifest.verify()
    measured_hash = result_sha256(
        result.events,
        result.time_tags,
        policy=manifest.result_hash_policy,
    )
    if manifest.result_sha256 is not None and measured_hash != manifest.result_sha256:
        raise SimulationError(
            "simulation result no longer matches its manifest; data was modified after the run"
        )
    if manifest.result_sha256 is None:
        manifest = replace(manifest, result_sha256=measured_hash)
    destination.parent.mkdir(parents=True, exist_ok=True)
    events = result.events
    tags = result.time_tags
    arrays: dict[str, Any] = {
        "bundle_schema": np.asarray(_BUNDLE_SCHEMA),
        "manifest_json": np.asarray(manifest.to_json(indent=None)),
        "event_time": np.asarray([event.time for event in events], dtype=np.float64),
        "event_mode": np.asarray([event.mode for event in events], dtype=np.int64),
        "event_amplitude_real": np.asarray([event.amplitude.real for event in events]),
        "event_amplitude_imag": np.asarray([event.amplitude.imag for event in events]),
        "event_shot": np.asarray([event.shot for event in events], dtype=np.int64),
        "event_roundtrips": np.asarray([event.roundtrips for event in events], dtype=np.int64),
        "event_metadata": _metadata_array([event.metadata for event in events]),
        "photon_time_bin": np.asarray([event.photon.time_bin for event in events], dtype=np.int64),
        "photon_metadata": _metadata_array([event.photon.metadata for event in events]),
        "wavepacket_width": np.asarray(
            [event.photon.wavepacket.temporal_width for event in events]
        ),
        "wavepacket_wavelength": np.asarray(
            [event.photon.wavepacket.wavelength for event in events]
        ),
        "wavepacket_polarization_real": np.asarray(
            [[value.real for value in event.photon.wavepacket.polarization] for event in events]
        ).reshape((-1, 2)),
        "wavepacket_polarization_imag": np.asarray(
            [[value.imag for value in event.photon.wavepacket.polarization] for event in events]
        ).reshape((-1, 2)),
        "wavepacket_purity": np.asarray([event.photon.wavepacket.purity for event in events]),
        "wavepacket_chirp": np.asarray([event.photon.wavepacket.chirp for event in events]),
        "wavepacket_label": _metadata_array([event.photon.wavepacket.label for event in events]),
        "wavepacket_profile": _metadata_array(
            [event.photon.wavepacket.profile for event in events]
        ),
        "tag_time": np.asarray([tag.time for tag in tags], dtype=np.float64),
        "tag_channel": np.asarray([tag.channel for tag in tags], dtype=np.int64),
        "tag_shot": np.asarray([tag.shot for tag in tags], dtype=np.int64),
        "tag_dark_count": np.asarray([tag.dark_count for tag in tags], dtype=np.bool_),
        "tag_metadata": _metadata_array([tag.metadata for tag in tags]),
    }
    np.savez_compressed(str(destination), **arrays)
    return destination


def _parse_json_array(values: np.ndarray) -> list[Any]:
    return [_restore_json(json.loads(str(value))) for value in values]


def _validate_bundle_arrays(archive: Any, schema: str) -> None:
    event_vectors = [
        "event_time",
        "event_mode",
        "event_amplitude_real",
        "event_amplitude_imag",
        "event_shot",
        "event_roundtrips",
        "event_metadata",
        "photon_time_bin",
        "photon_metadata",
        "wavepacket_width",
        "wavepacket_wavelength",
        "wavepacket_purity",
        "wavepacket_chirp",
        "wavepacket_label",
    ]
    if schema == _BUNDLE_SCHEMA:
        event_vectors.append("wavepacket_profile")
    tag_vectors = ("tag_time", "tag_channel", "tag_shot", "tag_dark_count", "tag_metadata")
    required = {
        "bundle_schema",
        "manifest_json",
        "wavepacket_polarization_real",
        "wavepacket_polarization_imag",
        *event_vectors,
        *tag_vectors,
    }
    missing = required.difference(archive.files)
    if missing:
        raise ValidationError(f"research bundle is missing arrays: {sorted(missing)!r}")
    event_count = len(archive["event_time"])
    for name in event_vectors:
        if archive[name].ndim != 1 or len(archive[name]) != event_count:
            raise ValidationError(f"research bundle event array {name!r} has invalid shape")
    for name in ("wavepacket_polarization_real", "wavepacket_polarization_imag"):
        if archive[name].shape != (event_count, 2):
            raise ValidationError(f"research bundle array {name!r} must have shape (events, 2)")
    tag_count = len(archive["tag_time"])
    for name in tag_vectors:
        if archive[name].ndim != 1 or len(archive[name]) != tag_count:
            raise ValidationError(f"research bundle tag array {name!r} has invalid shape")


def load_simulation_result(path: str | Path) -> SimulationResult:
    """Load and verify a :class:`SimulationResult` research bundle."""

    source = Path(path)
    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise SimulationError(f"could not read TBL research bundle {source}") from exc
    with archive:
        if "bundle_schema" not in archive.files:
            raise ValidationError("research bundle is missing bundle_schema")
        schema = str(archive["bundle_schema"].item())
        if schema not in {_BUNDLE_SCHEMA, _LEGACY_BUNDLE_SCHEMA}:
            raise ValidationError("unsupported simulation-result bundle schema")
        _validate_bundle_arrays(archive, schema)
        manifest = RunManifest.from_dict(json.loads(str(archive["manifest_json"].item())))
        manifest.verify()
        event_metadata = _parse_json_array(archive["event_metadata"])
        photon_metadata = _parse_json_array(archive["photon_metadata"])
        labels = _parse_json_array(archive["wavepacket_label"])
        profiles = (
            _parse_json_array(archive["wavepacket_profile"])
            if schema == _BUNDLE_SCHEMA
            else ["gaussian"] * len(labels)
        )
        events = []
        for index, time in enumerate(archive["event_time"]):
            polarization_real = archive["wavepacket_polarization_real"][index]
            polarization_imag = archive["wavepacket_polarization_imag"][index]
            polarization = (
                complex(polarization_real[0], polarization_imag[0]),
                complex(polarization_real[1], polarization_imag[1]),
            )
            packet = Wavepacket(
                arrival_time=float(time),
                temporal_width=float(archive["wavepacket_width"][index]),
                wavelength=float(archive["wavepacket_wavelength"][index]),
                polarization=polarization,
                purity=float(archive["wavepacket_purity"][index]),
                chirp=float(archive["wavepacket_chirp"][index]),
                label=labels[index],
                profile=profiles[index],
            )
            # Construction validates normalization. Restore the already
            # validated stored vector afterward so round-trips preserve every
            # floating-point bit instead of normalizing a second time.
            object.__setattr__(packet, "polarization", polarization)
            mode = int(archive["event_mode"][index])
            photon = Photon(
                packet,
                mode=mode,
                time_bin=int(archive["photon_time_bin"][index]),
                metadata=photon_metadata[index],
            )
            events.append(
                PhotonEvent(
                    photon,
                    float(time),
                    mode,
                    complex(
                        archive["event_amplitude_real"][index],
                        archive["event_amplitude_imag"][index],
                    ),
                    int(archive["event_shot"][index]),
                    int(archive["event_roundtrips"][index]),
                    event_metadata[index],
                )
            )
        tag_metadata = _parse_json_array(archive["tag_metadata"])
        tags = tuple(
            TimeTag(float(time), int(channel), int(shot), bool(dark), metadata)
            for time, channel, shot, dark, metadata in zip(
                archive["tag_time"],
                archive["tag_channel"],
                archive["tag_shot"],
                archive["tag_dark_count"],
                tag_metadata,
                strict=True,
            )
        )
    event_tuple = tuple(events)
    measured_hash = result_sha256(
        event_tuple,
        tags,
        policy=manifest.result_hash_policy,
    )
    if manifest.result_sha256 is None or measured_hash != manifest.result_sha256:
        raise SimulationError(
            "research bundle checksum mismatch; data may be corrupted or modified"
        )
    from .simulator import SimulationResult

    return SimulationResult(event_tuple, tags, manifest.shots, manifest.seed, manifest)
