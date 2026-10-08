"""Versioned quantum output bundles with bounded, pickle-free numerical payloads.

The descriptor is the commit record: a writer publishes it only after the unique
payload is durable. Readers validate both files, including uncompressed array
sizes, before exposing immutable arrays. These are native engine outputs, not a
second cache format.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from chemrefine.errors import ConfigError, OutputParseError

DEFAULT_MAX_BYTES = 268435456
MAX_DESCRIPTOR_BYTES = 1048576
_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*\Z")


class ArrayDescription(BaseModel):
    """Shape and numerical storage type of one named array."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    shape: tuple[StrictInt, ...]
    dtype: str


class BundleDescription(BaseModel):
    """Public, JSON-only schema of a native quantum artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    format_version: Literal[1] = 1
    kind: str = Field(min_length=1)
    payload: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arrays: dict[str, ArrayDescription]
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class QuantumBundle:
    """Validated manifest and detached, read-only numerical arrays."""

    description: BundleDescription
    arrays: Mapping[str, NDArray[Any]]

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a detached metadata view so callers cannot alter provenance."""
        return dict(self.description.model_dump(mode="json")["metadata"])


def _digest(path: Path) -> str:
    """Hash a payload in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _payload_path(path: Path, name: str) -> Path:
    """Resolve a single local filename, refusing traversal and external symlinks."""
    if not name or Path(name).name != name or "\\" in name or name in {".", ".."}:
        raise ValueError("bundle payload must be a filename beside its descriptor")
    target = path.parent / name
    if target.resolve().parent != path.parent.resolve():
        raise ValueError("bundle payload escapes its output directory")
    return target


def _array_bytes(description: Mapping[str, ArrayDescription], max_bytes: int) -> int:
    """Validate array descriptions and account for allocations without loading data."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    total = 0
    for name, spec in description.items():
        if not _NAME.fullmatch(name):
            raise ValueError(f"invalid array name {name!r}")
        dtype = np.dtype(spec.dtype)
        if dtype.kind not in "biufc" or any(size < 0 for size in spec.shape):
            raise ValueError("bundle arrays require numeric dtypes and nonnegative shapes")
        total += math.prod(spec.shape) * dtype.itemsize
        if total > max_bytes:
            raise ValueError(f"bundle arrays exceed max_bytes={max_bytes}")
    return total


def encode_bundle_descriptor(description: BundleDescription) -> bytes:
    """Encode the exact published JSON bytes, enforcing the reader's size limit."""
    document = json.dumps(description.model_dump(mode="json"), allow_nan=False, indent=2)
    encoded = (document + "\n").encode("utf-8")
    if len(encoded) > MAX_DESCRIPTOR_BYTES:
        raise ConfigError("quantum bundle descriptor exceeds its size limit")
    return encoded


def write_bundle(
    path: Path,
    *,
    kind: str,
    arrays: Mapping[str, NDArray[Any]],
    metadata: Mapping[str, Any],
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    """Publish a descriptor last, preserving any prior complete output on failure.

    Callers attach units, orbital order, tensor convention and scientific provenance
    to ``metadata``. Empty arrays are allowed for reports with no numerical payload.
    Unique payload names avoid overwriting arrays referenced by an older descriptor.
    """
    values = {name: np.asarray(value) for name, value in arrays.items()}
    descriptions = {
        name: ArrayDescription(shape=value.shape, dtype=value.dtype.str)
        for name, value in values.items()
    }
    try:
        _array_bytes(descriptions, max_bytes)
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError("bundle arrays must be finite")
        # Refuse invalid metadata before writing either output file.
        json.dumps(metadata, allow_nan=False)
        if not kind:
            raise ValueError("bundle kind cannot be empty")
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid quantum bundle: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = path.with_name(f"{path.stem}.{uuid4().hex}.npz")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with payload.open("xb") as stream:
            with ZipFile(stream, "w") as archive:
                for name, value in values.items():
                    with archive.open(f"{name}.npy", "w") as member:
                        np.lib.format.write_array(member, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = BundleDescription(
            kind=kind,
            payload=payload.name,
            sha256=_digest(payload),
            arrays=descriptions,
            metadata=dict(metadata),
        )
        encoded = encode_bundle_descriptor(descriptor)
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        payload.unlink(missing_ok=True)
        raise
    return path


def _description(path: Path) -> BundleDescription:
    """Read the bounded descriptor and reject non-standard JSON constants."""
    if path.stat().st_size > MAX_DESCRIPTOR_BYTES:
        raise ValueError("quantum bundle descriptor exceeds its size limit")
    raw = path.read_text(encoding="utf-8")
    # A round trip with allow_nan=False also rejects non-finite nested metadata.
    value = json.loads(raw)
    json.dumps(value, allow_nan=False)
    return BundleDescription.model_validate(value)


def bundle_dependencies(path: Path) -> dict[str, Path]:
    """Declare payload dependencies without importing a provider or executing code."""
    try:
        descriptor = _description(path)
        return {"payload": _payload_path(path, descriptor.payload)}
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read quantum bundle dependencies {path}: {exc}") from exc


def read_bundle(path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> QuantumBundle:
    """Validate descriptor, archive, digest and arrays before returning any output."""
    try:
        descriptor = _description(path)
        total = _array_bytes(descriptor.arrays, max_bytes)
        payload = _payload_path(path, descriptor.payload)
        # NPY headers are bounded by numpy's 10 KB default; ZIP overhead is bounded
        # separately so a forged archive cannot force an unbounded hash/read.
        overhead = 12000 * len(descriptor.arrays) + 65536
        if payload.stat().st_size > total + overhead:
            raise ValueError("bundle payload exceeds its declared allocation")
        if _digest(payload) != descriptor.sha256:
            raise ValueError("bundle payload digest mismatch")
        with ZipFile(payload) as archive:
            entries = archive.infolist()
            if sorted(entry.filename for entry in entries) != sorted(
                f"{name}.npy" for name in descriptor.arrays
            ):
                raise ValueError("bundle archive members differ from the descriptor")
            if sum(entry.file_size for entry in entries) > total + overhead:
                raise ValueError("bundle uncompressed payload exceeds its allocation")
            for name, spec in descriptor.arrays.items():
                with archive.open(f"{name}.npy") as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                    else:
                        raise ValueError("unsupported bundle NPY header version")
                    if shape != spec.shape or dtype.str != spec.dtype:
                        raise ValueError(f"bundle array {name!r} disagrees with its descriptor")
        result = {}
        with np.load(payload, allow_pickle=False) as loaded:
            for name, spec in descriptor.arrays.items():
                array = loaded[name]
                if array.shape != spec.shape or array.dtype.str != spec.dtype:
                    raise ValueError(f"bundle array {name!r} disagrees with its descriptor")
                if not np.isfinite(array).all():
                    raise ValueError(f"bundle array {name!r} contains non-finite values")
                array.flags.writeable = False
                result[name] = array
        return QuantumBundle(description=descriptor, arrays=result)
    except (OSError, ValueError, TypeError, BadZipFile, ValidationError) as exc:
        raise OutputParseError(f"invalid quantum bundle {path}: {exc}") from exc
