"""SDK-free discovery of file-valued leaves in validated component option models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from chemrefine.errors import ConfigError


@dataclass(frozen=True)
class TypedInputReference:
    """A logical option location and its explicitly declared optional file format."""

    location: tuple[str | int, ...]
    file_format: str | None = None


def typed_input_references(
    model: BaseModel, *, prefix: tuple[str | int, ...] = ()
) -> tuple[TypedInputReference, ...]:
    """Discover nested models, selected union branches and declared filename containers.

    Mark a string field, or a list/tuple/dictionary of strings, with
    ``json_schema_extra={"input_file": True}``. An optional ``file_format`` describes
    its parser. Unmarked strings are never interpreted as paths. ``None`` disables
    an optional reference. Models use canonical field names, matching model_dump().
    """
    references: list[TypedInputReference] = []
    active: set[int] = set()

    def walk(
        value: Any,
        location: tuple[str | int, ...],
        *,
        declared: bool = False,
        file_format: str | None = None,
    ) -> None:
        """Visit actual validated values, preserving dictionary keys and list indices."""
        if value is None:
            return
        if declared and isinstance(value, (str, Path)):
            if not str(value).strip():
                raise ConfigError("declared input filenames must be nonempty")
            references.append(TypedInputReference(location, file_format))
            return
        structured = isinstance(value, (BaseModel, Mapping, Sequence)) and not isinstance(
            value, (str, bytes, bytearray)
        )
        if not structured:
            if declared:
                raise ConfigError("declared input files require filenames or filename containers")
            return
        if id(value) in active:
            raise ConfigError("typed input references cannot contain cyclic option values")
        if len(location) > 64:
            raise ConfigError("typed input reference nesting exceeds 64 option levels")
        active.add(id(value))
        if isinstance(value, BaseModel):
            if declared:
                raise ConfigError("mark file fields inside nested models, not the model itself")
            for name, field in type(value).model_fields.items():
                extra = field.json_schema_extra
                marked = isinstance(extra, dict) and extra.get("input_file") is True
                format_name = (
                    extra.get("file_format") if isinstance(extra, dict) and marked else None
                )
                if format_name is not None and (
                    not isinstance(format_name, str) or not format_name.strip()
                ):
                    raise ConfigError("input file formats must be nonempty names")
                walk(
                    getattr(value, name),
                    (*location, name),
                    declared=marked,
                    file_format=format_name,
                )
        elif isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ConfigError("typed input option dictionaries require string keys")
                walk(item, (*location, key), declared=declared, file_format=file_format)
        else:
            for index, item in enumerate(value):
                walk(item, (*location, index), declared=declared, file_format=file_format)
        active.remove(id(value))

    walk(model, prefix)
    return tuple(references)
