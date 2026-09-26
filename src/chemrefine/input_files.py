"""Resolve engine-declared input files without teaching the config or cache their grammar.

The config retains its source directory, engines declare option locations and manifest
dependencies, and this registry-aware layer joins those facts. Files need not exist until
the producing upstream step runs. The cache receives ordinary named paths to digest.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from chemrefine.config import StepConfig
from chemrefine.engines.api import CalculationEngine, InputFileDependencies, InputFileOptions
from chemrefine.errors import ConfigError

OptionPath = tuple[str | int, ...]
"""A dictionary/list location relative to a step's options."""


def option_pointer(location: OptionPath) -> str:
    """Render an option location as an escaped JSON pointer, independent of disk paths."""
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in location)


def _locations(step: StepConfig, engine: CalculationEngine) -> tuple[OptionPath, ...]:
    """Check an engine's declaration before using it to address nested options."""
    if not isinstance(engine, InputFileOptions):
        return ()
    locations = engine.input_file_options(step.engine_options())
    seen: set[OptionPath] = set()
    for location in locations:
        if (
            not isinstance(location, tuple)
            or not location
            or any(
                isinstance(part, bool)
                or not isinstance(part, (str, int))
                or (isinstance(part, int) and part < 0)
                for part in location
            )
        ):
            raise ConfigError(f"engine {engine.name!r} declared an invalid input option path")
        if location in seen:
            raise ConfigError(f"duplicate input file option {option_pointer(location)}")
        seen.add(location)
    return locations


def _parent(options: dict[str, Any], location: OptionPath) -> tuple[Any, str | int]:
    """Find a declared leaf's container, refusing missing or mismatched option locations."""
    value: Any = options
    for part in location:
        if not (
            (isinstance(value, dict) and isinstance(part, str) and part in value)
            or (
                isinstance(value, list)
                and isinstance(part, int)
                and not isinstance(part, bool)
                and 0 <= part < len(value)
            )
        ):
            raise ConfigError(f"input file option {option_pointer(location)} does not exist")
        parent = value
        value = cast(Any, parent)[part]
    return parent, location[-1]


def resolve_input_file_options(step: StepConfig, engine: CalculationEngine) -> StepConfig:
    """Copy declared option filenames to absolute paths, retaining their written spelling.

    No files are opened here. The same operation can therefore run during config-text
    validation, preflight, dry-run and preparation, even for outputs of earlier steps.
    Repeated resolution preserves the original references; an edited resolved option
    is treated as a new declaration rather than reusing stale private provenance.
    """
    locations = _locations(step, engine)
    if not locations:
        if step.input_file_spellings:
            return step.with_input_files(step.options, ())
        return step
    options = deepcopy(step.options)
    previous = {
        location: (written, resolved) for location, written, resolved in step.input_file_spellings
    }
    spellings: list[tuple[OptionPath, str, str]] = []
    for location in locations:
        parent, key = _parent(options, location)
        value = parent[key]
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"input file option {option_pointer(location)} must be a nonempty string"
            )
        old = previous.get(location)
        written = old[0] if old is not None and old[1] == value else value
        filename = Path(value)
        if not filename.is_absolute():
            filename = step.source_dir / filename
        resolved = str(filename.absolute())
        parent[key] = resolved
        spellings.append((location, written, resolved))
    return step.with_input_files(options, tuple(spellings))


def normalized_input_options(
    step: StepConfig, engine: CalculationEngine, options: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace resolved file strings with portable references in a validated option dump."""
    resolved = resolve_input_file_options(step, engine)
    normalized = deepcopy(dict(options))
    for location, written, _filename in resolved.input_file_spellings:
        parent, key = _parent(normalized, location)
        # Absolute external references retain a basename plus their content identity,
        # matching model_path. Relative written references remain exactly as authored.
        parent[key] = Path(written).name if Path(written).is_absolute() else written
    return normalized


def declared_input_files(step: StepConfig, engine: CalculationEngine) -> dict[str, Path]:
    """Enumerate direct input files and any engine-defined manifest payload dependencies.

    Names include the option location and written reference, not an absolute filename.
    The dependency hook receives direct files by option location, while its own names
    are separately namespaced to prevent collisions with direct files or template inputs.
    All transitive references must be returned by the engine; no file-format guessing
    or recursive parsing lives here.
    """
    resolved = resolve_input_file_options(step, engine)
    roots: dict[str, Path] = {}
    files: dict[str, Path] = {}
    for location, written, filename in resolved.input_file_spellings:
        pointer = option_pointer(location)
        path = Path(filename)
        portable = Path(written).name if Path(written).is_absolute() else written
        roots[pointer] = path
        files[f"option:{pointer}:{portable}"] = path
    if isinstance(engine, InputFileDependencies):
        dependencies = engine.input_file_dependencies(
            resolved.engine_options(),
            {name: path for name, path in roots.items() if path.is_file()},
        )
        for name, path in dependencies.items():
            if not isinstance(name, str) or not name or Path(name).is_absolute():
                raise ConfigError("input dependency names must be nonempty, relative logical names")
            if not isinstance(path, Path) or not path.is_absolute():
                raise ConfigError(f"input dependency {name!r} must resolve to an absolute Path")
            files[f"dependency:{name}"] = path
    return files


def validate_input_files(step: StepConfig, engine: CalculationEngine) -> None:
    """Require declared files and manifest payloads at preparation time, before submission."""
    for name, path in declared_input_files(step, engine).items():
        try:
            with path.open("rb") as handle:
                handle.read(1)
        except OSError as e:
            raise ConfigError(f"input file {name!r} is unavailable at {path}: {e}") from e
