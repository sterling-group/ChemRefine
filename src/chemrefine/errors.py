"""Exception hierarchy for ChemRefine.

Every exception carries a non-zero ``exit_code`` so the CLI can translate
the failure mode into a deterministic process exit code. The root class is
:class:`ChemRefineError` — all package exceptions inherit from it so the
CLI can catch one type and dispatch on the code.
"""

from __future__ import annotations


class ChemRefineError(Exception):
    """Base class for every ChemRefine-raised exception."""

    exit_code: int = 1


class ConfigError(ChemRefineError):
    """The YAML config is malformed, missing required fields, or invalid."""

    exit_code = 2


class EngineNotFoundError(ChemRefineError):
    """A step requested an engine that is not in the registry."""

    exit_code = 3


class JobSubmissionError(ChemRefineError):
    """``sbatch`` (or the local runner) refused to accept a job."""

    exit_code = 4


class JobFailureError(ChemRefineError):
    """A submitted job finished but its output indicates failure."""

    exit_code = 5


class OutputParseError(ChemRefineError):
    """An engine output file could not be parsed."""

    exit_code = 6


class CacheError(ChemRefineError):
    """A step cache is corrupt, version-mismatched, or unwritable."""

    exit_code = 7
