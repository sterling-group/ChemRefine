"""Pydantic validator for MLIP engine YAML options.

Mirrors :class:`chemrefine.engines.pyscf.options.PyscfOptions` so a
new engine implementer adds an ``options.py`` per backend by the
same pattern. Both the direct (:class:`~chemrefine.engines.mlip.engine.MlipEngine`)
and ExtOpt (:class:`~chemrefine.engines.mlip.extopt_engine.MlipExtOptEngine`)
engines route ``step.options`` through this validator instead of
reading the raw YAML dict.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class MlipOptions(BaseModel):
    """Validated knobs for the MLIP backend.

    Two axes, mapped the same way across libraries (the mace↔fairchem
    equivalence): ``task_name`` is the **method/head** that selects the builder
    (see the table in :mod:`chemrefine.engines.mlip.calculator`), and
    ``model_name`` is the **weights** handed to it. Backend-natural YAML aliases
    are accepted (``task`` for ``task_name``; ``model``/``size`` for
    ``model_name``) so each backend reads naturally.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    model_name: str = Field(
        "uma-s-1p1",
        validation_alias=AliasChoices("model_name", "model", "size"),
    )
    """The model *weights* for the chosen backend.

    A MACE size (``small``/``medium``/``large``), a FAIRChem checkpoint
    (``uma-s-1``/``uma-s-1p1``/``esen-…``), a SevenNet id (``7net-0``), an ORB
    loader (``orb_v3_…``), or a local path. YAML aliases: ``model``, ``size``.
    ``uma-s-1p2`` (UMA-1.2) is newer but needs an updated fairchem.
    """

    task_name: str = Field(
        "omol",
        validation_alias=AliasChoices("task_name", "task"),
    )
    """The method / head / family — selects the builder.

    A FAIRChem head (``omol``/``omat``/``odac``/``oc20``/``oc22``/``oc25``/``omc``),
    ``mace_off``/``mace_mp``/``mace_omol``, ``custom_mace``, ``chgnet``,
    ``sevenn``, or ``orb``. YAML alias: ``task``.
    """

    model_path: str | None = None
    """Custom MACE checkpoint path (selects the ``custom_mace`` backend)."""

    device: Literal["cuda", "cpu"] = "cuda"
    """Compute device for the MLIP model."""

    cores: int = Field(1, ge=1)
    """Per-structure core budget (passed to the throttler when applicable)."""

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> MlipOptions:
        """Validate a raw ``step.options`` dict; empty/None yields defaults."""
        return cls(**(raw or {}))
