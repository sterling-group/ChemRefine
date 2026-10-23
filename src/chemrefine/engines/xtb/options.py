"""Pydantic validator for the xtb engine's YAML options.

One validated model, the single reader of the step's knobs, as in the other engines'
``options.py`` modules. xtb takes its method, solvation and convergence settings on its
command line rather than in its xcontrol input, so the YAML is where those natively live;
:mod:`chemrefine.engines.xtb.command` turns the model into the arguments. ``cores`` is the
inherited thread count, given to xtb as ``-P`` and ``OMP_NUM_THREADS``.

Every value here is checked against what xtb 6.7.1 actually accepts, because xtb takes
some wrong values without a word: ``--opt bogus`` runs at ``normal`` level, and an unknown
reference state is reported in the output of a job that still exits 0.
"""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from chemrefine.engines._options import EngineOptions

_SOLVENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9,()+\-.']*")
"""What a solvent name may be spelled with: the names xtb's ALPB/GBSA tables use
(``water``, ``ch2cl2``, ``woctanol``) and the Minnesota-database names CPCM-X takes
(``1,2-dichloroethane``, ``N,N-dimethylformamide``, ``2,2,2-trifluoroethanol``)."""


class XtbOptions(EngineOptions):
    """Validated knobs for the xtb engine.

    ``cores`` (inherited) is the thread count; ``device`` and ``backend_python`` are
    inherited too and refused at preflight
    (:meth:`~chemrefine.engines.xtb.engine.XtbEngine.check_step`), since xtb reads neither.
    """

    method: Literal["gfn2", "gfn1", "gfn0", "gfnff"] = "gfn2"
    """The Hamiltonian: ``--gfn 2`` / ``--gfn 1`` / ``--gfn 0``, or ``--gfnff``.

    The default is xtb's own, written on the command line every time so the job does not
    depend on what the binary defaults to. GFN0 reads its parameters from ``XTBPATH``
    (see :meth:`~chemrefine.engines.xtb.engine.XtbEngine.run_block`)."""

    solvation: Literal["alpb", "gbsa", "cpcmx"] | None = None
    """Implicit-solvation model: ``--alpb`` / ``--gbsa`` / ``--cpcmx``. **No default**, and
    required whenever ``solvent`` is set, so a model added later can never change what an
    existing config means.

    ``cpcmx`` gives no gradients, so it is accepted on ``operation: sp`` alone, and only
    some xtb builds include it. A binary built without it fails at once, and the parser
    quotes xtb's own message."""

    solvent: str | None = None
    """The solvent's name, as xtb spells it (``water``, ``ch2cl2``, ``thf``, …; for
    CPCM-X, a Minnesota Solvation Database name). Required with ``solvation``. xtb
    checks the name itself."""

    solvation_reference: Literal["gsolv", "bar1M", "reference"] | None = None
    """The ALPB/GBSA reference state of the solvation free energy. ``gsolv`` is xtb's
    default and adds no argument; ``bar1M`` (1 bar gas → 1 M solution) and ``reference``
    are passed after the solvent. Spelled exactly as xtb reads them: ``bar1m`` and an
    explicit ``gsolv`` are both unknown to xtb 6.7.1, which reports it and carries on."""

    opt_level: (
        Literal["crude", "sloppy", "loose", "lax", "normal", "tight", "vtight", "extreme"] | None
    ) = None
    """The optimisation level for ``opt_sp`` / ``opt_freq`` (``--opt LEVEL``); unset is
    xtb's ``normal``. Checked here because xtb runs an unknown level as ``normal``
    without a word."""

    accuracy: float | None = Field(None, gt=0)
    """The SCC accuracy, ``--acc`` (lower is tighter; xtb's default is 1.0)."""

    etemp: float | None = Field(None, gt=0)
    """The electronic temperature in K, ``--etemp`` (xtb's default is 300)."""

    @field_validator("solvent")
    @classmethod
    def _solvent_is_a_name(cls, v: str | None) -> str | None:
        """Hold the solvent to the characters a solvent name has.

        It reaches the job's command line. Quoting already keeps it one word there, but a
        name that is not a name — a space, a ``$``, a newline — is a mistake to say now
        rather than a job that fails on its own command line later.
        """
        if v is not None and not _SOLVENT_RE.fullmatch(v):
            raise ValueError(
                f"solvent {v!r} is not a solvent name xtb could read (letters, digits and "
                f"the punctuation of names such as 1,2-dichloroethane)"
            )
        return v

    @model_validator(mode="after")
    def _solvation_is_named_whole(self) -> Self:
        """``solvation`` and ``solvent`` come together; a reference state needs both.

        A solvent with no model is a step whose author meant solvent and would get gas
        phase; a model with no solvent has nothing to compute in. CPCM-X has no reference
        state argument, so one given with it would be dropped.
        """
        if (self.solvation is None) != (self.solvent is None):
            raise ValueError(
                "`solvation` and `solvent` are set together: name the model (alpb, gbsa "
                "or cpcmx) and the solvent, or neither"
            )
        if self.solvation_reference is not None and self.solvation not in ("alpb", "gbsa"):
            raise ValueError("`solvation_reference` is for `solvation: alpb` or `gbsa`")
        return self
