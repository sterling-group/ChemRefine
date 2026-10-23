"""The xtb command line for one calculation: run type, method, solvation, charge and spin.

xtb takes everything that decides *what* it computes on its command line; its xcontrol
input (the step's template) carries only what the command line cannot say. This module is
the one place that turns a step into those arguments. It is pure (values in, a tuple of
words out) and depends on nothing but the options model, so code outside the engine that
runs xtb itself (a solvation correction computing E(ALPB) - E(gas), say) builds its
command line the same way the engine does.

The run types, by ChemRefine ``operation``:

==============  =================================  ===========================================
``operation``   xtb flags                          what the output holds
==============  =================================  ===========================================
``sp``          ``--grad``                         energy and gradient
``opt_sp``      ``--opt [level] --grad``           + the optimised geometry and its verdict
``freq``        ``--hess --grad``                  + the Hessian, frequencies, thermochemistry
``opt_freq``    ``--ohess [level] --grad``         all of the above
==============  =================================  ===========================================

``--grad`` is always asked for: the gradient costs about as much as the SCC it follows,
and with it every structure carries forces. The one exception is CPCM-X, which gives no
gradient, so an ``sp`` under it runs ``--scc``.
"""

from __future__ import annotations

from chemrefine.engines.xtb.options import XtbOptions

RUN_TYPES: dict[str, tuple[str, ...]] = {
    "sp": (),
    "opt_sp": ("--opt",),
    "freq": ("--hess",),
    "opt_freq": ("--ohess",),
}
"""Each ``operation`` — the engine's whole vocabulary — and the run-type flag it adds
beside ``--grad``. A single point adds none: there ``--grad`` is the run type."""

OPTIMISING = frozenset({"opt_sp", "opt_freq"})
"""The operations that optimise, the only ones an ``opt_level`` means anything to."""

FREQUENCY = frozenset({"freq", "opt_freq"})
"""The operations that compute a Hessian, and so frequencies."""

_METHOD_FLAGS: dict[str, tuple[str, ...]] = {
    "gfn2": ("--gfn", "2"),
    "gfn1": ("--gfn", "1"),
    "gfn0": ("--gfn", "0"),
    "gfnff": ("--gfnff",),
}


def unpaired_electrons(multiplicity: int) -> int:
    """xtb's ``--uhf``: the number of unpaired electrons, ``multiplicity - 1``."""
    return multiplicity - 1


def arguments(
    options: XtbOptions, *, operation: str, charge: int, multiplicity: int
) -> tuple[str, ...]:
    """The xtb arguments that decide what one calculation computes, in a fixed order.

    The geometry, ``--input`` and the thread count are not here. They say where and how
    a job runs, not what it computes, and the caller knows them
    (:meth:`~chemrefine.engines.xtb.engine.XtbEngine.run_block`). Every argument is a
    word of its own, so the caller quotes each (the solvent is the only value from the
    config, and it is held to a name's characters by the options model).

    Raises :class:`ValueError` for an operation outside :data:`RUN_TYPES`, or CPCM-X with
    any operation but ``sp``. The engine refuses both before anything runs
    (:meth:`~chemrefine.engines.xtb.engine.XtbEngine.check_step`), so a caller reaching
    either has skipped that check.
    """
    if operation not in RUN_TYPES:
        raise ValueError(f"xtb has no run type for operation {operation!r}")
    if options.solvation == "cpcmx" and operation != "sp":
        raise ValueError("CPCM-X in xtb gives no gradient, so it runs only as `operation: sp`")
    words: list[str] = list(RUN_TYPES[operation])
    if operation in OPTIMISING and options.opt_level is not None:
        words.append(options.opt_level)
    words.append("--scc" if options.solvation == "cpcmx" else "--grad")
    words += _METHOD_FLAGS[options.method]
    # The options model holds `solvation` and `solvent` together, so one is set exactly
    # when the other is.
    if options.solvation is not None and options.solvent is not None:
        words += [f"--{options.solvation}", options.solvent]
        if options.solvation_reference not in (None, "gsolv"):
            words.append(options.solvation_reference)
    if options.accuracy is not None:
        words += ["--acc", repr(options.accuracy)]
    if options.etemp is not None:
        words += ["--etemp", repr(options.etemp)]
    words += ["--chrg", str(charge), "--uhf", str(unpaired_electrons(multiplicity))]
    return tuple(words)
