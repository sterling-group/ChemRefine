"""Read facts from a Q-Chem input template (the reader half of the Q-Chem input file).

The writer half — generating the per-structure ``.in`` — is
:mod:`chemrefine.engines.qchem.input`. This module only *reads* a template: the run type
(for parser dispatch, when a step omits ``operation:``), the ``JOBTYPE`` facts (for NMS)
and the declared ``mem_total`` (for the SLURM memory request), in one pass.

Every ``$rem`` block is scanned, not just the first: a Q-Chem opt→freq workflow is a
multi-job ``@@@`` input whose *second* job carries ``jobtype freq``, so a first-block-only
scan would refuse NMS to exactly the template shape NMS needs. ``!`` comments are stripped
per line first, so a commented-out ``! jobtype freq`` is never mistaken for a directive —
the same discipline as ORCA's inspector.

Frequencies can also be asked for without a second job: Q-Chem 6 runs them after an
optimisation when a ``$geom_opt`` block sets ``final_vibrational_analysis true`` — the
one-job idiom that has replaced the ``@@@`` chain in recent inputs — so that block is read
too, and ``has_freq`` answers for either spelling.

``mem_total`` follows qqchem's grammar (``mem_total`` with an optional ``=``, MB). The jobs
of an ``@@@`` chain run one after another, so the **max** across blocks is the run's peak
requirement — the number the SLURM request has to cover.

A job type whose output is many geometries — a relaxed surface scan, a reaction path, a
string, a trajectory — infers an ``operation`` of its own (:data:`_MULTI_GEOMETRY_OPERATIONS`)
rather than ``sp``: ``sp`` would file a scan's last frame as the structure's result, while a
named operation lets the parser dispatch decide — refusing the step while it does not know
the word, fanning the output out into its geometries once the parser exists, exactly as
ORCA's ``pes`` does. Where ORCA runs the same kind of job the word is shared, so a config
means one thing on either engine. Every ``jobtype`` the chain names is reported too, for
the refusal's message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Line-anchored like the writer's ``$molecule`` regex, and for the same reason: ``$rem``
# named in a ``$comment`` block's prose must not open a phantom block whose "jobtype ts"
# is the comment's own words. (``!`` comments are already stripped; ``$comment`` blocks
# are not.)
_REM_BLOCK_RE = re.compile(
    r"^[ \t]*\$rem\b(.*?)^[ \t]*\$end[ \t]*$", re.IGNORECASE | re.DOTALL | re.MULTILINE
)
# The optimiser's own block (Q-Chem 6, libopt3), scanned for the one setting that makes an
# ``opt`` job a frequency job as well. Anchored like ``$rem`` for the same reason.
_GEOM_OPT_BLOCK_RE = re.compile(
    r"^[ \t]*\$geom_opt\b(.*?)^[ \t]*\$end[ \t]*$", re.IGNORECASE | re.DOTALL | re.MULTILINE
)
_JOBTYPE_RE = re.compile(r"\bjobtype\s*=?\s*(\S+)", re.IGNORECASE)
_FINAL_VIB_RE = re.compile(r"\bfinal_vibrational_analysis\s*=?\s*(?:true|1)\b", re.IGNORECASE)
# qqchem's grammar, which submits real inputs with it: optional ``=``, value in MB.
_MEM_TOTAL_RE = re.compile(r"\bmem_total\s*=?\s*(\d+)", re.IGNORECASE)

_MULTI_GEOMETRY_OPERATIONS = {
    "pes_scan": "pes",
    "rpath": "irc",
    "aimd": "md",
    "fsm": "fsm",
    "gsm": "gsm",
    "pimd": "pimd",
    "pimc": "pimc",
    "bh": "bh",
}
"""Job types whose output is many geometries, and the ``operation`` each infers.

The word is shared with ORCA wherever ORCA runs the same kind of job, so one config means
one thing on either engine: ``pes_scan`` is the relaxed surface scan (``$scan`` — a sequence
of constrained optimisations, one frame per point) that ORCA's ``%geom Scan`` already
answers to as ``pes``; ``rpath`` is the intrinsic reaction coordinate (Fukui's IRC, from a
TS and its Hessian) that ORCA spells ``! IRC``, so ``irc``; ``aimd`` is a trajectory like
ORCA's ``%md``, so ``md``. The string methods (``fsm``, ``gsm`` — a chain of nodes whose
highest is the TS guess; ORCA's counterpart is ``NEB-TS``, a different algorithm) and the
path-integral and basin-hopping kinds (``pimd``, ``pimc``, ``bh`` — a global search, ORCA's
nearest being ``GOAT``) keep their own names. Dict order is precedence when a chain names
more than one.

The engine refuses a step whose operation the parser dispatch
(:func:`~chemrefine.engines.qchem.output.known_operations`) does not know, and runs it the
moment the dispatch learns the word — with no edit here. Note for the ``md`` parser to come:
Q-Chem writes the trajectory to the ``AIMD/`` directory under the job's scratch, which only
``save: true`` brings home."""


@dataclass(frozen=True)
class QchemInputInfo:
    """Facts read from a Q-Chem input template in one pass.

    ``operation`` is the parser key (see :mod:`chemrefine.engines.qchem.output`) —
    ``opt_sp`` for an optimisation or TS search, a multi-geometry job's own word
    (:data:`_MULTI_GEOMETRY_OPERATIONS`), else ``sp``. It is never ``freq``, because
    frequencies are parsed off the output unconditionally and ``has_freq`` carries that
    fact on its own. ``is_ts``
    marks a ``jobtype ts`` (drives the NMS ``ts`` target); ``has_freq`` marks a
    ``jobtype freq`` anywhere in the chain or a ``$geom_opt`` block asking for
    ``final_vibrational_analysis`` (gates NMS); ``mem_total_mb`` is the largest
    declared ``mem_total`` (``None`` if none is declared — absence means the header's
    memory policy stands, so it is not defaulted); ``jobtypes`` is every ``jobtype`` the
    chain names, lower-cased, so a refusal can say which one inferred the operation.
    """

    operation: str
    is_ts: bool
    has_freq: bool
    mem_total_mb: int | None
    jobtypes: frozenset[str]


def _strip_comments(text: str) -> str:
    """Drop Q-Chem ``!`` comments — everything from the first ``!`` on each line."""
    return "\n".join(line.split("!", 1)[0] for line in text.splitlines())


def inspect_template(template_path: str | Path) -> QchemInputInfo:
    """Infer the ``JOBTYPE`` facts and the declared ``mem_total`` in a single read."""
    text = _strip_comments(Path(template_path).read_text(encoding="utf-8", errors="replace"))
    jobtypes = {
        m.group(1).lower()
        for block in _REM_BLOCK_RE.findall(text)
        for m in [_JOBTYPE_RE.search(block)]
        if m
    }
    mem_totals = [
        int(m.group(1))
        for block in _REM_BLOCK_RE.findall(text)
        for m in _MEM_TOTAL_RE.finditer(block)
    ]
    final_vib = any(_FINAL_VIB_RE.search(block) for block in _GEOM_OPT_BLOCK_RE.findall(text))
    # A multi-geometry job anywhere in the chain shapes the whole output, so its word wins;
    # an opt or ts job makes the run an optimisation — the last geometry is a stationary
    # point, which is what `opt_sp` promises the parser dispatch; anything else parses
    # like a single point. `freq` is a flag, not an operation: frequencies are parsed off
    # the output unconditionally.
    multi = [op for jobtype, op in _MULTI_GEOMETRY_OPERATIONS.items() if jobtype in jobtypes]
    if multi:
        operation = multi[0]
    elif jobtypes & {"opt", "ts"}:
        operation = "opt_sp"
    else:
        operation = "sp"
    return QchemInputInfo(
        operation=operation,
        is_ts="ts" in jobtypes,
        has_freq="freq" in jobtypes or final_vib,
        mem_total_mb=max(mem_totals) if mem_totals else None,
        jobtypes=frozenset(jobtypes),
    )
