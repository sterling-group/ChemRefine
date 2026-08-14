"""Read facts from a Q-Chem input template (the reader half of the Q-Chem input file).

The writer half — generating the per-structure ``.in`` — is
:mod:`chemrefine.engines.qchem.input`. This module only *reads* a template: the ``JOBTYPE``
facts (for NMS) and the declared ``mem_total`` (for the SLURM memory request), in one pass.

Every ``$rem`` block is scanned, not just the first: a Q-Chem opt→freq workflow is a
multi-job ``@@@`` input whose *second* job carries ``jobtype freq``, so a first-block-only
scan would refuse NMS to exactly the template shape NMS needs. ``!`` comments are stripped
per line first, so a commented-out ``! jobtype freq`` is never mistaken for a directive —
the same discipline as ORCA's inspector.

``mem_total`` follows qqchem's grammar (``mem_total`` with an optional ``=``, MB). The jobs
of an ``@@@`` chain run one after another, so the **max** across blocks is the run's peak
requirement — the number the SLURM request has to cover.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_REM_BLOCK_RE = re.compile(r"\$rem\b(.*?)\$end", re.IGNORECASE | re.DOTALL)
_JOBTYPE_RE = re.compile(r"\bjobtype\s*=?\s*(\S+)", re.IGNORECASE)
# qqchem's grammar, which submits real inputs with it: optional ``=``, value in MB.
_MEM_TOTAL_RE = re.compile(r"\bmem_total\s*=?\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class QchemInputInfo:
    """Facts read from a Q-Chem input template in one pass.

    ``is_ts`` marks a ``jobtype ts`` (drives the NMS ``ts`` target); ``has_freq`` marks a
    ``jobtype freq`` anywhere in the chain (gates NMS); ``mem_total_mb`` is the largest
    declared ``mem_total`` (``None`` if none is declared — absence means the header's
    memory policy stands, so it is not defaulted).
    """

    is_ts: bool
    has_freq: bool
    mem_total_mb: int | None


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
    return QchemInputInfo(
        is_ts="ts" in jobtypes,
        has_freq="freq" in jobtypes,
        mem_total_mb=max(mem_totals) if mem_totals else None,
    )
