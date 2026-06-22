"""The script engine kind: run a user ``step{N}.py`` per structure.

A reusable building block (not a plugin): :class:`ScriptEngine` is a
:class:`~chemrefine.engines._job.JobEngine` whose per-structure input is a user Python
script; :mod:`.render` writes it and :mod:`.output` parses its JSON. The ``mlip`` and
``pyscf`` plugins subclass :class:`ScriptEngine`.
"""

from chemrefine.engines._script.engine import ScriptEngine

__all__ = ["ScriptEngine"]
