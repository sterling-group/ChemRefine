"""Template-driven ChemRefine engine that launches the modular Qiskit runner."""

from __future__ import annotations

import json
from typing import ClassVar

from chemrefine.engines._script import ScriptEngine
from chemrefine.engines.api import register
from chemrefine.engines.qiskit.backend import QiskitBackend
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.workflow import validate_options


@register("qiskit")
class QiskitEngine(
    QiskitBackend,
    ScriptEngine[QiskitOptions],
):
    """Direct Qiskit Nature electronic-structure engine."""

    name: ClassVar[str] = "qiskit"
    label: ClassVar[str] = "Qiskit Nature"
    options_cls: ClassVar[type[QiskitOptions]] = QiskitOptions

    def _vars_from(self, opts: QiskitOptions) -> dict[str, object]:
        """Expose safely escaped JSON while keeping the source template valid Python."""
        validate_options(opts)
        payload = json.dumps(opts.as_job_spec(), separators=(",", ":"), sort_keys=True)
        # The placeholder sits inside a double-quoted Python string.  Encoding the JSON
        # text as another JSON string and removing only its outer quotes leaves an escaped
        # string body that is safe to substitute, including for quotes and backslashes.
        escaped = json.dumps(payload)[1:-1]
        return {"QISKIT_OPTIONS_JSON": escaped}
