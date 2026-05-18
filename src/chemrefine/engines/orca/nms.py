"""Normal-mode sampling for ORCA frequency outputs.

TODO: port the v3 NMS pipeline from ``orca_interface.py`` on ``main``.
The reference functions are:

* ``parse_imaginary_frequency`` — find imaginary-mode indices in
  the ORCA ``.hess`` / output file
* ``parse_normal_modes_tensor`` — load the normal-mode displacement
  tensor of shape ``(3 N_atoms, n_modes)``
* ``displace_normal_modes`` — apply ±dq displacements per mode
* ``write_displaced_xyz`` — emit the displaced ``.xyz`` files
* ``select_lowest_imaginary_structures`` — pick the structures whose
  energy drops fastest along the imaginary mode

The new shape should expose a single :func:`normal_mode_sample`
function that takes :class:`~chemrefine.state.StepResults` plus the
:class:`~chemrefine.state.StepContext` and returns expanded
:class:`StepResults`. The work to implement is mostly already in v3 —
the tricky part is verifying displacements against a known reference
output, which needs a real frequency calculation.
"""

from __future__ import annotations

from chemrefine.state import StepContext, StepResults


def normal_mode_sample(results: StepResults, ctx: StepContext) -> StepResults:
    """Displace each structure along its imaginary modes.

    Placeholder — see module docstring for the port plan.
    """
    raise NotImplementedError(
        "ORCA normal-mode sampling not yet ported — see TODO in engines/orca/nms.py"
    )
