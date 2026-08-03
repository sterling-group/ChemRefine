"""MLIP training pipeline.

Trains a MACE-style MLIP on the structures + energies + forces
collected by a previous pipeline step. The training itself runs as a
single SLURM job (one ``mace_run_train`` invocation against a YAML
config); ChemRefine only writes the inputs, submits the job, and
returns the seed structures unchanged so downstream steps can keep
using them.

Every helper takes its inputs explicitly (no shared module state) so
it can be unit-tested with a ``tmp_path`` fixture. The actual MACE
training command (``mace_run_train``) is exercised only in the
integration suite that runs against a real CUDA stack.
"""

from __future__ import annotations

import logging
import re
from math import ceil
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io import write as ase_write

from chemrefine import ids, slurm
from chemrefine.engines.mlip.options import MlipTrainOptions
from chemrefine.errors import ConfigError
from chemrefine.quantities import HARTREE_TO_EV
from chemrefine.state import StepContext, StepResults

logger = logging.getLogger(__name__)

_JOB_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
"""Characters allowed in a ``job_name`` that is interpolated into an ``#SBATCH`` line."""


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def prepare_inputs(results: StepResults, ctx: StepContext) -> tuple[Path, Path]:
    """Write train / test ``extxyz`` files from ``results`` and return their paths.

    Energy is taken from ``Structure.energy_hartree`` (Hartree → eV on
    write); forces are taken from ``Structure.forces_ev_per_a`` if set,
    otherwise from a Hartree/Bohr ``forces_hartree_per_bohr`` info
    attribute the ORCA parser would have stored. The 90/10 split (and
    the seed) come from ``step.options.valid_fraction`` and
    ``step.options.seed`` respectively.

    Raises :class:`ValueError` when any structure is missing an energy
    or forces — MLIP training needs both.
    """
    opts = MlipTrainOptions.from_raw_lenient(ctx.step_cfg.options)
    valid_fraction, seed = opts.valid_fraction, opts.seed

    atoms_list: list[Atoms] = []
    for struct in results.structures:
        if struct.energy_hartree is None:
            raise ValueError(f"structure {struct.id} has no energy — cannot train")
        if struct.forces_ev_per_a is None:
            raise ValueError(f"structure {struct.id} has no forces — cannot train")
        atoms = struct.atoms.copy()
        atoms.info["DFT_energy"] = struct.energy_hartree * HARTREE_TO_EV
        # Forces are stored in eV/Å (already converted from Hartree/Bohr
        # by the ORCA parser).
        atoms.arrays["DFT_Forces"] = np.asarray(struct.forces_ev_per_a, dtype=float)
        atoms_list.append(atoms)

    if not atoms_list:
        raise ValueError("no usable structures for MLIP training")

    n = len(atoms_list)
    if n < 2:
        # A split needs at least one of each — fall back to "all train".
        train_set, test_set = atoms_list, []
    else:
        # Random validation split: ``ceil(valid_fraction * n)`` structures to
        # test, the rest to train (same sizing sklearn's train_test_split used).
        n_test = ceil(valid_fraction * n)
        if n - n_test == 0:
            raise ValueError(
                f"valid_fraction={valid_fraction} leaves no training structures "
                f"for n={n}; lower it."
            )
        perm = np.random.default_rng(seed).permutation(n)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        train_set = [atoms_list[i] for i in train_idx]
        test_set = [atoms_list[i] for i in test_idx]

    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    train_path = ctx.step_dir / "mace_train.xyz"
    test_path = ctx.step_dir / "mace_test.xyz"
    ase_write(str(train_path), train_set, format="extxyz")
    ase_write(str(test_path), test_set, format="extxyz")
    logger.info(
        "MLIP training: wrote %d train / %d test structures",
        len(train_set),
        len(test_set),
    )
    return train_path, test_path


# ---------------------------------------------------------------------------
# Config + SLURM
# ---------------------------------------------------------------------------


def write_training_config(*, train_path: Path, test_path: Path, ctx: StepContext) -> Path:
    """Render a MACE training YAML from the per-step template.

    The template is the step's ``template:`` override or the default
    ``<template_dir>/step{N}.inp`` (a YAML body). The dataset paths are patched
    and three output directories so MACE writes inside the step dir, then
    write the resolved config to ``<step_dir>/input.yaml``.
    """
    template_path = ids.require_template(ctx.template, label="MLIP training")
    raw = template_path.read_text(encoding="utf-8")
    config = yaml.safe_load(raw) or {}
    config["train_file"] = str(train_path)
    config["test_file"] = str(test_path)
    for key in ("log_dir", "checkpoints_dir", "results_dir"):
        config[key] = str(ctx.step_dir / key)

    config_path = ctx.step_dir / "input.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def write_training_slurm(*, ctx: StepContext, config_path: Path) -> Path:
    """Generate the MACE training SLURM script.

    Picks ``cuda.slurm.header`` or ``cpu.slurm.header`` from ``ctx.template_dir``
    based on ``step.options.device``, appends ``--job-name`` / ``--output`` /
    ``--error``, and writes a single ``mace_run_train --config <input.yaml>``
    command at the end.

    The device is read through :class:`~chemrefine.engines.mlip.options.MlipTrainOptions`,
    which declares training's ``cuda`` default, rather than repeating that default here. A
    literal here and a default there drift the first time either moves, leaving the header
    asking for a GPU the options model did not request.
    """
    opts = MlipTrainOptions.from_raw_lenient(ctx.step_cfg.options)
    header_path = ctx.template_dir / slurm.header_name_for_device(opts.device)
    if not header_path.is_file():
        raise ConfigError(f"SLURM header template not found: {header_path}")

    # `job_name` lands inside an #SBATCH directive, where a newline would start an
    # arbitrary extra directive and whitespace would split the value. The pattern that
    # enforces that is on the field, so it is declared once next to what it constrains
    # rather than re-checked wherever the value is read.
    job_name = opts.job_name
    script_path = ctx.step_dir / "train.slurm"
    header_text = header_path.read_text(encoding="utf-8").rstrip()
    body = (
        f"{header_text}\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --output={ctx.step_dir / 'slurm-%j.out'}\n"
        f"#SBATCH --error={ctx.step_dir / 'slurm-%j.err'}\n"
        "\n"
        "export MKL_THREADING_LAYER=GNU\n"
        f"mace_run_train --config {config_path}\n"
    )
    script_path.write_text(body, encoding="utf-8")
    return script_path


# ---------------------------------------------------------------------------
# Submit + wait
# ---------------------------------------------------------------------------


def submit_training(
    *,
    script_path: Path,
    poll_seconds: float = 30.0,
    dispatch: str = "auto",
    max_wait_seconds: float | None = None,
) -> str:
    """Submit the MLIP training SLURM job and block until it finishes.

    Waits through :func:`chemrefine.slurm.wait_for_jobs` rather than a loop of its own, so
    ``Config.job_timeout_seconds`` means the same thing here as on the other two waiting
    paths. A private loop would answer to nothing: a training job stuck in ``PD`` would block
    the pipeline with no diagnostic instead of failing with
    :class:`~chemrefine.errors.ThrottleTimeoutError`, and the knob would silently do nothing
    for this one step.
    """
    job_id = slurm.submit(script_path, dispatch=dispatch)
    logger.info("MLIP training submitted as job %s", job_id)
    slurm.wait_for_jobs(
        [job_id],
        poll_interval=poll_seconds,
        poll=slurm.poll_jobs,
        max_wait_seconds=max_wait_seconds,
    )
    logger.info("MLIP training job %s finished", job_id)
    return job_id


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_training(results: StepResults, ctx: StepContext) -> StepResults:
    """Train an MLIP on ``results`` and return the seed structures unchanged.

    The pipeline writes the training inputs + config + SLURM script
    into ``ctx.step_dir`` and submits one MACE training job. Downstream
    steps reuse the same seed structures as if the training step were a
    no-op.
    """
    train_path, test_path = prepare_inputs(results, ctx)
    config_path = write_training_config(train_path=train_path, test_path=test_path, ctx=ctx)
    script_path = write_training_slurm(ctx=ctx, config_path=config_path)
    submit_training(
        script_path=script_path,
        dispatch=ctx.dispatch,
        max_wait_seconds=ctx.job_timeout_seconds,
    )
    return results
