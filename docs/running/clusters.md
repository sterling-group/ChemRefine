# Local & cluster (SLURM)

The same config runs both ways. `dispatch: auto` — the default — submits through `sbatch`
when it is on `PATH` and runs the generated scripts locally with `bash` otherwise, so
moving a workflow from a laptop to a cluster changes nothing in the YAML.

Set `dispatch: local` to force the local runner even where an `sbatch` binary exists (a
workstation with SLURM client tools but no reachable cluster), or `dispatch: slurm` to
fail fast when `sbatch` is missing rather than quietly running everything locally.

## The core budget is not a job count

`max_cores` is a **total CPU budget** the throttler enforces across concurrent jobs, not
a limit on how many run at once. Each job is charged what its step declares — ORCA's
`%pal` from the template, Q-Chem's `cores` from the YAML — and the scheduler starts as
many as fit. A step with four 8-core jobs and `max_cores: 16` runs two at a time.
`max_gpus` works the same way for `device: cuda` steps; left unset it auto-resolves to
unlimited under SLURM (the scheduler places GPUs via `--gres`) and to the detected device
count locally.

## Your login shell never matters

ChemRefine does not run anything through it:

- **Locally**, every job script is launched as `bash <script>` — an explicit interpreter,
  not `$SHELL`.
- **On SLURM**, the submitted script's first line is `#!/bin/bash`, written by ChemRefine
  — `sbatch` honours the shebang regardless of the shell your account uses.

A cluster whose users live in `zsh`, `fish`, or anything else is fine. The one
requirement: **`bash` must exist on the compute nodes** (universal on Linux clusters).
Your SLURM header files (`*.slurm.header`) contribute `#SBATCH` directives and
environment lines; they do not need their own shebang.

## Provision on a login node, run anywhere

A managed backend environment is a plain directory of files, so offline compute nodes just
execute its `python` — but building one needs the internet, which compute nodes usually
lack. Do it once on a login node.

Environments live under `$CHEMREFINE_HOME` (default: alongside the install when writable,
else `~/.chemrefine/<interpreter tag>` — tagged because `$HOME` is routinely shared
across clusters, and two machines on different Pythons must not resolve each other's
envs). Point it at a project or shared filesystem to share provisioned
backends across machines:

```bash
export CHEMREFINE_HOME=/projects/mygroup/chemrefine   # optional; put it on shared storage
chemrefine backends install mlip-mace mlip-fairchem   # once, on the login node
```

The [workflow builder](../workflow/builder.md#from-a-cluster) belongs on the login node
too: `chemrefine gui` there prints the SSH forwarding recipe that puts it in the browser
on your own machine.

## Job arrays

`slurm_array: true` submits each step as SLURM job array(s) instead of one job per
structure — one queue entry for a step with 200 conformers rather than 200. It is ignored
when running locally. The throttle, the cache and the failure ledger behave identically
either way.

## Q-Chem on a cluster

A threaded Q-Chem job cannot span nodes, so carry `#SBATCH --nodes=1` in the header your
`qchem` steps use. `QCSCRATCH` needs no line anywhere — ChemRefine points it at the job's
own scratch directory. The MPI install facts (`QCRSH`/`QCMPI` exports, MPI module loads)
belong in the header too, beside the other machine environment.

## If a cluster run misbehaves

| Symptom | Likely cause |
|---------|--------------|
| `sbatch: command not found` | Either SLURM isn't installed locally — run the generated `.slurm` script with `bash` instead — or activate the cluster's SLURM module. |
| `sbatch failed …` on a machine that isn't a cluster | An `sbatch` binary on `PATH` made auto-detection pick SLURM; set `dispatch: local` to force the local runner. |

Everything else is in [When a run fails](when-a-run-fails.md).
