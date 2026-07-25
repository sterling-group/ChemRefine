---
name: Bug report
about: Report something that isn't working as expected
title: "[bug] "
labels: bug
---

**Describe the bug**
A clear description of what went wrong.

**To reproduce**
The command and the relevant `input.yaml` (trimmed):

```bash
chemrefine run input.yaml -v
```

```yaml
# input.yaml (trimmed to the failing step)
```

**Expected behaviour**
What you expected to happen.

**Environment**
- ChemRefine version (`chemrefine --version`):
- Python version:
- OS / cluster:
- Engine + backend (orca / mlip / pyscf, and which MLIP model):
- Running under SLURM or locally (`dispatch:`)?

**Logs / output**
Paste the error and relevant log lines (re-run with `-v` for debug logging).
`chemrefine run input.yaml --dry-run` output is often useful too.
