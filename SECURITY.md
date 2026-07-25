# Security Policy

## Supported versions

Security fixes are applied to the latest released `2.0.x` line.

| Version | Supported |
| ------- | --------- |
| 2.0.x   | yes       |
| < 2.0   | no        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report privately via GitHub's
[**Report a vulnerability**](https://github.com/sterling-group/ChemRefine/security/advisories/new)
form (Security → Advisories). Include a description, the affected version, and
steps to reproduce. You can expect an acknowledgement within a few days.

## Scope

ChemRefine executes several of its inputs as code — SLURM headers, `step{N}.py`
and ORCA templates, and the `executables:` a config names. Running an untrusted
workflow is therefore equivalent to running an untrusted script, and is not in
itself a vulnerability. Only run projects you trust.

Escaping a boundary ChemRefine *does* defend **is** in scope — for example
reaching the ExtOpt server without its token, or executing code merely by
loading a config or a cached result. The
[Security & Trust Boundaries](https://sterling-group.github.io/ChemRefine/concepts/security/)
page describes those boundaries in detail.
