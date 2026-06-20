# User Guide

This guide takes you from a fresh install to a running multi-step refinement
workflow.

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **1 — Installation**

    ---

    Install ChemRefine, pick the MLIP / PySCF backends you need, and verify the
    CLI works.

    [:octicons-arrow-right-24: Install ChemRefine](installation.md)

-   :material-file-cog:{ .lg .middle } **2 — Configuration**

    ---

    Write the single YAML file that describes your pipeline: top-level defaults,
    the per-step engines and operations, and the survivor filters.

    [:octicons-arrow-right-24: Configuration reference](configuration.md)

-   :material-console:{ .lg .middle } **3 — CLI Reference**

    ---

    Drive a run with `run` / `resume` / `rerun` / `rerun-errors` /
    `rebuild-cache` / `rebuild-nms`, plus the global flags.

    [:octicons-arrow-right-24: CLI reference](cli.md)

-   :material-school:{ .lg .middle } **4 — Tutorials**

    ---

    Worked end-to-end examples: conformer sampling, TS finding, docking,
    MLIP training, redox, and spin.

    [:octicons-arrow-right-24: Run a tutorial](../tutorials/index.md)

</div>

Once you understand the moving parts, the [Concepts](../concepts/index.md)
section explains *how* ChemRefine works — the pipeline data flow, the
fingerprint cache, filtering, and normal-mode sampling.
