# Tutorials Overview

This section contains full examples of ChemRefine workflows:

- [Conformer sampling](conformer_sampling.md) — a GOAT search funnelled through a
  Boltzmann filter into DFT.
- [Transition states](ts_finding.md) — PES scan, saddle-point optimisation, and
  normal-mode sampling to confirm exactly one imaginary mode.
- [Host–guest docking](host_guest.md) — docking poses refined by an MLIP, validated by
  DFT, then explicitly solvated.
- [Training a potential](mlip_training.md) — generate reference data, fine-tune a model,
  and use it in a later step.
- [Redox properties](redox.md) — oxidised and reduced states through one pipeline.
- [Spin states](spin.md) — multiple multiplicities of the same system.

Each opens with the shipped config, included verbatim from `examples/tutorials/` — the
same files CI validates on every run. To have a workflow written *for* you instead, see
the [workflow builder](../workflow/builder.md) and the [AI agents](../workflow/agents.md).

