# Examples

Every example is a self-contained directory with the same shape: the pipeline
config (`input.yaml`) and the input structures (`step1.xyz` / a SMILES `.csv`)
at the root, and the method definitions (ORCA/`.py` step templates, SLURM
headers) in `templates/`. Run any of them with:

```bash
cd <example> && chemrefine run input.yaml
```

| Example | Molecule | Demonstrates |
| --- | --- | --- |
| [quickstart](quickstart/) | N,N-dimethylaniline | The annotated tour of the config schema: MLIP screen → DFT refine → high-level DFT with normal-mode sampling. |
| [tutorials/conformational_sampling](tutorials/conformational_sampling/) | Pd(PPh₃)₄ | GOAT conformer ensemble, MLIP refinement, DFT level-of-theory ladder. |
| [tutorials/transition_state](tutorials/transition_state/) | C₁₃H₁₅N₃ | Relaxed PES scan, `max` sampling, OptTS, TS-targeted normal-mode sampling. |
| [tutorials/host_guest](tutorials/host_guest/) | macrocycle + Cl⁻ | DOCKER guest docking, per-step charge override, SOLVATOR microsolvation. |
| [tutorials/mlip_training](tutorials/mlip_training/) | C₁₀H₂₂ | Dataset building with random NMS, MACE training, trained-model validation. |
| [tutorials/redox/amines](tutorials/redox/amines/) | 8 amines (SMILES) | CSV seeding and redox charge/multiplicity ladders per molecule. |
| [tutorials/redox/dimethylaniline](tutorials/redox/dimethylaniline/) | N,N-dimethylaniline | GOAT + Boltzmann filter, −1/0/+1 redox ladder on MLIP and DFT. |
| [tutorials/spin/benzophenone](tutorials/spin/benzophenone/) | benzophenone | Singlet/triplet gaps and TDDFT. |
| [tutorials/spin/heme_catalyst](tutorials/spin/heme_catalyst/) | Fe heme model | Quintet/triplet/singlet spin ladder on DFT and MLIP. |

The tutorial workflows are the ones from the ChemRefine paper; see the
[documentation](https://sterling-group.github.io/ChemRefine/) for walkthroughs.
