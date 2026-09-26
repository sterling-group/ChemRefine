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
| [first_run](first_run/) | ethylene glycol | The README quickstart: an xTB screen (ORCA's bundled GFN2-xTB) and a DFT refinement — runs on a base install plus ORCA, in minutes. |
| [schema_tour](schema_tour/) | N,N-dimethylaniline | The annotated tour of the config schema: MLIP screen → DFT refine → high-level DFT with normal-mode sampling. |
| [tutorials/conformational_sampling](tutorials/conformational_sampling/) | Pd(PPh₃)₄ | GOAT conformer ensemble, MLIP refinement, DFT level-of-theory ladder. |
| [tutorials/transition_state](tutorials/transition_state/) | C₁₃H₁₅N₃ | Relaxed PES scan, `max` sampling, OptTS, TS-targeted normal-mode sampling. |
| [tutorials/host_guest](tutorials/host_guest/) | macrocycle + Cl⁻ | DOCKER guest docking, per-step charge override, SOLVATOR microsolvation. |
| [tutorials/mlip_training](tutorials/mlip_training/) | C₁₀H₂₂ | Dataset building with random NMS, MACE training, trained-model validation. |
| [tutorials/qiskit_sp](tutorials/qiskit_sp/) | H₂ | Modular Qiskit Nature single point with an active space, UCCSD, and VQE; switchable in YAML to exact or ADAPT-VQE and to reference statevector, lightweight shots, Aer statevector, or Aer finite-shot estimators. |
| [tutorials/qiskit_experiment](tutorials/qiskit_experiment/) | Quantum artifacts | Lattice and variational dynamics, local encodings, measurements, fermionic shadows and double-factorized trajectories with portable NPZ output; input structures pass through unchanged. |
| [tutorials/qiskit_cutting](tutorials/qiskit_cutting/) | Quantum circuits | Manual gate/wire cuts, partitions and automated width-constrained planning with signed reconstruction. |
| [tutorials/qiskit_resources](tutorials/qiskit_resources/) | Quantum resource models | Pauli-LCU/QPE, validated DF/THC costs and explicit surface-code/factory assumptions. |
| [tutorials/fairchem_finetune](tutorials/fairchem_finetune/) | water conformers | FAIRChem (UMA) fine-tuning: label with UMA, fine-tune via fairchem's own recipe, run the produced checkpoint. |
| [tutorials/pyscf_refine](tutorials/pyscf_refine/) | water | Direct-PySCF single-point refinement through a `step1.py` script template. |
| [tutorials/redox/amines](tutorials/redox/amines/) | 8 amines (SMILES) | CSV seeding and redox charge/multiplicity ladders per molecule. |
| [tutorials/redox/dimethylaniline](tutorials/redox/dimethylaniline/) | N,N-dimethylaniline | GOAT + Boltzmann filter, −1/0/+1 redox ladder on MLIP and DFT. |
| [tutorials/spin/benzophenone](tutorials/spin/benzophenone/) | benzophenone | Singlet/triplet gaps and TDDFT. |
| [tutorials/spin/heme_catalyst](tutorials/spin/heme_catalyst/) | Fe heme model | Quintet/triplet/singlet spin ladder on DFT and MLIP. |

The ChemRefine paper's workflows sit in `tutorials/`, beside examples added
since; see the
[documentation](https://sterling-group.github.io/ChemRefine/) for walkthroughs.
