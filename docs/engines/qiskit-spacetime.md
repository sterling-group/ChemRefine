# Experimental spacetime postselection

`spacetime_postselection` runs **endpoint coherent Pauli checks** around a bound
Clifford payload. Users choose signed Pauli checks; ChemRefine derives their
output conjugates, constructs the ancilla circuit, and keeps shots whose measured
syndromes are zero. The component requires a serialized **nonideal Aer noise
model**. It supports local noisy simulation, including supported Aer GPU methods.

This is the endpoint construction underlying coherent Pauli checks. It does not
implement Paulice's distributed low-weight check search, non-Markovian noise
learning, or hardware execution. IBM's [spacetime-code tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/spacetime-codes)
explains the distinction and the additional synthesis needed for distributed
checks. Its [non-Markovian postselection guide](https://quantum.cloud.ibm.com/docs/en/addons/qiskit-mitigation/guides/postselection-with-non-markovian-error-checks)
covers a separate hardware workflow.

## Scientific domain

For a payload unitary $U$, each selected Hermitian Pauli $P_j$ has a signed output
check $Q_j=UP_jU^\dagger$. Clifford conjugation makes every $Q_j$ another signed
Pauli. An ancilla in $|+\rangle$ controls $P_j$ before the payload and $Q_j$ after
it, then is measured in the X basis. The circuit builder implements negative
Pauli signs with a Z on the control ancilla; dropping that phase changes the
reported syndrome.

Input checks are inserted in the declared order and output checks in **reverse
order**. Thus even noncommuting checks cancel in the ideal circuit and preserve
an arbitrary input state. The construction does not require that the input be an
eigenstate of the checks. Optional preparation runs before the checks and is not
protected by them. Payloads with parameters, classical bits, measurements,
resets, delays, control flow, or unsupported non-Clifford operations are rejected.

A Pauli fault between the payload and output checks flips each syndrome whose
output check anticommutes with that fault. Errors during preparation, check gates,
and measurement can remain undetected or create incorrect syndromes. Additional
checks can decrease acceptance and introduce noise: improved accuracy is not
guaranteed. The estimates describe the selected noisy distribution, not an
unbiased reconstruction of the ideal state.

## YAML and retained artifacts

Run the complete example from its directory:

```bash
chemrefine run spacetime_postselection.yaml --dry-run
chemrefine run spacetime_postselection.yaml
```

The example lives in
`examples/tutorials/qiskit_experiment/spacetime_postselection.yaml`,
with `bell_clifford.qpy` and a fully serialized stochastic noise model.

The worker compiles the **whole checked circuit**, including measurements, through
the selected sampler. Data and check classical registers retain their meaning
after routing. At least one declared nonideal channel must match an operation
and its physical target qubits in the final compiled circuit; inactive gate or
qubit channels cannot label an effectively ideal run. Shot-aligned register bitstrings are paired before counting;
independent marginal counts would lose the syndrome/data correlation.

The manifest-backed artifact stores:

- Integer physical joint counts with syndrome and data bits, plus separate
  accepted data counts and rejected joint counts. None are signed or weighted.
- Packed bitstrings in declared register order, QPY bytes for the uncompiled
  checked payload, and signed input/output checks. The QPY excludes optional
  preparation and final measurement.
- Raw/accepted shot totals, acceptance rate, binomial plug-in standard error,
  Wilson confidence interval, and observed sampling overhead.
- Conditional means and standard errors for requested I/Z Pauli observables;
  zero acceptance produces null estimates and null overhead, and fewer than two
  accepted shots produces null conditional standard errors.
- The explicit noise model, sampler options, actual compiled circuit costs, and
  interpretation of the conditional estimates.

The Wilson interval stays nontrivial when all or none of the shots pass. Its
independent-shot assumption does not include correlated drift or uncertainty in
the supplied noise model. Conditional parity uncertainty includes ordinary shot
noise only. It does not propagate mitigation-model uncertainty. Signed PEC or
quasiprobability outputs are rejected by the count interface.

## Option knobs

Every public option below appears explicitly in the runnable example.

| Option | Default | Effect and domain |
| --- | --- | --- |
| `circuit_path` | required | One bound Clifford QPY payload; typed file dependency. |
| `preparation_path` | `null` | Optional bound, width-matched preparation QPY; zero state if absent. |
| `max_circuit_bytes` | 33554432 | Per-QPY input byte limit. |
| `sampler` | required | `aer` with a serialized nonideal `noise_model`; preserves the standard sampler's device, precision, method, seed and compilation controls. |
| `spacetime.checks` | required | Nonempty equal-width Pauli labels, optional `+`/`-`; identity-only checks rejected. |
| `spacetime.diagonal_observables` | `[]` | Signed I/Z strings evaluated on accepted data shots. |
| `spacetime.shots` | 10000 | Physical shot budget, from 1 to 10000000. |
| `spacetime.confidence` | 0.95 | Wilson acceptance confidence level strictly between zero and one. |
| `spacetime.max_qubits` | 128 | Data plus check ancilla allocation guard. |
| `spacetime.max_circuit_operations` | 100000 | Conservative construction guard plus actual prepared-circuit size check. |
| `spacetime.max_memory_mb` | 512 | Conservative shot-decoding memory guard; not a scheduler reservation. |

The engine's `max_output_bytes` guards native payload size before acquisition.
Scheduler grants control numerical worker threads and selected Aer CPU/GPU use.

## Python and verification

`build_spacetime_circuit` returns the checked unitary and signed conjugated checks
without sampling. `collect_spacetime_counts` selects a managed noisy sampler and
returns physical counts; `postselect_spacetime_counts` can reanalyze already
acquired counts without provider calls. Their options and results are exported
through `chemrefine.engines.qiskit.api`.

Tests exhaust all 24 single-qubit Clifford actions with signed noncommuting check
pairs. Independent dense matrices verify conjugation and the full checked unitary.
A complex arbitrary two-qubit input with three checks is tested against every
Pauli fault and its expected syndrome. Actual noisy Aer tests check register
alignment, resource cleanup, physical shot totals, artifact packing, and QPY
round trips.
