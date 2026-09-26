"""Explicit Runtime 0.50 primitives, bounded publications and durable job recovery.

Only component construction opens a service. Parsing options, reading journals and
loading result artifacts never constructs this provider adapter. The executor and
legacy V2 APIs are selected by explicit imports, insulating saved configurations from
future changes to the package's top-level aliases.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

from chemrefine.engines.qiskit.context import EstimatorResource, SamplerResource
from chemrefine.engines.qiskit.journal import JournaledJob, RequestJournal, RequestSummary
from chemrefine.engines.qiskit.runtime_options import (
    RuntimeEstimatorOptions,
    RuntimeSamplerOptions,
)
from chemrefine.errors import ConfigError

PrimitiveOptions = RuntimeEstimatorOptions | RuntimeSamplerOptions


def _backend_fingerprint(backend: Any) -> dict[str, Any]:
    """Identify the target and its instruction support without serializing service credentials."""
    target = getattr(backend, "target", None)
    instructions = []
    if target is not None:
        for name in sorted(target.operation_names):
            for qubits, _properties in sorted(
                target[name].items(), key=lambda entry: repr(entry[0])
            ):
                instructions.append(
                    {
                        "name": name,
                        "qubits": qubits,
                    }
                )
    return {
        "name": backend.name,
        "num_qubits": backend.num_qubits,
        "dt": getattr(target, "dt", None),
        "instructions": instructions,
    }


class _DigestWriter:
    """Hash QPY incrementally without retaining serialized circuits or their metadata."""

    def __init__(self, digest: Any, maximum: int) -> None:
        """Share one hash and byte budget across a complete submission."""
        self.digest, self.maximum, self.size = digest, maximum, 0

    def write(self, data: bytes) -> int:
        """Stop oversized request serialization before any provider submission."""
        self.size += len(data)
        if self.size > self.maximum:
            raise ConfigError("Runtime request exceeds max_request_bytes")
        self.digest.update(data)
        return len(data)

    def seekable(self) -> bool:
        """Use QPY's supported sequential output path without an in-memory byte buffer."""
        return False


def _circuit_digest(circuit: Any, stream: _DigestWriter) -> None:
    """Normalize irrelevant UUIDs/names while preserving the complete physical circuit."""
    from qiskit import qpy
    from qiskit.circuit import Parameter

    parameters = {
        item: Parameter(item.name, uuid=uuid5(NAMESPACE_URL, f"chemrefine/runtime/{i}/{item.name}"))
        for i, item in enumerate(circuit.parameters)
    }
    normalized = circuit.assign_parameters(parameters)
    normalized.name = "chemrefine_runtime_request"
    normalized.metadata = {}
    qpy.dump(normalized, stream)


def fingerprint_pubs(
    pubs: list[Any],
    *,
    options: dict[str, Any],
    maximum: int,
) -> str:
    """Hash effective execution options, compiled circuits, bindings and observables."""
    digest = hashlib.sha256()
    stream = _DigestWriter(digest, maximum)
    stream.write(json.dumps(options, sort_keys=True, allow_nan=False).encode())
    for pub in pubs:
        _circuit_digest(pub.circuit, stream)
        entry = {
            "parameters": pub.parameter_values.as_array(pub.circuit.parameters).tolist(),
            "precision": getattr(pub, "precision", None),
            "shots": getattr(pub, "shots", None),
        }
        if hasattr(pub, "observables"):
            entry["observables"] = pub.observables.tolist()
        stream.write(json.dumps(entry, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def sdk_options(options: PrimitiveOptions) -> dict[str, Any]:
    """Translate only supported typed controls into the selected released primitive schema."""
    twirling = options.twirling.model_dump(exclude_none=True)
    if options.implementation == "legacy_v2":
        twirling.pop("group")
    result: dict[str, Any] = {
        "max_execution_time": options.max_execution_time,
        "dynamical_decoupling": options.dynamical_decoupling.model_dump(),
        "twirling": twirling,
    }
    if options.seed_simulator is not None:
        result["simulator"] = {"seed_simulator": options.seed_simulator}
    if isinstance(options, RuntimeEstimatorOptions):
        trex = options.trex if options.trex is not None else options.resilience_level >= 1
        zne = (
            options.zne.enable if options.zne.enable is not None else options.resilience_level == 2
        )
        result.update(
            {
                "default_precision": options.default_precision,
                "resilience_level": options.resilience_level,
                "resilience": {
                    "measure_mitigation": trex,
                    "zne_mitigation": zne,
                    "pec_mitigation": options.pec.enable,
                    "zne": options.zne.model_dump(exclude={"enable"}),
                    "pec": options.pec.model_dump(exclude={"enable"}),
                },
            }
        )
        if options.default_shots is not None:
            result["default_shots"] = options.default_shots
    else:
        result["default_shots"] = options.default_shots
    if options.fake_backend and options.implementation == "legacy_v2":
        # The legacy local service warns about even disabled mitigation options.
        # Validators already reject requests that would enable these controls.
        for key in ("dynamical_decoupling", "twirling", "resilience", "resilience_level"):
            result.pop(key, None)
    return result


class RuntimeBatchJob:
    """A local aggregation of explicitly journaled provider jobs, preserving PUB order."""

    def __init__(self, jobs: list[JournaledJob]) -> None:
        """Keep job handles without polling or resubmitting."""
        self.jobs = tuple(jobs)

    @property
    def job_ids(self) -> tuple[str, ...]:
        """Expose every real provider ID, avoiding a synthetic identifier that resembles one."""
        return tuple(job.job_id() for job in self.jobs)

    def result(self) -> Any:
        """Gather PUB results in submission order and retain the provider metadata per job."""
        from qiskit.primitives.containers import PrimitiveResult

        values, metadata = [], []
        for job in self.jobs:
            result = job.result()
            values.extend(result)
            metadata.append(result.metadata)
        return PrimitiveResult(
            values, metadata={"provider_job_ids": self.job_ids, "jobs": metadata}
        )


class RuntimePrimitive:
    """V2-compatible PUB execution with explicit batching, calibration and recovery."""

    def __init__(
        self,
        primitive: Any,
        options: PrimitiveOptions,
        journal: RequestJournal,
        *,
        kind: Literal["estimator", "sampler"],
        backend: Any,
        service: Any = None,
        mode: Any = None,
    ) -> None:
        """Bind one owned primitive and resource budget to one provider resource lifetime."""
        if (kind == "estimator") != isinstance(options, RuntimeEstimatorOptions):
            raise ConfigError("Runtime primitive kind and typed options do not match")
        self.primitive, self.options, self.journal = primitive, options, journal
        self.kind, self.backend, self.service, self.mode = kind, backend, service, mode
        self.submissions = 0
        self.nominal_shots = 0
        self.retrieval_index = 0
        self.target_fingerprint = _backend_fingerprint(backend)

    def _submit(self, digest: str, summary: RequestSummary, operation: Any) -> JournaledJob:
        """Never retry a create call implicitly, and never submit when replay IDs are exhausted."""
        if self.submissions >= self.options.max_jobs:
            raise ConfigError("Runtime exceeds max_jobs")
        self.submissions += 1
        if self.options.retrieve_job_ids:
            if self.retrieval_index >= len(self.options.retrieve_job_ids):
                raise ConfigError(
                    "Runtime explicit retrieve_job_ids exhausted; no new job submitted"
                )
            identifier = self.options.retrieve_job_ids[self.retrieval_index]
            record = self.journal.matching_job(identifier, digest)
            self.retrieval_index += 1
            if self.service is None:
                raise ConfigError("Runtime retrieval requires a remote service")
            return JournaledJob(self.service.job(identifier), self.journal, record)
        return self.journal.submit(digest, summary, operation)

    def _summary(self, pubs: list[Any], nominal: int) -> RequestSummary:
        """Describe execution volume without persisting circuits, account names or raw inputs."""
        return RequestSummary(
            kind=self.kind,
            implementation=self.options.implementation,
            backend=self.backend.name,
            pub_count=len(pubs),
            parameter_sets=sum(pub.parameter_values.size for pub in pubs),
            nominal_shots=nominal,
            circuit_qubits=tuple(pub.circuit.num_qubits for pub in pubs),
            max_execution_time=self.options.max_execution_time,
        )

    def _learn_noise(self, pubs: list[Any], summary: RequestSummary) -> str | None:
        """Journal executor calibration and match each learned model to its actual layer."""
        options = self.options
        if not isinstance(options, RuntimeEstimatorOptions) or not options.noise_learning.enabled:
            return None
        from ibm_quantum_schemas.common import QpyModelV13ToV17
        from qiskit import QuantumCircuit
        from qiskit_ibm_runtime.noise_learner_v3 import NoiseLearnerV3

        layers = self.primitive.find_unique_layers(pubs, types="gates")
        if len(layers) > options.noise_learning.max_layers:
            raise ConfigError("Runtime noise learning exceeds max_layers")
        settings = options.noise_learning.model_dump(exclude={"enabled", "max_layers"})
        settings["max_execution_time"] = options.max_execution_time
        hasher = hashlib.sha256()
        stream = _DigestWriter(hasher, options.max_request_bytes)
        stream.write(
            json.dumps(settings | {"backend": self.target_fingerprint}, sort_keys=True).encode()
        )
        for layer in layers:
            circuit = QuantumCircuit(
                list(layer.qubits), list(layer.clbits), name="chemrefine_runtime_noise_layer"
            )
            circuit.append(layer)
            # Match the SDK's public serializer for its annotated BoxOp inputs;
            # plain QPY has no samplomatic annotation handlers by default.
            serialized = QpyModelV13ToV17.from_quantum_circuit(circuit, qpy_version=17)
            stream.write(serialized.model_dump_json().encode())
        digest = hasher.hexdigest()
        learning_summary = summary.model_copy(
            update={
                "kind": "noise_learner",
                "pub_count": len(layers),
                "nominal_shots": 0,
                "parameter_sets": 0,
                "circuit_qubits": tuple(len(item.qubits) for item in layers),
            }
        )
        if not layers:
            self.primitive.options.resilience.layer_noise_model = []
            return digest
        learner = NoiseLearnerV3(mode=self.mode or self.backend, options=settings)
        job = self._submit(digest, learning_summary, lambda: learner.run(layers))
        models = job.result().to_pauli_lindblad_maps()
        if len(models) != len(layers) or any(
            model.num_qubits != len(layer.qubits)
            for layer, model in zip(layers, models, strict=True)
        ):
            raise ConfigError("Runtime noise learning returned incompatible layer models")
        # The learned model is an input to the energy job. Fingerprint it without
        # persisting raw calibration data in the credential-free journal.
        try:
            serialized_models = json.dumps(
                [(model.num_qubits, model.to_sparse_list()) for model in models],
                sort_keys=True,
                allow_nan=False,
            ).encode()
        except ValueError as exc:
            raise ConfigError("Runtime noise learning returned nonfinite layer models") from exc
        stream.write(serialized_models)
        self.primitive.options.resilience.layer_noise_model = list(zip(layers, models, strict=True))
        return hasher.hexdigest()

    def run(
        self, pubs: Iterable[Any], *, precision: float | None = None, shots: int | None = None
    ) -> RuntimeBatchJob:
        """Coerce and bound every publication before creating the first provider job."""
        from qiskit.primitives.containers import EstimatorPub, SamplerPub

        if self.kind == "estimator" and shots is not None:
            raise ConfigError("Runtime estimator accepts precision, not sampler shots")
        if self.kind == "sampler" and precision is not None:
            raise ConfigError("Runtime sampler accepts shots, not estimator precision")
        if precision is not None and (not math.isfinite(precision) or precision <= 0):
            raise ConfigError("Runtime precision must be finite and strictly positive")
        if shots is not None and (
            isinstance(shots, bool) or not isinstance(shots, int) or shots < 1
        ):
            raise ConfigError("Runtime shots must be a positive integer")
        options = self.options
        coerced, nominal = [], 0
        pub_shots: list[int] = []
        for pub in pubs:
            if isinstance(options, RuntimeEstimatorOptions):
                value = EstimatorPub.coerce(pub, precision)
                if value.precision is not None and (
                    not math.isfinite(value.precision) or value.precision <= 0
                ):
                    raise ConfigError("Runtime PUB precision must be finite and strictly positive")
                effective_shots = (
                    math.ceil(1 / value.precision**2)
                    if value.precision is not None
                    else options.default_shots or math.ceil(1 / options.default_precision**2)
                )
            else:
                value = SamplerPub.coerce(
                    pub, shots if shots is not None else options.default_shots
                )
                effective_shots = value.shots
            if value.parameter_values.size > options.max_parameter_sets:
                raise ConfigError("Runtime PUB exceeds max_parameter_sets")
            nominal += value.size * effective_shots
            pub_shots.append(value.size * effective_shots)
            coerced.append(value)
            if len(coerced) > options.max_pubs_per_job * (options.max_jobs - self.submissions):
                raise ConfigError("Runtime publication batch exceeds remaining max_jobs")
        if not coerced:
            raise ConfigError("Runtime needs at least one publication")
        if nominal + self.nominal_shots > options.max_nominal_shots:
            raise ConfigError("Runtime exceeds max_nominal_shots")
        self.nominal_shots += nominal
        jobs = []
        settings = sdk_options(options) | {
            "implementation": options.implementation,
            "kind": self.kind,
            "backend": self.target_fingerprint,
        }
        chunks: list[list[Any]] = []
        chunk_shots: list[int] = []
        # Runtime's executor requires one precision/shots setting per job. Split
        # contiguous groups without reordering PUBs or changing their requested cost.
        setting = "precision" if self.kind == "estimator" else "shots"
        for value, count in zip(coerced, pub_shots, strict=True):
            if (
                not chunks
                or len(chunks[-1]) == options.max_pubs_per_job
                or getattr(value, setting) != getattr(chunks[-1][0], setting)
            ):
                chunks.append([])
                chunk_shots.append(0)
            chunks[-1].append(value)
            chunk_shots[-1] += count
        factor = (
            2
            if isinstance(options, RuntimeEstimatorOptions) and options.noise_learning.enabled
            else 1
        )
        if factor * len(chunks) > options.max_jobs - self.submissions:
            raise ConfigError("Runtime job batches/calibration exceed remaining max_jobs")
        # Validate serialization of all chunks before submitting any of them.
        digests = [
            fingerprint_pubs(chunk, options=settings, maximum=options.max_request_bytes)
            for chunk in chunks
        ]
        for chunk, digest, count in zip(chunks, digests, chunk_shots, strict=True):
            summary = self._summary(chunk, count)
            learned_digest = self._learn_noise(chunk, summary)
            if learned_digest is not None:
                digest = hashlib.sha256((digest + learned_digest).encode()).hexdigest()
            jobs.append(
                self._submit(digest, summary, lambda chunk=chunk: self.primitive.run(chunk))
            )
        return RuntimeBatchJob(jobs)


def build_runtime_resource(
    options: PrimitiveOptions,
    *,
    kind: Literal["estimator", "sampler"],
    device: str,
) -> EstimatorResource | SamplerResource:
    """Open the explicitly selected release API and close only modes owned by this resource."""
    if device != "cpu":
        raise ConfigError("IBM Runtime is a remote/local provider; device must be cpu")
    if (kind == "estimator") != isinstance(options, RuntimeEstimatorOptions):
        raise ConfigError("Runtime primitive kind and typed options do not match")
    directory = options.journal_dir or os.environ.get("CHEMREFINE_PROVIDER_JOURNAL_DIR")
    if directory is None:
        raise ConfigError("Runtime requires journal_dir or durable CHEMREFINE_PROVIDER_JOURNAL_DIR")
    history: list[str | Path] = list(options.journal_history_dirs)
    if options.journal_dir is None:
        # ChemRefine archives prior attempts before starting a fresh worker.
        # Search only the injected job's own archives, never an unrelated root.
        history.extend(sorted(Path(directory).parent.glob("attempt*/provider_jobs")))
    journal = RequestJournal(
        directory,
        history_directories=tuple(history),
        max_records=options.max_journal_records,
    )
    if not version("qiskit-ibm-runtime").startswith("0.50."):
        raise ConfigError(
            "Runtime adapter requires the explicitly supported qiskit-ibm-runtime0.50"
        )
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import Batch, QiskitRuntimeService, Session

    service = None
    if options.fake_backend:
        from qiskit.providers import BackendV2
        from qiskit_ibm_runtime import fake_provider

        factory = getattr(fake_provider, options.fake_backend, None)
        if not isinstance(factory, type) or not issubclass(factory, BackendV2):
            raise ConfigError(f"unknown public Runtime fake backend {options.fake_backend!r}")
        backend = factory()
    else:
        service = QiskitRuntimeService(
            **{
                key: value
                for key, value in {
                    "name": options.account_name,
                    "channel": options.channel,
                    "instance": options.instance,
                }.items()
                if value is not None
            }
        )
        backend = service.backend(options.backend_name)
    mode = None
    try:
        if options.mode != "job":
            mode_cls = Session if options.mode == "session" else Batch
            if options.mode_id:
                mode = mode_cls.from_id(options.mode_id, service=service)
            elif options.fake_backend:
                mode = mode_cls(backend=backend, max_time=options.max_time)
            else:
                # Session creation is a provider operation too. Its identifier is
                # recorded before a primitive is constructed or a quantum job starts.
                digest = hashlib.sha256(
                    json.dumps(
                        {
                            "mode": options.mode,
                            "backend": backend.name,
                            "max_time": options.max_time,
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                summary = RequestSummary(
                    kind=options.mode,
                    implementation=options.implementation,
                    backend=backend.name,
                    max_execution_time=options.max_execution_time,
                )
                record = journal.begin(digest, summary)
                try:
                    mode = mode_cls(backend=backend, max_time=options.max_time)
                except BaseException as exc:
                    journal.update(record, "submission_unknown", error_kind=type(exc).__name__)
                    raise
                identifier = mode.session_id
                if identifier is None:
                    journal.update(record, "submission_unknown", error_kind="MissingIdentifier")
                    raise ConfigError("Runtime session creation did not return an identifier")
                try:
                    journal.update(record, "submitted", job_id=identifier)
                except OSError as exc:
                    raise ConfigError(
                        f"Runtime mode {identifier!r} created but journal update failed"
                    ) from exc
        if options.implementation == "executor":
            if kind == "estimator":
                from qiskit_ibm_runtime.executor_estimator import Estimator as Primitive
            else:
                from qiskit_ibm_runtime.executor_sampler import Sampler as Primitive
        elif kind == "estimator":
            from qiskit_ibm_runtime.estimator import EstimatorV2 as Primitive
        else:
            from qiskit_ibm_runtime.sampler import SamplerV2 as Primitive
        primitive = Primitive(mode=mode or backend, options=sdk_options(options))
        transpiler = generate_preset_pass_manager(
            backend=backend,
            optimization_level=options.optimization_level,
            seed_transpiler=options.seed_transpiler,
            initial_layout=options.initial_layout,
        )
        resource_type = EstimatorResource if kind == "estimator" else SamplerResource
        from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2

        class JournalEstimator(RuntimePrimitive, BaseEstimatorV2):  # type: ignore[misc]
            """Expose the real V2 type contract with durable submission behavior."""

        class JournalSampler(RuntimePrimitive, BaseSamplerV2):  # type: ignore[misc]
            """Retain sampler isinstance checks used by ComputeUncompute and QNSPSA."""

        adapter = JournalEstimator if kind == "estimator" else JournalSampler
        return resource_type(
            adapter(
                primitive, options, journal, kind=kind, backend=backend, service=service, mode=mode
            ),
            close=mode.close if mode is not None and options.close_mode else lambda: None,
            transpiler=transpiler,
        )
    except BaseException:
        if mode is not None and options.close_mode:
            mode.close()
        raise
