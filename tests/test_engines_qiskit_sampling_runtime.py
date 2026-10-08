"""Whole-experiment budgets and recovery use actual Runtime adapters without remote jobs."""

from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import SparsePauliOp

from chemrefine.engines.qiskit.context import SamplerResource
from chemrefine.engines.qiskit.journal import RequestJournal, read_journal
from chemrefine.engines.qiskit.measurement import MeasurementOptions, measure_observable
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.registry import SAMPLERS, ComponentSpec
from chemrefine.engines.qiskit.runtime import RuntimePrimitive
from chemrefine.engines.qiskit.runtime_options import RuntimeSamplerOptions
from chemrefine.engines.qiskit.sampling import SamplingSession
from chemrefine.engines.qiskit.shadows import FermionicShadowOptions, collect_fermionic_shadows
from chemrefine.errors import ConfigError

pytestmark = pytest.mark.filterwarnings(
    "ignore:The SamplerV2 class is deprecated:DeprecationWarning"
)


def _acquire(kind, selection):
    """Run real multi-request acquisition through the selected adapter."""
    preparation = QuantumCircuit(2)
    preparation.x(0)
    if kind == "shadows":
        result = collect_fermionic_shadows(
            preparation,
            selection,
            FermionicShadowOptions(
                ensemble="majorana_clifford", num_settings=3, shots_per_setting=4
            ),
        )
        return result.counts
    measurement = measure_observable(
        preparation,
        SparsePauliOp.from_list([("XI", 1), ("ZI", 0.3)]),
        sampler=selection,
        options=MeasurementOptions(grouping="none", shots=20, pilot_shots=4),
    )
    return tuple(group["counts"] for group in measurement["groups"])


def _install_adapter(monkeypatch, directory):
    """Keep actual PUBs, provider results and journaled IDs behind an in-memory transport."""
    jobs, built, closed, retrieved = {}, [], [], []
    backend = BasicSimulator()

    def build(*, options, cores, device):
        """Construct the real cumulative adapter once per acquisition."""
        assert cores == 1 and device == "cpu"
        primitive = StatevectorSampler(seed=np.random.default_rng(11))

        def submit(pubs):
            """Record actual numerical PrimitiveJobs; retrieval must never call this path."""
            assert not options.retrieve_job_ids
            job = primitive.run(pubs)
            jobs[job.job_id()] = job
            return job

        def retrieve(identifier):
            """Return the previously completed physical result without new sampling."""
            retrieved.append(identifier)
            return jobs[identifier]

        adapter = RuntimePrimitive(
            SimpleNamespace(run=submit),
            options,
            RequestJournal(directory),
            kind="sampler",
            backend=backend,
            service=SimpleNamespace(job=retrieve),
        )
        built.append(adapter)
        return SamplerResource(adapter, close=lambda: closed.append(True))

    monkeypatch.setitem(
        SAMPLERS._specs, "offline_runtime", ComponentSpec(RuntimeSamplerOptions, build)
    )
    return jobs, built, closed, retrieved


@pytest.mark.parametrize("kind", ["shadows", "measurement"])
@pytest.mark.parametrize("limit,value", [("max_jobs", 1), ("max_nominal_shots", 5)])
def test_runtime_limits_cover_all_settings_and_measurement_stages(
    tmp_path, monkeypatch, kind, limit, value
):
    """A second request cannot reset the first request's job or shot consumption."""
    jobs, built, closed, _ = _install_adapter(monkeypatch, tmp_path)
    selection = ComponentSelection(
        name="offline_runtime", options={"backend_name": "offline", limit: value}
    )
    with pytest.raises(ConfigError, match=limit):
        _acquire(kind, selection)
    assert len(jobs) == len(built) == 1 and closed == [True]
    assert built[0].nominal_shots == 4
    assert len(read_journal(tmp_path)) == 1


@pytest.mark.parametrize("kind,requests", [("shadows", 3), ("measurement", 4)])
def test_runtime_retrieval_cursor_advances_across_complete_acquisition(
    tmp_path, monkeypatch, kind, requests
):
    """Replay consumes each distinct job ID once and never falls back to fresh submission."""
    jobs, built, closed, retrieved = _install_adapter(monkeypatch, tmp_path)
    selection = ComponentSelection(name="offline_runtime", options={"backend_name": "offline"})
    expected = _acquire(kind, selection)
    identifiers = tuple(jobs)
    replay = ComponentSelection(
        name="offline_runtime", options={"backend_name": "offline", "retrieve_job_ids": identifiers}
    )
    assert _acquire(kind, replay) == expected
    assert len(identifiers) == requests
    assert retrieved == list(identifiers)
    assert len(built) == 2 and built[-1].retrieval_index == requests
    assert len(jobs) == requests and len(read_journal(tmp_path)) == requests
    assert closed == [True, True]
    truncated = ComponentSelection(
        name="offline_runtime",
        options={"backend_name": "offline", "retrieve_job_ids": identifiers[:1]},
    )
    with pytest.raises(ConfigError, match="exhausted"):
        _acquire(kind, truncated)
    assert len(jobs) == requests and closed == [True, True, True]


@pytest.mark.parametrize("implementation", ["executor", "legacy_v2"])
def test_real_runtime_fake_sampler_uses_reproducible_independent_request_seeds(
    tmp_path, implementation
):
    """Both released local SDK paths apply each seed through their public options model."""
    pytest.importorskip("qiskit_ibm_runtime")
    preparation = QuantumCircuit(1)
    preparation.h(0)
    batches = []
    for repeat in range(2):
        selection = ComponentSelection(
            name="ibm_runtime",
            options={
                "fake_backend": "FakeManilaV2",
                "implementation": implementation,
                "seed_simulator": 3,
                "seed_transpiler": 7,
                "journal_dir": str(tmp_path / str(repeat)),
            },
        )
        with SamplingSession(selection) as session:
            batches.append([session.sample(preparation, shots=128) for _ in range(3)])
    assert [batch.counts for batch in batches[0]] == [batch.counts for batch in batches[1]]
    seeds = [batch.metadata["sampling_seed"] for batch in batches[0]]
    assert len(set(seeds)) == 3
    assert len({tuple(sorted(batch.counts.items())) for batch in batches[0]}) > 1
    records = read_journal(tmp_path / "0")
    assert len(records) == 3 and len({record.request_digest for record in records}) == 3
