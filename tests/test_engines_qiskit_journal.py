"""Provider recovery records survive ambiguous failures without leaking payloads or retrying."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.journal import (
    JournaledJob,
    JournalRecord,
    RequestJournal,
    RequestSummary,
    read_journal,
    request_digest,
)
from chemrefine.errors import ConfigError


def summary():
    """Persist only allowed execution facts; no account/token/payload field exists."""
    return RequestSummary(
        kind="estimator",
        implementation="executor",
        backend="ibm_example",
        pub_count=1,
        nominal_shots=4096,
    )


def test_intent_precedes_network_and_id_precedes_result(tmp_path):
    store = RequestJournal(tmp_path)
    calls = []

    def submit():
        (record,) = read_journal(tmp_path)
        assert record.state == "intent" and record.job_id is None
        calls.append("submit")

        def result():
            (record,) = read_journal(tmp_path)
            assert record.state in {"submitted", "completed"} and record.job_id == "known_id"
            calls.append("result")
            return 3

        return SimpleNamespace(job_id=lambda: "known_id", result=result, status=lambda: "DONE")

    job = store.submit(request_digest(b"secret circuit content"), summary(), submit)
    assert calls == ["submit"]
    assert job.job_id() == "known_id"
    assert job.status() == "DONE"
    assert job.result() == 3
    assert read_journal(tmp_path)[0].state == "completed"
    assert b"secret" not in next(tmp_path.glob("*.json")).read_bytes()
    assert job.result() == 3
    assert len(read_journal(tmp_path)) == 1


def test_submission_timeout_is_ambiguous_and_reading_never_retries(tmp_path):
    store, calls = RequestJournal(tmp_path), []

    def timeout():
        calls.append(True)
        raise TimeoutError("contains-secret-token-that-must-not-be-stored")

    with pytest.raises(TimeoutError):
        store.submit(request_digest(b"request"), summary(), timeout)
    (record,) = read_journal(tmp_path)
    assert record.state == "submission_unknown" and record.error_kind == "TimeoutError"
    assert record.job_id is None
    assert "secret" not in next(tmp_path.glob("*.json")).read_text()
    assert read_journal(tmp_path) == (record,)
    assert calls == [True]
    with pytest.raises(ConfigError, match="transition"):
        store.update(record, "submitted", job_id="guess")


def test_retrieval_failure_preserves_id_and_explicit_result_can_retry(tmp_path):
    store = RequestJournal(tmp_path)
    attempts = []

    def result():
        attempts.append(True)
        if len(attempts) == 1:
            raise ConnectionError("no response")
        return 7

    job = store.submit(
        request_digest(b"r"),
        summary(),
        lambda: SimpleNamespace(job_id=lambda: "job_7", result=result),
    )
    with pytest.raises(ConnectionError):
        job.result()
    (record,) = read_journal(tmp_path)
    assert record.state == "result_unavailable" and record.job_id == "job_7"
    assert job.result() == 7
    assert read_journal(tmp_path)[0].state == "completed"
    job.job.result = lambda: (_ for _ in ()).throw(TimeoutError())
    with pytest.raises(TimeoutError):
        job.result()
    assert read_journal(tmp_path)[0].state == "completed"


def test_explicit_matching_rejects_other_input_or_unknown_id(tmp_path):
    store = RequestJournal(tmp_path)
    digest = request_digest(b"one")
    job = store.submit(digest, summary(), lambda: SimpleNamespace(job_id=lambda: "stored"))
    assert store.matching_job("stored", digest) == job.record
    for identifier, other in (("other", digest), ("stored", request_digest(b"two"))):
        with pytest.raises(ConfigError, match="matching journal request digest"):
            store.matching_job(identifier, other)
    with pytest.raises(ConfigError, match="identifier cannot change"):
        store.update(job.record, "completed", job_id="other")
    with pytest.raises(ConfigError, match="transition"):
        store.update(job.record, "completed", token="never accept arbitrary updates")


def test_read_validation_and_bounded_records(tmp_path):
    assert read_journal(tmp_path / "absent") == ()
    with pytest.raises(ConfigError, match="max_records"):
        read_journal(tmp_path, max_records=0)
    store = RequestJournal(tmp_path)
    first = store.begin(request_digest(b"one"), summary())
    store.begin(request_digest(b"two"), summary())
    with pytest.raises(ConfigError, match="max_records"):
        read_journal(tmp_path, max_records=1)
    path = tmp_path / f"{first.request_id}.json"
    original = path.read_bytes()
    path.write_text('{"invalid":true}')
    with pytest.raises(ConfigError, match="invalid provider journal"):
        read_journal(tmp_path)
    path.write_bytes(original)
    renamed = tmp_path / "mismatch.json"
    path.rename(renamed)
    with pytest.raises(ConfigError, match="filename"):
        read_journal(tmp_path)
    renamed.rename(path)
    other = tmp_path / "linked.json"
    other.symlink_to(path)
    with pytest.raises(ConfigError, match="regular files"):
        read_journal(tmp_path)
    other.unlink()
    path.write_bytes(b" " * 1048577)
    with pytest.raises(ConfigError, match="1 MiB"):
        read_journal(tmp_path)


def test_schema_rejects_credentials_bad_states_and_negative_widths(tmp_path):
    with pytest.raises(ValidationError):
        summary().model_validate(summary().model_dump() | {"token": "sensitive"})
    with pytest.raises(ValidationError, match="non-negative"):
        RequestSummary(**(summary().model_dump() | {"circuit_qubits": [-1]}))
    with pytest.raises(ConfigError, match="absolute"):
        RequestJournal("relative")
    store = RequestJournal(tmp_path)
    record = store.begin(request_digest(b"r"), summary())
    with pytest.raises(ValidationError, match="inconsistent"):
        JournalRecord(**(record.model_dump() | {"state": "submitted"}))
    with pytest.raises(ValidationError, match="exception class"):
        JournalRecord(**(record.model_dump() | {"state": "submission_unknown"}))
    with pytest.raises(ConfigError, match="known provider identifier"):
        JournaledJob(None, store, record)


def test_failed_atomic_replace_leaves_intent_and_no_temporary_file(tmp_path, monkeypatch):
    store = RequestJournal(tmp_path)
    record = store.begin(request_digest(b"r"), summary())

    def fail(_self, _target):
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        store.update(record, "submission_unknown", error_kind="TimeoutError")
    assert read_journal(tmp_path)[0].state == "intent"
    assert not list(tmp_path.glob(".request-*"))


def test_returned_job_id_is_in_error_when_post_submission_disk_write_fails(tmp_path, monkeypatch):
    store = RequestJournal(tmp_path)
    actual = store._write

    def fail_on_identifier(record):
        if record.job_id:
            raise OSError("disk full")
        actual(record)

    monkeypatch.setattr(store, "_write", fail_on_identifier)
    with pytest.raises(ConfigError, match="returned job 'accepted_id'"):
        store.submit(
            request_digest(b"r"), summary(), lambda: SimpleNamespace(job_id=lambda: "accepted_id")
        )
    assert read_journal(tmp_path)[0].state == "intent"


def test_invalid_returned_id_retains_ambiguous_record(tmp_path):
    store = RequestJournal(tmp_path)
    with pytest.raises(ValidationError):
        store.submit(
            request_digest(b"r"), summary(), lambda: SimpleNamespace(job_id=lambda: "../unsafe")
        )
    assert read_journal(tmp_path)[0].state == "submission_unknown"


def test_journal_uses_file_sync_on_platforms_without_directory_descriptors(tmp_path, monkeypatch):
    """The Windows path still flushes the file before its atomic replacement."""
    from chemrefine.engines.qiskit import journal

    calls = []
    original = journal.os

    def sync(descriptor):
        calls.append(descriptor)
        return original.fsync(descriptor)

    monkeypatch.setattr(
        journal, "os", SimpleNamespace(name="nt", fdopen=original.fdopen, fsync=sync)
    )
    record = RequestJournal(tmp_path).begin(request_digest(b"r"), summary())
    assert read_journal(tmp_path) == (record,)
    assert len(calls) == 1


def test_explicit_retrieval_finds_readonly_archived_attempts(tmp_path):
    archive = tmp_path / "attempt0" / "provider_jobs"
    previous = RequestJournal(archive)
    digest = request_digest(b"prior run")
    submitted = previous.submit(
        digest, summary(), lambda: SimpleNamespace(job_id=lambda: "recover")
    )
    archive_data = next(archive.glob("*.json")).read_bytes()
    current = tmp_path / "provider_jobs"
    fresh = RequestJournal(current, history_directories=(archive, current, archive))
    record = fresh.matching_job("recover", digest)
    assert record == submitted.record
    assert list(current.glob("*.json")) == []
    job = JournaledJob(SimpleNamespace(result=lambda: 8), fresh, record)
    assert job.result() == 8
    assert next(archive.glob("*.json")).read_bytes() == archive_data
    assert fresh.matching_job("recover", digest).state == "completed"
    with pytest.raises(ConfigError, match="history exceeds max_records"):
        RequestJournal(current, history_directories=(archive,), max_records=1).matching_job(
            "recover", digest
        )
    # Identical IDs with conflicting request digests must not select whichever happens to match.
    RequestJournal(current).submit(
        request_digest(b"different"), summary(), lambda: SimpleNamespace(job_id=lambda: "recover")
    )
    with pytest.raises(ConfigError, match="conflicting"):
        fresh.matching_job("recover", digest)


@pytest.mark.parametrize(
    "history,maximum,match",
    [
        (("relative",), 10, "absolute"),
        ((), 0, "positive"),
        (("/unused",) * 129, 10, "128 history"),
    ],
)
def test_history_directory_and_combined_record_budgets(tmp_path, history, maximum, match):
    with pytest.raises(ConfigError, match=match):
        RequestJournal(tmp_path, history_directories=history, max_records=maximum)
