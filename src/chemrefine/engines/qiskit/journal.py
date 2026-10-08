"""Durable, credential-free records of provider submission intent and returned identifiers.

A journal is an audit trail, not an exactly-once submission service. A failed call
may have reached the provider without returning an identifier. Such an entry remains
explicitly ambiguous; neither reading the journal nor restarting submits anything.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from chemrefine.errors import ConfigError

JournalKind = Literal["estimator", "sampler", "noise_learner", "session", "batch"]
JournalState = Literal[
    "intent", "submitted", "submission_unknown", "completed", "result_unavailable"
]
ResubmissionPolicy = Literal["refuse_unresolved", "allow_unresolved"]


class RequestSummary(BaseModel):
    """Only execution facts are persisted; circuits, credentials and account names are excluded."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    kind: JournalKind
    implementation: Literal["executor", "legacy_v2"]
    backend: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.-]+$")
    pub_count: int = Field(0, ge=0)
    parameter_sets: int = Field(0, ge=0)
    nominal_shots: int = Field(0, ge=0)
    circuit_qubits: tuple[int, ...] = ()
    max_execution_time: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _widths(self) -> RequestSummary:
        """Reject malformed register widths without importing a provider."""
        if any(width < 0 for width in self.circuit_qubits):
            raise ValueError("circuit_qubits must be non-negative")
        return self


class JournalRecord(BaseModel):
    """One submission's state and optional provider identifier, using a stable schema."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary: RequestSummary
    state: JournalState = "intent"
    created_at: str
    updated_at: str
    job_id: str | None = Field(None, min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_.:-]+$")
    error_kind: str | None = Field(None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")

    @model_validator(mode="after")
    def _state_contract(self) -> JournalRecord:
        """Require an identifier for submitted/result states and none for ambiguous intent."""
        needs_id = self.state in {"submitted", "completed", "result_unavailable"}
        if needs_id != (self.job_id is not None):
            raise ValueError("journal state and provider identifier are inconsistent")
        if self.state in {"submission_unknown", "result_unavailable"} and self.error_kind is None:
            raise ValueError("journal error states require an exception class name")
        return self


def request_digest(payload: bytes) -> str:
    """Fingerprint an already canonical request; only this digest is persisted."""
    return hashlib.sha256(payload).hexdigest()


def read_journal(directory: str | Path, *, max_records: int = 10000) -> tuple[JournalRecord, ...]:
    """Validate local records only, with no service import, polling or submission."""
    if max_records < 1:
        raise ConfigError("provider journal max_records must be positive")
    paths = sorted(Path(directory).glob("*.json"))
    if len(paths) > max_records:
        raise ConfigError("provider journal exceeds max_records")
    records = []
    for path in paths:
        if path.is_symlink() or path.stat().st_size > 1048576:
            raise ConfigError("provider journal records must be regular files under 1 MiB")
        try:
            record = JournalRecord.model_validate_json(path.read_bytes())
        except (OSError, ValidationError) as exc:
            raise ConfigError(f"invalid provider journal record {path.name}") from exc
        if path.stem != record.request_id:
            raise ConfigError("provider journal filename does not match request_id")
        records.append(record)
    return tuple(records)


class RequestJournal:
    """Atomically replace bounded records, syncing contents and their directory to disk.

    A single owner updates each request; independent workers use distinct UUIDs.
    Explicit retrieval is a new action and never consumes an intent automatically.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        history_directories: tuple[str | Path, ...] = (),
        max_records: int = 10000,
    ) -> None:
        """Require an explicit absolute directory, such as the durable host job directory."""
        path = Path(directory).expanduser()
        if not path.is_absolute():
            raise ConfigError("provider journal directory must be absolute and durable")
        if max_records < 1 or len(history_directories) > 128:
            raise ConfigError(
                "provider journal requires positive max_records and at most 128 history directories"
            )
        if any(not Path(value).expanduser().is_absolute() for value in history_directories):
            raise ConfigError("provider journal history directories must be absolute")
        self.directory = path.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.history_directories = tuple(
            dict.fromkeys(
                Path(value).expanduser().resolve()
                for value in history_directories
                if Path(value).expanduser().resolve() != self.directory
            )
        )
        self.max_records = max_records
        self._lock = threading.RLock()
        self._owned_requests: set[str] = set()

    @contextmanager
    def _submission_claim(self) -> Iterator[None]:
        """Serialize scan-and-intent publication across owners of this journal directory.

        Exclusive file creation works on shared filesystems without advisory locking.
        A crash can leave the claim behind: fail closed until an operator verifies no
        writer remains and removes it. Historical directories are read-only and are
        expected to be quiescent; independent current directories are not coordinated.
        """
        claim = self.directory / ".submission.lock"
        try:
            stream = claim.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise ConfigError(
                "provider journal submission is claimed by another writer or an interrupted "
                f"process; inspect {claim} and remove it only after all writers have stopped"
            ) from exc
        try:
            with stream:
                stream.write("scan-and-intent publication in progress\n")
            yield
        finally:
            claim.unlink()

    def _records(self) -> tuple[JournalRecord, ...]:
        """Read a bounded current/history snapshot without making provider calls."""
        records: list[JournalRecord] = []
        for directory in (self.directory, *self.history_directories):
            records.extend(read_journal(directory, max_records=self.max_records))
            if len(records) > self.max_records:
                raise ConfigError("provider journal history exceeds max_records")
        return tuple(records)

    def _check_resubmission(self, records: tuple[JournalRecord, ...], digest: str) -> None:
        """Reject unresolved prior requests while allowing completed and live owned work."""
        completed = {record.request_id for record in records if record.state == "completed"}
        for record in records:
            if record.request_digest != digest or record.request_id in completed:
                continue
            if record.state == "submitted" and (
                record.request_id in self._owned_requests
                or record.summary.kind in {"session", "batch"}
            ):
                # An active adapter may legitimately submit repeated PUBs before
                # retrieving them. A known mode ID means mode creation succeeded.
                continue
            identifier = f"; known job ID {record.job_id!r}" if record.job_id else ""
            raise ConfigError(
                f"matching provider request remains {record.state}{identifier}; retrieve a known "
                "job explicitly or set resubmission_policy: allow_unresolved to acknowledge "
                "that a new submission may duplicate accepted work"
            )

    def _write(self, record: JournalRecord) -> None:
        """Flush the temporary file before atomic replacement and directory synchronization."""
        data = record.model_dump_json().encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(prefix=".request-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            Path(temporary).replace(self.directory / f"{record.request_id}.json")
            if os.name != "nt":
                directory_descriptor = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def begin(
        self,
        digest: str,
        summary: RequestSummary,
        *,
        resubmission_policy: ResubmissionPolicy = "refuse_unresolved",
    ) -> JournalRecord:
        """Check prior unresolved work and durably claim a new request before submission."""
        if resubmission_policy not in {"refuse_unresolved", "allow_unresolved"}:
            raise ConfigError("invalid provider resubmission_policy")
        now = datetime.now(UTC).isoformat()
        record = JournalRecord(
            request_id=uuid4().hex,
            request_digest=digest,
            summary=summary,
            created_at=now,
            updated_at=now,
        )
        with self._lock, self._submission_claim():
            records = self._records()
            if len(records) >= self.max_records:
                raise ConfigError("provider journal cannot add a request beyond max_records")
            if resubmission_policy == "refuse_unresolved":
                self._check_resubmission(records, digest)
            self._write(record)
            self._owned_requests.add(record.request_id)
        return record

    def update(self, record: JournalRecord, state: JournalState, **changes: Any) -> JournalRecord:
        """Validate each state transition, retaining the original request and identifier."""
        allowed = {
            "intent": {"submitted", "submission_unknown"},
            "submitted": {"completed", "result_unavailable"},
            "result_unavailable": {"completed", "result_unavailable"},
            "submission_unknown": set(),
            "completed": set(),
        }
        if state not in allowed[record.state] or set(changes) - {"job_id", "error_kind"}:
            raise ConfigError("invalid provider journal transition")
        if record.job_id is not None and changes.get("job_id", record.job_id) != record.job_id:
            raise ConfigError("provider journal identifier cannot change")
        updated = JournalRecord.model_validate(
            record.model_dump()
            | changes
            | {
                "state": state,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        with self._lock:
            self._write(updated)
        return updated

    def submit(
        self,
        digest: str,
        summary: RequestSummary,
        operation: Callable[[], Any],
        *,
        resubmission_policy: ResubmissionPolicy = "refuse_unresolved",
    ) -> JournaledJob:
        """Record IDs before returning control; failed calls remain explicitly ambiguous."""
        record = self.begin(digest, summary, resubmission_policy=resubmission_policy)
        try:
            job = operation()
            identifier = job.job_id()
            # Validate the identifier before mutating the durable record.
            checked = JournalRecord.model_validate(
                record.model_dump()
                | {
                    "state": "submitted",
                    "job_id": identifier,
                }
            )
        except BaseException as exc:
            self.update(record, "submission_unknown", error_kind=type(exc).__name__)
            raise
        try:
            record = self.update(record, "submitted", job_id=checked.job_id)
        except OSError as exc:
            raise ConfigError(
                f"provider returned job {identifier!r}, but its journal update failed; "
                "retain this ID for explicit recovery"
            ) from exc
        return JournaledJob(job, self, record)

    def matching_job(self, identifier: str, digest: str) -> JournalRecord:
        """Require an explicit known job and exactly matching request before retrieval."""
        records = self._records()
        identified = [record for record in records if record.job_id == identifier]
        if len({(record.request_id, record.request_digest) for record in identified}) > 1:
            raise ConfigError("provider job ID has conflicting journal requests")
        matches = [record for record in identified if record.request_digest == digest]
        if not matches:
            raise ConfigError("explicit provider job ID has no matching journal request digest")
        return max(matches, key=lambda record: record.updated_at)


class JournaledJob:
    """Preserve the provider result interface while recording result availability safely."""

    def __init__(self, job: Any, journal: RequestJournal, record: JournalRecord) -> None:
        """Retain a known provider job without polling it during construction."""
        if record.job_id is None:
            raise ConfigError("a journaled job requires a known provider identifier")
        self.job = job
        self.journal = journal
        self.record = record
        self._identifier = record.job_id

    def job_id(self) -> str:
        """Return the already journaled identifier without contacting the provider."""
        return self._identifier

    def result(self, *args: Any, **kwargs: Any) -> Any:
        """Fetch explicitly; failed retrieval retains the ID and may be tried again."""
        try:
            result = self.job.result(*args, **kwargs)
        except BaseException as exc:
            if self.record.state != "completed":
                self.record = self.journal.update(
                    self.record,
                    "result_unavailable",
                    error_kind=type(exc).__name__,
                )
            raise
        if self.record.state != "completed":
            self.record = self.journal.update(self.record, "completed", error_kind=None)
        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate optional explicit operations, such as status or cancellation, to the job."""
        return getattr(self.job, name)
