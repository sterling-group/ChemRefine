"""Run bounded quantum benchmarks with raw evidence and explicit unsupported cases.

Run from the repository root. Each case runs in its own process with a timeout;
warmups, setup, cold wall time and repeated operation timings remain distinct.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import cast

from quantum_validation import snapshot

ROOT = Path(__file__).resolve().parents[1]


def case_id(case: dict) -> str:
    """Use configuration content, independent of execution device, as the join key."""
    return hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:16]


def cases(suite: str) -> list[dict]:
    """Declare the finite Cartesian grids and explicit tutorial coverage."""
    result = []
    if suite in {"all", "primitives"}:
        for family, qubits, cores, precision in itertools.product(
            ("estimator", "sampler"), (4, 8, 12, 16, 20), (1, 8), ("single", "double")
        ):
            providers: tuple[str, ...] = (
                "statevector",
                "basic_backend",
                "aer_statevector",
                "aer_shots",
            )
            if family == "sampler":
                providers = ("statevector", "basic_backend", "aer")
            for provider in providers:
                if not provider.startswith("aer") and (precision == "single" or cores != 1):
                    continue
                methods: tuple[str, ...] = ("statevector",)
                if provider in {"aer", "aer_shots"}:
                    methods = (
                        "automatic",
                        "statevector",
                        "density_matrix",
                        "matrix_product_state",
                        "tensor_network",
                    )
                for method in methods:
                    for noisy in (False, True) if provider in {"aer", "aer_shots"} else (False,):
                        result.append(
                            {
                                "family": family,
                                "qubits": qubits,
                                "cores": cores,
                                "precision": precision,
                                "provider": provider,
                                "method": method,
                                "noisy": noisy,
                                "shots": 1024,
                                "layers": 4,
                                "seed": 19,
                            }
                        )
    if suite in {"all", "runtime"}:
        for family, qubits, implementation, mode in itertools.product(
            ("estimator", "sampler"), (2, 4), ("executor", "legacy_v2"), ("job", "session", "batch")
        ):
            for resilience in (0, 1, 2) if family == "estimator" else (0,):
                result.append(
                    {
                        "family": family,
                        "qubits": qubits,
                        "provider": "ibm_runtime",
                        "fake_backend": "FakeManilaV2",
                        "implementation": implementation,
                        "mode": mode,
                        "resilience_level": resilience,
                        "cores": 1,
                        "precision": "double",
                        "method": "automatic",
                        "noisy": True,
                        "shots": 1024,
                        "layers": 4,
                        "seed": 19,
                    }
                )
    if suite in {"all", "molecular", "regressions"}:
        from chemrefine.engines.qiskit.registry import ALGORITHMS, ANSATZE, MAPPERS, OPTIMIZERS

        for algorithm in sorted(ALGORITHMS.names()):
            for qubits in (4, 6):
                result.append(
                    {
                        "family": "molecular",
                        "algorithm": algorithm,
                        "qubits": qubits,
                        "mapper": "jordan_wigner",
                        "ansatz": "uccsd",
                        "optimizer": "slsqp",
                        "initial_state": "hartree_fock",
                        "initial_point": "zeros",
                        "seed": 19,
                    }
                )
        for mapper, ansatz, optimizer in itertools.product(
            sorted(MAPPERS.names()), sorted(ANSATZE.names()), sorted(OPTIMIZERS.names())
        ):
            result.append(
                {
                    "family": "molecular",
                    "algorithm": "vqe",
                    "qubits": 4,
                    "mapper": mapper,
                    "ansatz": ansatz,
                    "optimizer": optimizer,
                    "initial_state": "hartree_fock",
                    "initial_point": "zeros",
                    "seed": 19,
                }
            )
        for state, point in itertools.product(
            ("hartree_fock", "determinant", "zero"), ("zeros", "random")
        ):
            result.append(
                {
                    "family": "molecular",
                    "algorithm": "vqe",
                    "qubits": 4,
                    "mapper": "jordan_wigner",
                    "ansatz": "real_amplitudes",
                    "optimizer": "slsqp",
                    "initial_state": state,
                    "initial_point": point,
                    "seed": 19,
                }
            )
    if suite in {"all", "tutorials"}:
        for group in ("qiskit_experiment", "qiskit_cutting", "qiskit_resources"):
            for path in sorted((ROOT / "examples/tutorials" / group).glob("*.yaml")):
                result.append(
                    {"family": "tutorial", "path": str(path.relative_to(ROOT)), "seed": 19}
                )
    for case in result:
        if case.get("ansatz") == "ucc":
            case["ansatz_options"] = {"excitations": [[[0], [1]], [[2], [3]], [[0, 2], [1, 3]]]}
        if case.get("algorithm") == "skqd":
            case["algorithm_options"] = {"symmetrize_spin": True}
    if suite == "regressions":
        result = [
            case
            for case in result
            if (
                case["algorithm"] == "vqe"
                and (
                    case["ansatz"] == "ucc"
                    or (
                        case["ansatz"] == "excitation_preserving"
                        and case["mapper"] != "jordan_wigner"
                    )
                    or (
                        case["optimizer"] == "qnspsa"
                        and (
                            case["ansatz"] == "ucc_ranks"
                            or (case["ansatz"] == "uccsd" and case["mapper"] == "z2_tapered")
                        )
                    )
                )
            )
            or (case["algorithm"] == "skqd" and case["qubits"] == 6)
        ]
    return list({case_id(case): case for case in result}.values())


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a rectangular UTF-8 table; nested values remain JSON cells."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate only measured repeats, retaining failures instead of averaging them away."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    result = []
    for key, group in grouped.items():
        good = [r["seconds"] for r in group if r["status"] == "ok" and r["phase"] == "measure"]
        statuses = {r["status"] for r in group}
        result.append(
            {
                "case_id": key,
                "device": group[0]["device"],
                "family": group[0]["family"],
                "status": ",".join(sorted(statuses)),
                "samples": len(good),
                "median_seconds": statistics.median(good) if good else None,
                "min_seconds": min(good) if good else None,
                "max_seconds": max(good) if good else None,
                "stdev_seconds": statistics.stdev(good) if len(good) > 1 else None,
            }
        )
    return result


def worker(payload: dict) -> dict:
    """Prepare once and synchronously wait for every quantum job before timing ends."""
    import resource

    from quantum_workloads import Unsupported, prepare

    case = payload["case"]
    rows = []
    base = {
        **case,
        "case_id": case_id(case),
        "device": payload["device"],
        "requested_qubits": case.get("qubits"),
    }
    started = time.perf_counter()
    try:
        operation, setup = prepare(case, payload["device"])
        setup_seconds = time.perf_counter() - started
        for iteration in range(payload["warmups"] + payload["repeats"]):
            start = time.perf_counter()
            metrics = operation()
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    **base,
                    **setup,
                    **metrics,
                    "status": "ok",
                    "seconds": elapsed,
                    "setup_seconds": setup_seconds,
                    "iteration": iteration,
                    "phase": "warmup" if iteration < payload["warmups"] else "measure",
                    "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                }
            )
    except Exception as exc:
        rows.append(
            {
                **base,
                "status": "unsupported" if isinstance(exc, Unsupported) else "error",
                "phase": "failure",
                "iteration": len(rows),
                "seconds": None,
                "reason": str(exc),
                "exception": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        )
    from chemrefine.engines.qiskit.reporting import jsonable

    return cast("dict", jsonable({"rows": rows}))


def run_isolated_worker(command: list[str], *, cwd: Path, env: dict, log, timeout: float) -> int:
    """Reap the worker and kill its owned process group on completion or interruption."""
    with subprocess.Popen(  # noqa: S603 - fixed worker argv, no shell
        command,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ) as process:
        try:
            return process.wait(timeout=timeout)
        finally:
            # Compilation pools can outlive their parent; never kill unrelated host jobs.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main() -> int:
    """Run one isolated worker at a time and retain partial evidence after failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--suite",
        choices=("all", "primitives", "molecular", "tutorials", "regressions", "runtime"),
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--qiskit-processes",
        type=int,
        default=os.environ.get("QISKIT_NUM_PROCS", "1"),
        help="Bound transpiler workers independently of Aer threads (default: env or 1)",
    )
    parser.add_argument("--match", help="Only configurations containing this text in their JSON")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        payload = json.loads(args.worker.read_text())
        outcome = worker(payload)
        Path(payload["result"]).write_text(json.dumps(outcome, allow_nan=False) + "\n")
        return 0
    if (
        args.output is None
        or args.repeats < 1
        or args.warmups < 0
        or args.timeout <= 0
        or args.qiskit_processes < 1
    ):
        parser.error("require --output, repeats/processes >= 1, warmups >= 0 and timeout > 0")
    if os.name != "posix":
        parser.error("benchmark workers require POSIX process-group isolation")
    args.output.mkdir(parents=True, exist_ok=False)
    logs = args.output / "logs"
    logs.mkdir()
    selected = cases(args.suite)
    if args.match:
        selected = [case for case in selected if args.match in json.dumps(case, sort_keys=True)]
        if not selected:
            parser.error("--match selected no cases")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("limit must be positive")
        selected = selected[: args.limit]
    manifest = {
        **snapshot(),
        "command": sys.argv,
        "device": args.device,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "timeout_seconds": args.timeout,
        "qiskit_processes": args.qiskit_processes,
        "cases": [{"case_id": case_id(c), **c} for c in selected],
        "scope": "Finite declared matrix; not exhaustive over arbitrary options.",
        "timing": "operation wall clock including compilation; setup excluded; synchronous jobs",
    }
    manifest.setdefault("environment", {})["QISKIT_NUM_PROCS"] = str(args.qiskit_processes)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with (args.output / "samples.jsonl").open("w") as raw:
        for index, case in enumerate(selected):
            key = case_id(case)
            with tempfile.TemporaryDirectory(prefix="chemrefine-benchmark-") as temporary:
                request = Path(temporary) / "request.json"
                response = Path(temporary) / "result.json"
                request.write_text(
                    json.dumps(
                        {
                            "case": case,
                            "device": args.device,
                            "repeats": args.repeats,
                            "warmups": args.warmups,
                            "result": str(response),
                        }
                    )
                )
                env = dict(os.environ)
                env["CHEMREFINE_BENCHMARK_JOURNAL_DIR"] = str(
                    args.output.resolve() / "journals" / key
                )
                for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                    env[variable] = str(case.get("cores", 1))
                env["QISKIT_NUM_PROCS"] = str(args.qiskit_processes)
                start = time.perf_counter()
                failure = None
                with (logs / f"{key}.txt").open("w") as log:
                    try:
                        returncode = run_isolated_worker(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--worker",
                                str(request),
                            ],
                            cwd=ROOT,
                            env=env,
                            log=log,
                            timeout=args.timeout,
                        )
                        if returncode or not response.exists():
                            failure = f"worker exited {returncode}; see logs/{key}.txt"
                    except subprocess.TimeoutExpired:
                        failure = f"timeout after {args.timeout} seconds"
                if failure:
                    batch = [
                        {
                            **case,
                            "case_id": key,
                            "device": args.device,
                            "phase": "failure",
                            "status": "timeout" if failure.startswith("timeout") else "error",
                            "seconds": None,
                            "reason": failure,
                        }
                    ]
                else:
                    batch = json.loads(response.read_text())["rows"]
                for row in batch:
                    row["cold_process_seconds"] = time.perf_counter() - start
                    raw.write(json.dumps(row, allow_nan=False) + "\n")
                raw.flush()
                rows.extend(batch)
            if (index + 1) % 10 == 0 or index + 1 == len(selected):
                print(
                    f"{index + 1}/{len(selected)}: {case['family']} {batch[-1]['status']}",
                    flush=True,
                )
    write_csv(args.output / "samples.csv", rows)
    write_csv(args.output / "summary.csv", summarize(rows))
    return int(any(row["status"] in {"error", "timeout"} for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
