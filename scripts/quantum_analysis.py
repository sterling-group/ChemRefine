"""Consolidate completed quantum campaigns without confusing failures with timings."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from quantum_benchmarks import summarize, write_csv


def validate_samples(manifest: dict, rows: list[dict]) -> None:
    """Require unique, ordered iterations and truthful timing/failure records."""
    cases = manifest["cases"]
    ids = [case["case_id"] for case in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("manifest requires nonempty, unique case IDs")
    warmups, repeats = manifest["warmups"], manifest["repeats"]
    if (
        type(warmups) is not int
        or type(repeats) is not int
        or warmups < 0
        or repeats < 1
        or manifest["device"] not in {"cpu", "cuda"}
    ):
        raise ValueError("invalid campaign repetition or device settings")
    positions: Counter = Counter()
    failed = set()
    for row in rows:
        key = row["case_id"]
        if row["device"] != manifest["device"]:
            raise ValueError(f"{key}: sample device differs from manifest")
        if key in failed:
            raise ValueError(f"{key}: samples after terminal failure")
        iteration = positions[key]
        if row["status"] == "ok":
            if type(row.get("iteration")) is not int or row["iteration"] != iteration:
                raise ValueError(f"{key}: duplicate or out-of-order iteration")
            phase = "warmup" if iteration < warmups else "measure"
            if iteration >= warmups + repeats or row["phase"] != phase:
                raise ValueError(f"{key}: invalid iteration phase or repetition count")
            seconds = row.get("seconds")
            if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"{key}: successful timing must be finite and positive")
            positions[key] += 1
        elif row["status"] in {"error", "unsupported", "timeout"}:
            if row["phase"] != "failure" or row.get("seconds") is not None:
                raise ValueError(f"{key}: failure must have no timing")
            if "iteration" in row and (
                type(row["iteration"]) is not int or row["iteration"] != iteration
            ):
                raise ValueError(f"{key}: invalid failure iteration")
            failed.add(key)
        else:
            raise ValueError(f"{key}: unknown sample status")
    for key in ids:
        if key not in failed and positions[key] != warmups + repeats:
            raise ValueError(f"{key}: unexpected repetition count")


def load_campaign(path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Validate the manifest, repetition counts and completion before aggregating."""
    manifest = json.loads((path / "manifest.json").read_text())
    cases = {case["case_id"]: case for case in manifest["cases"]}
    rows = [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()]
    if {row["case_id"] for row in rows} != set(cases):
        raise ValueError(f"{path}: incomplete campaign or unknown case IDs")
    if not (path / "summary.csv").exists():
        raise ValueError(f"{path}: campaign has not finalized")
    validate_samples(manifest, rows)
    for row in rows:
        row["campaign"] = path.name
        row["requested_qubits"] = cases[row["case_id"]].get("qubits")
    aggregates = []
    for summary in summarize(rows):
        case = cases[summary["case_id"]]
        if summary["status"] == "ok" and summary["samples"] != manifest["repeats"]:
            raise ValueError(f"{path}: unexpected repetition count for {summary['case_id']}")
        measured: dict = next(
            (r for r in rows if r["case_id"] == summary["case_id"] and r["status"] == "ok"),
            {},
        )
        aggregates.append(
            {
                **case,
                **summary,
                "campaign": path.name,
                "requested_qubits": case.get("qubits"),
                "effective_qubits": measured.get("qubits"),
                "spatial_orbitals": measured.get("spatial_orbitals"),
                "electrons": measured.get("electrons"),
            }
        )
    return manifest, rows, aggregates


def provider_versions(manifest: dict) -> dict:
    """Identify the numerical provider stack, including the compiled Aer build."""
    versions = {
        key.lower(): value
        for key, value in manifest["packages"].items()
        if key.lower() in {"qiskit", "qiskit-aer", "qiskit-nature", "numpy", "scipy"}
    }
    versions["aer_build"] = next(
        (p["build"] for p in manifest.get("conda_packages", []) if p["name"] == "qiskit-aer"),
        "pip",
    )
    return versions


def comparisons(cpu: list[dict], gpu: list[dict]) -> list[dict]:
    """Join only fully successful cases; ratios above one mean GPU is faster."""
    left = {row["case_id"]: row for row in cpu if row["status"] == "ok"}
    result = []
    for row in gpu:
        key = row["case_id"]
        if key not in left or row["status"] != "ok":
            continue
        reference = left[key]
        result.append(
            {
                **{k: v for k, v in row.items() if not k.endswith("seconds")},
                "cpu_campaign": reference["campaign"],
                "gpu_campaign": row["campaign"],
                "cpu_median_seconds": reference["median_seconds"],
                "gpu_median_seconds": row["median_seconds"],
                "cpu_over_gpu": reference["median_seconds"] / row["median_seconds"],
            }
        )
    return result


def plot_scaling(output: Path, summaries: list[dict]) -> None:
    """Plot matching exact-estimator settings; matplotlib is an optional analysis tool."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in summaries
        if row.get("provider") == "aer_statevector"
        and row.get("precision") == "double"
        and row.get("cores") == 1
        and row["status"] == "ok"
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for campaign in sorted({row["campaign"] for row in selected}):
        values = sorted(
            (row for row in selected if row["campaign"] == campaign), key=lambda r: r["qubits"]
        )
        axis.errorbar(
            [row["qubits"] for row in values],
            [row["median_seconds"] for row in values],
            yerr=(
                [row["median_seconds"] - row["min_seconds"] for row in values],
                [row["max_seconds"] - row["median_seconds"] for row in values],
            ),
            marker="o",
            capsize=3,
            label=campaign,
        )
    axis.set(
        xlabel="Circuit qubits",
        ylabel="Median completed operation (seconds)",
        yscale="log",
        title="Exact Aer estimator: double precision, 1 CPU thread; median and range",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output / "estimator_scaling.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Export tidy data, diagnostics, provenance and explicitly selected comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--cpu", default="cpu-matched")
    parser.add_argument("--gpu", default="gpu0")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    output = args.root / "analysis"
    output.mkdir(exist_ok=True)
    manifests, aggregates = {}, {}
    all_rows, all_summaries, run_index = [], [], []
    for path in sorted(args.root.iterdir()):
        if not (path / "manifest.json").exists():
            continue
        manifest, rows, summaries = load_campaign(path)
        manifests[path.name], aggregates[path.name] = manifest, summaries
        all_rows.extend(rows)
        all_summaries.extend(summaries)
        run_index.append(
            {
                "campaign": path.name,
                "device": manifest["device"],
                "recorded_at": manifest["recorded_at"],
                "planned_cases": len(manifest["cases"]),
                "warmups": manifest["warmups"],
                "repeats": manifest["repeats"],
                "timeout_seconds": manifest["timeout_seconds"],
                "statuses": dict(Counter(row["status"] for row in summaries)),
                "provider_versions": provider_versions(manifest),
                "cuda_visible_devices": manifest["environment"]["CUDA_VISIBLE_DEVICES"],
            }
        )
    for name, records in (
        ("measurements", all_rows),
        ("case_summary", all_summaries),
        ("diagnostics", [r for r in all_rows if r["status"] != "ok"]),
        ("campaigns", run_index),
    ):
        write_csv(output / f"{name}.csv", records)
    (output / "campaigns.json").write_text(json.dumps(run_index, indent=2) + "\n")
    environments = {m["interpreter"]: m for m in manifests.values()}
    validations = []
    for path in sorted((args.root / "validation").glob("*.json")):
        record = json.loads(path.read_text())
        if "packages" in record:
            environments[record["interpreter"]] = record
        if "returncode" in record:
            validation = {
                "label": path.stem,
                "returncode": record["returncode"],
                "wall_seconds": record["wall_seconds"],
                "interpreter": record["interpreter"],
                "source_package": record.get("source_package"),
            }
            junit = path.with_suffix(".xml")
            if junit.exists():
                outcomes: Counter = Counter()
                for test in ET.parse(junit).iter("testcase"):  # noqa: S314 - local pytest artifacts
                    status = next(
                        (
                            tag
                            for tag in ("failure", "error", "skipped")
                            if test.find(tag) is not None
                        ),
                        "passed",
                    )
                    outcomes[status] += 1
                validation.update(outcomes)
            validations.append(validation)
    write_csv(output / "validation.csv", validations)
    write_csv(
        output / "installed_packages.csv",
        [
            {"interpreter": interpreter, "package": package, "version": version}
            for interpreter, record in sorted(environments.items())
            for package, version in sorted(record["packages"].items())
        ],
    )
    if args.cpu not in manifests or args.gpu not in manifests:
        raise ValueError("comparison campaigns are missing; provide --cpu and --gpu")
    if provider_versions(manifests[args.cpu]) != provider_versions(manifests[args.gpu]):
        raise ValueError("comparison provider versions/builds differ; use a matched CPU campaign")
    if manifests[args.cpu]["device"] != "cpu" or manifests[args.gpu]["device"] != "cuda":
        raise ValueError("comparison requires CPU and CUDA campaigns respectively")
    write_csv(output / "speedups.csv", comparisons(aggregates[args.cpu], aggregates[args.gpu]))
    if args.plots:
        plot_scaling(output, aggregates[args.cpu] + aggregates[args.gpu])


if __name__ == "__main__":
    main()
