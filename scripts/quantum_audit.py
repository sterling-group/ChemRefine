"""Export dependency changes, quantum import edges and the registered-name graph grid."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import itertools
import json
import subprocess
import time
import tomllib
from collections import Counter
from pathlib import Path

from packaging.requirements import Requirement
from pydantic import ValidationError
from quantum_validation import snapshot

ROOT = Path(__file__).resolve().parents[1]


def inventory(output: Path, baseline: str) -> None:
    """Compare declared requirements against a named revision and preserve import locations."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    old = tomllib.loads(
        subprocess.check_output(  # noqa: S603 - git revision read only
            ["git", "show", f"{baseline}:pyproject.toml"],  # noqa: S607 - repository inspection
            text=True,
        )
    )["project"]

    def requirements(metadata):
        groups = {"core": metadata["dependencies"], **metadata["optional-dependencies"]}
        return {
            (group, Requirement(raw).name.lower()): raw
            for group, values in groups.items()
            for raw in values
        }

    current, previous = requirements(project), requirements(old)
    previous_names = {name for _, name in previous}
    rows = [
        {
            "group": group,
            "package": name,
            "requirement": raw,
            "baseline_requirement": previous.get((group, name)),
            "change": "unchanged"
            if previous.get((group, name)) == raw
            else "modified"
            if (group, name) in previous
            else "added_to_group",
            "new_distribution": name not in previous_names,
        }
        for (group, name), raw in sorted(current.items())
    ]
    edges = []
    for path in sorted((ROOT / "src/chemrefine").rglob("*.py")):
        source = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
        for node in ast.walk(ast.parse(path.read_text())):
            imports = []
            if isinstance(node, ast.ImportFrom) and node.module:
                imports = [node.module]
                if node.module == "chemrefine.engines":
                    imports = [f"{node.module}.{alias.name}" for alias in node.names]
            elif isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            for target in imports:
                if source.startswith("chemrefine.engines.qiskit") or target.startswith(
                    "chemrefine.engines.qiskit"
                ):
                    edges.append(
                        {
                            "source": source,
                            "target": target,
                            "path": str(path.relative_to(ROOT)),
                            "line": node.lineno,
                            "internal": target.startswith("chemrefine."),
                            "external_quantum_boundary": source.startswith(
                                "chemrefine.engines.qiskit"
                            )
                            and target.startswith("chemrefine.")
                            and not target.startswith("chemrefine.engines.qiskit"),
                        }
                    )
    for name, records in (("dependencies", rows), ("import_edges", edges)):
        (output / f"{name}.json").write_text(json.dumps(records, indent=2) + "\n")
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (output / "baseline.json").write_text(
        json.dumps(
            {
                "baseline": baseline,
                "meaning": (
                    "Parent of the Qiskit estimator-provider introduction; all groups compared."
                ),
                "new_distributions": sorted({r["package"] for r in rows if r["new_distribution"]}),
                "quantum_internal_dependencies": sorted(
                    {r["target"] for r in edges if r["external_quantum_boundary"]}
                ),
            },
            indent=2,
        )
        + "\n"
    )


def graph_grid(output: Path) -> None:
    """Validate every registered-name combination on CPU and CUDA using fixed canonical knobs.

    This is schema/capability validation, not numerical execution. Numeric option
    spaces, SDK versions and molecule-dependent validity are separate dimensions.
    """
    from chemrefine.engines.qiskit.options import QiskitOptions
    from chemrefine.engines.qiskit.registry import REGISTRIES, validate_component_graph
    from chemrefine.errors import ConfigError

    categories = list(REGISTRIES)
    choices = [sorted(REGISTRIES[name].names()) for name in categories]
    canonical = {
        ("ansatz", "ucc"): {"excitations": [[[0], [1]], [[2], [3]], [[0, 2], [1, 3]]]},
        ("initial_state", "determinant"): {"alpha": [0], "beta": [0]},
        ("initial_point", "random"): {"seed": 19},
        ("estimator", "ibm_runtime"): {"fake_backend": "FakeManilaV2"},
        ("sampler", "ibm_runtime"): {"fake_backend": "FakeManilaV2"},
    }
    counts: Counter = Counter()
    started = time.perf_counter()
    with gzip.open(output / "component_graph.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*categories, "device", "status", "reason"])
        for names in itertools.product(*choices):
            raw = {
                category: {"name": name, "options": canonical.get((category, name), {})}
                for category, name in zip(categories, names, strict=True)
            }
            for device in ("cpu", "cuda"):
                status, reason = "accepted", ""
                try:
                    validate_component_graph(QiskitOptions.from_raw({**raw, "device": device}))
                except (ConfigError, ValidationError) as exc:
                    status, reason = "rejected", str(exc)
                counts[(device, status)] += 1
                writer.writerow([*names, device, status, reason])
    (output / "component_graph.json").write_text(
        json.dumps(
            {
                "environment": snapshot(),
                "categories": dict(zip(categories, choices, strict=True)),
                "canonical_options": [
                    {"category": key[0], "name": key[1], "options": value}
                    for key, value in canonical.items()
                ],
                "rows": sum(counts.values()),
                "counts": {f"{a}:{b}": n for (a, b), n in counts.items()},
                "seconds": time.perf_counter() - started,
                "scope": (
                    "All registered names, canonical knobs, both devices; "
                    "validation only, not execution."
                ),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Produce reusable structured audit evidence from the checkout being tested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", default="df3d37a^")
    parser.add_argument("--graph-grid", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory(args.output, args.baseline)
    if args.graph_grid:
        graph_grid(args.output)


if __name__ == "__main__":
    main()
