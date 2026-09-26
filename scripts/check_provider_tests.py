#!/usr/bin/env python3
"""Reject skipped, empty or failed numerical-provider JUnit reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree


def validate_report(report: Path, files: list[str]) -> None:
    """Require successful executed cases for every explicitly selected provider file."""
    # The trusted local pytest process writes this file within the CI job.
    cases = list(ElementTree.parse(report).getroot().iter("testcase"))  # noqa: S314
    if not cases or any(
        case.find(tag) is not None for case in cases for tag in ("skipped", "error", "failure")
    ):
        raise ValueError("provider tests must execute successfully without skips")
    modules = {part for case in cases for part in case.get("classname", "").split(".")}
    missing = [filename for filename in files if Path(filename).stem not in modules]
    if missing:
        raise ValueError(f"provider files have no executed cases: {', '.join(missing)}")


def main() -> None:
    """Check one provider job report using its declared test-file selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    try:
        validate_report(args.report, args.files)
    except (ValueError, OSError, ElementTree.ParseError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
