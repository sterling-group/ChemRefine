# Developer Guide

For contributors working on ChemRefine itself.

- [Contributing](contributing.md) — dev setup, the CI gates, conventions, releases.
- [Adding an Engine](adding-an-engine.md) — register a new calculation backend and
  where the code changes go.

The [Concepts](../concepts/index.md) section (especially
[Architecture & Code Flow](../concepts/architecture.md)) is the recommended
orientation before making changes.

## One thing to know about the test suite

ChemRefine holds 100% line and branch coverage, and the CI gate enforces it.
What that buys is precise: it proves every line ran, not that two components
agree with each other.

Covered defects live in the seams — an option knob read through a validated
model in one place and off the raw dict in another, with different defaults; a
value shell-quoted where it is executed but not where it is written to the
runlog. Both halves are exercised; the pair is not.

So when a change spans two modules, add an *invariant* test alongside the unit
tests: assert the two readers agree, across every engine rather than the one
you happened to touch. `tests/test_engines_invariants.py` collects them, and
[Contributing](contributing.md) has the details.
