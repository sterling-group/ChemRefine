"""Every full config the docs show must validate against the real schema.

The README quickstart and each tutorial open with a complete YAML config, and nothing ran
them: a schema change that renamed a field would leave every example teaching the old
spelling, discovered by the next new user as a validation error on their first run. Any
``yaml`` fence whose text contains ``steps:`` is a full config by this repo's convention,
so each must parse and pass :class:`chemrefine.config.Config` — the same validation a real
run applies first.

The count is deliberately not written down here: a page that pulls its config in with
``pymdownx.snippets`` contributes no fence of its own (the file it includes is validated
by ``test_examples`` instead), so a number in this docstring would be one more fact to
keep in sync. ``test_the_scanner_still_finds_the_readme_quickstart`` is what stops the
scanner silently matching nothing.

Schema only, deliberately: executing the examples end-to-end needs ORCA and an MLIP
stack, which is what the recorded e2e tier covers. ``engine:`` names are free-form at
load time (the registry lookup happens later), so this holds regardless of which extras
are installed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from chemrefine.config import Config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_YAML_FENCE_RE = re.compile(r"```yaml\n(.*?)```", re.DOTALL)


def _config_blocks() -> list[object]:
    """Every ``steps:``-carrying YAML block, id'd by file and block number.

    ``docs/`` ships in the repository but not in the sdist, so a distro packager's run
    validates the README's block alone rather than failing on a directory that was never
    promised to be there.
    """
    blocks: list[object] = []
    for path in [_REPO_ROOT / "README.md", *sorted((_REPO_ROOT / "docs").rglob("*.md"))]:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for n, match in enumerate(_YAML_FENCE_RE.finditer(text), 1):
            if "steps:" in match.group(1):
                blocks.append(
                    pytest.param(match.group(1), id=f"{path.relative_to(_REPO_ROOT)}:{n}")
                )
    return blocks


def test_the_scanner_still_finds_the_readme_quickstart():
    """Guards the fence regex itself: a scanner that finds nothing passes everything."""
    assert any("README.md" in str(getattr(p, "id", "")) for p in _config_blocks())


@pytest.mark.parametrize("block", _config_blocks())
def test_the_documented_config_validates(block: str):
    raw = yaml.safe_load(block)
    assert isinstance(raw, dict), "a steps: block must be a mapping — is the fence a fragment?"
    Config(**raw)


# ---------------------------------------------------------------------------
# The engine guide's Python — the one artefact a new engine author copies
# ---------------------------------------------------------------------------

_PY_FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_ENGINE_GUIDE = _REPO_ROOT / "docs" / "developer" / "adding-an-engine.md"


def _engine_guide_python() -> list[str]:
    """The engine guide's ``python`` fences that are meant to be code.

    A fence carrying an ``engines/<name>/`` placeholder is a shape, not a snippet — it names
    the file you create rather than one that exists — so it is not held to compiling. Every
    other fence is the worked engine, which is copied verbatim by whoever writes the next one.
    """
    if not _ENGINE_GUIDE.is_file():  # pragma: no cover - docs/ is absent from the sdist
        pytest.skip("docs/ not shipped in this tree")
    fences = _PY_FENCE_RE.findall(_ENGINE_GUIDE.read_text(encoding="utf-8"))
    return [f for f in fences if "<name>" not in f]


def test_the_engine_guides_python_parses():
    """A worked example that is not Python is not an example."""
    fences = _engine_guide_python()
    assert any("class DemoqmEngine" in f for f in fences), (
        "the scanner found no worked engine — have the fences been renamed?"
    )
    for fence in fences:
        compile(fence, str(_ENGINE_GUIDE), "exec")


@pytest.mark.parametrize(
    ("method", "returns"),
    [("build_input", "None"), ("run_block", "RunBlock"), ("parse_one", "list[ParsedResult]")],
)
def test_the_engine_guides_primitives_return_what_the_base_declares(method: str, returns: str):
    """The guide claims "every signature matches the real base". This is that claim, checked.

    ``run_block`` is why: it returned a bare ``str`` for four weeks after ``RunBlock`` landed,
    through five edits of this page — one of them titled "eleven things the documentation said
    that were not true". Copied, it fails with ``AttributeError: 'str' object has no attribute
    'cleanup'`` while the first job script is assembled. Nothing type-checked or executed the
    page's Python, so nothing could have said so.
    """
    import inspect

    from chemrefine.engines._job import JobEngine

    base = inspect.signature(getattr(JobEngine, method)).return_annotation
    assert base == returns, f"JobEngine.{method} now returns {base!r} — update the guide too"
    guide = "\n".join(_engine_guide_python())
    assert f"def {method}(" in guide, f"the guide's worked engine no longer defines {method}"
    assert f"-> {returns}:" in guide, (
        f"the guide's {method} does not return {returns}, which JobEngine declares"
    )
