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
    """Every ``steps:``-carrying YAML block, id'd by file and block number."""
    blocks: list[object] = []
    for path in [_REPO_ROOT / "README.md", *sorted((_REPO_ROOT / "docs").rglob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        for n, match in enumerate(_YAML_FENCE_RE.finditer(text), 1):
            if "steps:" in match.group(1):
                blocks.append(
                    pytest.param(match.group(1), id=f"{path.relative_to(_REPO_ROOT)}:{n}")
                )
    return blocks


def test_the_scanner_finds_the_readme_and_the_docs_configs():
    """Guards the fence regex and both roots: a scanner that finds nothing passes everything.

    ``docs/`` ships wherever this suite runs, so a docs page contributing no block means
    the scan broke, not that the pages went missing — the README alone would be validated.
    """
    files = {str(getattr(p, "id", "")).rsplit(":", 1)[0] for p in _config_blocks()}
    assert "README.md" in files
    assert any(name.startswith("docs/") for name in files)


@pytest.mark.parametrize("block", _config_blocks())
def test_the_documented_config_validates(block: str):
    raw = yaml.safe_load(block)
    assert isinstance(raw, dict), "a steps: block must be a mapping — is the fence a fragment?"
    Config(**raw)


# ---------------------------------------------------------------------------
# The README quickstart is the shipped example, verbatim
# ---------------------------------------------------------------------------

_FIRST_RUN = _REPO_ROOT / "examples" / "first_run"


def _first_run_files() -> list[str]:
    """What a reader pastes: the config, each template its steps read, each header they run under.

    Read off the loaded config as the run reads it, not listed by hand: a file the run needs
    and the README leaves out is the paste path's failure, so a hand-kept list was the one
    place it could hide — and not off the directory either, where the output of running
    the example in place would become a case. The seed is the reader's own.
    """
    from chemrefine.config import load_config
    from chemrefine.engines.api import JobExecutable, get_engine
    from chemrefine.state import PipelineState
    from chemrefine.step import build_context
    from chemrefine.validate import effective_header

    config = load_config(_FIRST_RUN / "input.yaml")
    read = {_FIRST_RUN / "input.yaml"}
    for step in config.steps:
        engine = get_engine(step.engine)
        template = build_context(config, step, PipelineState(), engine).template
        if template is not None:
            read.add(template)
        if isinstance(engine, JobExecutable):
            read.add(config.template_dir / effective_header(config, step, engine))
    return sorted(str(path.resolve().relative_to(_FIRST_RUN.resolve())) for path in read)


@pytest.mark.parametrize("relative", _first_run_files())
def test_the_readme_quickstart_is_the_shipped_example(relative: str):
    """The README's quickstart fences and ``examples/first_run/`` are one text, not two.

    The README teaches the pipeline and the example directory runs it (``test_examples``
    sweeps the directory: seed, templates, header, the operation the templates infer). A
    reader who pastes the README and one who copies the directory must get the same run —
    the schema test above validates the fence, but only this holds it to the files.
    """
    text = (_FIRST_RUN / relative).read_text(encoding="utf-8")
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(rf"```[a-z]*\n{re.escape(text)}```", readme), (
        f"{relative} is not a README quickstart fence"
    )


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


@pytest.mark.parametrize("method", ["build_input", "run_block", "parse_one"])
def test_the_engine_guides_primitives_take_what_the_base_passes(method: str):
    """Each primitive the guide defines takes the base's parameters, by name and in order.

    ``prepare`` hands ``build_input`` every argument by keyword, so a copied primitive that
    still spells a parameter the base renamed fails with ``TypeError: unexpected keyword
    argument`` at the step's first structure, after every earlier step has run.
    """
    import ast
    import inspect

    from chemrefine.engines._job import JobEngine

    base = list(inspect.signature(getattr(JobEngine, method)).parameters)
    defined = [
        node
        for fence in _engine_guide_python()
        for node in ast.walk(ast.parse(fence))
        if isinstance(node, ast.FunctionDef) and node.name == method
    ]
    assert defined, f"the guide's worked engine no longer defines {method}"
    for node in defined:
        spelled = [a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
        assert spelled == base, f"the guide's {method} takes {spelled}; JobEngine's takes {base}"
