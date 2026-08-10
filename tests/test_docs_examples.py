"""Every full config the docs show must validate against the real schema.

The README quickstart and each tutorial open with a complete YAML config, and nothing ran
them: a schema change that renamed a field would leave every example teaching the old
spelling, discovered by the next new user as a validation error on their first run. Any
``yaml`` fence whose text contains ``steps:`` is a full config by this repo's convention
(all eleven current blocks are), so each must parse and pass :class:`chemrefine.config.
Config` — the same validation a real run applies first.

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
