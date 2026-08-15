"""MkDocs hook: publish the workflow builder as a static playground under /playground/.

The GUI's assets are written to serve two worlds (relative paths, an ``/api`` probe
that falls back to static mode), so publishing them on the docs site is a copy plus one
generated file: ``schema.json``, the same document ``chemrefine schema`` prints, baked
at build time from the installed package — the playground's dropdowns and forms are
exactly as current as the release the docs describe. Runs post-build so ``--strict``
page validation is untouched; the docs workflow imports chemrefine anyway (mkdocstrings
renders live signatures), so the import here adds no new requirement.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def on_post_build(config: Any, **kwargs: Any) -> None:
    """Copy the GUI assets into ``site/playground/`` and bake ``schema.json`` beside them.

    ``docs/playground.md`` renders a stub page to the same URL first — that is what puts
    **Playground** in the site's top navigation tabs — and this hook then replaces the
    stub with the app, so clicking the tab opens the full-screen builder directly.
    """
    from chemrefine.gui.app import STATIC_DIR
    from chemrefine.introspect import schema_document

    playground = Path(config["site_dir"]) / "playground"
    if playground.exists():
        shutil.rmtree(playground)
    playground.mkdir(parents=True)
    shutil.copytree(STATIC_DIR, playground / "static")
    # The page sits above its assets, exactly as the Flask app serves it.
    (playground / "static" / "index.html").rename(playground / "index.html")
    (playground / "schema.json").write_text(
        json.dumps(schema_document()), encoding="utf-8"
    )
