"""Tests for the ORCA template inspector (``engines/orca/inspect.py``).

The inspector lets a step omit ``operation``: ChemRefine reads the template's
keywords to pick the parser and flag TS / frequency runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chemrefine.engines.orca.inspect import inspect_template


@pytest.mark.parametrize(
    ("body", "operation"),
    [
        ("! GOAT XTB\n* xyzfile 0 1 geom.xyz\n", "goat"),
        ("! DOCKER\n", "docker"),
        ("! SOLVATOR\n", "solvator"),
        ("! B3LYP def2-SVP Opt\n%geom Scan B 0 1 = 1.0, 2.0, 10 end\nend\n", "pes"),
        ("! B3LYP def2-SVP Opt Freq\n", "opt_sp"),
        ("! B3LYP def2-SVP OptTS\n", "opt_sp"),
        ("! PBE def2-SVP Freq\n", "sp"),  # frequency-only: a single point + Hessian
        ("! HF def2-SVP\n", "sp"),  # no run-type keyword → ORCA's SP fallback
        ("! B3LYP def2-SVP IRC\n%irc MaxIter 50 end\n", "irc"),
        ("! B3LYP def2-SVP OptTS Freq IRC\n", "irc"),  # the path is the product
        ('! XTB NEB-CI\n%neb NEB_End_XYZFile "product.xyz" end\n', "neb"),
        ("! XTB ZOOM-NEB-CI\n", "neb"),
        ("! XTB Opt NEB-IDPP\n", "neb"),  # the band wins over the optimisation beside it
        ("! XTB\n%md\n  timestep 0.5_fs\n  run 100\nend\n", "md"),
        ("! B3LYP def2-SVP Opt\n%md timestep 0.5_fs end\n", "md"),
    ],
)
def test_inspect_picks_parser_from_keywords(body: str, operation: str):
    assert inspect_template(body).operation == operation


@pytest.mark.parametrize(
    ("body", "operation"),
    [
        ("! XTB ScanTS\n%geom scan B 1 0 = 2.0, 1.0, 15 end end\n", "opt_sp"),
        ("! XTB ScanTS IRC\n%geom scan B 1 0 = 2.0, 1.0, 15 end end\n", "irc"),
        ('! r2SCAN-3c NEB-TS\n%neb NEB_End_XYZFile "product.xyz" end\n', "opt_sp"),
        ("! XTB FAST-NEB-TS\n", "opt_sp"),
        ("! XTB Zoom-NEB-TS\n", "opt_sp"),
        ("! XTB NEB-TS IRC\n", "irc"),
    ],
    ids=["scants", "scants-then-irc", "neb-ts", "fast-neb-ts", "zoom-neb-ts", "neb-ts-then-irc"],
)
def test_a_run_that_ends_in_a_ts_optimisation_is_that_optimisation(body: str, operation: str):
    """The saddle is the product of a ``ScanTS`` or ``NEB-TS`` run, flagged as a TS search.

    Its scan or band is the stage before the optimisation, so neither names the run; an
    ``IRC`` beside it is the stage after, and the path stays the product.
    """
    run = inspect_template(body)
    assert (run.operation, run.is_ts) == (operation, True)


def test_a_band_that_ends_in_no_ts_optimisation_stays_a_band():
    """``NEB``, ``NEB-CI`` and ``NEB-IDPP`` stop at the band: no saddle is refined."""
    for keyword in ("NEB", "NEB-CI", "NEB-IDPP", "LOOSE-NEB"):
        run = inspect_template(f"! XTB {keyword}\n")
        assert (run.operation, run.is_ts) == ("neb", False), keyword


def test_a_keyword_that_merely_contains_a_run_kind_does_not_name_it():
    """Whole tokens only, as for ``opt`` and ``freq``: ``NEBULA`` is not a band, ``MDCI`` no MD."""
    assert inspect_template("! NEBULA def2-SVP\n").operation == "sp"
    assert inspect_template("! MDCI def2-SVP\n").operation == "sp"
    assert inspect_template("! XTB GOATS\n").operation == "sp"
    assert inspect_template("! XTB GOAT-WALK\n").operation == "sp"
    assert inspect_template("# ! IRC later\n! HF def2-SVP\n").operation == "sp"


@pytest.mark.parametrize(
    "keyword",
    ["GOAT", "GOAT-ENTROPY", "goat-explore", "GOAT-React", "GOAT-DIVERSITY", "Goat-Coarse"],
)
def test_every_goat_keyword_infers_a_goat_run(keyword: str):
    """ORCA runs GOAT from six keywords, and each writes the same ensemble sidecar."""
    assert inspect_template(f"! XTB {keyword}\n").operation == "goat"


def test_inspect_is_case_insensitive():
    assert inspect_template("! goat xtb\n").operation == "goat"
    assert inspect_template("! GoAt XtB\n").operation == "goat"


def test_inspect_empty_template_is_single_point():
    assert inspect_template("* xyzfile 0 1 geom.xyz\n").operation == "sp"


@pytest.mark.parametrize(
    "keyword",
    [
        "Opt",
        "OptTS",
        "COpt",
        "ExtOpt",
        "SloppyOpt",
        "LooseOpt",
        "NormalOpt",
        "TightOpt",
        "VeryTightOpt",
    ],
)
def test_every_optimisation_keyword_orca_accepts_reads_as_an_optimisation(keyword: str):
    """The list is what ORCA 6.1.1 actually takes, checked against the binary.

    Guessing here fails silently in both directions: an accepted keyword this does not
    recognise picks the wrong parser and, through the NMS frequency gate, admits or refuses
    a step for a reason that is not true.
    """
    assert inspect_template(f"! HF STO-3G {keyword}\n").operation == "opt_sp"


@pytest.mark.parametrize(
    "keyword", ["TightOptTS", "VeryTightOptTS", "LooseOptTS", "COptTS", "ExtOptTS"]
)
def test_a_ts_search_is_only_ever_the_bare_keyword(keyword: str):
    """ORCA rejects every prefixed `…OptTS` spelling, so none can reach a real run.

    This is what lets `is_ts` test for the exact token: a tightness level for a saddle-point
    search is written as its own keyword (`! OptTS TightOpt`), read here as two tokens.
    """
    assert inspect_template(f"! HF STO-3G {keyword}\n").is_ts is False
    assert inspect_template(f"! HF STO-3G OptTS {keyword}\n").is_ts is True


def test_inspect_flags_optts_as_ts():
    run = inspect_template("! B3LYP def2-SVP OptTS Freq\n")
    # OptTS still parses like a normal opt (single optimized structure)…
    assert run.operation == "opt_sp"
    # …but is flagged as a TS search and a frequency run.
    assert run.is_ts is True
    assert run.has_freq is True


def test_inspect_plain_opt_is_not_ts_and_not_freq():
    run = inspect_template("! B3LYP def2-SVP Opt\n")
    assert run.is_ts is False
    assert run.has_freq is False


@pytest.mark.parametrize(
    ("template", "has_freq"),
    [
        ("! B3LYP def2-SVP Opt\n%freq NumFreq true end\n", True),
        ("! B3LYP def2-SVP Opt\n%FREQ AnFreq TRUE END\n", True),
        ("! B3LYP def2-SVP Opt\n%freq Hybrid_Hess {0 1} end NumFreq true end\n", True),
        ("! B3LYP def2-SVP Opt\n%freq CentralDiff false end\n", False),
        ("! B3LYP def2-SVP Opt\n%freq Temp 290, 300 end\n", False),
        ("! B3LYP def2-SVP Opt\n%freq NumFreq false end\n", False),
        ("! B3LYP def2-SVP Opt\n# %freq NumFreq true end\n", False),
    ],
    ids=[
        "numfreq-on",
        "anfreq-on-any-case",
        "a-sub-block-first",
        "a-scheme-only",
        "a-temperature-only",
        "switched-off",
        "commented-out",
    ],
)
def test_a_freq_block_that_switches_a_hessian_on_requests_frequencies(
    template: str, has_freq: bool
):
    """ORCA takes ``NumFreq true`` / ``AnFreq true`` in ``%freq`` as it takes the keyword."""
    assert inspect_template(template).has_freq is has_freq


def test_inspect_detects_numfreq():
    assert inspect_template("! PBE def2-SVP NumFreq\n").has_freq is True


def test_inspect_ignores_keywords_in_non_bang_lines():
    # 'goat' only counts on a ``!`` keyword line, not in a coordinate comment.
    body = "! B3LYP def2-SVP Opt\n# a goat wandered through the docker yard\n"
    assert inspect_template(body).operation == "opt_sp"


def test_inspect_ignores_inline_comment_keywords():
    """An inline ``#`` comment on a ``!`` line is stripped before keyword scanning."""
    run = inspect_template("! B3LYP def2-SVP Opt  # goat solvator freq later\n")
    assert run.operation == "opt_sp"  # 'goat'/'solvator' in the comment are ignored
    assert run.has_freq is False  # so is the commented-out 'freq'


def test_inspect_ignores_commented_out_scan_block():
    """A ``%geom Scan`` hidden behind comments is not treated as a PES scan."""
    body = "! B3LYP def2-SVP Opt\n# %geom Scan B 0 1 = 1.0, 2.0, 10 end end\n"
    assert inspect_template(body).operation == "opt_sp"


@pytest.mark.parametrize(
    ("body", "operation"),
    [
        # The block alone runs the ensemble, as the keyword does (ORCA 6.1.1).
        ('! XTB\n%DOCKER\n  GUEST "cl.xyz"\nEND\n', "docker"),
        ("! XTB ALPB(WATER)\n%solvator nsolv 2 end\n", "solvator"),
        # The ensemble is the product, not the optimisation beside it.
        ('! XTB Opt\n%docker guest "cl.xyz" end\n', "docker"),
        # A solvation reads a `%docker` block as the settings of its own docking.
        ("! XTB\n%solvator nsolv 2 end\n%docker docklevel quick end\n", "solvator"),
        # A run-kind keyword outranks a block.
        ("! XTB SOLVATOR\n%docker docklevel quick end\n", "solvator"),
        ('! GOAT XTB\n%docker guest "cl.xyz" end\n', "goat"),
    ],
)
def test_a_docker_or_solvator_block_infers_its_run_kind(body: str, operation: str):
    """ORCA runs a docking or a solvation from its ``%`` block alone.

    Read from the ``!`` line only, such a run inferred a single point, and the step kept one
    structure of the ensemble ORCA wrote.
    """
    assert inspect_template(body).operation == operation


def test_a_commented_out_docker_block_is_not_a_docking():
    """A ``%docker`` block hidden behind a comment is not a directive."""
    body = '! XTB Opt\n# %docker guest "cl.xyz" end\n'
    assert inspect_template(body).operation == "opt_sp"


@pytest.mark.parametrize(
    "body",
    [
        # The word in a later block's quoted path.
        '! B3LYP def2-SVP Opt Freq\n%geom MaxIter 300 end\n%pointcharges "/data/scan/q.pc"\n'
        "%pal nprocs 8 end\n",
        # The word as a later block's value.
        '! B3LYP def2-SVP Opt Freq\n%geom\n  MaxIter 300\nend\n%base "scan"\n',
        # The word on the coordinate line that follows the block.
        "! B3LYP def2-SVP Opt Freq\n%geom MaxIter 300 end\n* xyzfile 0 1 scan.xyz\n",
        # The word on a keyword line that follows the block.
        "%geom MaxIter 300 end\n! B3LYP def2-SVP Opt Freq scan\n",
    ],
    ids=["later-block-path", "later-block-value", "coordinate-line", "keyword-line"],
)
def test_scan_outside_the_geom_block_body_is_not_a_scan(body: str):
    """``scan`` names a relaxed scan only inside the ``%geom`` block that declares it.

    An ``Opt Freq`` step read as ``pes`` is parsed by the scan reader, which reports no
    frequency table — so the run keeps its energies and silently loses its thermochemistry
    and every imaginary mode NMS would act on.
    """
    run = inspect_template(body)
    assert run.operation == "opt_sp"
    assert run.has_freq is True


@pytest.mark.parametrize(
    "body",
    [
        # The sub-block on its own lines, its ``end`` before the block's own.
        "! B3LYP def2-SVP Opt\n%geom\n  Scan\n    B 0 1 = 1.0, 2.0, 10\n  end\nend\n",
        # A second sub-block after the scan, then another block and the coordinates.
        "! B3LYP def2-SVP Opt\n%geom Scan B 0 1 = 1.0, 2.0, 10 end\n"
        "  Constraints { B 2 3 C } end\nend\n%pal nprocs 4 end\n* xyzfile 0 1 geom.xyz\n",
        # The scan block after another block.
        "! B3LYP def2-SVP Opt\n%pal nprocs 4 end\n%GEOM SCAN D 0 1 2 3 = 0, 180, 19 END END\n",
    ],
    ids=["multi-line", "with-constraints", "after-another-block"],
)
def test_a_scan_sub_block_anywhere_in_the_geom_body_is_a_scan(body: str):
    assert inspect_template(body).operation == "pes"


def test_inspect_keeps_a_hash_inside_a_quoted_filename():
    """A ``#`` in a quoted path is part of the filename, not the start of a comment.

    Cutting the line at the ``#`` truncates whatever follows, so a template naming
    ``"lig#3.xyz"`` loses the rest of its own line — here the `Freq` that gates NMS and the
    `end` that closes the scan block — and the step is classified with the wrong parser, or
    refused NMS for a reason that is not true: a chemistry-shaped failure caused by a
    character in a filename.
    """
    run = inspect_template('! B3LYP def2-SVP Opt "lig#3.xyz" Freq\n')
    assert run.has_freq is True, "the keyword after the quoted '#' was swallowed"
    assert run.operation == "opt_sp"


def test_inspect_still_strips_a_real_comment_after_a_quoted_string():
    """Quote tracking must not cost us actual comment stripping."""
    run = inspect_template('! B3LYP def2-SVP Opt "lig#3.xyz"  # Freq later\n')
    assert run.has_freq is False


# ---------------------------------------------------------------------------
# Keyword matching is by whole token, checked against the real templates
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_extopt_is_an_optimisation():
    """ORCA's ``ExtOpt`` runs its optimiser over an external program's gradients.

    Read from a real tutorial template rather than a fabricated one: this is the
    keyword every ``mlip-extopt`` / ``pyscf-extopt`` step uses, and tokenising the
    keyword line without allowing for it reclassified all of them as single points.
    """
    template = REPO_ROOT / "examples/tutorials/redox/amines/templates/step2.inp"
    assert "!ExtOpt" in template.read_text(encoding="utf-8")
    assert inspect_template(template.read_text(encoding="utf-8")).operation == "opt_sp"


def test_every_shipped_extopt_template_is_an_optimisation():
    """Whatever the tutorials use, the classification has to hold for all of it."""
    templates = sorted(
        p
        for root in ("tests/data/e2e/cases", "examples")
        for p in (REPO_ROOT / root).rglob("templates/*.inp")
        if any(
            line.lstrip().lstrip("!").split()[:1] == ["ExtOpt"]
            for line in p.read_text(errors="replace").splitlines()
            if line.lstrip().startswith("!")
        )
    )
    assert templates, "no ExtOpt templates found — has the layout moved?"
    assert {inspect_template(p.read_text(encoding="utf-8")).operation for p in templates} == {
        "opt_sp"
    }


def test_a_keyword_merely_containing_opt_is_not_an_optimisation():
    """The point of matching whole tokens: substrings must not decide the run type."""
    assert inspect_template("! B3LYP def2-SVP Optimizer-Is-Not-A-Keyword\n").operation == "sp"


# ---------------------------------------------------------------------------
# %maxcore — the memory declaration the SLURM request is derived from
# ---------------------------------------------------------------------------


def test_inspect_reads_maxcore():
    """``%maxcore`` is per-core MB, read with qorca's grammar, case-insensitively."""
    info = inspect_template("! B3LYP Opt\n%MaxCore 3000\n%pal nprocs 4 end\n")
    assert info.maxcore == 3000
    assert info.pal == 4


def test_a_template_without_maxcore_declares_no_memory():
    """Absence is ``None``, not a default: no declaration means the header's policy stands."""
    assert inspect_template("! B3LYP Opt\n").maxcore is None


def test_a_commented_maxcore_is_not_a_declaration():
    """The same comment-stripping the keywords get: ``# %maxcore 9000`` declares nothing."""
    assert inspect_template("! B3LYP Opt\n# %maxcore 9000\n").maxcore is None
