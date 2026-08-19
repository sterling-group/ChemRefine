"""Tests for the managed backend-env provisioner (``engines/_provision.py``) + launch seam.

Everything is faked — no real ``uv`` / ``conda`` subprocess runs, no real backend imports.
The requirement DTOs come from the engines' own ``backend_requirement`` (Protocol capability),
so these tests also pin the end-to-end wiring: managed env → server command / script command.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from email.message import Message
from pathlib import Path

import pytest
from ase import Atoms
from packaging.requirements import Requirement

from chemrefine.config import StepConfig
from chemrefine.engines import _provision as provision
from chemrefine.engines import preflight_backends
from chemrefine.engines.api import BackendRequirement, get_engine
from chemrefine.errors import BackendProvisionError, ConfigError
from chemrefine.state import PipelineState, StepContext, Structure

_REQ = BackendRequirement(extra="mlip-mace", import_name="mace")


@pytest.fixture(autouse=True)
def _cold_python_lookups():
    """Every test sees the derived-Python lookups uncached.

    They are `lru_cache`d in the module because a run asks them once per provisionable step
    and the answer cannot change inside a process — but a test that fakes this dist's
    metadata is exactly the case where it can.
    """
    cached = (provision._candidate_pythons, provision._supported_pythons)  # the real ones
    for lookup in cached:
        lookup.cache_clear()
    yield
    for lookup in cached:  # held from setup: a test may have monkeypatched the module attribute
        lookup.cache_clear()


def _provisioned(tmp_path: Path, extra: str) -> Path:
    """Create a fake managed env for ``extra`` under ``tmp_path`` and return its python.

    A symlink to this interpreter rather than an empty file, matching what CI's
    `provisioned-backend` job creates (`ln -s $(command -v python)`) — so a test that
    reaches the shared-env probe meets something that can actually answer.
    """
    py = tmp_path / "backends" / extra / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.symlink_to(sys.executable)
    return py


def _ctx(tmp_path: Path, *, engine: str, options: dict | None) -> StepContext:
    """A minimal real ``StepContext`` for launch-seam tests."""
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    return StepContext(
        step_cfg=StepConfig(step=1, engine=engine, operation="opt_sp", options=options or {}),
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=tmp_path / "templates",
        template=tmp_path / "templates" / f"step1.{get_engine(engine).template_suffix}",
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=(seed,)),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        executables={},
    )


# ---------------------------------------------------------------------------
# chemrefine_home / backend_env_path
# ---------------------------------------------------------------------------


def test_chemrefine_home_explicit_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path / "crh"))
    assert provision.chemrefine_home() == tmp_path / "crh"


def test_chemrefine_home_alongside_writable_prefix(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("CHEMREFINE_HOME", raising=False)
    monkeypatch.setattr(provision.sys, "prefix", str(tmp_path))
    assert provision.chemrefine_home() == tmp_path / "share" / "chemrefine"


def test_chemrefine_home_falls_back_to_user_home(monkeypatch, tmp_path: Path):
    """The $HOME fallback is namespaced by interpreter tag.

    $HOME is routinely shared across machines on HPC, so two clusters running
    different Pythons would otherwise resolve the same managed-env interpreter
    and one would silently run the other's env.
    """
    import sys

    monkeypatch.delenv("CHEMREFINE_HOME", raising=False)
    monkeypatch.setattr(provision.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(provision.os, "access", lambda _p, _m: False)
    monkeypatch.setattr(provision.Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    home = provision.chemrefine_home()
    assert home == tmp_path / "home" / ".chemrefine" / sys.implementation.cache_tag
    assert home.parent.name == ".chemrefine"


def test_backend_env_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    assert provision.backend_env_path("pyscf") == tmp_path / "backends" / "pyscf"


# ---------------------------------------------------------------------------
# resolve_launcher / require_backend
# ---------------------------------------------------------------------------


def test_resolve_launcher_override_wins(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    _provisioned(tmp_path, _REQ.extra)  # even with a managed env present
    assert provision.resolve_launcher(_REQ, "/custom/python") == "/custom/python"


def test_resolve_launcher_managed_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    py = _provisioned(tmp_path, _REQ.extra)
    assert provision.resolve_launcher(_REQ) == str(py)


def test_resolve_launcher_falls_back_to_sys_executable(monkeypatch, tmp_path: Path):
    """Single-env case: the orchestrator's own interpreter runs the backend."""
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    assert provision.resolve_launcher(_REQ) == sys.executable


def test_require_backend_override_ok(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    provision.require_backend(_REQ, "/custom/python")  # no raise


def test_require_backend_managed_ok(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    _provisioned(tmp_path, _REQ.extra)
    provision.require_backend(_REQ)  # no raise


def test_require_backend_importable_ok(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", lambda _n: object())
    provision.require_backend(_REQ)  # no raise


def test_require_backend_missing_raises_actionable(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", lambda _n: None)
    with pytest.raises(ConfigError, match="chemrefine backends install mlip-mace"):
        provision.require_backend(_REQ)


def test_require_backend_does_not_advise_an_install_that_would_do_nothing(
    monkeypatch, tmp_path: Path
):
    """On a Python the extra excludes, "install it into this environment" is not the fix.

    pip would report success having installed nothing — every requirement the extra declares
    is marker-excluded there — so the only advice that helps is the managed env, which gets
    built on a Python the backend supports.
    """
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", lambda _n: None)
    monkeypatch.setattr(provision, "_supported_pythons", lambda _e: ("3.12",))
    with pytest.raises(ConfigError) as excinfo:
        provision.require_backend(BackendRequirement(extra="mlip-orb", import_name="orb_models"))
    message = str(excinfo.value)
    assert "does not install on Python" in message and "it needs 3.12" in message
    assert "into this environment" not in message


# ---------------------------------------------------------------------------
# preflight_backends — fail fast before any job submits
# ---------------------------------------------------------------------------


def test_preflight_skips_non_provisionable_engines(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", lambda _n: None)
    steps = [StepConfig(step=1, engine="fake", operation="opt_sp")]
    preflight_backends(steps)  # fake isn't provisionable → nothing to check


def test_preflight_raises_for_unavailable_backend(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", lambda _n: None)
    steps = [
        StepConfig(step=1, engine="fake", operation="opt_sp"),
        StepConfig(step=2, engine="mlip", operation="opt_sp", options={"task_name": "mace_off"}),
    ]
    with pytest.raises(ConfigError, match="mlip-mace"):
        preflight_backends(steps)


def test_preflight_passes_with_managed_envs(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", lambda _n: None)
    _provisioned(tmp_path, "mlip-mace")
    _provisioned(tmp_path, "mlip-fairchem")
    steps = [
        StepConfig(step=1, engine="mlip", operation="opt_sp", options={"task": "mace_off"}),
        StepConfig(step=2, engine="mlip", operation="opt_sp", options={"task_name": "omol"}),
    ]
    preflight_backends(steps)  # both resolve → no raise


# ---------------------------------------------------------------------------
# preflight — the server half of an ExtOpt backend
# ---------------------------------------------------------------------------


def _importable(*names: str):
    """A ``find_spec`` stand-in that resolves exactly ``names`` and nothing else."""
    return lambda name: object() if name in names else None


def _extopt_step(**extra_options) -> StepConfig:
    return StepConfig(
        step=1,
        engine="mlip-extopt",
        operation="opt_sp",
        options={"task_name": "mace_off", **extra_options},
    )


def test_preflight_requires_server_deps_for_extopt_in_the_single_env(monkeypatch, tmp_path: Path):
    """A hand-assembled env — backend importable, no ``[server]`` — fails before submission.

    The report behind this: ``pip install -e .`` beside a pre-existing backend passed
    preflight, and the gradient server died on ``import waitress`` *inside the job*, after
    the upstream step had spent 19 hours. The message must name both fixes, like the
    backend check's does.
    """
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", _importable("mace"))

    with pytest.raises(ConfigError, match=r"chemrefine\[server\]") as excinfo:
        preflight_backends([_extopt_step()])

    assert "flask and waitress" in str(excinfo.value)
    assert "backends install mlip-mace" in str(excinfo.value)


def test_preflight_names_only_the_server_dep_that_is_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", _importable("mace", "flask"))

    with pytest.raises(ConfigError, match="needs waitress —"):
        preflight_backends([_extopt_step()])


def test_preflight_passes_when_the_single_env_carries_the_server_deps(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        provision.importlib.util, "find_spec", _importable("mace", "flask", "waitress")
    )
    preflight_backends([_extopt_step()])  # no raise


def test_preflight_trusts_a_managed_env_for_the_server_deps(monkeypatch, tmp_path: Path):
    """A managed env is built as ``chemrefine[<extra>]``, which cross-references ``[server]``.

    Probing it from here is not possible anyway — ``find_spec`` answers for *this*
    interpreter — and not necessary: the env's own install is the guarantee.
    """
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    _provisioned(tmp_path, "mlip-mace")
    monkeypatch.setattr(provision.importlib.util, "find_spec", _importable())
    preflight_backends([_extopt_step()])  # no raise


def test_preflight_trusts_a_backend_python_override_for_the_server_deps(
    monkeypatch, tmp_path: Path
):
    """The documented escape hatch stays an escape hatch — trusted whole."""
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", _importable())
    preflight_backends([_extopt_step(backend_python="/envs/x/bin/python")])  # no raise


def test_preflight_asks_nothing_of_a_direct_engine_about_the_server(monkeypatch, tmp_path: Path):
    """The direct engines run scripts, not a server — flask/waitress are not their business."""
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", _importable("mace"))
    steps = [
        StepConfig(step=1, engine="mlip", operation="opt_sp", options={"task_name": "mace_off"})
    ]
    preflight_backends(steps)  # no raise


# ---------------------------------------------------------------------------
# Which Pythons an extra installs on — derived from our own metadata
# ---------------------------------------------------------------------------


def _metadata_with(monkeypatch, *, requires_python: str, classifiers: tuple[str, ...]) -> None:
    """Fake this dist's core metadata — the object `importlib.metadata` really returns."""
    message = Message()
    message["Requires-Python"] = requires_python
    for classifier in classifiers:
        message["Classifier"] = classifier
    monkeypatch.setattr(provision.importlib.metadata, "metadata", lambda _n: message)


def test_candidates_are_the_classifiers_inside_requires_python(monkeypatch):
    """Newest first, and a classifier the floor excludes is not a candidate.

    The bare `:: 3` classifier is not a version and must not be read as one.
    """
    _metadata_with(
        monkeypatch,
        requires_python=">=3.11",
        classifiers=(
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ),
    )
    assert provision._candidate_pythons() == ("3.12", "3.11")


def test_candidates_fall_back_to_this_interpreter_without_classifiers(monkeypatch):
    _metadata_with(monkeypatch, requires_python=">=3.11", classifiers=())
    here = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert provision._candidate_pythons() == (here,)


def test_candidates_fall_back_to_this_interpreter_without_a_dist(monkeypatch):
    """A tree run without an installed dist: the honest answer is "the Python I am"."""

    def _raise(_name):
        raise provision.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(provision.importlib.metadata, "metadata", _raise)
    here = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert provision._candidate_pythons() == (here,)


def test_an_extra_is_supported_where_it_contributes_everything(monkeypatch):
    """The rule: full requirement set, not a non-empty one.

    Every backend extra cross-references `chemrefine[server]`, which hatchling flattens into
    flask/waitress — so "the extra installs something" is true on every Python, capped or
    not, and only "installs everything it declares" separates them.
    """
    requirements = [
        Requirement("flask>=3.0; extra == 'demo'"),
        Requirement("wheelless; python_version < '3.13' and extra == 'demo'"),
        Requirement("numpy>=1.26"),  # a core dependency: unmarked, never an extra's
        Requirement("other; extra == 'unrelated'"),
    ]
    assert provision._supported_from(requirements, "demo", ("3.14", "3.13", "3.12")) == ("3.12",)
    assert provision._supported_from(requirements, "unrelated", ("3.14", "3.12")) == (
        "3.14",
        "3.12",
    )


def test_supported_pythons_reads_the_installed_metadata(monkeypatch):
    """The claim the provisioner acts on is the string pip resolves — this dist's own."""
    monkeypatch.setattr(
        provision.importlib.metadata,
        "requires",
        lambda _n: ["orb; python_version < '3.13' and extra == 'mlip-orb'"],
    )
    _metadata_with(
        monkeypatch,
        requires_python=">=3.11",
        classifiers=(
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
        ),
    )
    assert provision._supported_pythons("mlip-orb") == ("3.12",)


def test_supported_pythons_survives_a_dist_that_declares_nothing(monkeypatch):
    monkeypatch.setattr(provision.importlib.metadata, "requires", lambda _n: None)
    assert provision._supported_pythons("mlip-orb") == provision._candidate_pythons()


def test_every_backend_extra_claims_the_pythons_it_can_install_on():
    """The caps in pyproject say what we mean — read from the file, not from an install.

    `_supported_pythons` reads *installed* metadata, which is frozen at install time; an
    editable checkout can therefore be a pyproject edit ahead of it. This holds the claim
    where it is written, so the gate is on the source rather than on how recently someone
    ran `pip install -e .`.

    Capped extras are named; every other backend extra must claim the whole matrix. A new
    backend is therefore covered the moment it registers — with no cap, or with a cap and a
    line here saying why.
    """
    from chemrefine.engines import known_backend_extras

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    candidates = tuple(
        sorted(
            (
                match.group(1)
                for classifier in declared["classifiers"]
                if (match := provision._PYTHON_CLASSIFIER.fullmatch(classifier))
            ),
            key=lambda v: int(v.split(".")[1]),
            reverse=True,
        )
    )
    # `extra == …` is a marker, so it is `and`-ed onto whatever marker the entry already
    # carries — exactly what the packaging backend does when it builds Requires-Dist.
    requirements = [
        Requirement(f"{raw} and extra == '{extra}'" if ";" in raw else f"{raw}; extra == '{extra}'")
        for extra, raws in declared["optional-dependencies"].items()
        for raw in raws
    ]
    capped = {
        "mlip-orb": ("3.12",),  # orb pins dm-tree==0.1.8, whose newest wheels are cp312
        "mlip-mace": ("3.13", "3.12", "3.11"),  # our torch<2.9 pin has no cp314 wheels
        "mlip-chgnet": ("3.12", "3.11"),  # chgnet 0.4.2 ships cp310-cp312 only
    }
    for extra in sorted(known_backend_extras()):
        expected = capped.get(extra, candidates)
        assert provision._supported_from(requirements, extra, candidates) == expected, extra


# ---------------------------------------------------------------------------
# detect_env_tool
# ---------------------------------------------------------------------------


def test_detect_env_tool_conda(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(provision.sys, "prefix", str(tmp_path))
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    assert provision.detect_env_tool() == "conda"


def test_detect_env_tool_uv(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(provision.sys, "prefix", str(tmp_path))
    (tmp_path / "pyvenv.cfg").write_text("home = /x\nuv = 0.5.0\n", encoding="utf-8")
    monkeypatch.setattr(provision.shutil, "which", lambda _n: "/usr/bin/uv")
    assert provision.detect_env_tool() == "uv"


def test_detect_env_tool_venv_without_uv_binary(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(provision.sys, "prefix", str(tmp_path))
    (tmp_path / "pyvenv.cfg").write_text("uv = 0.5.0\n", encoding="utf-8")
    monkeypatch.setattr(provision.shutil, "which", lambda _n: None)
    assert provision.detect_env_tool() == "venv"


def test_detect_env_tool_venv_without_uv_marker(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(provision.sys, "prefix", str(tmp_path))
    (tmp_path / "pyvenv.cfg").write_text("home = /x\n", encoding="utf-8")
    monkeypatch.setattr(provision.shutil, "which", lambda _n: "/usr/bin/uv")
    assert provision.detect_env_tool() == "venv"


# ---------------------------------------------------------------------------
# resolve_base_python — which Python a managed env is created on, and from where
# ---------------------------------------------------------------------------


def _supports(monkeypatch, versions: tuple[str, ...]) -> None:
    """Pin what the extra under test claims, without depending on today's pyproject."""
    monkeypatch.setattr(provision, "_supported_pythons", lambda _e: versions)


def test_the_orchestrators_own_interpreter_is_used_where_the_extra_installs(monkeypatch):
    """The common case, and what this module did unconditionally before it could choose."""
    _supports(monkeypatch, ("3.14", provision.this_python(), "3.11"))
    base = provision.resolve_base_python("mlip-mace", "venv")
    assert base == provision.BasePython(provision.this_python(), "venv", sys.executable)


def test_an_extra_that_claims_nothing_still_builds_here(monkeypatch):
    """No claim is not a claim of "nowhere" — an unknown extra behaves as it always did."""
    _supports(monkeypatch, ())
    assert provision.resolve_base_python("mystery", "venv").interpreter == sys.executable


@pytest.mark.parametrize("tool", ["conda", "uv"])
def test_conda_and_uv_are_asked_for_a_version_they_can_produce(monkeypatch, tool):
    """Neither needs an interpreter on the machine: conda resolves one, uv downloads one."""
    _supports(monkeypatch, ("3.12", "3.11"))
    monkeypatch.setattr(provision.shutil, "which", lambda _n: None)  # nothing on PATH at all
    assert provision.resolve_base_python("mlip-orb", tool) == provision.BasePython("3.12", tool)


def test_a_venv_is_created_by_the_canonical_interpreter_for_that_version(monkeypatch):
    _supports(monkeypatch, ("3.12",))
    monkeypatch.setattr(
        provision.shutil, "which", lambda n: "/usr/bin/python3.12" if n == "python3.12" else None
    )
    base = provision.resolve_base_python("mlip-orb", "venv")
    assert base == provision.BasePython("3.12", "venv", "/usr/bin/python3.12")


def test_uv_supplies_the_python_a_venv_machine_lacks(monkeypatch):
    """The escalation: uv creates the env even though uv did not create the current one.

    It is the only tool on such a machine that can still produce the interpreter, and an env
    it makes is installed into by uv — which is why the tool travels with the version.
    """
    _supports(monkeypatch, ("3.12",))
    monkeypatch.setattr(provision.shutil, "which", lambda n: "/usr/bin/uv" if n == "uv" else None)
    assert provision.resolve_base_python("mlip-orb", "venv") == provision.BasePython("3.12", "uv")


def test_no_interpreter_and_no_uv_is_refused_with_the_ways_out(monkeypatch):
    _supports(monkeypatch, ("3.12",))
    monkeypatch.setattr(provision.shutil, "which", lambda _n: None)
    with pytest.raises(BackendProvisionError) as excinfo:
        provision.resolve_base_python("mlip-orb", "venv")
    message = str(excinfo.value)
    assert "installs on Python 3.12" in message
    assert "pip install uv" in message and "--python" in message
    assert excinfo.value.exit_code == 9


def test_the_tool_is_detected_when_it_is_not_given(monkeypatch):
    _supports(monkeypatch, (provision.this_python(),))
    monkeypatch.setattr(provision, "detect_env_tool", lambda: "conda")
    assert provision.resolve_base_python("pyscf").tool == "conda"


# --- the --python override -------------------------------------------------


def test_an_overriding_version_is_looked_up_for_venv(monkeypatch):
    monkeypatch.setattr(
        provision.shutil, "which", lambda n: "/usr/bin/python3.11" if n == "python3.11" else None
    )
    base = provision.resolve_base_python("mlip-orb", "venv", "3.11")
    assert base == provision.BasePython("3.11", "venv", "/usr/bin/python3.11")


def test_an_overriding_version_needs_no_lookup_for_conda(monkeypatch):
    monkeypatch.setattr(provision.shutil, "which", lambda _n: None)
    assert provision.resolve_base_python("pyscf", "conda", "3.11") == provision.BasePython(
        "3.11", "conda"
    )


def test_an_overriding_version_venv_cannot_find_is_refused(monkeypatch):
    monkeypatch.setattr(provision.shutil, "which", lambda _n: None)
    with pytest.raises(BackendProvisionError, match="names no interpreter on PATH"):
        provision.resolve_base_python("mlip-orb", "venv", "3.11")


def test_an_overriding_path_reports_its_own_version(monkeypatch):
    """conda takes `python=X.Y` and never a path, so the version is read back off it."""
    monkeypatch.setattr(provision.shutil, "which", lambda n: n)
    base = provision.resolve_base_python("mlip-orb", "conda", sys.executable)
    assert base == provision.BasePython(provision.this_python(), "conda", sys.executable)


def test_an_overriding_name_that_is_not_an_interpreter_is_refused(monkeypatch):
    monkeypatch.setattr(provision.shutil, "which", lambda _n: None)
    with pytest.raises(BackendProvisionError, match=r"neither an `X\.Y` version"):
        provision.resolve_base_python("mlip-orb", "venv", "/opt/nothing/python")


def test_an_interpreter_that_cannot_be_run_is_refused(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(provision.shutil, "which", lambda n: n)

    def _boom(argv, **_k):
        raise OSError(8, "Exec format error")

    monkeypatch.setattr(provision.subprocess, "run", _boom)
    with pytest.raises(BackendProvisionError, match="could not be run to ask for its version"):
        provision.resolve_base_python("mlip-orb", "venv", str(tmp_path / "python"))


# ---------------------------------------------------------------------------
# build_backend_env — mirrors the detected tool; idempotent
# ---------------------------------------------------------------------------


def test_build_commands_per_tool(monkeypatch, tmp_path: Path):
    """Each tool creates the env on the version it is given; only venv needs an interpreter."""
    monkeypatch.setattr(provision, "_direct_url", lambda: None)  # index install
    env = tmp_path / "e"
    uv = provision._build_commands(provision.BasePython("3.12", "uv"), env, "mlip-mace")
    assert uv[0] == ["uv", "venv", "--python", "3.12", str(env)]
    assert uv[1][:4] == ["uv", "pip", "install", "--python"]
    conda = provision._build_commands(provision.BasePython("3.12", "conda"), env, "pyscf")
    assert conda[0][:4] == ["conda", "create", "-y", "-p"]
    assert conda[0][-1] == "python=3.12"
    venv = provision._build_commands(
        provision.BasePython("3.12", "venv", "/usr/bin/python3.12"), env, "mlip-orb"
    )
    assert venv[0] == ["/usr/bin/python3.12", "-m", "venv", str(env)]
    # Every tool installs the extra pinned to the orchestrator's version.
    for cmds in (uv, conda, venv):
        assert cmds[1][-1].startswith("chemrefine[") and "==" in cmds[1][-1]


def test_uv_is_handed_an_explicit_interpreter_as_it_was_given(monkeypatch, tmp_path: Path):
    """uv's `--python` takes a path as happily as a version, so `--python` passes through."""
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    base = provision.BasePython("3.12", "uv", "/opt/py312/bin/python3")
    create, _ = provision._build_commands(base, tmp_path / "e", "mlip-orb")
    assert create[:4] == ["uv", "venv", "--python", "/opt/py312/bin/python3"]


def test_build_backend_env_installs_into_an_env_that_already_exists(monkeypatch, tmp_path: Path):
    """An existing env is extended, not skipped — creation is what is idempotent.

    Returning early on `<env>/bin/python` made `backends install pyscf-gpu` a silent no-op
    wherever a `pyscf` env already stood, because the two share a directory: the command
    reported success having installed nothing, and the GPU step then ran on CPU. pip is
    idempotent when the requirement is already satisfied, so letting it run is what makes
    "install the CPU stack, then add the GPU one" work at all.
    """
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    py = _provisioned(tmp_path, "pyscf")
    calls: list[list[str]] = []
    monkeypatch.setattr(provision.subprocess, "run", lambda argv, **_k: calls.append(argv))

    assert provision.build_backend_env("pyscf-gpu") == py  # the *pyscf* env, extended

    assert len(calls) == 1, "the env exists, so only the install runs — no second create"
    assert calls[0][-1].startswith("chemrefine[pyscf-gpu]")


def test_build_backend_env_runs_the_detected_tool(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    monkeypatch.setattr(provision, "detect_env_tool", lambda: "venv")
    calls: list[list[str]] = []
    monkeypatch.setattr(provision.subprocess, "run", lambda argv, **_k: calls.append(argv))
    python = provision.build_backend_env("mlip-mace")
    assert python == tmp_path / "backends" / "mlip-mace" / "bin" / "python"
    assert len(calls) == 2 and calls[0][1:3] == ["-m", "venv"]


def test_build_backend_env_explicit_tool(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    calls: list[list[str]] = []
    monkeypatch.setattr(provision.subprocess, "run", lambda argv, **_k: calls.append(argv))
    provision.build_backend_env("pyscf", tool="uv")
    assert calls[0][0] == "uv"


def test_a_fresh_env_is_created_on_the_python_the_extra_supports(monkeypatch, tmp_path: Path):
    """The whole point: the backend's Python, not the orchestrator's."""
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    _supports(monkeypatch, ("3.12",))
    monkeypatch.setattr(
        provision.shutil, "which", lambda n: "/usr/bin/python3.12" if n == "python3.12" else None
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(provision.subprocess, "run", lambda argv, **_k: calls.append(argv))

    provision.build_backend_env("mlip-orb", tool="venv")

    assert calls[0][:3] == ["/usr/bin/python3.12", "-m", "venv"]


def test_an_env_built_on_a_python_the_extra_excludes_is_refused(monkeypatch, tmp_path: Path):
    """The hollow env: pip succeeds against it having installed nothing.

    Every requirement the extra declares is marker-excluded on that Python, so what comes
    back is the `[server]` half and exit 0 — and the env then passes preflight by name while
    the step fails on the backend import inside the job.
    """
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    _supports(monkeypatch, ("3.12",))
    env_path = provision.backend_env_path("mlip-orb")
    _provisioned(tmp_path, "mlip-orb")
    (env_path / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    monkeypatch.setattr(provision.subprocess, "run", lambda *_a, **_k: pytest.fail("no install"))

    with pytest.raises(BackendProvisionError, match=r"was built on Python 3\.13"):
        provision.build_backend_env("mlip-orb", tool="conda")


def test_an_env_this_extra_does_install_on_is_extended(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    _supports(monkeypatch, ("3.12",))
    env_path = provision.backend_env_path("mlip-orb")
    _provisioned(tmp_path, "mlip-orb")
    (env_path / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    calls: list[list[str]] = []
    monkeypatch.setattr(provision.subprocess, "run", lambda argv, **_k: calls.append(argv))

    provision.build_backend_env("mlip-orb", tool="conda")

    assert len(calls) == 1 and calls[0][-1].startswith("chemrefine[mlip-orb]")


def test_an_env_whose_layout_says_nothing_is_left_alone(monkeypatch, tmp_path: Path):
    """Unknown is not evidence of wrong — an env that cannot be read is not refused."""
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    _supports(monkeypatch, ("3.12",))
    _provisioned(tmp_path, "mlip-orb")  # bin/python and nothing else
    monkeypatch.setattr(provision.subprocess, "run", lambda argv, **_k: None)

    provision.build_backend_env("mlip-orb", tool="conda")  # no raise


def test_python_is_refused_against_an_env_that_already_exists(monkeypatch, tmp_path: Path):
    """`--python` only applies where an env is created; ignoring it would report a lie."""
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    _provisioned(tmp_path, "mlip-orb")
    with pytest.raises(BackendProvisionError, match="already exists"):
        provision.build_backend_env("mlip-orb", tool="venv", python="3.12")


def test_an_existing_uv_made_env_is_installed_into_by_uv(monkeypatch, tmp_path: Path):
    """uv leaves its marker in the env's own pyvenv.cfg, and leaves no pip beside it.

    That env is reached from a machine whose *own* env venv made — which is the case uv gets
    used for here — so the detected tool would otherwise send `python -m pip` into an
    interpreter that has none.
    """
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    env_path = provision.backend_env_path("mlip-orb")
    _provisioned(tmp_path, "mlip-orb")
    (env_path / "pyvenv.cfg").write_text("home = /x\nuv = 0.12.5\n", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(provision.subprocess, "run", lambda argv, **_k: calls.append(argv))

    provision.build_backend_env("mlip-orb", tool="venv")

    assert len(calls) == 1 and calls[0][:3] == ["uv", "pip", "install"]


def test_an_existing_plain_env_is_installed_into_by_its_own_pip(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    env_path = provision.backend_env_path("mlip-orb")
    py = _provisioned(tmp_path, "mlip-orb")
    (env_path / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(provision.subprocess, "run", lambda argv, **_k: calls.append(argv))

    provision.build_backend_env("mlip-orb", tool="venv")

    assert calls[0][:4] == [str(py), "-m", "pip", "install"]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(subprocess.CalledProcessError(1, ["pip"]), id="tool-exits-nonzero"),
        pytest.param(FileNotFoundError(2, "No such file"), id="tool-not-on-path"),
    ],
)
def test_a_failed_build_carries_an_exit_code(monkeypatch, tmp_path: Path, failure: Exception):
    """Provisioning failures stay inside the exit-code contract the CLI depends on.

    `cli.backends_install` catches `ChemRefineError` and nothing else, so a bare
    `CalledProcessError` (no network, resolver conflict) or `FileNotFoundError` (conda is a
    shell function, not a binary) would reach the user as a traceback.
    """
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision, "_direct_url", lambda: None)

    def _boom(argv, **_k):
        raise failure

    monkeypatch.setattr(provision.subprocess, "run", _boom)
    with pytest.raises(BackendProvisionError, match="mlip-mace") as excinfo:
        provision.build_backend_env("mlip-mace", tool="venv")
    assert excinfo.value.exit_code == 9


def test_a_half_built_env_is_not_left_behind(monkeypatch, tmp_path: Path):
    """A create that succeeded and an install that failed must not look provisioned.

    `build_backend_env` short-circuits on `<env>/bin/python`, so a leftover env would be
    handed to every later run as though complete and fail on the backend import instead.
    """
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    env_path = provision.backend_env_path("mlip-mace")

    def _create_then_fail(argv, **_k):
        if argv[1:3] != ["-m", "venv"]:  # only the create step succeeds
            raise subprocess.CalledProcessError(1, argv)
        (env_path / "bin").mkdir(parents=True)
        (env_path / "bin" / "python").write_text("", encoding="utf-8")

    monkeypatch.setattr(provision.subprocess, "run", _create_then_fail)
    with pytest.raises(BackendProvisionError):
        provision.build_backend_env("mlip-mace", tool="venv")
    assert not env_path.exists()


# ---------------------------------------------------------------------------
# _install_target — PEP 610 source-matching
# ---------------------------------------------------------------------------


def test_install_target_index_install_pins_version(monkeypatch):
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    assert provision._install_target("pyscf") == f"chemrefine[pyscf]=={provision.__version__}"


def test_install_target_editable_local_dir(monkeypatch, tmp_path: Path):
    url = tmp_path.as_uri()
    direct = {"url": url, "dir_info": {"editable": True}}
    monkeypatch.setattr(provision, "_direct_url", lambda: direct)
    assert provision._install_target("mlip-mace") == f"chemrefine[mlip-mace] @ {url}"


def test_install_target_missing_source_dir_raises(monkeypatch, tmp_path: Path):
    url = (tmp_path / "gone").as_uri()
    direct = {"url": url, "dir_info": {"editable": True}}
    monkeypatch.setattr(provision, "_direct_url", lambda: direct)
    with pytest.raises(ConfigError, match="no longer exists"):
        provision._install_target("mlip-mace")


def test_install_target_git_install_pins_commit(monkeypatch):
    monkeypatch.setattr(
        provision,
        "_direct_url",
        lambda: {
            "url": "https://github.com/sterling-group/ChemRefine.git",
            "vcs_info": {"vcs": "git", "commit_id": "abc123"},
            "subdirectory": "pkg",
        },
    )
    assert provision._install_target("pyscf") == (
        "chemrefine[pyscf] @ git+https://github.com/sterling-group/ChemRefine.git"
        "@abc123#subdirectory=pkg"
    )


def test_install_target_git_install_without_ref(monkeypatch):
    direct = {"url": "https://example.com/repo.git", "vcs_info": {"vcs": "git"}}
    monkeypatch.setattr(provision, "_direct_url", lambda: direct)
    assert (
        provision._install_target("pyscf") == "chemrefine[pyscf] @ git+https://example.com/repo.git"
    )


def test_install_target_remote_archive_passes_url_through(monkeypatch):
    direct = {"url": "https://example.com/chemrefine.tar.gz", "archive_info": {}}
    monkeypatch.setattr(provision, "_direct_url", lambda: direct)
    assert provision._install_target("pyscf") == (
        "chemrefine[pyscf] @ https://example.com/chemrefine.tar.gz"
    )


def test_install_target_metadata_without_url_pins_version(monkeypatch):
    monkeypatch.setattr(provision, "_direct_url", lambda: {"dir_info": {}})
    assert provision._install_target("pyscf") == f"chemrefine[pyscf]=={provision.__version__}"


def test_direct_url_none_when_dist_missing(monkeypatch):
    def _raise(_name):
        raise provision.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(provision.importlib.metadata, "distribution", _raise)
    assert provision._direct_url() is None


def test_direct_url_none_without_metadata_file(monkeypatch):
    class _Dist:
        def read_text(self, _name):
            return None  # direct_url.json absent → index install

    monkeypatch.setattr(provision.importlib.metadata, "distribution", lambda _n: _Dist())
    assert provision._direct_url() is None


def _dist_with(monkeypatch, raw: str) -> None:
    class _Dist:
        def read_text(self, _name):
            return raw

    monkeypatch.setattr(provision.importlib.metadata, "distribution", lambda _n: _Dist())


def test_direct_url_parses_the_metadata(monkeypatch):
    _dist_with(monkeypatch, '{"url": "file:///src", "dir_info": {"editable": true}}')
    assert provision._direct_url() == {"url": "file:///src", "dir_info": {"editable": True}}


def test_direct_url_none_on_corrupt_metadata(monkeypatch):
    _dist_with(monkeypatch, "{not json")
    assert provision._direct_url() is None


def test_direct_url_none_on_non_dict_metadata(monkeypatch):
    _dist_with(monkeypatch, "[1, 2]")
    assert provision._direct_url() is None


# ---------------------------------------------------------------------------
# Launch seam — the resolved interpreter reaches the server + script commands
# ---------------------------------------------------------------------------


def test_extopt_server_cmd_uses_managed_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    py = _provisioned(tmp_path, "mlip-mace")
    ctx = _ctx(tmp_path, engine="mlip-extopt", options={"task_name": "mace_off"})
    cmd = get_engine("mlip-extopt")._server_cmd(ctx)
    assert cmd.startswith(f"{py} -m chemrefine.engines._backend_server.server")


def test_extopt_server_cmd_defaults_to_sys_executable(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    ctx = _ctx(tmp_path, engine="mlip-extopt", options={"task_name": "mace_off"})
    cmd = get_engine("mlip-extopt")._server_cmd(ctx)
    assert cmd.startswith(f"{sys.executable} -m chemrefine.engines._backend_server.server")


def test_script_run_block_uses_managed_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    py = _provisioned(tmp_path, "mlip-fairchem")
    ctx = _ctx(tmp_path, engine="mlip", options={"task_name": "omol"})
    block = get_engine("mlip").run_block(ctx, Path("step1_0.py"), Path("step1_0.json")).body
    assert block.splitlines()[-1] == f"{py} step1_0.py"


def test_script_run_block_defaults_to_sys_executable(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    ctx = _ctx(tmp_path, engine="mlip", options=None)
    block = get_engine("mlip").run_block(ctx, Path("step1_0.py"), Path("step1_0.json")).body
    assert block.splitlines()[-1] == f"{sys.executable} step1_0.py"


def test_script_run_block_honours_backend_python_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    ctx = _ctx(tmp_path, engine="mlip", options={"backend_python": "/envs/x/bin/python"})
    block = get_engine("mlip").run_block(ctx, Path("step1_0.py"), Path("step1_0.json")).body
    assert block.splitlines()[-1] == "/envs/x/bin/python step1_0.py"


def test_non_provisionable_script_engine_uses_own_interpreter(monkeypatch, tmp_path: Path):
    """A third-party ScriptEngine without ``backend_requirement`` runs this interpreter."""
    from chemrefine.engines._script import ScriptEngine

    class _PlainScript(ScriptEngine):
        name = "plain-script-test"
        label = "Plain"

    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    ctx = _ctx(tmp_path, engine="mlip", options={})
    block = _PlainScript().run_block(ctx, Path("step1_0.py"), Path("step1_0.json")).body
    assert block.splitlines()[-1] == f"{sys.executable} step1_0.py"


def test_non_provisionable_extopt_engine_uses_own_interpreter(monkeypatch, tmp_path: Path):
    """A third-party ExtOpt engine without ``backend_requirement`` serves from this interpreter."""
    from chemrefine.engines._options import EngineOptions
    from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator
    from chemrefine.engines.orca.extopt.engine import ExtOptOrcaEngine

    class _PlainExtOpt(ExtOptOrcaEngine):
        name = "plain-extopt-test"
        backend = "mlip"
        wrapper_filename = "plain.sh"
        options_cls = EngineOptions
        calculator_cls = MlipExtOptCalculator

    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    ctx = _ctx(tmp_path, engine="mlip-extopt", options={})
    cmd = _PlainExtOpt()._server_cmd(ctx)
    assert cmd.startswith(f"{sys.executable} -m chemrefine.engines._backend_server.server")


# ---------------------------------------------------------------------------
# known_backend_extras + the `chemrefine backends` CLI group
# ---------------------------------------------------------------------------


def test_known_backend_extras_is_registration_driven():
    """The union of every provisionable engine's declared extras — no hardcoded list."""
    from chemrefine.engines import known_backend_extras

    extras = known_backend_extras()
    assert {"mlip-fairchem", "mlip-mace", "mlip-sevenn", "mlip-orb", "mlip-chgnet", "pyscf"} <= (
        extras
    )


def test_backends_cli_list_and_path(monkeypatch, tmp_path: Path):
    from typer.testing import CliRunner

    from chemrefine.cli import app

    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    py = _provisioned(tmp_path, "pyscf")
    runner = CliRunner()

    result = runner.invoke(app, ["backends", "list"])
    assert result.exit_code == 0
    assert str(py) in result.output  # provisioned → shows the env python
    assert "not provisioned" in result.output  # the others aren't

    ok = runner.invoke(app, ["backends", "path", "pyscf"])
    assert ok.exit_code == 0 and ok.output.strip() == str(py)
    missing = runner.invoke(app, ["backends", "path", "mlip-mace"])
    assert missing.exit_code == 1


def test_backends_cli_install_validates_and_builds(monkeypatch, tmp_path: Path):
    from typer.testing import CliRunner

    import chemrefine.engines as engines_pkg
    from chemrefine.cli import app

    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    runner = CliRunner()

    bad = runner.invoke(app, ["backends", "install", "not-a-backend"])
    assert bad.exit_code != 0

    built: list[str] = []

    def _fake_build(extra: str, **_kwargs):
        built.append(extra)
        return _provisioned(tmp_path, extra)

    monkeypatch.setattr(engines_pkg, "build_backend_env", _fake_build)
    ok = runner.invoke(app, ["backends", "install", "mlip-mace", "pyscf"])
    assert ok.exit_code == 0
    assert built == ["mlip-mace", "pyscf"]


def test_backends_cli_install_surfaces_chemrefine_errors(monkeypatch, tmp_path: Path):
    """A ConfigError from provisioning (e.g. vanished source tree) exits cleanly, no traceback."""
    from typer.testing import CliRunner

    import chemrefine.engines as engines_pkg
    from chemrefine.cli import app

    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))

    def _fail(extra: str, **_kwargs):
        raise ConfigError("ChemRefine was installed from /gone, which no longer exists")

    monkeypatch.setattr(engines_pkg, "build_backend_env", _fail)
    result = CliRunner().invoke(app, ["backends", "install", "pyscf"])
    assert result.exit_code == ConfigError.exit_code
    assert "no longer exists" in result.output


def test_backends_cli_install_names_the_python_and_passes_the_override(monkeypatch, tmp_path):
    """The command says which Python it is building on, and `--python` reaches the builder."""
    from typer.testing import CliRunner

    import chemrefine.engines as engines_pkg
    from chemrefine.cli import app

    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    seen: list[str | None] = []

    def _fake_build(extra: str, **kwargs):
        seen.append(kwargs.get("python"))
        return _provisioned(tmp_path, extra)

    monkeypatch.setattr(engines_pkg, "build_backend_env", _fake_build)
    monkeypatch.setattr(
        engines_pkg, "resolve_base_python", lambda _e, **_k: provision.BasePython("3.12", "venv")
    )
    result = CliRunner().invoke(app, ["backends", "install", "mlip-orb", "--python", "3.12"])
    assert result.exit_code == 0
    assert "on Python 3.12" in result.output
    assert seen == ["3.12"]


def test_backends_cli_install_does_not_choose_for_an_env_that_exists(monkeypatch, tmp_path):
    """An existing env is extended on the Python it has — nothing to resolve, nothing to say."""
    from typer.testing import CliRunner

    import chemrefine.engines as engines_pkg
    from chemrefine.cli import app

    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    _provisioned(tmp_path, "pyscf")

    def _refuse(*_a, **_k):
        raise AssertionError("an existing env must not be asked which Python to build on")

    monkeypatch.setattr(engines_pkg, "resolve_base_python", _refuse)
    monkeypatch.setattr(engines_pkg, "build_backend_env", lambda extra, **_k: Path("/x"))
    result = CliRunner().invoke(app, ["backends", "install", "pyscf"])
    assert result.exit_code == 0 and "on Python" not in result.output


def test_a_gpu_step_is_refused_by_an_env_holding_only_the_base_stack(monkeypatch, tmp_path: Path):
    """The shared env's name proves nothing, so the env is asked.

    `backends/pyscf` exists whether it was built from `[pyscf]` or `[pyscf-gpu]`. Trusting
    the name let a GPU step start against the CPU stack and finish on CPU, reporting
    success. The refusal names the command that repairs it in place.
    """
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    _provisioned(tmp_path, "pyscf")  # the base stack: a real interpreter, no gpu4pyscf

    with pytest.raises(ConfigError, match="cannot import 'gpu4pyscf'"):
        provision.require_backend(BackendRequirement(extra="pyscf-gpu", import_name="gpu4pyscf"))


def test_an_unambiguous_env_is_trusted_without_being_run(monkeypatch, tmp_path: Path):
    """Only a shared env is probed; everywhere else the directory name is the proof.

    Probing every backend would make CI's `provisioned-backend` job — which symlinks a bare
    interpreter precisely to show the suite does not need real backends — require MACE and
    FAIRChem to be installed before it could pass.
    """
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    _provisioned(tmp_path, "mlip-mace")
    monkeypatch.setattr(
        provision, "_importable_by", lambda *_a: pytest.fail("an unshared env was probed")
    )

    provision.require_backend(BackendRequirement(extra="mlip-mace", import_name="mace"))


def test_the_probe_reports_what_an_interpreter_can_and_cannot_import(tmp_path: Path):
    """`_importable_by` asks the *other* interpreter, and survives one that cannot be run."""
    assert provision._importable_by(Path(sys.executable), "json") is True
    assert provision._importable_by(Path(sys.executable), "no_such_module_at_all") is False
    assert provision._importable_by(tmp_path / "not-an-interpreter", "json") is False


def test_a_failed_extension_leaves_the_existing_env_intact(monkeypatch, tmp_path: Path):
    """Only an env *this call* created is torn down on failure.

    A half-built env is worse than none, because its `bin/python` would be served to every
    later run — but that reasoning covers the env this call made, not one that was already
    working. Removing `backends/pyscf` because a later `pyscf-gpu` install hit a resolver
    wall would destroy a CPU stack the user still has every right to run.
    """
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    py = _provisioned(tmp_path, "pyscf")

    def _explode(argv, **_kwargs):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(provision.subprocess, "run", _explode)

    with pytest.raises(BackendProvisionError, match="pyscf-gpu"):
        provision.build_backend_env("pyscf-gpu")

    assert py.is_file(), "the pre-existing pyscf env was destroyed by a failed GPU install"
