"""
inputgen.py

Workflow YAML generation + normalization + validation (+ optional LLM integration).

Goal:
- Take a user request (text) -> produce a ChemRefine workflow config dict
- Normalize (auto-correct common issues)
- Validate (raise on semantic issues)
- Optionally: run an LLM with a strict JSON Schema to draft the config

This module does NOT generate ORCA input files or MLFF training templates.
Those remain in templates/ and the step{n}.inp files you already use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import logging
import ollama
import json
logger = logging.getLogger(__name__)




# ---------------------------
# Canonical allowed values
# ---------------------------

VALID_OPERATIONS = {
    "OPT+SP",
    "GOAT",
    "PES",
    "DOCKER",
    "SOLVATOR",
    "MLFF_TRAIN",
    "MLIP_TRAIN",
}

VALID_ENGINES = {"dft", "mlff", "mlip"}

# Defaults for convenience (these match your run() defaults)
DEFAULT_TOP_LEVEL = {
    "charge": 0,
    "multiplicity": 1,
}

DEFAULT_MLFF = {
    "model_name": "medium",
    "task_name": "mace_off",
    "bind": "127.0.0.1:8888",
    "device": "cuda",
}

DEFAULT_MLIP = {
    "model_name": "chgnet",
    "device": "cuda",
}

# Training keys actually used by MLFFTrainer today
ALLOWED_TRAINER_KEYS = {"device", "valid_fraction", "seed", "model_name", "task_name"}  # model/task optional metadata
ALLOWED_MLFF_KEYS = {"model_name", "task_name", "bind", "device"}
ALLOWED_MLIP_KEYS = {"model_name", "task_name", "bind", "device"}  # adjust if you later diverge


# ---------------------------
# Errors + diagnostics
# ---------------------------


from abc import ABC, abstractmethod

def ollama_health_check(host: str = "http://localhost:11434") -> None:
    """
    Raise a friendly error if Ollama isn't reachable.
    Uses the ollama Python library.
    """
    try:
        client = ollama.Client(host=host)
        client.list()  # should succeed if server is up
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {host}. Start Ollama (app running or `ollama serve`). Error: {e}"
        ) from e


def ollama_list_models(host: str = "http://localhost:11434") -> List[str]:
    """
    Return a list of installed Ollama model names (e.g., ['llama3.1:8b']).
    """
    client = ollama.Client(host=host)
    out = client.list()
    models = out.get("models", []) if isinstance(out, dict) else []
    return [m.get("name") for m in models if isinstance(m, dict) and m.get("name")]

class LLMClient(ABC):
    """
    Abstract base class for LLM backends used by inputgen.py.
    """

    @abstractmethod
    def generate_config(
        self,
        user_request: str,
        *,
        schema: Dict[str, Any],
        system_prompt: str,
    ) -> Dict[str, Any]:
        """
        Return a Python dict matching the provided JSON schema.
        """
        raise NotImplementedError


@dataclass
class ValidationIssue:
    path: str
    message: str
    fix: Optional[str] = None


class WorkflowConfigError(ValueError):
    """Raised when a workflow config fails validation."""

    def __init__(self, issues: List[ValidationIssue]):
        self.issues = issues
        msg = "\n".join([f"- {i.path}: {i.message}" + (f" | fix: {i.fix}" if i.fix else "") for i in issues])
        super().__init__(msg)


# ---------------------------
# Normalization helpers
# ---------------------------

def _normalize_operation(op: Any) -> str:
    s = str(op or "").strip()
    s_up = s.upper()

    # Remove separators for alias matching
    compact = re.sub(r"[\s_\-]+", "", s_up)

    aliases = {
        "OPTSP": "OPT+SP",
        "OPTSINGLEPOINT": "OPT+SP",
        "MLFFTRAIN": "MLFF_TRAIN",
        "MLIPTRAIN": "MLIP_TRAIN",
        "DOCK": "DOCKER",
    }
    return aliases.get(compact, s_up)


def _normalize_engine(engine: Any) -> str:
    s = str(engine or "dft").strip().lower()
    aliases = {
        "orca": "dft",
        "dft": "dft",
        "mlff": "mlff",
        "mlip": "mlip",
        # if user says "ml" treat as mlff by default
        "ml": "mlff",
    }
    return aliases.get(s, s)


def normalize_workflow_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and auto-correct common issues without raising.

    - Fix known typos (orca_excutable -> orca_executable)
    - Fill top-level defaults (charge, multiplicity)
    - Normalize casing for operation/engine
    - Ensure sample_type.parameters exists if sample_type present
    - Auto-fill mlff/mlip blocks with defaults if engine requires them
    - Renumber steps sequentially (1..N) and sort by provided step
    """
    cfg = dict(cfg or {})

    # typo fix
    if "orca_executable" not in cfg and "orca_excutable" in cfg:
        cfg["orca_executable"] = cfg.pop("orca_excutable")

    # defaults
    for k, v in DEFAULT_TOP_LEVEL.items():
        cfg.setdefault(k, v)

    cfg.setdefault("steps", [])
    if not isinstance(cfg["steps"], list):
        cfg["steps"] = []

    steps_in = cfg["steps"]
    norm_steps: List[Dict[str, Any]] = []

    for idx, step in enumerate(steps_in, start=1):
        if not isinstance(step, dict):
            continue
        s = dict(step)

        # normalize op/engine
        s["operation"] = _normalize_operation(s.get("operation"))
        op = s["operation"]

        # keep original step number for sorting if user provided it
        raw_step_num = s.get("step", idx)
        try:
            raw_step_num = int(raw_step_num)
        except Exception:
            raw_step_num = idx
        s["_raw_step"] = raw_step_num

        if op not in {"MLFF_TRAIN", "MLIP_TRAIN"}:
            s["engine"] = _normalize_engine(s.get("engine", "dft"))

        # normalize sample_type structure
        if "sample_type" in s and s["sample_type"] is not None:
            if not isinstance(s["sample_type"], dict):
                s["sample_type"] = {"method": str(s["sample_type"]), "parameters": {}}
            s["sample_type"].setdefault("parameters", {})

        # auto-fill mlff/mlip blocks
        eng = s.get("engine")
        if eng == "mlff":
            mlff = s.get("mlff") or {}
            if isinstance(mlff, dict):
                merged = dict(DEFAULT_MLFF)
                merged.update(mlff)
                s["mlff"] = merged

        if eng == "mlip":
            mlip = s.get("mlip") or {}
            if isinstance(mlip, dict):
                merged = dict(DEFAULT_MLIP)
                merged.update(mlip)
                s["mlip"] = merged

        # training: force trainer dict
        if op in {"MLFF_TRAIN", "MLIP_TRAIN"}:
            trainer = s.get("trainer") or {}
            if not isinstance(trainer, dict):
                trainer = {}
            s["trainer"] = trainer

        norm_steps.append(s)

    # sort by user-provided step, then renumber sequentially
    norm_steps.sort(key=lambda x: x.get("_raw_step", 10**9))
    for i, s in enumerate(norm_steps, start=1):
        s["step"] = i
        s.pop("_raw_step", None)

    cfg["steps"] = norm_steps
    return cfg


# ---------------------------
# Validation
# ---------------------------

def validate_against_request(cfg: Dict[str, Any], user_request: str) -> None:
    issues: List[ValidationIssue] = []
    text = (user_request or "").lower()

    nm_requested = any(
        kw in text for kw in [
            "normal mode", "normal-mode", "normal mode sampling",
            "vibrational mode", "imaginary mode", "mode displacement"
        ]
    )

    for step in cfg.get("steps", []):
        sn = step.get("step", "?")
        if step.get("normal_mode_sampling") is True and not nm_requested:
            issues.append(
                ValidationIssue(
                    f"steps[{sn}].normal_mode_sampling",
                    "normal_mode_sampling was enabled but the user did not request it.",
                    fix="Remove normal_mode_sampling (or explicitly request it in the user prompt).",
                )
            )

        # also disallow stray parameters if nm_sampling is not true
        if "normal_mode_sampling_parameters" in step and not step.get("normal_mode_sampling"):
            issues.append(
                ValidationIssue(
                    f"steps[{sn}].normal_mode_sampling_parameters",
                    "normal_mode_sampling_parameters present but normal_mode_sampling is not enabled.",
                    fix="Remove normal_mode_sampling_parameters.",
                )
            )

    if issues:
        raise WorkflowConfigError(issues)


def validate_workflow_cfg(cfg: Dict[str, Any], *, require_paths: bool = True) -> None:
    """
    Validate normalized config. Raises WorkflowConfigError with issues list.

    require_paths=True means top-level dirs/executable must be present.
    If you want a "template YAML" with placeholders, call with require_paths=False.
    """
    issues: List[ValidationIssue] = []
    cfg = cfg or {}

    # -----------------------
    # Top-level required paths
    # -----------------------
    if require_paths:
        for k in ("template_dir", "scratch_dir", "output_dir", "orca_executable"):
            if not cfg.get(k):
                issues.append(ValidationIssue(k, "Missing or empty.", fix=f"Set '{k}' at top-level."))

    # -----------------------
    # Steps existence/type
    # -----------------------
    steps = cfg.get("steps")
    if not steps or not isinstance(steps, list):
        issues.append(ValidationIssue("steps", "Must be a non-empty list.", fix="Add at least one workflow step."))

    if issues:
        raise WorkflowConfigError(issues)

    # -----------------------
    # Per-step validation
    # -----------------------
    for step in steps:
        sn = step.get("step", "?")
        op = step.get("operation")

        # ---- operation enum ----
        if op not in VALID_OPERATIONS:
            issues.append(
                ValidationIssue(
                    f"steps[{sn}].operation",
                    f"Invalid operation '{op}'.",
                    fix=f"Use one of: {sorted(VALID_OPERATIONS)}",
                )
            )
            continue

        is_training = op in {"MLFF_TRAIN", "MLIP_TRAIN"}

        # ---- trainer allowed only for training steps ----
        if (not is_training) and ("trainer" in step) and (step.get("trainer") not in (None, {})):
            issues.append(
                ValidationIssue(
                    f"steps[{sn}].trainer",
                    "trainer is only allowed for MLFF_TRAIN / MLIP_TRAIN steps.",
                    fix="Remove trainer from this step; use sample_type for sampling behavior.",
                )
            )

        # -----------------------
        # Training steps
        # -----------------------
        if is_training:
            # training can't be first step
            if sn == 1:
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}]",
                        "Training cannot be step 1 (requires previous coords/energies/forces).",
                        fix="Move training to step >= 2.",
                    )
                )

            # optional-but-recommended: forbid inference blocks and engine on training steps
            if "engine" in step and step.get("engine") is not None:
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].engine",
                        "Training steps should not specify an engine.",
                        fix="Remove engine from training steps.",
                    )
                )
            if "mlff" in step and step.get("mlff") not in (None, {}):
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].mlff",
                        "Training steps should not include an mlff inference block.",
                        fix="Remove mlff from training steps (trainer controls training).",
                    )
                )
            if "mlip" in step and step.get("mlip") not in (None, {}):
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].mlip",
                        "Training steps should not include an mlip inference block.",
                        fix="Remove mlip from training steps (trainer controls training).",
                    )
                )

            trainer = step.get("trainer", {})
            if not isinstance(trainer, dict):
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].trainer",
                        "Must be a mapping/dict.",
                        fix="Example: trainer: {device: cuda, valid_fraction: 0.1, seed: 42}",
                    )
                )
                continue

            unknown = set(trainer.keys()) - ALLOWED_TRAINER_KEYS
            if unknown:
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].trainer",
                        f"Unknown trainer keys: {sorted(unknown)}",
                        fix=f"Allowed: {sorted(ALLOWED_TRAINER_KEYS)}",
                    )
                )

            # require device + valid_fraction (don’t silently default these)
            if "device" not in trainer:
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].trainer.device",
                        "Missing trainer.device.",
                        fix="Set trainer.device to 'cuda' or 'cpu'.",
                    )
                )
            if "valid_fraction" not in trainer:
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].trainer.valid_fraction",
                        "Missing trainer.valid_fraction.",
                        fix="Set trainer.valid_fraction, e.g. 0.1",
                    )
                )

            device = trainer.get("device", "cuda")
            if device not in {"cuda", "cpu"}:
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].trainer.device",
                        f"Invalid device '{device}'.",
                        fix="Use 'cuda' or 'cpu'.",
                    )
                )

            vf = trainer.get("valid_fraction", None)
            if vf is not None:
                try:
                    vf = float(vf)
                    if not (0.0 < vf < 1.0):
                        issues.append(
                            ValidationIssue(
                                f"steps[{sn}].trainer.valid_fraction",
                                "Must be between 0 and 1.",
                                fix="Example: 0.1 (10% validation split)",
                            )
                        )
                except Exception:
                    issues.append(
                        ValidationIssue(
                            f"steps[{sn}].trainer.valid_fraction",
                            "Must be a float.",
                            fix="Example: 0.1",
                        )
                    )

            continue  # training steps stop here

        # -----------------------
        # Non-training steps: engine required
        # -----------------------
        eng = step.get("engine")
        if eng not in VALID_ENGINES:
            issues.append(
                ValidationIssue(
                    f"steps[{sn}].engine",
                    f"Invalid engine '{eng}'.",
                    fix=f"Use one of: {sorted(VALID_ENGINES)}",
                )
            )
            continue

        # -----------------------
        # Cross-field semantics: engine vs blocks
        # -----------------------
        if eng == "dft":
            if "mlff" in step and step.get("mlff") not in (None, {}):
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].mlff",
                        "mlff block is not allowed when engine is dft.",
                        fix="Remove mlff block or set engine to mlff.",
                    )
                )
            if "mlip" in step and step.get("mlip") not in (None, {}):
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].mlip",
                        "mlip block is not allowed when engine is dft.",
                        fix="Remove mlip block or set engine to mlip.",
                    )
                )

        if eng == "mlff":
            if "mlip" in step and step.get("mlip") not in (None, {}):
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].mlip",
                        "mlip block is not allowed when engine is mlff.",
                        fix="Remove mlip block or set engine to mlip.",
                    )
                )
            mlff = step.get("mlff")
            if not isinstance(mlff, dict):
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].mlff",
                        "engine=mlff requires an mlff block.",
                        fix="Provide mlff: {model_name, task_name, bind, device}",
                    )
                )
            else:
                unknown = set(mlff.keys()) - ALLOWED_MLFF_KEYS
                if unknown:
                    issues.append(
                        ValidationIssue(
                            f"steps[{sn}].mlff",
                            f"Unknown keys: {sorted(unknown)}",
                            fix=f"Allowed: {sorted(ALLOWED_MLFF_KEYS)}",
                        )
                    )

        if eng == "mlip":
            if "mlff" in step and step.get("mlff") not in (None, {}):
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].mlff",
                        "mlff block is not allowed when engine is mlip.",
                        fix="Remove mlff block or set engine to mlff.",
                    )
                )
            mlip = step.get("mlip")
            if not isinstance(mlip, dict):
                issues.append(
                    ValidationIssue(
                        f"steps[{sn}].mlip",
                        "engine=mlip requires an mlip block.",
                        fix="Provide mlip: {...}",
                    )
                )
            else:
                unknown = set(mlip.keys()) - ALLOWED_MLIP_KEYS
                if unknown:
                    issues.append(
                        ValidationIssue(
                            f"steps[{sn}].mlip",
                            f"Unknown keys: {sorted(unknown)}",
                            fix=f"Allowed: {sorted(ALLOWED_MLIP_KEYS)}",
                        )
                    )

    if issues:
        raise WorkflowConfigError(issues)



# ---------------------------
# JSON Schema (for LLM structured outputs)
# ---------------------------

def workflow_json_schema(require_paths: bool = False) -> Dict[str, Any]:
    """
    JSON Schema for "LLM drafts config dict".

    You can pass this to providers that support structured output / JSON schema.
    (OpenAI Structured Outputs, Claude Structured Outputs, Gemini Structured Outputs, etc.)
    """
    required_top = ["steps"]
    if require_paths:
        required_top += ["template_dir", "scratch_dir", "output_dir", "orca_executable"]

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "template_dir": {"type": "string"},
            "scratch_dir": {"type": "string"},
            "output_dir": {"type": "string"},
            "orca_executable": {"type": "string"},
            "initial_xyz": {"type": "string"},
            "charge": {"type": "integer"},
            "multiplicity": {"type": "integer"},
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/step"},
            },
        },
        "required": required_top,
        "$defs": {
            "sample_type": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "method": {"type": "string"},
                    "parameters": {"type": "object"},
                },
                "required": ["method", "parameters"],
            },
            "mlff_block": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "model_name": {"type": "string"},
                    "task_name": {"type": "string"},
                    "bind": {"type": "string"},
                    "device": {"type": "string"},
                },
            },
            "trainer_block": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "device": {"type": "string", "enum": ["cuda", "cpu"]},
                    "valid_fraction": {"type": "number"},
                    "seed": {"type": "integer"},
                    "model_name": {"type": "string"},
                    "task_name": {"type": "string"},
                },
            },
            "step": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "step": {"type": "integer"},
                    "operation": {"type": "string", "enum": sorted(list(VALID_OPERATIONS))},
                    "engine": {"type": "string", "enum": sorted(list(VALID_ENGINES))},
                    "charge": {"type": "integer"},
                    "multiplicity": {"type": "integer"},
                    "sample_type": {"$ref": "#/$defs/sample_type"},
                    "normal_mode_sampling": {"type": "boolean"},
                    "normal_mode_sampling_parameters": {"type": "object"},
                    "mlff": {"$ref": "#/$defs/mlff_block"},
                    "mlip": {"$ref": "#/$defs/mlff_block"},
                    "trainer": {"$ref": "#/$defs/trainer_block"},
                },
                "required": ["step", "operation"],
            },
        },
    }
    return schema


# ---------------------------
# LLM interface (provider-agnostic)
# ---------------------------

class OllamaClient(LLMClient):
    """
    Ollama-backed local LLM client using the `ollama` Python library.

    Uses chat() with:
      - messages: system + user
      - format: JSON schema (structured outputs)
      - options: temperature=0 for determinism

    Requires:
      pip install ollama
      ollama serve  (or the Ollama desktop app running)
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        self.model = model
        self.host = host
        self.temperature = float(temperature)

    def generate_config(
        self,
        user_request: str,
        *,
        schema: Dict[str, Any],
        system_prompt: str,
    ) -> Dict[str, Any]:
        try:
            client = ollama.Client(host=self.host)

            resp = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Create a ChemRefine workflow configuration that satisfies the schema.\n"
                            "Return ONLY JSON (no markdown, no commentary).\n\n"
                            f"USER REQUEST:\n{user_request}"
                        ),
                    },
                ],
                # Key part: structured output
                format=schema,
                options={"temperature": self.temperature},
            )
        except Exception as e:
            raise RuntimeError(
                f"Ollama chat failed. Is Ollama running at {self.host}? "
                f"Model='{self.model}'. Error: {e}"
            ) from e

        # Ollama returns resp["message"]["content"] as a string (JSON)
        try:
            content = resp["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected ollama response: {resp}") from e

        # Parse JSON content
        try:
            obj = json.loads(content)
        except Exception as e:
            raise RuntimeError(
                f"Ollama returned non-JSON content (first 500 chars): {str(content)[:500]}"
            ) from e

        if not isinstance(obj, dict):
            raise RuntimeError(f"Ollama returned JSON but not an object/dict: {type(obj)}")

        return obj



def default_system_prompt() -> str:
    return (
        "You are generating a ChemRefine workflow configuration as JSON.\n"
        "Rules:\n"
        "- Output must be valid JSON matching the provided JSON Schema.\n"
        "- Use operation enums exactly (OPT+SP, GOAT, PES, DOCKER, SOLVATOR, MLFF_TRAIN, MLIP_TRAIN).\n"
        "- For non-training steps include engine (dft/mlff/mlip). For training steps include trainer.\n"
        "- Keep it minimal; specifics belong to ORCA/MACE template files.\n"
    )


# ---------------------------
# High-level orchestration
# ---------------------------

def generate_workflow_cfg(
    user_request: str,
    *,
    llm: Optional[LLMClient] = None,
    require_paths: bool = False,
) -> Dict[str, Any]:
    """
    Generate workflow config dict from a user request.

    If llm is provided:
      - Ask LLM for a draft config (dict)
      - Normalize
      - Validate (raise if invalid)

    If llm is None:
      - Raise NotImplementedError (you can implement a rule-based parser later)
    """
    if llm is None:
        raise NotImplementedError(
            "No LLM client provided. Either pass llm=... or implement a rule-based generator."
        )

    schema = workflow_json_schema(require_paths=require_paths)
    draft = llm.generate_config(
        user_request,
        schema=schema,
        system_prompt=default_system_prompt(),
    )

    cfg = normalize_workflow_cfg(draft)
    validate_workflow_cfg(cfg, require_paths=require_paths)
    return cfg


def generate_workflow_yaml(
    user_request: str,
    *,
    llm: Optional[LLMClient] = None,
    require_paths: bool = False,
) -> str:
    """
    Convenience: return YAML string (not saved to disk).
    """
    cfg = generate_workflow_cfg(user_request, llm=llm, require_paths=require_paths)

    # Lazy import so ChemRefine core doesn't need yaml unless this path is used
    import yaml
    return yaml.safe_dump(cfg, sort_keys=False)


def generate_with_autocorrect(
    user_request: str,
    *,
    llm: LLMClient,
    require_paths: bool = False,
    max_attempts: int = 2,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    LLM-backed generate -> normalize -> validate loop with "error feedback".

    Attempt 1: generate config
    If invalid: feed the validation errors back to the LLM and retry (up to max_attempts).

    verbose=True prints each attempt + errors.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_err: Optional[WorkflowConfigError] = None
    schema = workflow_json_schema(require_paths=require_paths)
    base_prompt = default_system_prompt()

    for attempt in range(1, max_attempts + 1):
        extra = ""
        if last_err is not None:
            extra = (
                "\nThe previous output failed validation with these issues:\n"
                + "\n".join([f"- {i.path}: {i.message} (fix: {i.fix})" for i in last_err.issues])
                + "\nReturn a corrected JSON that satisfies the schema and resolves all issues.\n"
            )

        if verbose:
            print(f"\n===== LLM ATTEMPT {attempt}/{max_attempts} =====")
            if extra:
                print("Validation feedback sent to LLM:")
                print(extra)

        # 1) Ask LLM for a draft (dict)
        draft = llm.generate_config(
            user_request,
            schema=schema,
            system_prompt=base_prompt + extra,
        )

        if verbose:
            import json as _json
            print("Raw LLM output (dict):")
            print(_json.dumps(draft, indent=2))

        # 2) Normalize
        cfg = normalize_workflow_cfg(draft)

        if verbose:
            import json as _json
            print("After normalization:")
            print(_json.dumps(cfg, indent=2))

        # 3) Validate
        try:
            validate_workflow_cfg(cfg, require_paths=require_paths)
            validate_against_request(cfg, user_request=user_request)
            if verbose:
                print("✓ Validation passed")
            return cfg
        except WorkflowConfigError as e:
            last_err = e
            if verbose:
                print("✗ Validation failed:")
                print(str(e))

    # If still invalid after retries, raise the last error
    assert last_err is not None
    raise last_err





