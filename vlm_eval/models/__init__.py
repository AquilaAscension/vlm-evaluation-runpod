from pathlib import Path
from typing import Optional

from importlib import import_module
from vlm_eval.util.interfaces import VLM

# Avoid importing heavy deps unless needed
_FAMILY2MODULE = {
    "instruct-blip": ("vlm_eval.models.instructblip", "InstructBLIP"),
    "prismatic": ("vlm_eval.models.prismatic", "PrismaticVLM"),
    "open-hf": ("vlm_eval.models.open_hf", "OpenHF"),
    "anthropic": ("vlm_eval.models.anthropic_loader", "AnthropicVLM"),
    "google": ("vlm_eval.models.google_loader", "GeminiVLM"),
    "janus": ("vlm_eval.models.janus", "JanusVLM"),
}

# Initializer dispatch table (values lazily filled)
FAMILY2INITIALIZER = {k: None for k in _FAMILY2MODULE}

def _get_initializer(family: str):
    if FAMILY2INITIALIZER[family] is None:
        mod_name, cls_name = _FAMILY2MODULE[family]
        mod = import_module(mod_name)
        FAMILY2INITIALIZER[family] = getattr(mod, cls_name)
    return FAMILY2INITIALIZER[family]


def load_vlm(
    model_family: str,
    model_id: str,
    run_dir: Path,
    hf_token: Optional[str] = None,
    ocr: Optional[bool] = False,
    load_precision: str = "bf16",
    max_length=128,
    temperature=1.0,
) -> VLM:
    if model_family == "open-hf":
        # ------------------------------------------------------------------
        # Pass the **full repo-id** (run_dir) to OpenHF as its first arg
        # ------------------------------------------------------------------
        cls = _get_initializer("open-hf")
        return cls(
            str(run_dir),
            hf_token=hf_token,
            load_precision=load_precision,
            max_length=max_length,
            temperature=temperature,
            ocr=ocr,
        )

    # all other families stay unchanged
    assert model_family in _FAMILY2MODULE, \
        f"Model family `{model_family}` not supported!"
    cls = _get_initializer(model_family)
    return cls(
        model_family=model_family,
        model_id=model_id,
        run_dir=run_dir,
        hf_token=hf_token,
        load_precision=load_precision,
        max_length=max_length,
        temperature=temperature,
        ocr=ocr,
    )

