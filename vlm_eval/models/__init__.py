from pathlib import Path
from typing import Optional

from vlm_eval.util.interfaces import VLM

from .instructblip import InstructBLIP
#from .llava import LLaVa
from .prismatic import PrismaticVLM
from .open_hf import OpenHF
from .anthropic_loader import AnthropicVLM
from .google_loader import GeminiVLM
from .janus import JanusVLM

# === Initializer Dispatch by Family ===
FAMILY2INITIALIZER = {
    "instruct-blip": InstructBLIP,
    "prismatic":      PrismaticVLM,
    "open-hf":        OpenHF,
    "anthropic":      AnthropicVLM,
    "google":         GeminiVLM,
    "janus": JanusVLM
}

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
        return OpenHF(
            str(run_dir),                 # <— deepseek-ai/Janus-Pro-7B
            hf_token=hf_token,
            load_precision=load_precision,
            max_length=max_length,
            temperature=temperature,
            ocr=ocr,
        )

    # all other families stay unchanged
    assert model_family in FAMILY2INITIALIZER, \
        f"Model family `{model_family}` not supported!"
    return FAMILY2INITIALIZER[model_family](
        model_family=model_family,
        model_id=model_id,
        run_dir=run_dir,
        hf_token=hf_token,
        load_precision=load_precision,
        max_length=max_length,
        temperature=temperature,
        ocr=ocr,
    )

