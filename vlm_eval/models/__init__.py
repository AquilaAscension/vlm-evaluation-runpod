from pathlib import Path
from typing import Optional
from vlm_eval.util.interfaces import VLM

FAMILY2INITIALIZER = {}

try:
    from .instructblip import InstructBLIP
    FAMILY2INITIALIZER["instruct-blip"] = InstructBLIP
except ImportError:
    pass

try:
    from .prismatic import PrismaticVLM
    FAMILY2INITIALIZER["prismatic"] = PrismaticVLM
except ImportError:
    pass

try:
    from .open_hf import OpenHF
    FAMILY2INITIALIZER["open-hf"] = OpenHF
except ImportError:
    pass

try:
    from .anthropic_loader import AnthropicVLM
    FAMILY2INITIALIZER["anthropic"] = AnthropicVLM
except ImportError:
    pass

try:
    from .google_loader import GeminiVLM
    FAMILY2INITIALIZER["google"] = GeminiVLM
except ImportError:
    pass

try:
    from .janus import JanusVLM
    FAMILY2INITIALIZER["janus"] = JanusVLM
except ImportError:
    pass

try:
    from .pixtral import PixtralVLLM
    FAMILY2INITIALIZER["pixtral"] = PixtralVLLM
except ImportError:
    pass



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
    
    if "pixtral" in model_id.lower():
        from .pixtral import PixtralVLLM
        return PixtralVLLM(
            model_id,
            hf_token=hf_token,
            max_length=max_length,
            temperature=temperature,
        )

    if model_family == "open-hf":
        # ------------------------------------------------------------------
        # Pass the **full repo-id** (run_dir) to OpenHF as its first arg
        # ------------------------------------------------------------------
        return OpenHF(
            str(run_dir),
            hf_token=hf_token,
            load_precision=load_precision,
            max_length=max_length,
            temperature=temperature,
            ocr=ocr,
        )
    
    elif model_family == "open-hf" and "pixtral" in model_id.lower():
        return PixtralVLLM(model_id=model_id, hf_token=hf_token)


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

