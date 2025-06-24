"""
OpenHF – generic loader for any open-weights VLM on HuggingFace.
"""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# <-- our new smart helper
from vlm_eval.util.auto_tokenizer import build_intelligent_tokenizer


class OpenHF:
    """
    Thin wrapper that fits the VLM interface used by the evaluation harness.
    """

    def __init__(
        self,
        model_id: str,
        model_dir: str = "",
        hf_token: Optional[str] = None,
        **_,
    ):
        # ------------------------------------------------------------------ #
        #  figure out which repo path to load
        # ------------------------------------------------------------------ #
        repo = model_dir or model_id          # if model_dir == "", fall back to model_id
        auth = {"token": hf_token} if hf_token else {}

        # ------------------------------------------------------------------ #
        #  🔑 robust tokenizer loading
        # ------------------------------------------------------------------ #
        self.tokenizer = build_intelligent_tokenizer(repo, token=hf_token)
        strategy = getattr(self.tokenizer, "_meta", {}).get("strategy", "auto")
        print(f"[OpenHF] tokenizer loaded via «{strategy}»")

        # ------------------------------------------------------------------ #
        #  processor & model
        # ------------------------------------------------------------------ #
        self.processor = AutoProcessor.from_pretrained(
            repo,
            trust_remote_code=True,
            resume_download=True,
            **auth,
        )

        self.model = AutoModel.from_pretrained(
            repo,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            **auth,
        )

    # ====================================================================== #
    #  Required interface methods
    # ====================================================================== #
    def generate(self, image_path: str | Path, prompt: str) -> str:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(
            self.model.device
        )
        output_ids = self.model.generate(**inputs, max_new_tokens=64)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    @property
    def image_processor(self):
        return self.processor

    def get_prompt_fn(self, dataset_family: str):
        # most vision-llms use the plain question for VQA; override if you need special tokens
        return None
