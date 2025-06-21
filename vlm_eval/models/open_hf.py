"""
OpenHF  – generic loader for any open-weights VLM hosted on Hugging Face.

It works with either call pattern that `load_vlm` might use:
  • load_vlm("open-hf", model_id, run_dir)
  • load_vlm("open-hf", model_id, run_dir, model_dir=repo_path)

If only model_id is provided, we assume it *is* the HF repo path.
"""

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


class OpenHF:
    def __init__(self, model_id: str, model_dir: str = "", **_):
        # Accept either argument style
        repo = model_dir or model_id  # -- if model_dir is "", fall back to model_id

        # Load tokenizer / processor / model with trust_remote_code=True
        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            repo, trust_remote_code=True, resume_download=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )

    # --------------------------------------------------------------------- #
    # Public API expected by evaluate.py
    # --------------------------------------------------------------------- #
    def generate(self, image_path: str | Path, prompt: str) -> str:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            images=img, text=prompt, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=64)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
