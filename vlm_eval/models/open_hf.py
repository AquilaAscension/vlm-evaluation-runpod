"""
OpenHF – generic loader for any open-weights VLM hosted on Hugging Face.

It works with either call pattern that `load_vlm` might use:
  • load_vlm("open-hf", model_id, run_dir)
  • load_vlm("open-hf", model_id, run_dir, model_dir=repo_path)

If only model_id is provided, we assume it *is* the HF repo path.
"""
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import sentencepiece as spm

def _safe_load_tokenizer(repo: str, **auth) -> Tuple[object, str]:
    last_err = None

    # 1️⃣  Try the repo’s own (usually fast) tokenizer ― this is what Pixtral expects
    try:
        tok = AutoTokenizer.from_pretrained(
            repo,
            trust_remote_code=True,   # lets Pixtral’s custom class register itself
            use_fast=True,            # prefers tokenizer.json → avoids SentencePiece
            legacy=False,
            **auth
        )
        return tok, "auto-fast/custom"
    except Exception as e:
        last_err = e                     # remember the error and fall through

    # 2️⃣  Slow fallback (for odd cases that really do need the .model)
    try:
        tok = AutoTokenizer.from_pretrained(
            repo,
            trust_remote_code=True,
            use_fast=False,
            legacy=True,
            **auth
        )
        return tok, "auto-slow"
    except Exception as e:
        last_err = e

    # 3️⃣  If both routes explode, re-raise
    raise RuntimeError(f"[OpenHF] failed to load tokenizer for {repo}") from last_err

class OpenHF:
    def __init__(self, model_id: str, model_dir: str = "", hf_token: Optional[str] = None, **_):
        # Accept either argument style
        repo = model_dir or model_id  # -- if model_dir is "", fall back to model_id

        auth = {"token": hf_token} if hf_token else {}

        # ---- tokenizer (robust) ------------------------------------------------
        self.tokenizer, strategy = _safe_load_tokenizer(repo, **auth)
        print(f"[OpenHF] tokenizer loaded via «{strategy}»")

        # ---- processor & model -------------------------------------------------
        self.processor = AutoProcessor.from_pretrained(
            repo, trust_remote_code=True, resume_download=True, **auth
        )
        self.model = AutoModel.from_pretrained(
            repo,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            **auth,
        )

    def generate(self, image_path: str | Path, prompt: str) -> str:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=64)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @property
    def image_processor(self):
        return self.processor

    def get_prompt_fn(self, dataset_family: str):
        # ... (unchanged prompt-handling logic) ...
        return None
