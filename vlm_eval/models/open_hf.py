"""
OpenHF – generic loader for any open-weights VLM on HuggingFace.
"""

from pathlib import Path, PosixPath
import os, shutil
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoConfig, AutoModelForCausalLM

from vlm_eval.util.resolve_tokenizer import resolve_tokenizer

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
        hf_auth = {"token": hf_token} if hf_token else {}

        # ------------------------------------------------------------------ #
        #  🔑 robust tokenizer loading
        # ------------------------------------------------------------------ #
        self.tokenizer = resolve_tokenizer(repo)
        strategy = getattr(self.tokenizer, "_meta", {}).get("strategy", "auto")
        print(f"[OpenHF] tokenizer loaded via «{strategy}»")

        # ------------------------------------------------------------------ #
        #  processor & model
        # ------------------------------------------------------------------ #

        try:
            self.processor = AutoProcessor.from_pretrained(
                repo,
                trust_remote_code=True,
                resume_download=True,
                **hf_auth,
            )
        except Exception as e:
            print(f"[OpenHF] No processor found → {e}")
            self.processor = None  # or just use tokenizer directly

        snap = Path(model_dir)               # model_dir is already passed in

        src = snap / "consolidated.safetensors"
        dst = snap / "model.safetensors"     # one of HF’s recognised names

        if src.exists() and not dst.exists():
            try:
                os.symlink(src, dst)         # cheap if filesystem allows
            except OSError:
                shutil.copyfile(src, dst)    # fallback inside containers

        is_pixtral = "pixtral" in repo.lower() or "janus" in repo.lower()  # add others as needed

        if is_pixtral:
            # 🔧 CONFIG-FIX ────────────────────────────────────────────────
            from glob import glob
            import json

            # 1) look for any alt config that already mentions 12B
            alt_cfg = next(
                (Path(p) for p in snap.glob("**/config*12*.json")), None
            )
            if alt_cfg:
                cfg = AutoConfig.from_pretrained(alt_cfg, trust_remote_code=True)
            else:
                # 2) load default and patch obvious size fields
                cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True, **hf_auth)
                if cfg.hidden_size == 4096:            # clearly the 7-B template
                    cfg.hidden_size         = 5120
                    cfg.intermediate_size   = 13824     # 5120 * 2.7  (Mistral ratio)
                    cfg.num_attention_heads = 40        # 5120 / 128
                    cfg.num_key_value_heads = 8
                    print("[OpenHF] patched config → 12-B dims (hidden 5120)")
            # ──────────────────────────────────────────────────────────────

            self.model = AutoModelForCausalLM.from_pretrained(
                repo,
                config=cfg,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                **hf_auth,
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
        return lambda q: q
