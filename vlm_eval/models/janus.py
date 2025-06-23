# vlm_eval/models/janus.py
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from janus.models import MultiModalityCausalLM, VLChatProcessor   # Janus library
from vlm_eval.util.interfaces import VLM                          # repo-wide base class


class JanusVLM(VLM):
    """
    Thin adapter that lets the DeepSeek-Janus VLM comply with this repo’s VLM interface.
    """

    def __init__(
        self,
        model_family: str,                 # ← ignored but kept for uniform signature
        model_id: str,                     # ← ignored (we rely on `run_dir`)
        run_dir: Path | str,               # full HF repo-id, e.g. "deepseek-ai/Janus-Pro-7B"
        hf_token: Optional[str] = None,
        load_precision: str = "bf16",
        **_,
    ):
        repo = str(run_dir)                # load everything from this HF repo
        dtype = torch.bfloat16 if load_precision.startswith("bf") else torch.float16
        auth  = {"token": hf_token} if hf_token else {}

        # --- Janus tokenizer / processor / model ---------------------------------
        self.processor = VLChatProcessor.from_pretrained(repo, **auth)
        self.tokenizer = self.processor.tokenizer

        self.model = MultiModalityCausalLM.from_pretrained(
            repo,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            **auth,
        )

        # Janus does its own image preprocessing inside VLChatProcessor
        self.image_processor = None

    # -------------------------------------------------------------------------
    def get_prompt_fn(self, dataset_family: str):
        """
        Janus VQA prompts need the <image_placeholder> token.
        """
        def _wrap(question: str) -> str:
            return f"<image_placeholder>\n{question}"
        return _wrap

    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, image_path: str | Path, prompt: str) -> str:
        img = Image.open(image_path).convert("RGB")

        conv = [
            {"role": "<|User|>",      "content": f"<image_placeholder>\n{prompt}", "images": [img]},
            {"role": "<|Assistant|>", "content": ""},
        ]

        inputs = self.processor(
            conversations=conv,
            images=[img],
            force_batchify=True
        ).to(self.model.device, dtype=self.model.dtype)

        embeds = self.model.prepare_inputs_embeds(**inputs)
        output = self.model.language_model.generate(
            inputs_embeds=embeds,
            attention_mask=inputs.attention_mask,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
