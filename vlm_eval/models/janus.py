# vlm_eval/models/janus.py
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from PIL import Image

from transformers import AutoTokenizer  # only for decode convenience
from janus.models import MultiModalityCausalLM, VLChatProcessor

from vlm_eval.util.interfaces import VLM   # base class in the repo


class JanusVLM(VLM):
    """
    Thin wrapper so Janus fits the repo's VLM interface:
    ´generate(image_path, prompt) -> str´
    """

    def __init__(
        self,
        model_family: str,
        model_id: str,
        model_dir: str,
        hf_token: Optional[str] = None,
        load_precision: str = "bfloat16",
        **kw,
    ):
        auth = {"token": hf_token} if hf_token else {}
        dtype = torch.bfloat16 if load_precision.startswith("bf") else torch.float16

        self.processor = VLChatProcessor.from_pretrained(model_dir, **auth)
        self.tokenizer = self.processor.tokenizer

        self.model = MultiModalityCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            **auth,
        )

        # one-liner wrapper that repo’s evaluation harness expects
        self.image_processor = None   # not used by Janus

    # ------------------------------------------------------------------
    def get_prompt_fn(self, dataset_family: str):
        """
        Returns a closure the harness will call to wrap each VQA question.
        For Janus we need to insert <image_placeholder>.
        """
        def _fn(question: str) -> str:
            return f"<image_placeholder>\n{question}"
        return _fn

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, image_path: str, prompt: str) -> str:
        img = Image.open(image_path).convert("RGB")

        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{prompt}", "images": [img]},
            {"role": "<|Assistant|>", "content": ""},
        ]

        inputs = self.processor(
            conversations=conversation,
            images=[img],
            force_batchify=True
        ).to(self.model.device, dtype=self.model.dtype)

        embeds = self.model.prepare_inputs_embeds(**inputs)
        out = self.model.language_model.generate(
            inputs_embeds=embeds,
            attention_mask=inputs.attention_mask,
            max_new_tokens=64,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
