"""
PixtralVLLM – minimal vLLM-based wrapper that mimics the VLM interface.
"""

from pathlib import Path
from typing import Optional
import torch
from PIL import Image
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer

class PixtralVLLM:
    def __init__(self, model_id: str, hf_token: Optional[str] = None, **_):
        # vLLM auto-downloads the vision shards listed in tekken.json
        self.llm = LLM(
            model           = model_id,
            tokenizer_mode  = "mistral",
            config_format   = "mistral",
            download_dir    = Path.home() / ".cache" / "pixtral",
            trust_remote_code=True,
            dtype="float16",
            tokenizer_kwargs = {"trust_remote_code": True},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=hf_token
        )

        # vLLM uses an <image> token. No extra processor object needed —
        # the internal loader embeds the image for us.
        self.image_token = "<image>"
        self.sampler = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=64)

    # ---------------- harness API ----------------
    def generate(self, image_path: str | Path, prompt: str) -> str:
        full_prompt = f"{self.image_token} {prompt.strip()}"
        # vLLM expects PIL image list + prompt list
        img = Image.open(image_path).convert("RGB")
        outputs = self.llm.generate(
            prompts=[full_prompt],
            images=[[img]],          # list of list for batch axis
            sampling_params=self.sampler,
        )
        return outputs[0].outputs[0].text.strip()

    @property
    def image_processor(self):
        # Not used by this loader – harness passes raw image paths.
        return None

    def get_prompt_fn(self, dataset_family: str):
        return lambda q: q
