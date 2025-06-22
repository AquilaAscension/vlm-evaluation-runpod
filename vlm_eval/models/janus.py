
from pathlib import Path
from typing import Optional
import torch
from janus.modeling import JanusForCausalLM  # hypothetical; import from DeepSeek repo
from transformers import AutoTokenizer, AutoProcessor

from vlm_eval.util.interfaces import VLM

class JanusVLM(VLM):
    def __init__(
        self,
        model_family: str,
        model_id: str,
        run_dir: Path,
        hf_token: Optional[str] = None,
        load_precision="bf16",
        **kw,
    ):
        repo = model_id if "/" in model_id else f"deepseek-ai/{model_id}"
        auth = {"token": hf_token} if hf_token else {}
        self.tokenizer  = AutoTokenizer.from_pretrained(repo, **auth)
        self.processor  = AutoProcessor.from_pretrained(repo, **auth)
        self.model      = JanusForCausalLM.from_pretrained(
                              repo,
                              torch_dtype=getattr(torch, load_precision),
                              device_map="auto",
                              **auth)
        # Janus’ own generate args may differ:
        self.gen_kwargs = dict(max_new_tokens=64, temperature=0.2)

    def get_prompt_fn(self, dataset_name):          # required by harness
        return lambda q: q

    def generate(self, image_path, prompt):
        inputs = self.processor(images=image_path, text=prompt, return_tensors="pt").to(self.model.device)
        out_ids = self.model.generate(**inputs, **self.gen_kwargs)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
