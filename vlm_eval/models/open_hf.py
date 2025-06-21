from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch, os, base64
from PIL import Image
from io import BytesIO

class OpenHF:
    def __init__(self, model_dir, **kw):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, resume_download=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )

    def generate(self, image_path, prompt):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=64)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
