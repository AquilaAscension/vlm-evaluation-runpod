#!/usr/bin/env bash
set -e
echo "🔧  Applying RunPod-friendly patches …"

# 1 ─── Update dependencies ───────────────────────────────────
apply_pyproject_fix() {
  sed -i 's/^transformers.*$/transformers = ">=4.38,<4.40"/' pyproject.toml
  grep -q 'anthropic' pyproject.toml || \
    sed -i '/^pydantic/a anthropic = "^0.26"\ngoogle-generativeai = "^0.5"\nerniebot = "^1.0"' pyproject.toml
}
apply_pyproject_fix

# 2 ─── Remove LLaVA auto-import to stop duplicate key crash ─
sed -i '/from \.llava /s/^/#/' vlm_eval/models/__init__.py
sed -i '/"llava-v15":/d'       vlm_eval/models/__init__.py

# 3 ─── Add Open-HF + closed-API loaders ─────────────────────
cat > vlm_eval/models/open_hf.py <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch, base64
from PIL import Image
class OpenHF:
    def __init__(self, model_dir, **kw):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
    def generate(self, image_path, prompt, **_):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=64)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
PY

cat > vlm_eval/models/anthropic_loader.py <<'PY'
import os, base64, requests, json
class AnthropicVLM:
    def __init__(self, model_id="claude-3-sonnet-20240229"):
        self.key = os.environ["ANTHROPIC_API_KEY"]; self.model=model_id
    def generate(self, image_path, prompt, **_):
        img_b64 = base64.b64encode(open(image_path,"rb").read()).decode()
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key":self.key,"anthropic-version":"2023-06-01"},
            json={"model":self.model,"max_tokens":64,"messages":[{"role":"user","content":[
              {"type":"image","source":{"type":"base64","data":img_b64,"media_type":"image/jpeg"}},
              {"type":"text","text":prompt}]}]},timeout=60)
        return r.json()["content"][0]["text"].strip()
PY

cat > vlm_eval/models/google_loader.py <<'PY'
import os, base64, google.generativeai as genai
class GeminiVLM:
    def __init__(self, model_id="gemini-2.5-flash"):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_id)
    def generate(self, image_path, prompt, **_):
        img_bytes = open(image_path,"rb").read()
        img = {"mime_type":"image/jpeg","data":img_bytes}
        r = self.model.generate_content([img,prompt],max_output_tokens=64)
        return r.text.strip()
PY

cat > vlm_eval/models/ernie_loader.py <<'PY'
import os, base64, erniebot
class ErnieVLM:
    def __init__(self, model_id="ernie-4.5-turbo"):
        erniebot.api_type="aistudio"
        erniebot.access_token=os.environ["ERNIE_API_KEY"]
        self.model=model_id
    def generate(self, image_path, prompt, **_):
        img_b64=base64.b64encode(open(image_path,"rb").read()).decode()
        r = erniebot.ChatCompletion.create(
            model=self.model,
            messages=[{"role":"user","content":[
              {"type":"image","image":{"type":"base64","data":img_b64}},
              {"type":"text","text":prompt}]}],
            max_tokens=64)
        return r.result.strip()
PY

# 4 ─── Register new loaders ─────────────────────────────────
python - <<'PY'
import re, pathlib, textwrap, sys
f = pathlib.Path("vlm_eval/models/__init__.py")
txt = f.read_text()
block = textwrap.dedent("""
    from .open_hf import OpenHF
    from .anthropic_loader import AnthropicVLM
    from .google_loader import GeminiVLM
    from .ernie_loader import ErnieVLM
    FAMILY2INITIALIZER.update({
        "open-hf": OpenHF,
        "anthropic": AnthropicVLM,
        "google": GeminiVLM,
        "ernie": ErnieVLM,
    })
""")
if "open-hf" not in txt:
    f.write_text(txt + "\n" + block)
PY

# 5 ─── Patch GQA mirror URL ────────────────────────────────
sed -i 's_https://downloads.cs.stanford.edu/nlp/data/gqa/_https://huggingface.co/datasets/allenai/gqa/resolve/main/_' \
    vlm_eval/tasks/download.py

# 6 ─── Install with deps ───────────────────────────────────
pip install -U pip setuptools wheel
pip install -e .

echo "✅  Patch & install complete."
