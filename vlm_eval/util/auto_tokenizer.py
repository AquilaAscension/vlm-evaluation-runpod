from __future__ import annotations

import json, os
from pathlib import Path
from typing import Tuple, Dict, Any, List

from huggingface_hub import list_repo_files, snapshot_download
from openai import OpenAI
from transformers import (
    AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast
)
import sentencepiece as spm
import tempfile

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# --------------------------------------------------------------------------------------
# 1) query OpenAI
# --------------------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an expert in Hugging-Face model repositories.
Given the file names inside a repo, decide which TRANSFORMERS tokenizer
class should load it and what files are needed.
Respond strictly as valid JSON with keys:
{
  "tokenizer_class": "<AutoTokenizer|LlamaTokenizer|LlamaTokenizerFast|SentencePiece>",
  "tokenizer_files": ["file1", "file2", ...],   // relative paths in the repo
  "special": "<short human hint>"
}"""


def _ask_openai(repo_id: str, file_list: List[str]) -> Dict[str, Any]:
    prompt = f"Repo: {repo_id}\nFiles:\n" + "\n".join(file_list)
    chat = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=chat, temperature=0)
    return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------------
# 2) Build tokenizer from the recipe (final version)
# ---------------------------------------------------------------------
from transformers import PreTrainedTokenizerFast

def _build_from_recipe(repo_id: str, recipe: dict, **hf_auth):
    klass  = recipe["tokenizer_class"]
    files  = recipe["tokenizer_files"]
    local  = snapshot_download(repo_id, allow_patterns=files, **hf_auth)
    local  = Path(local)

    json_vocab = [f for f in files if f.lower().endswith(".json")]
    if len(json_vocab) == 1:
        vocab_path = Path(local) / json_vocab[0]
        tok = PreTrainedTokenizerFast(
            tokenizer_file=str(vocab_path),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tok._meta = {"strategy": f"json-vocab:{json_vocab[0]}"}
        return tok, tok._meta["strategy"]

    # ---------- normal AutoTokenizer route ----------
    if klass == "AutoTokenizer":
        try:
            tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True,
                                                legacy=True, **hf_auth)
            tok._meta = {"strategy": "auto"}
            return tok, "auto"
        except Exception:
            pass                                    # fall through

    # ---------- fallback #1 : tokenizer.json ----------
    json_file = next(local.rglob("tokenizer.json"), None)
    if json_file is not None:
        tok = PreTrainedTokenizerFast(tokenizer_file=str(json_file))
        tok._meta = {"strategy": f"tokenizer.json:{json_file.name}"}
        return tok, tok._meta["strategy"]

    # ---------- fallback #2 : SentencePiece ----------
    spm_file = next(local.rglob("*.model"), None)
    if spm_file is not None:
        tok = LlamaTokenizerFast(vocab_file=str(spm_file),
                                 bos_token="<s>", eos_token="</s>", unk_token="<unk>")
        tok._meta = {"strategy": f"spm:{spm_file.name}"}
        return tok, tok._meta["strategy"]
    
    # ❶ Append at END of _build_from_recipe(), just before the final `raise`
    # ------------------------------------------------------
    # last-chance: ask GPT to parse README for a tokenizer snippet
    readme_files = [f for f in files if "readme" in f.lower()]
    if readme_files:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        readme_path = Path(local) / readme_files[0]
        with open(readme_path) as fh:
            readme_txt = fh.read()[:8000]        # 8 k tokens max to stay cheap
        chat = [
            {"role": "system",
            "content": "You're an expert at Hugging-Face repos. "
                        "Given this README, tell me the exact Python code snippet "
                        "needed to create the tokenizer, in JSON:\n"
                        "{ \"code\": \"import ...\" }"},
            {"role": "user", "content": readme_txt},
        ]
        rsp = client.chat.completions.create(
            model="gpt-4o-mini", messages=chat, temperature=0)
        try:
            code = json.loads(rsp.choices[0].message.content)["code"]
            # ⚠️ eval-free exec: write snippet to a temp file and import it
            temp = tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False)
            temp.write(code + "\n__tok__ = tokenizer")  # user must name var 'tokenizer'
            temp.close()
            import importlib.util, runpy
            spec = importlib.util.spec_from_file_location("tokmod", temp.name)
            tokmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(tokmod)
            tok = tokmod.__tok__
            tok._meta = {"strategy": "README+GPT"}
            return tok, "README+GPT"
        except Exception as e:
            print("[OpenHF] README-GPT fallback failed →", e)
    # ------------------------------------------------------


    # ---------- give up ----------
    raise RuntimeError(f"No usable tokenizer assets found in {repo_id}")


# --------------------------------------------------------------------------------------
# 3) Public entry point
# --------------------------------------------------------------------------------------
def build_intelligent_tokenizer(repo_id: str, *, token: str | None = None):
    hf_auth = {"token": token} if token else {}
    files = list_repo_files(repo_id, **hf_auth)
    recipe = _ask_openai(repo_id, files)
    tok, strategy = _build_from_recipe(repo_id, recipe, **hf_auth)
    return tok
