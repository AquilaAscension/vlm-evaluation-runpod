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
# 2) Build a tokenizer from the OpenAI “recipe”
# ---------------------------------------------------------------------
def _build_from_recipe(repo_id: str, recipe: dict, **hf_auth):
    """
    Returns
    -------
    tok       : the built tokenizer
    strategy  : short string for logging
    """
    klass   = recipe["tokenizer_class"]          # e.g. "AutoTokenizer"
    needed  = recipe["tokenizer_files"]          # list of file names
    local   = snapshot_download(repo_id,
                                allow_patterns=needed,
                                **hf_auth)

    # ---------- 1. try the normal AutoTokenizer route -----------------
    if klass == "AutoTokenizer":
        try:
            tok = AutoTokenizer.from_pretrained(
                    local,
                    trust_remote_code=True,
                    legacy=True,          # allow older tokenizers
                    **hf_auth)
            tok._meta = {"strategy": "auto"}
            return tok, "auto"
        except Exception:                 # <-- anything goes wrong → fall through
            pass                          # we’ll attempt a manual SPM build

    # ---------- 2. manual SentencePiece fallback ----------------------
    spm_file = next(Path(local).rglob("*.model"), None)
    if spm_file is None:
        raise RuntimeError(f"No *.model file found in {repo_id} - cannot build tokenizer")

    tok = LlamaTokenizerFast(vocab_file=str(spm_file),
                             bos_token="<s>",
                             eos_token="</s>",
                             unk_token="<unk>")
    tok._meta = {"strategy": f"spm-direct:{spm_file.name}"}
    return tok, tok._meta["strategy"]


# --------------------------------------------------------------------------------------
# 3) Public entry point
# --------------------------------------------------------------------------------------
def build_intelligent_tokenizer(repo_id: str, *, token: str | None = None):
    hf_auth = {"token": token} if token else {}
    files = list_repo_files(repo_id, **hf_auth)
    recipe = _ask_openai(repo_id, files)
    tok, strategy = _build_from_recipe(repo_id, recipe, **hf_auth)
    return tok
