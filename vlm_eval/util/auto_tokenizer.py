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


# --------------------------------------------------------------------------------------
# 2) Build tokenizer from the answer
# --------------------------------------------------------------------------------------
def _build_from_recipe(repo_id: str, recipe: Dict[str, Any], **hf_auth) -> Tuple[Any, str]:
    klass = recipe["tokenizer_class"]
    needed = recipe["tokenizer_files"]
    local_dir = snapshot_download(repo_id, allow_patterns=needed, **hf_auth)

    if klass == "AutoTokenizer":
        # Pixtral pattern: tekken.json + tekken.model
        model_files = [f for f in needed if f.endswith(".model")]
        if model_files:
            sp_model = Path(local_dir) / model_files[0]
            tok = LlamaTokenizerFast(vocab_file=str(sp_model))
            tok._meta = {"strategy": f"spm-direct:{sp_model.name}"}
        else:
            tok = AutoTokenizer.from_pretrained(
                local_dir, trust_remote_code=True, **hf_auth
            )
    elif klass == "LlamaTokenizerFast":
        tok = LlamaTokenizerFast(vocab_file=str(Path(local_dir) / needed[0]))
    elif klass == "LlamaTokenizer":
        tok = LlamaTokenizer(vocab_file=str(Path(local_dir) / needed[0]))
    elif klass == "SentencePiece":
        # bare sentence-piece -> wrap in LlamaTokenizerFast (works for Pixtral)
        model_file = Path(local_dir) / needed[0]
        tok = LlamaTokenizerFast(vocab_file=str(model_file))
    else:
        raise ValueError(f"Unknown tokenizer class {klass}")

    # record meta for logging
    tok._meta = {"strategy": f"openai:{klass}"}
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
