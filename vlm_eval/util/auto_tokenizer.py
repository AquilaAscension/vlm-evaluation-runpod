"""
auto_tokenizer.py
=================
Given a HF repo-id, build the right `transformers` tokenizer *without*
hard-coding per-model logic.

Order of battle
---------------
1.  Read the repos README / model-card (<= 8 kB) and ask GPT-4o-mini for a
    *minimal working Python snippet* that creates `tokenizer`.
2.  If #1 fails, ask GPT-4o-mini (again) **just** about the file-listing
    (“which files + which tokenizer class?”) and try that recipe.
3.  If #2 fails, use pragmatic heuristics (tokenizer.json, *.model, …).

The function exported to callers is:

    build_intelligent_tokenizer(repo_id, token=None) → tokenizer
"""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
import tempfile
from pathlib import Path
from typing import Dict, List

from huggingface_hub import list_repo_files, snapshot_download
from openai import OpenAI
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
)

# --------------------------------------------------------------------------- #
# OpenAI client (reads key from .keys.env/.env or real environment)           #
# --------------------------------------------------------------------------- #
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# --------------------------------------------------------------------------- #
# 1. Ask GPT to read README / model-card and spit out a code snippet          #
# --------------------------------------------------------------------------- #
_README_SYSTEM = (
    "You are an expert in Hugging-Face repos.\n"
    "Given this README, reply ONLY with JSON:\n"
    '{ "code": "<python that defines `tokenizer`>" }'
)


def _try_readme_snippet(repo_id: str, files: List[str], **hf_auth):
    readme_file = next((f for f in files if "readme" in f.lower()), None)
    if not readme_file:
        return None

    # Download just the README so we stay fast/cheap
    local_dir = snapshot_download(
        repo_id, allow_patterns=[readme_file], **hf_auth, local_dir=tempfile.mkdtemp()
    )
    readme_path = Path(local_dir) / readme_file
    text = readme_path.read_text(encoding="utf-8")[:8000]  # truncate

    chat = [
        {"role": "system", "content": _README_SYSTEM},
        {"role": "user", "content": text},
    ]
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini", messages=chat, temperature=0
        )
        code = json.loads(rsp.choices[0].message.content)["code"]

        # Write to temp       → import     → grab `tokenizer`
        f = tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False)
        f.write(code + "\n__tok__ = tokenizer\n")
        f.close()

        spec = importlib.util.spec_from_file_location("tokmod", f.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        tok = mod.__tok__
        tok._meta = {"strategy": "README+GPT"}
        return tok
    except Exception as e:
        print(f"[auto_tokenizer] README path failed → {e}")
        return None


# --------------------------------------------------------------------------- #
# 2. Ask GPT to choose tokenizer class + files from file-listing              #
# --------------------------------------------------------------------------- #
_FILES_SYSTEM = (
    "You are an expert in HF model repos.\n"
    "Given a file list, reply ONLY with valid JSON:\n"
    "{"
    '  "tokenizer_class": "AutoTokenizer|LlamaTokenizer|LlamaTokenizerFast|SentencePiece",'
    '  "tokenizer_files": ["file1", "file2"]'
    "}"
)


def _ask_gpt_recipe(repo_id: str, files: List[str]):
    prompt = f"Repo: {repo_id}\nFiles:\n" + "\n".join(files)
    chat = [
        {"role": "system", "content": _FILES_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    rsp = client.chat.completions.create(
        model="gpt-4o-mini", messages=chat, temperature=0
    )
    return json.loads(rsp.choices[0].message.content)


def _build_from_recipe(repo_id: str, recipe: Dict, **hf_auth):
    klass = recipe["tokenizer_class"]
    needed = recipe["tokenizer_files"]
    local = Path(
        snapshot_download(repo_id, allow_patterns=needed, **hf_auth, local_dir=tempfile.mkdtemp())
    )

    if klass == "AutoTokenizer":
        tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True, legacy=True)
    elif klass == "LlamaTokenizerFast":
        tok = LlamaTokenizerFast(vocab_file=str(local / needed[0]))
    elif klass == "LlamaTokenizer":
        tok = LlamaTokenizer(vocab_file=str(local / needed[0]))
    elif klass == "SentencePiece":
        spm_path = local / needed[0]
        tok = LlamaTokenizerFast(vocab_file=str(spm_path))
    else:
        raise ValueError(f"GPT suggested unknown class {klass}")

    tok._meta = {"strategy": f"GPT:{klass}"}
    return tok


# --------------------------------------------------------------------------- #
# 3. Heuristics (tokenizer.json, *.model)                                     #
# --------------------------------------------------------------------------- #
def _heuristic_tokenizer(local_dir: Path):
    # tokenizer.json
    json_tok = next(local_dir.rglob("tokenizer.json"), None)
    if json_tok:
        tok = PreTrainedTokenizerFast(tokenizer_file=str(json_tok))
        tok._meta = {"strategy": "heuristic:tokenizer.json"}
        return tok

    # *.model
    spm_tok = next(local_dir.rglob("*.model"), None)
    if spm_tok:
        tok = LlamaTokenizerFast(
            vocab_file=str(spm_tok),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tok._meta = {"strategy": f"heuristic:{spm_tok.name}"}
        return tok

    return None

import re
_TOKENIZER_RE = re.compile(
    r'from_pretrained\(\s*["\']([\w\-_]+\/[\w\-.]+)["\']\s*\)', re.IGNORECASE
)


def _external_repo_in_readme(readme_path: Path) -> str | None:
    text = readme_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    snippet = "\n".join(text[:1000])  # keep it quick
    m = _TOKENIZER_RE.search(snippet)
    return m.group(1) if m else None

# --------------------------------------------------------------------------- #
# 4. Public entry point                                                      #
# --------------------------------------------------------------------------- #
def build_intelligent_tokenizer(repo_id: str, *, token: str | None = None):
    hf_auth = {"token": token} if token else {}

    local_tmp = Path(snapshot_download(repo_id, allow_patterns=["README*"], **hf_auth,
                                       local_dir=tempfile.mkdtemp()))
    readme = next(local_tmp.rglob("*README*"), None)

    # ➊ external-tokenizer shortcut
    if readme:
        ext_repo = _external_repo_in_readme(readme)
        if ext_repo:
            print(f"[auto_tokenizer] README points to external tokenizer → {ext_repo}")
            tok = AutoTokenizer.from_pretrained(ext_repo, trust_remote_code=True, **hf_auth)
            tok._meta = {"strategy": f"external:{ext_repo}"}
            return tok

    files = list_repo_files(repo_id, **hf_auth)

    # 1️⃣ README route
    tok = _try_readme_snippet(repo_id, files, **hf_auth)
    if tok:
        return tok

    # 2️⃣ GPT file-list route
    try:
        recipe = _ask_gpt_recipe(repo_id, files)
        tok = _build_from_recipe(repo_id, recipe, **hf_auth)
        return tok
    except Exception as e:
        print(f"[auto_tokenizer] GPT recipe path failed → {e}")

    # 3️⃣ Heuristic route
    local_all = Path(snapshot_download(repo_id, **hf_auth, local_dir=tempfile.mkdtemp()))
    tok = _heuristic_tokenizer(local_all)
    if tok:
        return tok

    raise RuntimeError(f"[auto_tokenizer] Could not build tokenizer for {repo_id}")
