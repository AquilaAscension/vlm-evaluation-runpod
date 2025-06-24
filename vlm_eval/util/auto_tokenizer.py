"""
auto_tokenizer.py
=================
Smart (but pragmatic) tokenizer loader for arbitrary HF repos.
"""

from __future__ import annotations
import importlib.util, json, os, re, tempfile
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
# OpenAI                                                                      #
# --------------------------------------------------------------------------- #
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# --------------------------------------------------------------------------- #
# hard-coded one-offs                                                         #
# --------------------------------------------------------------------------- #
_OVERRIDES: dict[str, str] = {
    "mistralai/Pixtral-12B-2409": "mistralai/Mistral-7B-v0.1",
}

# --------------------------------------------------------------------------- #
# README → GPT small-snippet route                                            #
# --------------------------------------------------------------------------- #
_README_SYSTEM = (
    "You are an expert in Hugging-Face repos.\n"
    "Given this README, reply ONLY with JSON: "
    '{ "code": "<python that defines `tokenizer`>" }'
)

def _try_readme_snippet(repo_id: str, files: List[str], **hf_auth):
    readme_file = next((f for f in files if "readme" in f.lower()), None)
    if not readme_file:
        return None

    local_dir = snapshot_download(
        repo_id, allow_patterns=[readme_file],
        local_dir=tempfile.mkdtemp(), **hf_auth
    )
    text = (Path(local_dir) / readme_file).read_text(encoding="utf-8")[:8000]

    chat = [
        {"role": "system", "content": _README_SYSTEM},
        {"role": "user",   "content": text},
    ]
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini", messages=chat, temperature=0
        )
        raw = rsp.choices[0].message.content.strip()
        code = json.loads(raw)["code"]        # <- may raise, we catch below

        tmp = tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False)
        tmp.write(code + "\n__tok__ = tokenizer\n")
        tmp.close()

        spec = importlib.util.spec_from_file_location("toktmp", tmp.name)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # type: ignore
        tok = mod.__tok__
        tok._meta = {"strategy": "README+GPT"}
        return tok
    except Exception as e:
        print(f"[auto_tokenizer] README path failed → {e}")
        return None

# --------------------------------------------------------------------------- #
# GPT file-list recipe route                                                  #
# --------------------------------------------------------------------------- #
_FILES_SYSTEM = (
    "You are an expert in HF model repos.\n"
    "Given a file list, reply ONLY with JSON:\n"
    '{ "tokenizer_class": "...", "tokenizer_files": [...] }'
)

def _ask_gpt_recipe(repo_id: str, files: List[str]):
    chat = [
        {"role": "system", "content": _FILES_SYSTEM},
        {"role": "user",   "content": f"Repo: {repo_id}\nFiles:\n" + "\n".join(files)},
    ]
    rsp  = client.chat.completions.create(
        model="gpt-4o-mini", messages=chat, temperature=0
    )
    return json.loads(rsp.choices[0].message.content)

def _build_from_recipe(repo_id: str, recipe: Dict, **hf_auth):
    cls, needed = recipe["tokenizer_class"], recipe["tokenizer_files"]
    local = Path(snapshot_download(repo_id, allow_patterns=needed,
                                   local_dir=tempfile.mkdtemp(), **hf_auth))

    if cls == "AutoTokenizer":
        tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True, legacy=True)
    elif cls == "LlamaTokenizerFast":
        tok = LlamaTokenizerFast(vocab_file=str(local / needed[0]))
    elif cls == "LlamaTokenizer":
        tok = LlamaTokenizer(vocab_file=str(local / needed[0]))
    elif cls == "SentencePiece":
        tok = LlamaTokenizerFast(vocab_file=str(local / needed[0]))
    else:
        raise ValueError(f"GPT suggested unknown class {cls}")

    tok._meta = {"strategy": f"GPT:{cls}"}
    return tok

# --------------------------------------------------------------------------- #
# heuristics                                                                  #
# --------------------------------------------------------------------------- #
def _heuristic_tokenizer(local: Path):
    j = next(local.rglob("tokenizer.json"), None)
    if j:
        t = PreTrainedTokenizerFast(tokenizer_file=str(j))
        t._meta = {"strategy": "heuristic:tokenizer.json"}
        return t

    spm = next(local.rglob("*.model"), None)
    if spm:
        t = LlamaTokenizerFast(
            vocab_file=str(spm), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        t._meta = {"strategy": f"heuristic:{spm.name}"}
        return t
    return None

# --------------------------------------------------------------------------- #
# naïve README scan for “tokenizer = X”                                       #
# --------------------------------------------------------------------------- #
_PATTERNS = [
    re.compile(r"tokenizer\s*[:=]\s*[\"']([\w\-./]+)[\"']", re.I),
    re.compile(r"AutoTokenizer\.from_pretrained\(\s*[\"']([\w\-./]+)[\"']", re.I),
]

def _external_repo_in_readme(readme: Path):
    try:
        txt = readme.read_text(encoding="utf-8", errors="ignore")[:4096]
    except Exception:
        return None
    for pat in _PATTERNS:
        m = pat.search(txt)
        if m:
            return m.group(1)
    return None

# --------------------------------------------------------------------------- #
# public entry                                                                #
# --------------------------------------------------------------------------- #
def build_intelligent_tokenizer(repo_id: str, *, token: str | None = None):
    # 0) hard-coded overrides --------------------------------------------------
    if repo_id in _OVERRIDES:
        tgt = _OVERRIDES[repo_id]
        print(f"[auto_tokenizer] override → {tgt}")
        tok = AutoTokenizer.from_pretrained(tgt, trust_remote_code=True, token=token)
        tok._meta = {"strategy": f"override:{tgt}"}
        return tok

    hf_auth = {"token": token} if token else {}

    # 1) README explicit external-repo hint -----------------------------------
    readme_dir = snapshot_download(repo_id, allow_patterns=["README*"],
                                   local_dir=tempfile.mkdtemp(), **hf_auth)
    readme = next(Path(readme_dir).rglob("*README*"), None)
    if readme:
        ext = _external_repo_in_readme(readme)
        if ext:
            if "/" not in ext:                       # no org → inherit
                ext = f"{repo_id.split('/')[0]}/{ext}"
            print(f"[auto_tokenizer] README points to → {ext}")
            tok = AutoTokenizer.from_pretrained(ext, trust_remote_code=True, **hf_auth)
            tok._meta = {"strategy": f"external:{ext}"}
            return tok

    # 2) README-snippet route --------------------------------------------------
    files = list_repo_files(repo_id, **hf_auth)
    t = _try_readme_snippet(repo_id, files, **hf_auth)
    if t:
        return t

    # 3) GPT file-list route ---------------------------------------------------
    try:
        recipe = _ask_gpt_recipe(repo_id, files)
        return _build_from_recipe(repo_id, recipe, **hf_auth)
    except Exception as e:
        print(f"[auto_tokenizer] GPT recipe path failed → {e}")

    # 4) heuristic fallback ----------------------------------------------------
    local = Path(snapshot_download(repo_id, local_dir=tempfile.mkdtemp(), **hf_auth))
    t = _heuristic_tokenizer(local)
    if t:
        return t

    raise RuntimeError(f"[auto_tokenizer] Could not build tokenizer for {repo_id}")
