# vlm_eval/util/resolve_tokenizer.py
from __future__ import annotations
import json, os, pathlib
from transformers import AutoTokenizer
import importlib

# --------------------------------------------------
# 0 · tiny on-disk cache  (~400 B/entry)
# --------------------------------------------------
_CACHE_PATH = pathlib.Path.home() / ".vlm_tokenizer_cache.json"
try:
    _TOK_CACHE: dict[str, str] = json.loads(_CACHE_PATH.read_text())
except FileNotFoundError:
    _TOK_CACHE = {}

def _cache_write() -> None:
    _CACHE_PATH.write_text(json.dumps(_TOK_CACHE))

# --------------------------------------------------
# 1 · your GPT-4-assisted helper (if present)
# --------------------------------------------------

try:
    _at = importlib.import_module("vlm_eval.util.auto_tokenizer")
    build_or_lookup_tokenizer = getattr(_at, "build_or_lookup_tokenizer", None)
except ModuleNotFoundError:
    build_or_lookup_tokenizer = None        # module truly absent

# --------------------------------------------------
# 2 · deterministic heuristic fallbacks
# --------------------------------------------------
_HEURISTIC_CHAIN = [
    lambda src: src,                                           # exact repo
    lambda src: src.replace("-vision", ""),
    lambda src: src.replace("-mm", ""),
    lambda _src: "mistralai/Mistral-7B-v0.1",                  # Pixtral et al.
    lambda _src: "hf-internal-testing/llama-tokenizer",        # generic BPE
]

# --------------------------------------------------
# 3 · public entry point
# --------------------------------------------------
def resolve_tokenizer(repo_id: str, **hf_kwargs):
    """
    Return a *transformers* tokenizer for any HuggingFace repo.

    Search order:
      1. on-disk cache
      2. GPT-4 README / file-map helper  (build_or_lookup_tokenizer)
      3. heuristic repo guesses (ordered)
    Caches the first successful repo string for next time.
    """
    # —— cached hit?
    if repo_id in _TOK_CACHE:
        return AutoTokenizer.from_pretrained(_TOK_CACHE[repo_id], **hf_kwargs)

    # —— GPT-4o path
    if build_or_lookup_tokenizer is not None:
        try:
            tok = build_or_lookup_tokenizer(repo_id, **hf_kwargs)
            if tok is not None:
                _TOK_CACHE[repo_id] = tok.name_or_path
                _cache_write()
                return tok
        except Exception as e:
            print(f"[auto_tokenizer ⚠️] {e}")

    # —— heuristics
    for derive in _HEURISTIC_CHAIN:
        guess_repo = derive(repo_id)
        try:
            tok = AutoTokenizer.from_pretrained(guess_repo, **hf_kwargs)
            _TOK_CACHE[repo_id] = guess_repo
            _cache_write()
            return tok
        except Exception:
            continue  # try next guess

    # —— lights out
    raise RuntimeError(f"resolve_tokenizer: cannot locate tokenizer for '{repo_id}'")
