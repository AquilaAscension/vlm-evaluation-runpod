# vlm_eval/util/resolve_tokenizer.py
"""
First try the repo’s own README/LLM-assisted resolver (your auto_tokenizer).
If that throws or returns None, fall back to heuristic guesses that never
touch the network.  Single import line inside open_hf.py keeps call-sites tidy.
"""
from transformers import AutoTokenizer

# --- your original GPT-4-assisted helper ---------------------------
try:
    from vlm_eval.util.auto_tokenizer import build_or_lookup_tokenizer
except ModuleNotFoundError:
    build_or_lookup_tokenizer = None          # unit-tests without OpenAI

# --- offline heuristics --------------------------------------------
PREFERRED = [
    lambda repo: repo,                                      # exact
    lambda repo: repo.replace("-vision", ""),
    lambda repo: repo.replace("-mm", ""),
    lambda repo: "mistralai/Mistral-7B-v0.1",
    lambda repo: "hf-internal-testing/llama-tokenizer",
]

def _heuristic(repo_id, **kw):
    for guess in PREFERRED:
        try:
            return AutoTokenizer.from_pretrained(guess(repo_id), **kw)
        except Exception:
            continue
    return None

# -------- public single-call entry ---------------------------------
def resolve_tokenizer(repo_id: str, **kw):
    # 1️⃣ LLM-assisted route
    if build_or_lookup_tokenizer is not None:
        try:
            tok = build_or_lookup_tokenizer(repo_id, **kw)
            if tok is not None:
                return tok
        except Exception as e:
            print(f"[auto_tokenizer❌] {e}")

    # 2️⃣ local heuristics
    tok = _heuristic(repo_id, **kw)
    if tok is not None:
        return tok

    raise RuntimeError(f"Could not load tokenizer for {repo_id}")
