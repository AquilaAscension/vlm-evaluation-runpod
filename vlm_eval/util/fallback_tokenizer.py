from transformers import AutoTokenizer

PREFERRED = [
    lambda repo: repo.replace("-vision", ""),          # many vision forks
    lambda repo: repo.replace("-mm", ""),
    lambda repo: "mistralai/Mistral-7B-v0.1",          # covers Pixtral forks
    lambda repo: "hf-internal-testing/llama-tokenizer" # last-ditch LLaMA-ish
]

def safe_load_tokenizer(repo_id, **kwargs):
    for guess in [repo_id] + [f(repo_id) for f in PREFERRED]:
        try:
            return AutoTokenizer.from_pretrained(guess, **kwargs)
        except Exception:
            continue
    raise RuntimeError(f"💥 could not resolve tokenizer for {repo_id}")
