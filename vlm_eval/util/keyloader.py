# vlm_eval/util/keyloader.py
"""
Side-effect importer that guarantees HF_TOKEN / OPENAI_API_KEY are in
os.environ, loading them from `.keys.env` if needed.
"""

import os, pathlib, dotenv  # dotenv is already required for vlm_eval; if not, pip install python-dotenv

ROOT = pathlib.Path(__file__).resolve().parents[2]   # repo root
ENV_FILE = ROOT / ".keys.env"

def _load_from_dotenv():
    if ENV_FILE.exists():
        dotenv.load_dotenv(ENV_FILE, override=False)  # only gaps are filled

def ensure_keys():
    # 1️⃣ first, try your existing helper
    try:
        import load_keys        # your file in repo root
        if hasattr(load_keys, "load_keys_from_env"):
            load_keys.load_keys_from_env(str(ENV_FILE))
        elif hasattr(load_keys, "load"):              # generic name
            load_keys.load(str(ENV_FILE))
        else:
            _load_from_dotenv()
    except ModuleNotFoundError:
        _load_from_dotenv()

    # 2️⃣ normalise alias names (inside .keys.env you might have HUGGINGFACE_TOKEN)
    if "HUGGINGFACE_TOKEN" in os.environ and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
