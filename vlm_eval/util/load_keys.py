import os
from pathlib import Path
from dotenv import load_dotenv

def load_keys():
    """
    Load .keys.env (if present) and, if HF_TOKEN is set,
    auto-write .hf_token for dataset downloads.
    """
    env_file = Path(".keys.env")
    if env_file.exists():
        load_dotenv(env_file)
        # auto-sync Hugging Face token for prepare.py
        if os.getenv("HF_TOKEN") and not Path(".hf_token").exists():
            with open(".hf_token", "w") as f:
                f.write(os.environ["HF_TOKEN"])
    else:
        print("[load_keys] .keys.env not found – skipping.")
