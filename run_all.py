#!/usr/bin/env python3
"""
run_all.py  –  Evaluate one model on TextVQA-Slim, VQA-v2-Slim, and GQA

Usage:
  # open-weight HF repo (string or URL)
  python run_all.py mistralai/Pixtral-12B
  python run_all.py https://huggingface.co/deepseek-ai/janus-pro-7b

  # closed-API shortcuts (keys must be in .keys.env or environment)
  python run_all.py claude          # Claude 4 Sonnet
  python run_all.py gemini          # Gemini 2.5 Flash
"""

import argparse, os, pathlib, re, subprocess, sys, tempfile

# ---------------- ensure repo root on sys.path ----------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# ---- load keys (works whether in utils/ or repo root) -------
try:
    from vlm_eval.util.load_keys import load_keys
except ModuleNotFoundError:
    from load_keys import load_keys  # fallback for old placement
load_keys()

# ---------------- constants ----------------------------------
DATASETS = ["text-vqa-slim", "vqa-v2-slim", "gqa-slim"]
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/home/ubuntu/prismatic-vlms/results")
ROOT_DATA = os.environ.get("ROOT_DATA", "/home/ubuntu/datasets/vlm-evaluation")

# ---------------- helper funcs -------------------------------
def classify(arg: str):
    """Return dict with model_family, model_id, model_dir fields."""
    arg = arg.strip()
    # closed APIs
    if re.fullmatch(r"claude(-sonnet4|-4-sonnet)?", arg, re.I):
        return dict(model_family="anthropic",
                    model_id="claude-4-sonnet-20250522",
                    model_dir="")
    if re.fullmatch(r"gemini(?:-?2\.5)?(?:-flash)?", arg, re.I):
        return dict(model_family="google",
                    model_id="gemini-2.5-flash",
                    model_dir="")
    m = re.fullmatch(r"(?:https?://huggingface.co/)?deepseek-ai/(janus[-_]pro[-_]7b)", arg, re.I)
    if m:
        repo = f"deepseek-ai/{m.group(1).replace('_','-')}"
        return dict(model_family="janus",
                    model_id=m.group(1),        # Janus-Pro-7B
                    model_dir=repo)
    # open-weight HF
    repo = arg.replace("https://huggingface.co/", "").rstrip("/")
    model_id = repo.split("/")[-1]
    return dict(model_family="open-hf", model_id=model_id, model_dir=repo)

def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

# ---------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF repo/URL or closed-API keyword (claude, gemini)")
    args = ap.parse_args()

    cfg = classify(args.model)
    print(f"→ Using loader {cfg['model_family']}   model_id={cfg['model_id']}")

    for dset in DATASETS:
        # build flat YAML expected by Draccus
        yaml_str = f"""\
model_family: {cfg['model_family']}
model_id: {cfg['model_id']}
model_dir: {cfg['model_dir']}
run_dir: {cfg['model_dir']}

dataset:
  type: {dset}
  root_dir: {ROOT_DATA}

results_dir: {RESULTS_DIR}
"""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as tmp:
            tmp.write(yaml_str)
            cfg_path = tmp.name

        # evaluate + score
        run(["python", "scripts/evaluate.py", "--config_path", cfg_path])
        run([
            "python", "scripts/score.py",
            "--model_id", cfg["model_id"],
            "--dataset.type", dset,
            "--dataset.root_dir", ROOT_DATA,
            "--results_dir", RESULTS_DIR,
        ])

        print(f"✓ finished {cfg['model_id']} on {dset}\n"
              f"  results → {RESULTS_DIR}/{cfg['model_id']}/{dset}_scores.json",
              flush=True)

        os.unlink(cfg_path)

if __name__ == "__main__":
    if not (REPO_ROOT / "scripts" / "evaluate.py").exists():
        sys.exit("Run this script from the repo root (where scripts/ lives).")
    main()
