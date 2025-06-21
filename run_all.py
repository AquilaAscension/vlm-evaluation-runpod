#!/usr/bin/env python3
"""
run_all.py — One-shot benchmark runner for three datasets
Usage examples:
  # open-weights (HF)
  python run_all.py mistralai/Pixtral-12B
  # or full URL
  python run_all.py https://huggingface.co/mistralai/Pixtral-12B

  # closed-API models
  python run_all.py claude                       # Claude 4 Sonnet
  python run_all.py gemini                       # Gemini 2.5 Flash
"""

import argparse, os, re, subprocess, sys, tempfile

DATASETS = ["text-vqa-slim", "vqa-v2-slim", "gqa"]
RESULTS_DIR = "/home/ubuntu/prismatic-vlms/results"
ROOT_DATA = "/home/ubuntu/datasets/vlm-evaluation"

def classify_model(arg: str):
    arg = arg.strip()
    # ---------- closed models ----------
    if re.fullmatch(r"claude(-sonnet4)?", arg, re.I):
        return dict(family="anthropic",
                    model_id="claude-3-sonnet-20240229",
                    model_dir="")
    if re.fullmatch(r"gemini(-flash)?", arg, re.I):
        return dict(family="google",
                    model_id="gemini-2.5-flash",
                    model_dir="")
    # ---------- open HF ----------
    repo = arg.replace("https://huggingface.co/", "").rstrip("/")
    model_id = repo.split("/")[-1]
    return dict(family="open-hf", model_id=model_id, model_dir=repo)

def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF repo/URL or closed-model keyword")
    args = ap.parse_args()

    cfg = classify_model(args.model)
    print(f"→ Using loader {cfg['family']}  model_id={cfg['model_id']}")

    for dset in DATASETS:
        # build temp YAML
        yaml_content = f"""\
model:
  model_family: {cfg['family']}
  model_id: {cfg['model_id']}
  model_dir: {cfg['model_dir']}
dataset:
  type: {dset}
  root_dir: {ROOT_DATA}
results_dir: {RESULTS_DIR}
"""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as fd:
            fd.write(yaml_content)
            cfg_path = fd.name

        # evaluate
        run(["python", "scripts/evaluate.py", "--config_path", cfg_path])

        # score
        run([
            "python", "scripts/score.py",
            "--model_id", cfg["model_id"],
            "--dataset.type", dset,
            "--dataset.root_dir", ROOT_DATA,
            "--results_dir", RESULTS_DIR,
        ])
        os.unlink(cfg_path)

if __name__ == "__main__":
    if not os.path.exists("scripts/evaluate.py"):
        sys.exit("Run this script from repo root (where scripts/ lives).")
    main()
