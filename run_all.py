# run_all.py  (overwrite or patch in-place)
import argparse, json, os, random, subprocess, sys, time
from pathlib import Path
from datetime import datetime
import torch

SEED = 42          # global reproducibility knob
random.seed(SEED); torch.manual_seed(SEED)

DATASETS = ["text-vqa-slim", "vqav2-slim", "gqa-slim"]
EVAL_CMD = ["python", "scripts/evaluate.py"]
SCORE_CMD = ["python", "scripts/score.py"]

def mark_done(model, ds, tag):
    Path(f"~/prismatic-vlms/results/{model}/{ds}/{tag}.done"
         .replace("~", os.path.expanduser("~"))).touch()

def already_done(model, ds, tag):
    return Path(f"~/prismatic-vlms/results/{model}/{ds}/{tag}.done"
                .replace("~", os.path.expanduser("~"))).exists()

def main(repo_id_list):
    for repo in repo_id_list:
        model_name = repo.split("/")[-1]
        print(f"\n▶ {model_name}  —  {datetime.now().isoformat(timespec='seconds')}")
        # 1-shot model download so the GPU never waits
        snapshot_path = download_once(repo)

        for ds in DATASETS:
            # deterministic resume: skip if both passes finished
            if already_done(model_name, ds, "_scored"):
                print(f"✓ {model_name}/{ds} already scored")
                continue

            # EVALUATE -------------------------------------------------------
            if not already_done(model_name, ds, "_eval"):
                run(EVAL_CMD + ["--model", repo, "--dataset", ds,
                                "--local_model_path", snapshot_path])
                mark_done(model_name, ds, "_eval")

            # SCORE ----------------------------------------------------------
            if not already_done(model_name, ds, "_scored"):
                run(SCORE_CMD + ["--model", model_name, "--dataset", ds])
                mark_done(model_name, ds, "_scored")

def download_once(repo):
    """Sync repo to HF cache once, regardless of earlier runs."""
    from huggingface_hub import snapshot_download
    cache_dir = os.environ.get("HF_HOME", "~/.cache/huggingface")
    dest = snapshot_download(repo_id=repo, cache_dir=cache_dir,
                             resume_download=True, local_files_only=False)
    return dest

def run(cmd):
    print("  $", *cmd)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("‼ subprocess failed:", e); sys.exit(1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="HF repos, one or many")
    main(p.parse_args().models)
