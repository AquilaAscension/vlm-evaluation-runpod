# tools/prefetch_models.py
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
import sys, os

def fetch(repo):
    print("→ downloading", repo)
    snapshot_download(repo_id=repo, resume_download=True)

if __name__ == "__main__":
    repos = sys.argv[1:]
    # 8 threads makes HF saturate a 1 Gbps line without over-spawning
    with ThreadPoolExecutor(max_workers=8) as ex:
        ex.map(fetch, repos)
