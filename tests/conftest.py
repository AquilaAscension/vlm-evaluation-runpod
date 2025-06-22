import sys, pathlib, os
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ["PYTHONHASHSEED"] = "0"   # reproducible
