import io, os, json, tempfile, importlib, types, pathlib
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------
# 1) load_keys behaviour without touching real env
# ---------------------------------------------------------------------
def test_load_keys(tmp_path, monkeypatch):
    dummy_env = tmp_path / ".keys.env"
    dummy_env.write_text("HF_TOKEN=hf_TEST\nDUMMY_API=xyz\n")
    monkeypatch.chdir(tmp_path)           # pretend repo root is tmp
    sys.modules.pop("load_keys", None)    # force reload
    from vlm_eval.util import load_keys
    load_keys.load_keys()
    assert os.getenv("DUMMY_API") == "xyz"
    assert (tmp_path / ".hf_token").read_text() == "hf_TEST"

# ---------------------------------------------------------------------
# 2) Draccus config round-trip
# ---------------------------------------------------------------------
def test_yaml_schema(tmp_path):
    txt = """\
model_family: open-hf
model_id: mock-model
model_dir: mock/model
dataset:
  type: text-vqa-slim
  root_dir: /data
results_dir: /results
"""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(txt)
    import scripts.evaluate as e
    cfg = e.parse(config_path=str(cfg_file))
    # minimal expectations
    assert cfg.model_family == "open-hf"
    assert cfg.dataset.type == "text-vqa-slim"

# ---------------------------------------------------------------------
# 3) run_all classify logic
# ---------------------------------------------------------------------
import run_all as RA

@pytest.mark.parametrize("arg,expect", [
    ("claude", ("anthropic", "claude-4-sonnet-20250522")),
    ("gemini", ("google",   "gemini-2.5-flash")),
    ("mistralai/Pixtral-12B", ("open-hf", "Pixtral-12B")),
])
def test_classify(arg, expect):
    out = RA.classify(arg)
    assert (out["model_family"], out["model_id"]) == expect

# ---------------------------------------------------------------------
# 4) mock the heavy loaders so evaluate.py can run end-to-end
# ---------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_heavy_loaders(monkeypatch):
    """Replace heavy VLM classes with lightweight echoes."""
    class Dummy:
        def __init__(self, *a, **k): pass
        def generate(self, image_path, prompt):
            return f"dummy-answer({pathlib.Path(image_path).name})"
        image_processor = None
    for name in ["OpenHF", "AnthropicVLM", "GeminiVLM"]:
        monkeypatch.setitem(
            sys.modules["vlm_eval.models"].__dict__, name, Dummy
        )

def test_evaluate_stub(tmp_path, monkeypatch):
    # fake 1-image dataset
    img_dir = tmp_path / "text-vqa-slim"; img_dir.mkdir()
    (img_dir / "dummy.jpg").write_bytes(b"JPEG")  # no need to be real
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
model_family: open-hf
model_id: mock
model_dir: mock
dataset:
  type: text-vqa-slim
  root_dir: {tmp_path}
results_dir: {tmp_path}
""")
    # run evaluate.py main() with patched sys.argv
    monkeypatch.chdir(REPO_ROOT)
    import scripts.evaluate as ev
    ev.main = getattr(ev, "evaluate") if hasattr(ev, "evaluate") else ev.main
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--config_path", str(cfg)])
    ev.main()
    # verify a results jsonl exists
    assert any(p.suffix == ".jsonl" for p in tmp_path.iterdir())
