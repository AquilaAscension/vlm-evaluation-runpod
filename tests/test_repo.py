import os, sys, pathlib, tempfile, json, pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------
#  Autouse fixture: monkey-patch heavy loaders *and* the initializer map
# -----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_heavy_loaders(monkeypatch):
    """Replace heavy VLM classes with lightweight echoes, and patch initializer map."""
    import vlm_eval.models as mdl            # ensure module is imported

    class Dummy:
        def __init__(self, *a, **k): pass
        def generate(self, image_path, prompt):
            return f"dummy-answer({pathlib.Path(image_path).name})"
        image_processor = None

    for name, fam in [("OpenHF", "open-hf"),
                      ("AnthropicVLM", "anthropic"),
                      ("GeminiVLM", "google")]:
        monkeypatch.setattr(mdl, name, Dummy, raising=False)
        if hasattr(mdl, "FAMILY2INITIALIZER"):
            mdl.FAMILY2INITIALIZER[fam] = Dummy

# -----------------------------------------------------------------------
# 1) .keys.env → env vars & .hf_token
# -----------------------------------------------------------------------
def test_load_keys(tmp_path, monkeypatch):
    dummy_env = tmp_path / ".keys.env"
    dummy_env.write_text("HF_TOKEN=hf_TEST\nDUMMY_API=xyz\n")

    # clear any host-level token so the test is deterministic
    monkeypatch.delenv("HF_TOKEN", raising=False)

    monkeypatch.chdir(tmp_path)      # pretend repo root is tmp
    from vlm_eval.util import load_keys
    load_keys.load_keys()
    assert os.getenv("DUMMY_API") == "xyz"
    assert (tmp_path / ".hf_token").read_text() == "hf_TEST"

# -----------------------------------------------------------------------
# 2) YAML schema round-trip via Draccus
# -----------------------------------------------------------------------
def test_yaml_schema(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("""\
model_family: open-hf
model_id: mock
model_dir: mock
dataset:
  type: text-vqa-slim
  root_dir: /data
results_dir: /results
""")
    from draccus import parse  # call Draccus directly
    from scripts import evaluate as ev
    cfg_obj = parse(ev.EvaluationConfig, config_path=str(cfg))
    assert cfg_obj.model_family == "open-hf"
    assert cfg_obj.dataset.type == "text-vqa-slim"

# -----------------------------------------------------------------------
# 3) run_all classify() mapping
# -----------------------------------------------------------------------
import run_all as RA
@pytest.mark.parametrize("arg,exp", [
    ("claude",   ("anthropic", "claude-4-sonnet-20250522")),
    ("gemini",   ("google",    "gemini-2.5-flash")),
    ("mistralai/Pixtral-12B", ("open-hf", "Pixtral-12B")),
])
def test_classify(arg, exp):
    out = RA.classify(arg)
    assert (out["model_family"], out["model_id"]) == exp

# -----------------------------------------------------------------------
# 4) End-to-end stub (no internet)
# -----------------------------------------------------------------------
def test_evaluate_stub(tmp_path, monkeypatch):
    # create tiny fake dataset
    img_dir = tmp_path / "text-vqa-slim"; img_dir.mkdir()
    (img_dir / "dummy.jpg").write_bytes(b"JPEG")

    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""\
model_family: open-hf
model_id: mock
model_dir: mock
dataset:
  type: text-vqa-slim
  root_dir: {tmp_path}
results_dir: {tmp_path}
""")

    monkeypatch.chdir(REPO_ROOT)
    import scripts.evaluate as ev
    main_fn = ev.evaluate if hasattr(ev, "evaluate") else ev.main
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--config_path", str(cfg)])
    main_fn()

    # a JSONL result file should have been written
    assert any(p.suffix == ".jsonl" for p in tmp_path.iterdir())
