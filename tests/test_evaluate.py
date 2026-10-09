import json

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from jevprobe import evaluate
from jevprobe.data.transitive import build_dataset, save_jsonl
from jevprobe.hooks import capture
from jevprobe.readout import forward_score, letter_score


def test_revision_resolved_from_hub_not_transformers_config(monkeypatch):
    from types import SimpleNamespace

    class Hub:
        def model_info(self, repo, revision):
            assert (repo, revision) == ("test/model", "v1")
            return SimpleNamespace(sha="immutable")

    monkeypatch.setattr(evaluate, "HfApi", Hub)
    assert evaluate.resolve_revision("test/model", "v1") == "immutable"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_direct_score_matches_captured_readout_without_lm_head(dtype):
    torch.manual_seed(8)
    cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = AutoModelForCausalLM.from_config(cfg).to(dtype).eval()
    inputs = torch.randint(0, 128, (1, 12))
    resid, _ = capture(model, inputs)
    expected = letter_score(model, resid[-1, -1], (32, 33))

    def reject_full_head(*args):
        raise AssertionError("behavior-only evaluation must not project the full vocabulary")

    handle = model.lm_head.register_forward_hook(reject_full_head)
    try:
        assert forward_score(model, inputs, (32, 33)) == pytest.approx(expected, abs=1e-5)
    finally:
        handle.remove()


@pytest.fixture
def run_setup(tmp_path, monkeypatch):
    data, out = tmp_path / "test.jsonl", tmp_path / "run"
    examples = build_dataset(2, "test", 0, ds=[1, 2])
    save_jsonl(examples, data)

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [32 if text == "A" else 33]

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    monkeypatch.setattr(evaluate, "resolve_revision", lambda *a: "abc123")
    monkeypatch.setattr(evaluate, "load_model", lambda *a, **kw: (object(), Tokenizer()))
    monkeypatch.setattr(evaluate, "forward_score", lambda *a: 0.5)
    monkeypatch.setattr(evaluate, "write_summary", lambda *a: None)
    argv = ["--data", str(data), "--model", "test/model@v1", "--out", str(out), "--device", "cpu"]
    return argv, out, data, examples


def test_depth_gate_extends_same_run_and_completed_run_does_not_load(run_setup, monkeypatch):
    argv, out, _, _ = run_setup
    evaluate.main(argv + ["--depths", "1"])
    assert json.loads((out / "status.json").read_text())["state"] == "partial"
    evaluate.main(argv)
    rows = [json.loads(x) for x in (out / "meta.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["uid"] for r in rows}) == 4
    assert {r["d"] for r in rows} == {1, 2}
    assert all("rank_x" in r and "prompt_sha256" in r for r in rows)
    monkeypatch.setattr(evaluate, "load_model", lambda *a, **kw: pytest.fail("loaded completed model"))
    evaluate.main(argv)
    assert json.loads((out / "status.json").read_text())["state"] == "complete"


def test_interruption_and_torn_tail_resume_without_duplicate_rows(run_setup, monkeypatch):
    argv, out, _, _ = run_setup
    scores = iter([0.5])
    monkeypatch.setattr(evaluate, "forward_score", lambda *a: next(scores))
    with pytest.raises(StopIteration):
        evaluate.main(argv)
    assert json.loads((out / "status.json").read_text())["completed"] == 1
    with (out / "meta.jsonl").open("ab") as f:
        f.write(b'{"uid": "torn')
    monkeypatch.setattr(evaluate, "forward_score", lambda *a: -0.5)
    evaluate.main(argv)
    rows = [json.loads(x) for x in (out / "meta.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["uid"] for r in rows}) == 4
    assert rows[0]["score"] == 0.5
    assert all(r["score"] == -0.5 for r in rows[1:])


@pytest.mark.parametrize("change", ["data", "model", "dtype", "source"])
def test_resume_rejects_changed_provenance(run_setup, monkeypatch, change):
    argv, out, data, examples = run_setup
    evaluate.main(argv + ["--depths", "1"])
    original = (out / "meta.jsonl").read_bytes()
    if change == "data":
        save_jsonl(list(reversed(examples)), data)
    elif change == "model":
        argv[argv.index("--model") + 1] = "other/model"
    elif change == "dtype":
        argv += ["--dtype", "float32"]
    else:
        monkeypatch.setattr(evaluate, "source_digest", lambda: "changed")
    with pytest.raises(ValueError, match="provenance mismatch"):
        evaluate.main(argv)
    assert (out / "meta.jsonl").read_bytes() == original


def test_completed_rows_reject_duplicates_and_malformed_full_lines(run_setup):
    argv, out, _, examples = run_setup
    evaluate.main(argv)
    path = out / "meta.jsonl"
    original = path.read_bytes()
    path.write_bytes(original + original.splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate.read_completed(path, examples)
    path.write_bytes(original + b"broken\n")
    with pytest.raises(json.JSONDecodeError):
        evaluate.read_completed(path, examples)


def test_concurrent_writer_rejected(tmp_path):
    with evaluate.run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another evaluator"):
            with evaluate.run_lock(tmp_path):
                pass
