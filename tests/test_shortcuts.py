import json
import subprocess
import sys
from dataclasses import replace

import pytest

from jevprobe.data.shortcuts import GATED, bow_cv_acc, gate, shortcut_report
from jevprobe.data.transitive import MAX_D, build_dataset


@pytest.fixture(scope="module")
def test_set():
    return build_dataset(200, "test", seed=1)


def test_report_is_per_depth_and_passes_gate(test_set):
    report = shortcut_report(test_set)
    assert set(report) == {"mention_order", "name_frequency", "bow_cv", "endpoint_cover", "rank_x_auroc"}
    for metric in report.values():
        assert set(metric) == {str(d) for d in range(1, MAX_D + 1)}
    assert all(v == 0.0 for v in report["endpoint_cover"].values())
    assert gate(report, n_per_d=200) == []


def test_bow_cv_detects_a_planted_leak(test_set):
    leaky = [replace(e, rel="richer" if e.answer else "louder") for e in test_set]
    assert bow_cv_acc(leaky) > 0.9


def test_gate_flags_a_leaky_metric():
    report = {m: {"1": 0.5} for m in GATED} | {"endpoint_cover": {"1": 0.0}}
    report["mention_order"]["1"] = 0.9
    assert gate(report, n_per_d=200) == ["mention_order d=1: 0.900"]


def test_make_data_cli(tmp_path):
    out = tmp_path / "ds"
    subprocess.run(
        [sys.executable, "-m", "jevprobe.data.make_data", "--out", str(out),
         "--train-per-d", "10", "--test-per-d", "40", "--pairs-per-d", "0", "--seed", "0"],
        check=True,
    )
    assert sum(1 for _ in open(out / "train.jsonl")) == 60
    assert sum(1 for _ in open(out / "test.jsonl")) == 240
    report = json.load(open(out / "shortcuts.json"))
    assert set(report["test"]) >= set(GATED)
