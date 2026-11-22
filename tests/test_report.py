import json

import numpy as np
import pytest

from coreml_flipcheck import report
from coreml_flipcheck.predict import swap_top2, synthetic
from coreml_flipcheck.reference import select_probe
from coreml_flipcheck.stats import decisions, margins
from coreml_flipcheck.workdir import write_json


def _write_prediction(wd, tag, arrays, meta):
    wd.ensure(wd.predictions)
    if arrays:
        np.savez(wd.predictions / f"{tag}.npz", **arrays)
    write_json(wd.predictions / f"{tag}.json", meta)


def test_probe_select_takes_lowest_margins(fake_run):
    wd, _ = fake_run
    p = select_probe(wd.root, n=50)
    mg = margins(np.load(wd.reference / "probe_pool.npy"))
    assert sorted(p["low_margin"]) == sorted(np.argsort(mg, kind="stable")[:50].tolist())
    assert max(mg[p["low_margin"]]) <= np.sort(mg)[49] + 1e-12
    assert len(set(p["random"])) == 50


def test_report_from_hand_made_prediction_with_known_flips(fake_run):
    wd, data_dir = fake_run
    p = select_probe(wd.root, n=50)
    ref_eval = np.load(wd.reference / "eval.npy")
    ref_pool = np.load(wd.reference / "probe_pool.npy")
    pidx = np.unique(np.array(p["low_margin"] + p["random"]))
    flip_eval = np.arange(0, 1000, 40)  # 25 known flips, so 2.5%
    new_pool = swap_top2(ref_pool, p["low_margin"][:3])  # 3 of the low margin probes flip
    _write_prediction(
        wd, "fake__cpu",
        {"eval": swap_top2(ref_eval, flip_eval), "probe": new_pool[pidx], "probe_idx": pidx,
         "eval_repeat": swap_top2(ref_eval, flip_eval)},
        {"recipe": "fake", "compute_unit": "cpu", "ok": True})
    _write_prediction(wd, "broken__all", {}, {"recipe": "broken", "compute_unit": "all", "ok": False,
                                               "error": "RuntimeError: boom"})
    rep = report.build(wd.root, data_dir, bootstrap=300)
    rows = {r["recipe"]: r for r in rep["rows"]}
    r = rows["fake"]
    assert r["flips"] == 25 and r["flip_rate"] == pytest.approx(0.025)
    assert r["probe_low_margin"] == 3 and r["flagged"] is True
    assert r["probe_random"] == len(set(p["random"]) & set(p["low_margin"][:3]))
    assert r["coreml_repeat_flips"] == 0 and r["coreml_repeat_bitwise"] is True
    assert r["verdict"] == "above noise"
    assert rows["broken"]["ok"] is False
    md = (wd.root / "report.md").read_text()
    assert "25 / 1000" in md and "prediction failed: RuntimeError: boom" in md
    assert "FAKE" not in md  # no synthetic rows here
    assert json.loads((wd.root / "report.json").read_text())["rows"]


def test_synthetic_predictions_are_labelled_and_exact(fake_run):
    wd, data_dir = fake_run
    select_probe(wd.root, n=50)
    synthetic(wd.root, "syn", flip_rate=0.02, seed=4)
    synthetic(wd.root, "syn_uniform", flip_rate=0.05, seed=4, low_margin_bias=False)
    rep = report.build(wd.root, data_dir, bootstrap=300)
    rows = {r["recipe"]: r for r in rep["rows"]}
    assert rows["syn"]["flips"] == 20 and rows["syn"]["synthetic"]
    assert rows["syn_uniform"]["flips"] == 50
    assert rep["any_synthetic"]
    assert "FAKE predictions" in (wd.root / "report.md").read_text()


def test_flip_free_prediction_is_not_flagged(fake_run):
    wd, data_dir = fake_run
    p = select_probe(wd.root, n=50)
    ref_eval = np.load(wd.reference / "eval.npy")
    ref_pool = np.load(wd.reference / "probe_pool.npy")
    pidx = np.unique(np.array(p["low_margin"] + p["random"]))
    noisy = ref_eval + np.float32(1e-6)
    assert (decisions(noisy) == decisions(ref_eval)).all()
    _write_prediction(wd, "clean__cpu", {"eval": noisy, "probe": ref_pool[pidx], "probe_idx": pidx},
                      {"recipe": "clean", "compute_unit": "cpu", "ok": True})
    rep = report.build(wd.root, data_dir, bootstrap=300)
    r = rep["rows"][0]
    assert r["flips"] == 0 and r["verdict"] == "no flips" and r["flagged"] is False
    assert "coreml_repeat_flips" not in r
