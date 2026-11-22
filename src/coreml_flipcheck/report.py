"""Build report.json and report.md from reference logits, the probe set and prediction files."""
from __future__ import annotations

import numpy as np

from . import data
from .stats import accuracy_equivalent, compare, probe_flips, verdict
from .workdir import Workdir, read_json, write_json


def _pct(x, digits=2):
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _ci(ci, digits=2):
    return "n/a" if ci is None else f"[{100 * ci[0]:.{digits}f}, {100 * ci[1]:.{digits}f}]"


def _ops(d: dict) -> str:
    return ", ".join(f"{k} x{v}" for k, v in d.items()) if d else "none"


def build(workdir, data_dir, probe_threshold: int = 1, bootstrap: int = 2000) -> dict:
    wd = Workdir(workdir)
    ref_meta = read_json(wd.reference / "reference.json")
    probe = read_json(wd.probe / "probe.json")
    conv = read_json(wd.models / "convert.json") if (wd.models / "convert.json").exists() else {"recipes": {}}
    ref_eval = np.load(wd.reference / "eval.npy")
    ref_pool = np.load(wd.reference / "probe_pool.npy")
    _, y_eval, _ = data.load(data_dir, "eval")
    noise = float(ref_meta["noise_floor"])
    rows = []
    for js in sorted(wd.predictions.glob("*.json")):
        meta = read_json(js)
        row = {"recipe": meta["recipe"], "compute_unit": meta["compute_unit"],
               "synthetic": bool(meta.get("synthetic")), "ok": bool(meta.get("ok"))}
        if not row["ok"]:
            row["error"] = meta.get("error")
            rows.append(row)
            continue
        z = np.load(js.with_suffix(".npz"))
        st = compare(ref_eval, z["eval"], y_eval, b=bootstrap)
        row.update(st.to_dict())
        row["verdict"] = verdict(st, noise)
        row["accuracy_equivalent"] = accuracy_equivalent(st)
        pidx = z["probe_idx"]
        pos = {int(i): j for j, i in enumerate(pidx)}
        for group in ("low_margin", "random"):
            sel = np.array([pos[i] for i in probe[group]], dtype=np.int64)
            row[f"probe_{group}"] = probe_flips(ref_pool[pidx][sel], z["probe"][sel])
        row["probe_n"] = probe["n"]
        row["flagged"] = row["probe_low_margin"] >= probe_threshold
        if "eval_repeat" in z.files:
            row["coreml_repeat_flips"] = probe_flips(z["eval"], z["eval_repeat"])
            row["coreml_repeat_bitwise"] = bool(np.array_equal(z["eval"], z["eval_repeat"]))
        for k in ("compute_plan", "devices", "eval_secs"):
            if k in meta:
                row[k] = meta[k]
        rows.append(row)
    rep = {"model": ref_meta["model"], "n_eval": int(len(ref_eval)), "noise_floor": noise,
           "controls": ref_meta.get("controls"), "reference_host": ref_meta.get("host"),
           "probe": {"n": probe["n"], "pool_size": probe["pool_size"], "threshold": probe_threshold},
           "conversion": conv.get("recipes", {}), "convert_host": conv.get("host"), "rows": rows,
           "any_synthetic": any(r["synthetic"] for r in rows)}
    write_json(wd.root / "report.json", rep)
    (wd.root / "report.md").write_text(markdown(rep), encoding="utf-8")
    return rep


def markdown(rep: dict) -> str:
    L = []
    L.append(f"# coreml-flipcheck report: {rep['model']}")
    L.append("")
    if rep["any_synthetic"]:
        L.append("> **Rows marked `synthetic` are FAKE predictions made by `synth-predictions` to test the report.** "
                 "They say nothing about Core ML.")
        L.append("")
    c = rep.get("controls") or {}
    L.append(f"Eval inputs: {rep['n_eval']}. Probe: the {rep['probe']['n']} lowest margin inputs of a "
             f"{rep['probe']['pool_size']} input pool; a stack is flagged when at least "
             f"{rep['probe']['threshold']} of them flip.")
    L.append("")
    L.append("## Noise floor (reference against itself)")
    L.append("")
    L.append("| control | bitwise equal | flips | max abs logit diff |")
    L.append("|---|---|---|---|")
    for k, v in c.items():
        L.append(f"| {k} | {v['bitwise']} | {v['flips']} | {v['max_abs_logit_diff']:.2e} |")
    L.append("")
    L.append(f"Noise floor F = {_pct(rep['noise_floor'])}. A row is `above noise` when the exact 95% lower bound "
             "of its flip rate exceeds F.")
    L.append("")
    L.append("## Per input decisions")
    L.append("")
    L.append("| recipe | compute unit | flips | flip rate [95% bootstrap] | exact 95% | acc ref / new | "
             "acc change [95%] | McNemar p | verdict | low margin probe | flagged | random probe | "
             "Core ML repeat flips | max abs logit diff |")
    L.append("|" + "---|" * 14)
    for r in rep["rows"]:
        if not r["ok"]:
            L.append(f"| {r['recipe']} | {r['compute_unit']} | prediction failed: {r.get('error')} |" + " |" * 11)
            continue
        cu = r["compute_unit"]
        L.append(
            f"| {r['recipe']} | {cu} | {r['flips']} / {r['n']} | {_pct(r['flip_rate'])} {_ci(r['flip_ci_boot'])} | "
            f"{_ci(r['flip_ci_exact'])} | {_pct(r['acc_ref'], 1)} / {_pct(r['acc_new'], 1)} | "
            f"{100 * r['dacc']:+.2f} pp {_ci(r['dacc_ci_boot'])} | {r['mcnemar_p']:.3g} | {r['verdict']} | "
            f"{r['probe_low_margin']} / {r['probe_n']} | {'YES' if r['flagged'] else 'no'} | "
            f"{r['probe_random']} / {r['probe_n']} | {r.get('coreml_repeat_flips', 'n/a')} | "
            f"{r['max_abs_logit_diff']:.2e} |")
    L.append("")
    hidden = [r for r in rep["rows"] if r["ok"] and r["flips"] > 0 and r.get("accuracy_equivalent")]
    if hidden:
        L.append("Rows whose accuracy change is inside +/- 1 pp (paired bootstrap) yet flip decisions: "
                 + ", ".join(f"`{r['recipe']}/{r['compute_unit']}` ({r['flips']} flips)" for r in hidden)
                 + ". Accuracy alone would call these conversions lossless.")
        L.append("")
    plans = [r for r in rep["rows"] if r.get("compute_plan")]
    if plans:
        L.append("## Where ops were planned to run (MLComputePlan, preferred device per op)")
        L.append("")
        for r in plans:
            L.append(f"* {r['recipe']} / {r['compute_unit']}: {r['compute_plan']}")
        L.append(f"* compute devices visible to Core ML: {plans[0].get('devices')}")
        L.append("")
    L.append("## Conversion")
    L.append("")
    L.append("| recipe | target | converted | size MB | compressed weight ops | weights left uncompressed | error |")
    L.append("|---|---|---|---|---|---|---|")
    for name, r in rep["conversion"].items():
        if r.get("ok"):
            L.append(f"| {name} | {r['target']} | yes | {r['size_bytes'] / 1e6:.1f} | {_ops(r['compressed_ops'])} | {r.get('skipped_weights', 0)} | |")
        else:
            L.append(f"| {name} | {r['target']} | no | | | | `{r['error']}` |")
    L.append("")
    ch, rh = rep.get("convert_host") or {}, rep.get("reference_host") or {}
    L.append(f"Reference host: {rh.get('platform')} torch {rh.get('torch')}. "
             f"Convert host: {ch.get('platform')} coremltools {ch.get('coremltools')}.")
    L.append("")
    return "\n".join(L)
