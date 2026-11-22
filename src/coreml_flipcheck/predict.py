"""Core ML predictions (macOS only).

For each converted recipe and each requested compute unit setting, runs every eval input and every
probe input one at a time and stores the logits. A second pass over the eval inputs (`repeats=2`)
measures whether the Core ML runtime itself is deterministic on this machine.

The compute unit setting is a request. Core ML decides per op where it runs; the compute plan
(device per op, recorded when the API is available) shows that decision.
"""
from __future__ import annotations

import collections
import time

import numpy as np

from . import data
from .convert import INPUT_NAME
from .workdir import Workdir, host_info, is_macos, read_json, write_json

COMPUTE_UNITS = {
    "cpu": "CPU_ONLY",
    "gpu": "CPU_AND_GPU",
    "ne": "CPU_AND_NE",
    "all": "ALL",
}


def probe_indices(workdir) -> np.ndarray:
    p = read_json(Workdir(workdir).probe / "probe.json")
    return np.unique(np.array(p["low_margin"] + p["random"], dtype=np.int64))


def _compute_plan(mlmodel, cu) -> dict | None:
    """Preferred device per op type from MLComputePlan (coremltools 8+, macOS 14.4+). None if unavailable."""
    try:
        from coremltools.models.compute_plan import MLComputePlan

        plan = MLComputePlan.load_from_path(path=mlmodel.get_compiled_model_path(), compute_units=cu)
        prog = plan.model_structure.program
        counts = collections.Counter()
        for op in prog.functions["main"].block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = "none" if usage is None else type(usage.preferred_compute_device).__name__
            counts[dev] += 1
        return dict(counts)
    except Exception as e:  # the plan is diagnostic only
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def _devices() -> list[str]:
    try:
        from coremltools.models.compute_device import get_all_compute_devices

        return [type(d).__name__ for d in get_all_compute_devices()]
    except Exception as e:
        return [f"unavailable: {type(e).__name__}"]


def _predict_all(mlmodel, out_name, xf) -> np.ndarray:
    rows = [np.asarray(mlmodel.predict({INPUT_NAME: xf[i : i + 1]})[out_name], np.float32).reshape(-1)
            for i in range(len(xf))]
    return np.stack(rows) if rows else np.zeros((0, 0), np.float32)


def run(workdir, data_dir, compute_units=("cpu", "all"), recipes=None, repeats: int = 2) -> list[dict]:
    if not is_macos():
        raise SystemExit("predict needs macOS: coremltools runs Core ML models only there")
    import coremltools as ct

    wd = Workdir(workdir)
    wd.ensure(wd.predictions)
    manifest = read_json(wd.models / "convert.json")
    out_name = manifest["output"]
    x_eval = data.to_float(data.load(data_dir, "eval")[0])
    pidx = probe_indices(workdir)
    x_probe = data.to_float(data.load(data_dir, "probe_pool")[0][pidx])
    devices = _devices()
    done = []
    for name, rec in manifest["recipes"].items():
        if recipes and name not in recipes:
            continue
        if not rec.get("ok"):
            print(f"skip {name}: conversion failed", flush=True)
            continue
        for cu_key in compute_units:
            cu = getattr(ct.ComputeUnit, COMPUTE_UNITS[cu_key])
            tag = f"{name}__{cu_key}"
            meta = {"recipe": name, "compute_unit": cu_key, "host": host_info(), "devices": devices}
            try:
                t0 = time.time()
                mlmodel = ct.models.MLModel(str(wd.models / rec["path"]), compute_units=cu)
                meta["load_secs"] = round(time.time() - t0, 3)
                t0 = time.time()
                ev = _predict_all(mlmodel, out_name, x_eval)
                meta["eval_secs"] = round(time.time() - t0, 3)
                arrays = {"eval": ev, "probe": _predict_all(mlmodel, out_name, x_probe), "probe_idx": pidx}
                if repeats > 1:
                    arrays["eval_repeat"] = _predict_all(mlmodel, out_name, x_eval)
                meta["compute_plan"] = _compute_plan(mlmodel, cu)
                np.savez(wd.predictions / f"{tag}.npz", **arrays)
                meta["ok"] = True
            except Exception as e:
                meta.update(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:400] if str(e) else ''}")
            write_json(wd.predictions / f"{tag}.json", meta)
            print(f"predict {tag}: {'ok' if meta['ok'] else meta['error']}", flush=True)
            done.append(meta)
    return done


def synthetic(workdir, name: str, flip_rate: float, seed: int = 0, low_margin_bias: bool = True) -> dict:
    """Write a FAKE prediction file for testing the report: reference logits with a chosen share of
    decisions moved to the runner up class. Clearly labelled as synthetic in its metadata and in the report.
    With low_margin_bias the flipped inputs are drawn with probability falling with reference margin,
    which mimics what real conversions do; otherwise uniformly."""
    from .stats import margins

    wd = Workdir(workdir)
    wd.ensure(wd.predictions)
    rng = np.random.default_rng(seed)
    pidx = probe_indices(workdir)
    arrays = {"probe_idx": pidx}
    flipped = {}
    for split, ref in (("eval", np.load(wd.reference / "eval.npy")),
                       ("probe", np.load(wd.reference / "probe_pool.npy")[pidx])):
        k = int(round(flip_rate * len(ref)))
        if low_margin_bias:
            w = 1.0 / (margins(ref) + 0.05)
            pick = rng.choice(len(ref), size=k, replace=False, p=w / w.sum())
        else:
            pick = rng.choice(len(ref), size=k, replace=False)
        arrays[split] = swap_top2(ref, pick)
        flipped[split] = int(k)
    arrays["eval_repeat"] = arrays["eval"].copy()
    tag = f"{name}__synthetic"
    np.savez(wd.predictions / f"{tag}.npz", **arrays)
    meta = {"recipe": name, "compute_unit": "synthetic", "synthetic": True, "ok": True,
            "flip_rate_requested": flip_rate, "flipped": flipped, "host": host_info()}
    write_json(wd.predictions / f"{tag}.json", meta)
    return meta


def swap_top2(logits: np.ndarray, rows) -> np.ndarray:
    """Copy of logits where, in the given rows, the top 1 and top 2 logits trade places (a guaranteed flip)."""
    z = np.array(logits, dtype=np.float32, copy=True)
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return z
    order = np.argsort(-z[rows], axis=1, kind="stable")
    a, b = order[:, 0], order[:, 1]
    va, vb = z[rows, a].copy(), z[rows, b].copy()
    z[rows, a], z[rows, b] = vb, va
    # break exact ties so the runner up becomes the unique top 1
    tie = va == vb
    if tie.any():
        z[rows[tie], b[tie]] += np.float32(1e-3)
    return z
