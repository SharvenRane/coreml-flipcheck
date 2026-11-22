"""Reference logits and the noise floor.

The reference is the PyTorch model on CPU in float32, batch 32. Two control runs measure how much the
reference moves under changes that should not matter:
  rerun    same inputs, same order, same batch size (expected bitwise identical)
  reorder  same inputs, shuffled order, batch size 7 (different kernels and summation order)
The noise floor is the larger flip rate of the two controls against the reference.
"""
from __future__ import annotations

import numpy as np

from . import data, models
from .stats import decisions, margins
from .workdir import Workdir, host_info, write_json


def run(model_name: str, data_dir, workdir, threads: int = 4, batch: int = 32, seed: int = 0) -> dict:
    import torch

    torch.set_num_threads(threads)
    wd = Workdir(workdir)
    wd.ensure(wd.reference)
    model = models.build(model_name)
    meta = {"model": model_name, "threads": threads, "batch": batch, "host": host_info(), "splits": {}}
    for split in ("eval", "probe_pool"):
        x, y, _ = data.load(data_dir, split)
        xf = data.to_float(x)
        logits = models.run(model, xf, batch=batch)
        np.save(wd.reference / f"{split}.npy", logits)
        meta["splits"][split] = {"n": len(x), "acc": float((decisions(logits) == y).mean()) if len(x) else None}
        if split == "eval":
            rerun = models.run(model, xf, batch=batch)
            order = np.random.default_rng(seed).permutation(len(xf))
            reorder = models.run(model, xf, batch=7, order=order)
            np.save(wd.reference / "eval_rerun.npy", rerun)
            np.save(wd.reference / "eval_reorder.npy", reorder)
            d = decisions(logits)
            meta["controls"] = {
                "rerun": {"bitwise": bool(np.array_equal(rerun, logits)),
                          "flips": int((decisions(rerun) != d).sum()),
                          "max_abs_logit_diff": float(np.abs(rerun - logits).max())},
                "reorder": {"bitwise": bool(np.array_equal(reorder, logits)),
                            "flips": int((decisions(reorder) != d).sum()),
                            "max_abs_logit_diff": float(np.abs(reorder - logits).max())},
            }
            meta["noise_floor"] = max(v["flips"] for v in meta["controls"].values()) / len(xf)
            meta["eval_margin_median"] = float(np.median(margins(logits)))
        print(f"reference {model_name} {split}: n={len(x)} acc={meta['splits'][split]['acc']}", flush=True)
    write_json(wd.reference / "reference.json", meta)
    return meta


def select_probe(workdir, n: int = 200, seed: int = 0) -> dict:
    """The n lowest margin inputs of the probe pool (by reference margin), plus n random ones for contrast."""
    wd = Workdir(workdir)
    wd.ensure(wd.probe)
    pool = np.load(wd.reference / "probe_pool.npy")
    if n > len(pool):
        raise ValueError(f"probe size {n} exceeds probe pool size {len(pool)}")
    mg = margins(pool)
    low = np.argsort(mg, kind="stable")[:n]
    rnd = np.random.default_rng(seed).permutation(len(pool))[:n]
    out = {"n": n, "seed": seed, "pool_size": len(pool),
           "low_margin": low.tolist(), "random": rnd.tolist(),
           "low_margin_max": float(mg[low].max()), "pool_margin_median": float(np.median(mg))}
    write_json(wd.probe / "probe.json", out)
    print(f"probe: {n} lowest margin of {len(pool)} (max margin {out['low_margin_max']:.4f}, "
          f"pool median {out['pool_margin_median']:.4f})", flush=True)
    return out
