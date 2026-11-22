import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


HAVE_TORCH = _have("torch")
HAVE_CT = _have("coremltools")


@pytest.fixture
def fake_run(tmp_path):
    """A workdir and data dir with random reference logits and labels (no torch needed)."""
    from coreml_flipcheck.workdir import Workdir, write_json

    rng = np.random.default_rng(1)
    n_eval, n_pool, c = 1000, 400, 50
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ref_eval = rng.normal(size=(n_eval, c)).astype(np.float32)
    ref_pool = rng.normal(size=(n_pool, c)).astype(np.float32)
    # labels: the reference is right on about 70% of inputs
    y = ref_eval.argmax(1)
    wrong = rng.random(n_eval) < 0.3
    y[wrong] = (y[wrong] + 1) % c
    for split, n, yy in (("eval", n_eval, y), ("probe_pool", n_pool, np.zeros(n_pool, np.int64)),
                         ("calib", 4, np.zeros(4, np.int64))):
        np.savez(data_dir / f"{split}.npz", x=np.zeros((n, 3, 2, 2), np.uint8), y=yy,
                 name=np.array([f"img{i}" for i in range(n)]))
    wd = Workdir(tmp_path / "run")
    wd.ensure(wd.reference)
    np.save(wd.reference / "eval.npy", ref_eval)
    np.save(wd.reference / "probe_pool.npy", ref_pool)
    write_json(wd.reference / "reference.json", {
        "model": "fake", "noise_floor": 0.0, "host": {},
        "controls": {"rerun": {"bitwise": True, "flips": 0, "max_abs_logit_diff": 0.0}}})
    return wd, data_dir
