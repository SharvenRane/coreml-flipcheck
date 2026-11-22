import numpy as np
import pytest

from conftest import HAVE_TORCH
from coreml_flipcheck import cli, data
from coreml_flipcheck.recipes import DEFAULT_RECIPES, RECIPES, parse


def _fake_imagenette(root, per_class=6):
    from PIL import Image

    rng = np.random.default_rng(0)
    for part in ("train", "val"):
        for wnid in list(data.WNID2IDX)[:3]:
            d = root / part / wnid
            d.mkdir(parents=True)
            for i in range(per_class):
                w, h = int(rng.integers(160, 400)), int(rng.integers(160, 400))
                arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{wnid}_{i}.JPEG")


def test_prepare_writes_disjoint_fixed_size_splits(tmp_path):
    src = tmp_path / "imagenette2-160"
    _fake_imagenette(src)
    meta = data.prepare(tmp_path / "data", src, n_eval=10, n_probe_pool=8, n_calib=4, seed=0)
    assert meta["splits"] == {"eval": 10, "probe_pool": 8, "calib": 4}
    x, y, names = data.load(tmp_path / "data", "eval")
    assert x.shape == (10, 3, 224, 224) and x.dtype == np.uint8
    assert set(y.tolist()) <= set(data.WNID2IDX.values())
    assert all(n.startswith("val/") for n in names)
    _, _, pool = data.load(tmp_path / "data", "probe_pool")
    _, _, calib = data.load(tmp_path / "data", "calib")
    assert not set(pool) & set(calib)
    assert all(n.startswith("train/") for n in list(pool) + list(calib))
    # deterministic
    data.prepare(tmp_path / "again", src, n_eval=10, n_probe_pool=8, n_calib=4, seed=0)
    assert np.array_equal(data.load(tmp_path / "again", "eval")[0], x)


def test_prepare_rejects_too_many(tmp_path):
    src = tmp_path / "imagenette2-160"
    _fake_imagenette(src, per_class=2)
    with pytest.raises(ValueError):
        data.prepare(tmp_path / "data", src, n_eval=100)


def test_to_float_is_imagenet_normalisation():
    x = np.full((1, 3, 2, 2), 255, np.uint8)
    out = data.to_float(x)
    assert out.dtype == np.float32
    assert np.allclose(out[0, :, 0, 0], (1 - data.MEAN.ravel()) / data.STD.ravel())


def test_recipe_parsing():
    assert [r.name for r in parse("default")] == DEFAULT_RECIPES
    assert len(parse("all")) == len(RECIPES)
    assert [r.name for r in parse("fp16, w8")] == ["fp16", "w8"]
    with pytest.raises(ValueError):
        parse("fp8")


def test_cli_lists_recipes(capsys):
    assert cli.main(["recipes"]) == 0
    out = capsys.readouterr().out
    for name in RECIPES:
        assert name in out


def test_cli_probe_select_and_report(fake_run, capsys):
    wd, data_dir = fake_run
    assert cli.main(["probe-select", "--workdir", str(wd.root), "--n", "20"]) == 0
    assert cli.main(["synth-predictions", "--workdir", str(wd.root), "--name", "s", "--flip-rate", "0.01"]) == 0
    assert cli.main(["report", "--workdir", str(wd.root), "--data", str(data_dir)]) == 0
    assert (wd.root / "report.md").exists()


@pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
def test_reordered_run_returns_rows_in_original_order():
    import torch

    from coreml_flipcheck import models

    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 5)).eval()
    x = np.random.default_rng(0).normal(size=(23, 3, 2, 2)).astype(np.float32)
    a = models.run(m, x, batch=4)
    b = models.run(m, x, batch=7, order=np.random.default_rng(1).permutation(23))
    assert np.allclose(a, b, atol=1e-6)


@pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
def test_unknown_model_name_is_rejected():
    from coreml_flipcheck import models

    with pytest.raises(ValueError):
        models.build("not_a_model")
