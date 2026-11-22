"""Conversion tests. Need torch and coremltools (Linux or macOS); skipped elsewhere."""
import os
import sys

import numpy as np
import pytest

from conftest import HAVE_CT, HAVE_TORCH

pytestmark = pytest.mark.skipif(not (HAVE_TORCH and HAVE_CT), reason="needs torch and coremltools")

EXPECTED_OP = {
    "w8": "constexpr_affine_dequantize",
    "w4": "constexpr_blockwise_shift_scale",
    "pal6": "constexpr_lut_to_dense",
    "pal4": "constexpr_lut_to_dense",
    "pal4g16": "constexpr_lut_to_dense",
    "sparse50": "constexpr_sparse_to_dense",
}


def _tiny_traced():
    import torch

    torch.manual_seed(0)
    m = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, stride=4), torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3, stride=4), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(64, 1000),
    ).eval()
    return torch.jit.trace(m, torch.zeros(1, 3, 224, 224))


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    from coreml_flipcheck import convert

    monkeypatch.setattr(convert, "trace", lambda name: _tiny_traced())
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    np.savez(data_dir / "calib.npz", x=rng.integers(0, 255, (4, 3, 224, 224), dtype=np.uint8),
             y=np.zeros(4, np.int64), name=np.array(["a", "b", "c", "d"]))
    return convert, tmp_path / "run", data_dir


def test_every_weight_recipe_converts_and_compresses(tiny):
    convert, wd, data_dir = tiny
    from coreml_flipcheck import recipes

    names = ["fp32", "fp16"] + list(EXPECTED_OP)
    m = convert.run("tiny", wd, recipes.parse(",".join(names)), data_dir=data_dir, threads=2)
    for n in names:
        r = m["recipes"][n]
        assert r["ok"], (n, r.get("error"))
        assert (wd / "models" / f"{n}.mlpackage").is_dir()
        if n in EXPECTED_OP:
            assert r["compressed_ops"].get(EXPECTED_OP[n], 0) >= 1, (n, r["compressed_ops"])
        else:
            assert r["compressed_ops"] == {}
    # the 1000 class Linear has 1000 output channels, not a multiple of the group size 16, so coremltools
    # leaves it uncompressed and logs a skip; the manifest must record that
    assert m["recipes"]["pal4g16"]["skipped_weights"] >= 1
    assert m["recipes"]["w8"]["skipped_weights"] == 0
    # fp16 halves the weight file, int8 halves it again, 4 bit palettes shrink it further
    size = {n: m["recipes"][n]["size_bytes"] for n in names}
    assert size["fp16"] < 0.6 * size["fp32"]
    assert size["w8"] < 0.6 * size["fp16"]
    assert size["pal4"] < size["w8"]


@pytest.mark.skipif(sys.platform == "darwin", reason="on macOS the calibration runs")
def test_activation_quantization_needs_macos(tiny):
    convert, wd, data_dir = tiny
    from coreml_flipcheck import recipes

    m = convert.run("tiny", wd, recipes.parse("w8a8"), data_dir=data_dir, threads=2)
    r = m["recipes"]["w8a8"]
    assert r["ok"] is False
    assert "only supported on macOS" in r["error"]


def test_failure_of_one_recipe_does_not_stop_the_others(tiny, monkeypatch):
    convert, wd, data_dir = tiny
    from coreml_flipcheck import recipes

    real = convert.compress

    def flaky(recipe, mlmodel, calib=None):
        if recipe.name == "w8":
            raise ValueError("synthetic failure")
        return real(recipe, mlmodel, calib)

    monkeypatch.setattr(convert, "compress", flaky)
    m = convert.run("tiny", wd, recipes.parse("w8,fp16"), threads=2)
    assert m["recipes"]["w8"]["ok"] is False and m["recipes"]["w8"]["error"] == "ValueError: synthetic failure"
    assert m["recipes"]["fp16"]["ok"] is True


@pytest.mark.skipif(sys.platform == "darwin", reason="predict runs on macOS")
def test_predict_refuses_off_macos(tmp_path):
    from coreml_flipcheck import predict

    with pytest.raises(SystemExit, match="macOS"):
        predict.run(tmp_path, tmp_path)


@pytest.mark.skipif(not os.environ.get("FLIPCHECK_SLOW"), reason="set FLIPCHECK_SLOW=1 (downloads weights)")
def test_real_torchvision_model_converts(tmp_path):
    from coreml_flipcheck import convert, recipes

    m = convert.run("mobilenet_v3_small", tmp_path, recipes.parse("default"), threads=2)
    assert all(m["recipes"][n]["ok"] for n in recipes.DEFAULT_RECIPES)
