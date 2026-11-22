"""Fixed input arrays shared by every step.

Images come from Imagenette (fastai, Apache 2.0), a 10 class subset of ImageNet whose classes map to
ImageNet 1k indices, so any torchvision or timm ImageNet classifier can be scored on it without
fine tuning. The images themselves are ImageNet photos: treat them as research use only.

Each split is stored once as uint8 (N, 3, 224, 224) after resize (shorter side 256, bilinear) and
center crop 224, so the reference model and the Core ML model see byte identical inputs.
Normalisation to float happens in `to_float`, which is the only place it happens.
"""
from __future__ import annotations

import json
import pathlib
import tarfile
import urllib.request

import numpy as np

URLS = {
    "160": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    "320": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
}

# Imagenette wnid -> ImageNet 1k class index.
WNID2IDX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
SPLITS = ("eval", "probe_pool", "calib")


def to_float(x_u8: np.ndarray) -> np.ndarray:
    """uint8 NCHW -> ImageNet normalised float32 NCHW, computed the same way for every runtime."""
    x = x_u8.astype(np.float32) / np.float32(255.0)
    return ((x - MEAN) / STD).astype(np.float32)


def crop224(path) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((left, top, left + 224, top + 224)), dtype=np.uint8).transpose(2, 0, 1)


def list_images(root: pathlib.Path, part: str) -> list[tuple[pathlib.Path, int]]:
    base = root / part
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found; expected an extracted imagenette2 folder")
    out = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if d.name not in WNID2IDX:
            raise ValueError(f"unknown class folder {d.name}")
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".jpeg", ".jpg", ".png"):
                out.append((f, WNID2IDX[d.name]))
    return out


def download(variant: str, dest: pathlib.Path) -> pathlib.Path:
    """Download and extract imagenette2-<variant>; returns the extracted folder. Skips if present."""
    if variant not in URLS:
        raise ValueError(f"variant must be one of {sorted(URLS)}")
    dest.mkdir(parents=True, exist_ok=True)
    folder = dest / f"imagenette2-{variant}"
    if (folder / "val").is_dir():
        return folder
    tgz = dest / f"imagenette2-{variant}.tgz"
    if not tgz.exists():
        print(f"downloading {URLS[variant]}", flush=True)
        urllib.request.urlretrieve(URLS[variant], tgz)
    with tarfile.open(tgz) as tf:
        members = [m for m in tf.getmembers() if m.isfile() or m.isdir()]
        for m in members:  # refuse absolute paths and parent references
            if m.name.startswith("/") or ".." in pathlib.PurePosixPath(m.name).parts:
                raise ValueError(f"unsafe path in archive: {m.name}")
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tf.extractall(dest, members=members, **kw)
    return folder


def _subset(items, n, rng):
    if n > len(items):
        raise ValueError(f"asked for {n} images, only {len(items)} available")
    pick = np.sort(rng.permutation(len(items))[:n])
    return [items[i] for i in pick]


def prepare(
    out: pathlib.Path,
    source: pathlib.Path,
    n_eval: int = 1000,
    n_probe_pool: int = 1000,
    n_calib: int = 64,
    seed: int = 0,
) -> dict:
    """Write eval.npz (from val), probe_pool.npz and calib.npz (disjoint, from train)."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    val = list_images(source, "val")
    train = list_images(source, "train")
    ev = _subset(val, n_eval, rng)
    tr = _subset(train, n_probe_pool + n_calib, rng)
    order = rng.permutation(len(tr))
    pool = [tr[i] for i in np.sort(order[:n_probe_pool])]
    calib = [tr[i] for i in np.sort(order[n_probe_pool:])]
    meta = {"source": str(source), "seed": seed, "splits": {}}
    for name, items in (("eval", ev), ("probe_pool", pool), ("calib", calib)):
        x = np.stack([crop224(f) for f, _ in items]) if items else np.zeros((0, 3, 224, 224), np.uint8)
        y = np.array([c for _, c in items], np.int64)
        names = np.array([str(f.relative_to(source)).replace("\\", "/") for f, _ in items])
        np.savez(out / f"{name}.npz", x=x, y=y, name=names)
        meta["splits"][name] = len(items)
        print(f"{name}: {x.shape}", flush=True)
    (out / "data.json").write_text(json.dumps(meta, indent=1))
    return meta


def load(data_dir: pathlib.Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x uint8, y int64, names) for a split written by `prepare`."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    z = np.load(pathlib.Path(data_dir) / f"{split}.npz")
    return z["x"], z["y"], z["name"]
