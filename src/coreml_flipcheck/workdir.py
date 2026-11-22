"""Layout of a run folder and small helpers shared by the steps.

<workdir>/
  reference/       eval.npy, probe_pool.npy, eval_rerun.npy, eval_reorder.npy, reference.json
  probe/           probe.json (indices into probe_pool, low margin and random groups)
  models/          <recipe>.mlpackage, convert.json
  predictions/     <recipe>__<compute_unit>.npz + .json
  report.md, report.json
"""
from __future__ import annotations

import json
import pathlib
import platform
import sys


class Workdir:
    def __init__(self, root):
        self.root = pathlib.Path(root)

    @property
    def reference(self) -> pathlib.Path:
        return self.root / "reference"

    @property
    def probe(self) -> pathlib.Path:
        return self.root / "probe"

    @property
    def models(self) -> pathlib.Path:
        return self.root / "models"

    @property
    def predictions(self) -> pathlib.Path:
        return self.root / "predictions"

    def ensure(self, *dirs: pathlib.Path) -> None:
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


def read_json(p) -> dict:
    return json.loads(pathlib.Path(p).read_text())


def write_json(p, obj) -> None:
    pathlib.Path(p).write_text(json.dumps(obj, indent=1, default=_default))


def _default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def is_macos() -> bool:
    return sys.platform == "darwin"


def host_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    for mod in ("numpy", "torch", "torchvision", "coremltools", "timm"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info
