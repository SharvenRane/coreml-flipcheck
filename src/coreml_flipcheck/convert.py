"""PyTorch -> Core ML conversion for each recipe. Runs on Linux and macOS (coremltools has no Windows wheels).

Each recipe is converted independently and its outcome (success, or the error line) is recorded in
models/convert.json, so one failing recipe never hides the others.
"""
from __future__ import annotations

import collections
import contextlib
import logging
import pathlib
import shutil
import time
import traceback

import numpy as np

from . import data, models
from .recipes import Recipe, compress
from .workdir import Workdir, host_info, is_macos, read_json, write_json

INPUT_NAME = "x"
OUTPUT_NAME = "logits"
COMPRESSED_OPS = (
    "constexpr_affine_dequantize",
    "constexpr_blockwise_shift_scale",
    "constexpr_lut_to_dense",
    "constexpr_sparse_to_dense",
    "constexpr_cast",
    "quantize",
    "dequantize",
)


def _targets():
    import coremltools as ct

    return {"iOS17": ct.target.iOS17, "iOS18": ct.target.iOS18}


def op_counts(spec) -> dict:
    """Count MIL op types in the main function of an mlprogram spec."""
    counts = collections.Counter()
    fn = spec.mlProgram.functions["main"]
    for block in fn.block_specializations.values():
        stack = list(block.operations)
        while stack:
            op = stack.pop()
            counts[op.type] += 1
            for b in op.blocks:
                stack.extend(b.operations)
    return dict(sorted(counts.items()))


def dir_size(p: pathlib.Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def trace(model_name: str):
    import torch

    model = models.build(model_name)
    example = torch.zeros(1, 3, 224, 224)
    return torch.jit.trace(model, example).eval()


def _convert_base(traced, precision: str, target: str):
    import coremltools as ct

    prec = {"fp32": ct.precision.FLOAT32, "fp16": ct.precision.FLOAT16}[precision]
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=INPUT_NAME, shape=(1, 3, 224, 224), dtype=np.float32)],
        outputs=[ct.TensorType(name=OUTPUT_NAME)],
        convert_to="mlprogram",
        compute_precision=prec,
        minimum_deployment_target=_targets()[target],
        skip_model_load=not is_macos(),
    )


def _calib_samples(data_dir, limit: int = 64):
    x, _, _ = data.load(data_dir, "calib")
    xf = data.to_float(x[:limit])
    return [{INPUT_NAME: xf[i : i + 1]} for i in range(len(xf))]


class _SkipCollector(logging.Handler):
    """Collects the 'Skipped this op' warnings coremltools logs for weights it leaves uncompressed."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.skipped: list[str] = []
        self.reasons: list[str] = []  # the warning lines that explain the skips, deduplicated

    def emit(self, record):
        msg = record.getMessage()
        if "Skipped this op" in msg:
            self.skipped.append(msg)
        elif msg not in self.reasons and len(self.reasons) < 20:
            self.reasons.append(msg)


@contextlib.contextmanager
def collect_skips():
    h = _SkipCollector()
    # coremltools logs through the "coremltools" logger and, in optimize/_utils.py, through the root
    # logger; records of the first propagate to the root, so one handler on the root sees both
    log = logging.getLogger()
    log.addHandler(h)
    try:
        yield h
    finally:
        log.removeHandler(h)


def error_line(e: BaseException) -> str:
    msg = str(e).strip().splitlines()
    return f"{type(e).__name__}: {msg[0] if msg else ''}"[:500]


def run(model_name: str, workdir, recipes: list[Recipe], data_dir=None, threads: int = 4) -> dict:
    import coremltools as ct
    import torch

    torch.set_num_threads(threads)
    wd = Workdir(workdir)
    wd.ensure(wd.models)
    manifest_path = wd.models / "convert.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"recipes": {}}
    manifest.update(model=model_name, host=host_info(), input=INPUT_NAME, output=OUTPUT_NAME)
    traced = trace(model_name)
    for r in recipes:
        out = wd.models / f"{r.name}.mlpackage"
        rec = {"description": r.description, "precision": r.precision, "compression": r.compression,
               "options": r.options, "target": r.target}
        t0 = time.time()
        try:
            if r.needs_predictions and data_dir is None:
                raise ValueError("this recipe needs --data for calibration inputs")
            mlmodel = _convert_base(traced, r.precision, r.target)
            calib = _calib_samples(data_dir) if r.needs_predictions else None
            with collect_skips() as skips:
                mlmodel = compress(r, mlmodel, calib)
            rec["skipped_weights"] = len(skips.skipped)
            rec["skipped_messages"] = skips.skipped[:50]
            rec["compression_warnings"] = skips.reasons
            if out.exists():
                shutil.rmtree(out)
            mlmodel.short_description = f"{model_name} converted with recipe {r.name} by coreml-flipcheck"
            mlmodel.save(str(out))
            reloaded = ct.models.MLModel(str(out), skip_model_load=True)
            ops = op_counts(reloaded.get_spec())
            rec.update(ok=True, path=out.name, size_bytes=dir_size(out), ops_total=sum(ops.values()),
                       compressed_ops={k: v for k, v in ops.items() if k in COMPRESSED_OPS}, ops=ops)
        except Exception as e:
            rec.update(ok=False, error=error_line(e), traceback=traceback.format_exc(limit=4)[-2000:])
        rec["secs"] = round(time.time() - t0, 2)
        manifest["recipes"][r.name] = rec
        status = "ok" if rec["ok"] else f"FAILED ({rec['error']})"
        extra = (f" {rec['size_bytes'] / 1e6:.1f} MB {rec['compressed_ops']}, "
                 f"{rec.get('skipped_weights', 0)} weights left uncompressed" if rec["ok"] else "")
        print(f"convert {model_name} {r.name}: {status}{extra} in {rec['secs']} s", flush=True)
        write_json(manifest_path, manifest)
    return manifest
