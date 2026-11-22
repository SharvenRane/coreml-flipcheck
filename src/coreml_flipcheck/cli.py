"""coreml-flipcheck command line.

Linux or macOS:  prepare-data, reference, probe-select, convert, report, synth-predictions
macOS only:      predict
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from . import __version__


def _p(s):
    return pathlib.Path(s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="coreml-flipcheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare-data", help="build eval, probe pool and calibration arrays from Imagenette")
    s.add_argument("--out", type=_p, required=True)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--source-dir", type=_p, help="an extracted imagenette2-160 or imagenette2-320 folder")
    g.add_argument("--download", choices=["160", "320"], help="download imagenette2-<size> into --cache")
    s.add_argument("--cache", type=_p, default=_p(".cache"))
    s.add_argument("--n-eval", type=int, default=1000)
    s.add_argument("--n-probe-pool", type=int, default=1000)
    s.add_argument("--n-calib", type=int, default=64)
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("reference", help="PyTorch reference logits and noise floor controls")
    s.add_argument("--model", required=True, help="torchvision name (resnet18) or timm/<name>")
    s.add_argument("--data", type=_p, required=True)
    s.add_argument("--workdir", type=_p, required=True)
    s.add_argument("--threads", type=int, default=4)

    s = sub.add_parser("probe-select", help="pick the N lowest margin probe inputs")
    s.add_argument("--workdir", type=_p, required=True)
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("convert", help="convert the model with each recipe")
    s.add_argument("--model", required=True)
    s.add_argument("--workdir", type=_p, required=True)
    s.add_argument("--recipes", default="default", help="comma list, 'default' or 'all' (see `recipes`)")
    s.add_argument("--data", type=_p, help="needed by recipes that calibrate (w8a8)")
    s.add_argument("--threads", type=int, default=4)

    sub.add_parser("recipes", help="list conversion recipes")

    s = sub.add_parser("predict", help="(macOS) run Core ML predictions for every converted recipe")
    s.add_argument("--workdir", type=_p, required=True)
    s.add_argument("--data", type=_p, required=True)
    s.add_argument("--compute-units", default="cpu,all", help="comma list of cpu, gpu, ne, all")
    s.add_argument("--recipes", default=None, help="comma list; default every converted recipe")
    s.add_argument("--repeats", type=int, default=2, help="2 = run eval twice to test Core ML determinism")

    s = sub.add_parser("report", help="write report.md and report.json")
    s.add_argument("--workdir", type=_p, required=True)
    s.add_argument("--data", type=_p, required=True)
    s.add_argument("--probe-threshold", type=int, default=1)

    s = sub.add_parser("synth-predictions", help="write a FAKE prediction file (for testing the report)")
    s.add_argument("--workdir", type=_p, required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--flip-rate", type=float, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--uniform", action="store_true", help="flip uniformly instead of favouring low margins")

    a = ap.parse_args(argv)

    if a.cmd == "prepare-data":
        from . import data

        src = a.source_dir if a.source_dir else data.download(a.download, a.cache)
        data.prepare(a.out, src, a.n_eval, a.n_probe_pool, a.n_calib, a.seed)
    elif a.cmd == "reference":
        from . import reference

        reference.run(a.model, a.data, a.workdir, threads=a.threads)
    elif a.cmd == "probe-select":
        from . import reference

        reference.select_probe(a.workdir, a.n, a.seed)
    elif a.cmd == "convert":
        from . import convert, recipes

        m = convert.run(a.model, a.workdir, recipes.parse(a.recipes), data_dir=a.data, threads=a.threads)
        return 0 if any(r.get("ok") for r in m["recipes"].values()) else 1
    elif a.cmd == "recipes":
        from .recipes import DEFAULT_RECIPES, RECIPES

        for r in RECIPES.values():
            tags = [r.target] + (["default"] if r.name in DEFAULT_RECIPES else []) + (
                ["calibrates on macOS"] if r.needs_predictions else [])
            print(f"{r.name:10s} {r.description}  ({', '.join(tags)})")
    elif a.cmd == "predict":
        from . import predict

        cus = [c.strip() for c in a.compute_units.split(",") if c.strip()]
        bad = [c for c in cus if c not in predict.COMPUTE_UNITS]
        if bad:
            ap.error(f"unknown compute unit(s) {bad}")
        rs = a.recipes.split(",") if a.recipes else None
        res = predict.run(a.workdir, a.data, cus, rs, a.repeats)
        return 0 if res and all(r["ok"] for r in res) else 1
    elif a.cmd == "report":
        from . import report

        rep = report.build(a.workdir, a.data, a.probe_threshold)
        print(f"wrote {a.workdir / 'report.md'} ({len(rep['rows'])} rows)")
    elif a.cmd == "synth-predictions":
        from . import predict

        m = predict.synthetic(a.workdir, a.name, a.flip_rate, a.seed, low_margin_bias=not a.uniform)
        print(f"wrote synthetic predictions {a.name}: {m['flipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
