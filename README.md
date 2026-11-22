# coreml-flipcheck

Check whether converting a PyTorch image classifier to Core ML changes the decision on individual
inputs, not only the aggregate accuracy.

For every conversion recipe (fp32, fp16, int8 and int4 weights, palettization, pruning, int8
activations) and every Core ML compute unit setting, the tool reports how many top 1 decisions differ
from the PyTorch reference, with intervals, against a measured noise floor, and whether a cheap probe
set of 200 low margin inputs would have caught it.

## Why accuracy is not enough

A converted model can match the source accuracy while giving different answers on different inputs:
the wins and losses cancel in the average. The preprint *One Checkpoint, Four Converters: Per Input
Disagreement After On Device Conversion and a Probe Set That Catches It* (Sharven Rane, 2026,
[doi:10.5281/zenodo.24433993](https://doi.org/10.5281/zenodo.24433993)) measured this for ExecuTorch,
LiteRT, ONNX Runtime and TensorRT: ONNX Runtime static INT8 flipped 4.07% of ResNet 50 decisions
(95% CI 3.65 to 4.53) while accuracy moved by 0.15 points, every strict FP32 conversion reproduced
every decision, and the 200 training inputs with the smallest source margin flagged every stack that
flipped at least 0.5% of decisions. That study had no Core ML results because Core ML predictions need
macOS. This tool applies the same method to Core ML.

## What it measures

| quantity | definition |
|---|---|
| flip | an input whose top 1 class under Core ML differs from the PyTorch reference |
| flip rate | flips / n, with a 95% percentile bootstrap interval (2,000 resamples, seed 0) and an exact Clopper Pearson interval |
| noise floor F | flip rate of the reference against itself: a rerun, and a run with shuffled order and batch size 7 |
| verdict | `above noise` when the exact lower bound of the flip rate exceeds F |
| accuracy change | paired bootstrap interval, plus an exact McNemar test on the inputs where exactly one model is right |
| low margin probe | the 200 inputs of a separate pool with the smallest reference margin (top 1 minus top 2 logit); a stack is `flagged` when at least one of them flips |
| random probe | 200 random pool inputs, for comparison with the low margin probe |
| Core ML repeat flips | flips between two passes of the same Core ML model on the same machine |
| compute plan | preferred device per op from `MLComputePlan`, recorded with each prediction |

The reference is PyTorch on CPU in fp32. Inputs are Imagenette (10 ImageNet classes, mapped to their
ImageNet 1k indices) after resize to 256 and center crop to 224, stored once as uint8 so both models
see byte identical inputs.

## Where each step runs

coremltools converts on Linux and macOS but runs predictions only on macOS, and it publishes no
Windows wheels. The steps are split accordingly.

| step | Linux | macOS | output |
|---|---|---|---|
| `prepare-data` | yes | yes | `data/{eval,probe_pool,calib}.npz` |
| `reference` | yes | yes | `reference/*.npy`, noise floor |
| `probe-select` | yes | yes | `probe/probe.json` |
| `convert` | yes (except `w8a8`) | yes | `models/<recipe>.mlpackage`, `models/convert.json` |
| `predict` | no | yes | `predictions/<recipe>__<unit>.npz` |
| `report` | yes | yes | `report.md`, `report.json` |

## Install

```bash
pip install torch torchvision
pip install -e .            # pulls coremltools on Linux and macOS
pip install -e ".[timm]"    # optional, for timm/<name> models
```

Tested locally with Python 3.12, torch 2.11.0, torchvision 0.26.0 and coremltools 9.0 on Linux
(WSL2). coremltools 9.0 prints that it was last tested with torch 2.7; the conversions below worked
with 2.11 regardless.

## Quickstart

On Linux (or macOS), prepare everything up to the converted models:

```bash
coreml-flipcheck prepare-data --download 320 --out data          # 341 MB download, 2,064 images kept
coreml-flipcheck reference    --model resnet18 --data data --workdir runs/resnet18
coreml-flipcheck probe-select --workdir runs/resnet18 --n 200
coreml-flipcheck convert      --model resnet18 --workdir runs/resnet18 --recipes all --data data
coreml-flipcheck recipes                                        # list the recipes
```

Copy `data/` and `runs/resnet18/` to a Mac, then:

```bash
coreml-flipcheck predict --workdir runs/resnet18 --data data --compute-units cpu,gpu,ne,all
coreml-flipcheck report  --workdir runs/resnet18 --data data
```

Run `reference` on the Mac as well if you want the noise floor measured on the same machine that runs
Core ML; it overwrites the Linux reference and keeps the comparison on one host.

## Recipes

| name | precision | compression | target |
|---|---|---|---|
| `fp32` | fp32 | none | iOS17 |
| `fp16` | fp16 | none | iOS17 |
| `w8` | fp16 | int8 linear symmetric weights, per channel | iOS17 |
| `w4` | fp16 | int4 linear symmetric weights, per block of 32 | iOS18 |
| `pal6` | fp16 | 6 bit kmeans palettization, per tensor | iOS17 |
| `pal4` | fp16 | 4 bit kmeans palettization, per tensor | iOS17 |
| `pal4g16` | fp16 | 4 bit kmeans palettization, per grouped channel (16) | iOS18 |
| `sparse50` | fp16 | 50% magnitude pruning | iOS17 |
| `w8a8` | fp16 | int8 activations calibrated on 64 images, then int8 weights | iOS17, macOS only |

Every recipe is converted on its own; a failure is recorded with its error line and the rest continue.
coremltools skips weights it cannot compress with a log warning and still returns a model, so
`convert.json` and the report count the weights left uncompressed and keep the warnings that explain
why.

### What coremltools 9.0 did on Linux

Measured on WSL2, torch 2.11.0, coremltools 9.0:

| model | recipe | result |
|---|---|---|
| resnet18, mobilenet_v3_small | `fp32`, `fp16`, `w8`, `pal6`, `pal4`, `sparse50` | converted, every eligible weight compressed |
| resnet18 | `w4` | converted; 1 of 21 weights left in fp16: `Invalid block_sizes; On 1th axis, the dim size 3 is not divisible by block size 32.` (the first conv) |
| mobilenet_v3_small | `w4` | converted; 29 of 45 weights left in fp16 (input dims 1, 24, 88 and others are not multiples of 32), so the model is 2.1 MB against 1.4 MB for `pal4` |
| resnet18 | `pal4g16` | converted; the classifier was skipped: `The number of channels at 0th axis (1000) is not divisible by channel_group_size (16).` |
| mobilenet_v3_small | `pal4g16` | converted; 12 of 45 weights skipped for the same reason (88, 24, 40, 120 channels) |
| both | `w8a8` | fails off macOS: `Exception: Model prediction is only supported on macOS version 10.13 or later.` (activation calibration runs the model) |

Not supported: Hugging Face text classifiers. The preprint found that coremltools 9.0 cannot convert
DistilBERT (`PyTorch convert function for op 'new_ones' not implemented`), so this version covers
image classifiers only.

## CI on a macOS runner

`.github/workflows/coreml.yml` runs the whole pipeline on a GitHub hosted `macos-15` runner (free for
public repositories): install, download Imagenette (cached), reference, probe selection, conversion
with every recipe, predictions for `cpu`, `gpu`, `ne` and `all`, then the report. The report is
appended to the job summary and uploaded as an artifact with the JSON files. Start it from the Actions
tab (`workflow_dispatch`, with model list, recipes, compute units and eval size as inputs); it also
runs weekly. `.github/workflows/tests.yml` runs the unit tests on Linux and macOS.

GitHub hosted macOS runners are virtual machines with a paravirtual GPU. Do not read the `ne` or `all`
rows from CI as Neural Engine results: check the compute plan section and the list of compute devices
Core ML reports, and run `predict` on a physical Mac for device numbers.

## Example report

> **Example from synthetic data.** The prediction rows below are FAKE: `synth-predictions` took the
> real resnet18 reference logits and swapped the top two classes on a chosen share of inputs. No Core ML
> predictions have been run yet. The noise floor and conversion rows are real (WSL2, coremltools 9.0).

```bash
coreml-flipcheck synth-predictions --workdir runs/resnet18 --name synthetic_1pct --flip-rate 0.01
coreml-flipcheck report --workdir runs/resnet18 --data data
```

| recipe | compute unit | flips | flip rate [95% bootstrap] | exact 95% | acc ref / new | acc change [95%] | McNemar p | verdict | low margin probe | flagged | random probe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic_0pct | synthetic | 0 / 1000 | 0.00% [0.00, 0.00] | [0.00, 0.37] | 77.1% / 77.1% | +0.00 pp [0.00, 0.00] | 1 | no flips | 0 / 200 | no | 0 / 200 |
| synthetic_1pct | synthetic | 10 / 1000 | 1.00% [0.50, 1.70] | [0.48, 1.83] | 77.1% / 76.9% | -0.20 pp [-0.70, 0.30] | 0.688 | above noise | 4 / 200 | YES | 1 / 200 |
| synthetic_5pct_uniform | synthetic | 50 / 1000 | 5.00% [3.70, 6.40] | [3.73, 6.54] | 77.1% / 74.0% | -3.10 pp [-4.40, -1.80] | 3.12e-06 | above noise | 9 / 200 | YES | 13 / 200 |

The `synthetic_1pct` row is the case the tool exists for: the accuracy change interval sits inside
one point, McNemar finds nothing, and 10 decisions still flipped. The synthetic flips favour low margin
inputs, which is what the preprint observed for real conversions; the low margin probe catches them
with 4 of 200 against 1 of 200 for a random probe. The uniform row is a contrast, not a model of any
real converter.

Noise floor on the same run: the rerun and the reordered run were both bitwise identical to the
reference (F = 0).

## Tests

```bash
pytest -q                       # unit tests; conversion tests skip without coremltools
FLIPCHECK_SLOW=1 pytest -q      # adds a real torchvision conversion (downloads weights)
```

The tests cover the statistics (a synthetic converted output with a known number of flips gives that
flip rate, Clopper Pearson closed forms, McNemar against `scipy.stats.binomtest`), probe selection,
the report built from hand made prediction files, data preparation from a fake Imagenette folder, and
conversion of a small network with every weight recipe. Core ML predictions are covered only by the
macOS workflow.

## Data and licences

Imagenette is published by fastai under Apache 2.0. Its images are ImageNet photos, so treat them as
research use only; the tool stores them only in your local `data/` folder and the reports contain
predictions, not images. torchvision weights are trained on ImageNet and carry the same caveat.

## License

MIT, see [LICENSE](LICENSE).
