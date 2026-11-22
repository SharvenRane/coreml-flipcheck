"""Conversion recipes.

Every recipe starts from `ct.convert(..., convert_to="mlprogram")` with a compute precision, then
optionally applies one `coremltools.optimize.coreml` post training step to the converted model.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Recipe:
    name: str
    precision: str  # "fp32" or "fp16"
    description: str
    compression: str | None = None  # key into _COMPRESSORS
    options: dict = field(default_factory=dict)
    target: str = "iOS17"  # minimum deployment target
    needs_predictions: bool = False  # calibration runs the model, which coremltools can do only on macOS


RECIPES: dict[str, Recipe] = {
    r.name: r
    for r in [
        Recipe("fp32", "fp32", "float32 compute precision, no compression"),
        Recipe("fp16", "fp16", "float16 compute precision (coremltools default), no compression"),
        Recipe("w8", "fp16", "fp16 + int8 linear symmetric weights, per channel", "linear",
               {"mode": "linear_symmetric", "dtype": "int8", "granularity": "per_channel"}),
        Recipe("w4", "fp16", "fp16 + int4 linear symmetric weights, per block of 32", "linear",
               {"mode": "linear_symmetric", "dtype": "int4", "granularity": "per_block", "block_size": 32},
               target="iOS18"),
        Recipe("pal6", "fp16", "fp16 + 6 bit kmeans palettization, per tensor", "palettize",
               {"mode": "kmeans", "nbits": 6, "granularity": "per_tensor"}),
        Recipe("pal4", "fp16", "fp16 + 4 bit kmeans palettization, per tensor", "palettize",
               {"mode": "kmeans", "nbits": 4, "granularity": "per_tensor"}),
        Recipe("pal4g16", "fp16", "fp16 + 4 bit kmeans palettization, per grouped channel (16)", "palettize",
               {"mode": "kmeans", "nbits": 4, "granularity": "per_grouped_channel", "group_size": 16},
               target="iOS18"),
        Recipe("sparse50", "fp16", "fp16 + 50% magnitude pruning of weights", "prune",
               {"target_sparsity": 0.5}),
        Recipe("w8a8", "fp16", "fp16 + int8 activations (calibrated) + int8 weights", "w8a8",
               {"mode": "linear_symmetric"}, needs_predictions=True),
    ]
}

DEFAULT_RECIPES = ["fp32", "fp16", "w8", "w4", "pal6", "pal4"]


def parse(spec: str) -> list[Recipe]:
    """Comma separated recipe names, or 'all' / 'default'."""
    if spec == "all":
        names = list(RECIPES)
    elif spec == "default":
        names = DEFAULT_RECIPES
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in RECIPES]
    if unknown:
        raise ValueError(f"unknown recipe(s) {unknown}; known: {sorted(RECIPES)}")
    return [RECIPES[n] for n in names]


def compress(recipe: Recipe, mlmodel, calib_samples=None):
    """Apply the recipe's post training compression to a converted MLModel."""
    import coremltools.optimize.coreml as cto

    kind = recipe.compression
    if kind is None:
        return mlmodel
    if kind == "linear":
        cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(**recipe.options))
        return cto.linear_quantize_weights(mlmodel, config=cfg)
    if kind == "palettize":
        cfg = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(**recipe.options))
        return cto.palettize_weights(mlmodel, config=cfg)
    if kind == "prune":
        cfg = cto.OptimizationConfig(global_config=cto.OpMagnitudePrunerConfig(**recipe.options))
        return cto.prune_weights(mlmodel, config=cfg)
    if kind == "w8a8":
        if not calib_samples:
            raise ValueError("w8a8 needs calibration samples")
        cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(**recipe.options))
        a8 = cto.linear_quantize_activations(mlmodel, cfg, calib_samples)
        return cto.linear_quantize_weights(a8, config=cfg)
    raise ValueError(f"unknown compression {kind}")
