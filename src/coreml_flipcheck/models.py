"""Reference PyTorch classifiers.

Model names:
  resnet18, mobilenet_v3_small, ...   any torchvision classification model, default weights
  timm/<name>                         a timm model with pretrained weights (needs `pip install timm`)

All models take a normalised float32 (N, 3, 224, 224) tensor and return ImageNet 1k logits.
"""
from __future__ import annotations


def build(name: str):
    import torch

    if name.startswith("timm/"):
        try:
            import timm
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("timm models need `pip install timm`") from e
        model = timm.create_model(name.split("/", 1)[1], pretrained=True)
        cfg = getattr(model, "pretrained_cfg", {}) or {}
        size = tuple(cfg.get("input_size", (3, 224, 224)))
        if size != (3, 224, 224):
            raise ValueError(f"{name} expects input {size}; this tool feeds 3x224x224")
    else:
        import torchvision

        if name not in torchvision.models.list_models(module=torchvision.models):
            raise ValueError(f"{name} is not a torchvision classification model")
        model = torchvision.models.get_model(name, weights="DEFAULT")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(0)
    return model


def run(model, x_float, batch: int = 32, order=None):
    """Logits for x_float (numpy float32 NCHW). `order` permutes the inputs before batching; outputs are
    returned in the original order, so a reordered run can be compared row by row with a plain run."""
    import numpy as np
    import torch

    n = len(x_float)
    order = np.arange(n) if order is None else np.asarray(order)
    out = np.empty((n, 0), np.float32)
    chunks = []
    with torch.inference_mode():
        for s in range(0, n, batch):
            ii = order[s : s + batch]
            chunks.append(model(torch.from_numpy(x_float[ii])).float().numpy())
    out = np.concatenate(chunks) if chunks else out
    res = np.empty_like(out)
    res[order] = out
    return res
