"""Per input decision statistics.

Everything here works on plain numpy arrays so it can be tested without torch or coremltools.

Definitions
-----------
decision      top 1 class index of a logit row.
flip          an input whose decision under the converted model differs from the reference decision.
flip rate     flips / n.
margin        top 1 logit minus top 2 logit of the reference model (small margin = fragile decision).
noise floor   flip rate of the reference model against itself under a harmless change (rerun, reordered
              batches). A converted model should not be judged against zero if the reference itself moves.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_BOOTSTRAP = 2000


def decisions(logits: np.ndarray) -> np.ndarray:
    """Top 1 index per row. Ties resolve to the lowest index (numpy argmax)."""
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"expected 2D logits, got shape {logits.shape}")
    return logits.argmax(axis=1)


def margins(logits: np.ndarray) -> np.ndarray:
    """Top 1 minus top 2 logit per row (float64)."""
    z = np.asarray(logits, dtype=np.float64)
    if z.shape[1] < 2:
        raise ValueError("need at least two classes for a margin")
    top2 = np.partition(z, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def bootstrap_indices(n: int, b: int = DEFAULT_BOOTSTRAP, seed: int = 0) -> np.ndarray:
    """b resamples of n input indices with replacement, fixed seed so reports are reproducible."""
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    return float(np.percentile(values, 100 * a)), float(np.percentile(values, 100 * (1 - a)))


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two sided binomial interval for k successes in n trials."""
    from scipy import stats

    if n <= 0:
        raise ValueError("n must be positive")
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def mcnemar_exact(b: int, c: int) -> float:
    """Two sided exact McNemar test (binomial test on the discordant pairs, p = 0.5).

    b = inputs the reference gets right and the converted model gets wrong,
    c = inputs the reference gets wrong and the converted model gets right.
    Returns 1.0 when there are no discordant pairs.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return float(min(1.0, 2.0 * tail))


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve of `score` for the boolean `positive` labels (rank formula, ties averaged)."""
    from scipy import stats

    y = np.asarray(positive, dtype=bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@dataclass
class FlipStats:
    n: int
    flips: int
    flip_rate: float
    flip_ci_boot: tuple[float, float]
    flip_ci_exact: tuple[float, float]
    acc_ref: float | None = None
    acc_new: float | None = None
    dacc: float | None = None
    dacc_ci_boot: tuple[float, float] | None = None
    ref_right_new_wrong: int | None = None
    ref_wrong_new_right: int | None = None
    mcnemar_p: float | None = None
    margin_auroc: float | None = None
    max_abs_logit_diff: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compare(
    ref_logits: np.ndarray,
    new_logits: np.ndarray,
    labels: np.ndarray | None = None,
    b: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    level: float = 0.95,
) -> FlipStats:
    """Flip rate of `new_logits` against `ref_logits`, with intervals, and paired accuracy if labels exist."""
    ref_logits = np.asarray(ref_logits)
    new_logits = np.asarray(new_logits)
    if ref_logits.shape != new_logits.shape:
        raise ValueError(f"shape mismatch: reference {ref_logits.shape} vs new {new_logits.shape}")
    n = ref_logits.shape[0]
    dref, dnew = decisions(ref_logits), decisions(new_logits)
    flip = dref != dnew
    k = int(flip.sum())
    idx = bootstrap_indices(n, b, seed)
    out = FlipStats(
        n=n,
        flips=k,
        flip_rate=k / n,
        flip_ci_boot=percentile_ci(flip[idx].mean(axis=1), level),
        flip_ci_exact=clopper_pearson(k, n, level),
        max_abs_logit_diff=float(np.max(np.abs(ref_logits.astype(np.float64) - new_logits.astype(np.float64)))),
    )
    if k > 0 and k < n:
        out.margin_auroc = auroc(-margins(ref_logits), flip)
    if labels is not None:
        y = np.asarray(labels)
        if y.shape[0] != n:
            raise ValueError("labels length does not match logits")
        ok_ref, ok_new = dref == y, dnew == y
        out.acc_ref = float(ok_ref.mean())
        out.acc_new = float(ok_new.mean())
        out.dacc = out.acc_new - out.acc_ref
        out.dacc_ci_boot = percentile_ci(ok_new[idx].mean(axis=1) - ok_ref[idx].mean(axis=1), level)
        out.ref_right_new_wrong = int((ok_ref & ~ok_new).sum())
        out.ref_wrong_new_right = int((~ok_ref & ok_new).sum())
        out.mcnemar_p = mcnemar_exact(out.ref_right_new_wrong, out.ref_wrong_new_right)
    return out


def verdict(stats_: FlipStats, noise_floor: float) -> str:
    """'above noise' when the exact lower bound of the flip rate clears the noise floor, else 'within noise'."""
    if stats_.flips == 0:
        return "no flips"
    return "above noise" if stats_.flip_ci_exact[0] > noise_floor else "within noise"


def accuracy_equivalent(stats_: FlipStats, band: float = 0.01) -> bool | None:
    """True when the paired bootstrap interval of the accuracy change lies inside +/- band."""
    if stats_.dacc_ci_boot is None:
        return None
    lo, hi = stats_.dacc_ci_boot
    return lo > -band and hi < band


def probe_flips(ref_logits: np.ndarray, new_logits: np.ndarray) -> int:
    return int((decisions(ref_logits) != decisions(new_logits)).sum())
