import math

import numpy as np
import pytest

from coreml_flipcheck import stats
from coreml_flipcheck.predict import swap_top2


def test_decisions_and_margins():
    z = np.array([[1.0, 3.0, 2.0], [5.0, 0.0, 4.5]])
    assert stats.decisions(z).tolist() == [1, 0]
    assert np.allclose(stats.margins(z), [1.0, 0.5])


def test_swap_top2_always_flips_even_on_ties():
    z = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0], [3.0, 3.0, 3.0]], np.float32)
    out = swap_top2(z, [0, 1, 2])
    assert (stats.decisions(out) != stats.decisions(z)).all()
    assert np.array_equal(swap_top2(z, []), z)


@pytest.mark.parametrize("k", [0, 1, 37, 250])
def test_known_flips_give_known_flip_rate(k):
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(1000, 20)).astype(np.float32)
    rows = rng.choice(1000, size=k, replace=False)
    new = swap_top2(ref, rows)
    s = stats.compare(ref, new, b=500)
    assert s.flips == k
    assert s.flip_rate == pytest.approx(k / 1000)
    lo, hi = s.flip_ci_exact
    assert lo <= k / 1000 <= hi
    blo, bhi = s.flip_ci_boot
    assert blo <= k / 1000 <= bhi
    if k == 0:
        assert s.max_abs_logit_diff == 0.0
        assert stats.verdict(s, 0.0) == "no flips"


def test_clopper_pearson_reference_values():
    # k = 0 of 1000: the upper bound has the closed form 1 - 0.025 ** (1 / 1000)
    lo, hi = stats.clopper_pearson(0, 1000)
    assert lo == 0.0
    assert hi == pytest.approx(1 - 0.025 ** (1 / 1000), rel=1e-9)
    lo, hi = stats.clopper_pearson(1000, 1000)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 1000), rel=1e-9)
    lo, hi = stats.clopper_pearson(10, 100)
    assert lo == pytest.approx(0.0490, abs=1e-3) and hi == pytest.approx(0.1762, abs=1e-3)


def test_mcnemar_exact_matches_binomial_test():
    from scipy.stats import binomtest

    for b, c in [(0, 0), (5, 0), (3, 7), (20, 9), (12, 12)]:
        expected = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
        assert stats.mcnemar_exact(b, c) == pytest.approx(expected, rel=1e-12)


def test_accuracy_change_and_discordant_counts():
    ref = np.eye(4, dtype=np.float32)[[0, 1, 2, 3, 0, 1]]
    y = np.array([0, 1, 2, 3, 1, 1])  # reference right on 5 of 6
    new = ref.copy()
    new[0] = np.eye(4)[1]  # right -> wrong
    new[4] = np.eye(4)[1]  # wrong -> right
    s = stats.compare(ref, new, y, b=200)
    assert s.flips == 2
    assert s.acc_ref == pytest.approx(5 / 6) and s.acc_new == pytest.approx(5 / 6)
    assert s.dacc == 0.0
    assert (s.ref_right_new_wrong, s.ref_wrong_new_right) == (1, 1)
    assert s.mcnemar_p == 1.0
    assert stats.accuracy_equivalent(s) is False  # 6 inputs: the interval is wide


def test_auroc_low_margin_predicts_flips():
    score = np.array([0.1, 0.2, 0.3, 0.9])
    assert stats.auroc(score, np.array([0, 0, 1, 1])) == 1.0
    assert stats.auroc(score, np.array([1, 1, 0, 0])) == 0.0
    assert math.isnan(stats.auroc(score, np.zeros(4)))


def test_verdict_against_noise_floor():
    ref = np.random.default_rng(3).normal(size=(1000, 10))
    s = stats.compare(ref, swap_top2(ref, np.arange(30)), b=200)
    assert stats.verdict(s, 0.0) == "above noise"
    assert stats.verdict(s, 0.05) == "within noise"


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        stats.compare(np.zeros((3, 4)), np.zeros((3, 5)))
