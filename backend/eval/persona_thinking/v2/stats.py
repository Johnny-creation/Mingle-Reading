# -*- coding: utf-8 -*-
"""Small, dependency-free statistics for the v2 evaluation.

The v1 design reported raw mean gaps with no uncertainty. v2 attaches a 95%
bootstrap CI and a paired permutation p-value to every headline contrast, so a
claim like "full beats neutral on distinctive items" comes with a falsifiable
significance statement rather than an eyeballed number.

All contrasts are PAIRED by item (the same probe answered under two conditions),
so we resample / permute within item — the correct structure for this design.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Contrast:
    name: str
    n: int
    mean_a: float
    mean_b: float
    diff: float          # mean(a) - mean(b)
    ci_low: float
    ci_high: float
    p_value: float       # two-sided paired permutation

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "diff": round(self.diff, 4),
            "ci95": [round(self.ci_low, 4), round(self.ci_high, 4)],
            "p_value": round(self.p_value, 4),
            "significant_0.05": self.p_value < 0.05,
        }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def paired_bootstrap_ci(
    a: list[float], b: list[float], *, n_boot: int = 10000, ci: float = 95.0, seed: int = 7
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(a) - mean(b), resampling item indices."""
    rng = random.Random(seed)
    m = len(a)
    diffs = []
    idx = range(m)
    for _ in range(n_boot):
        sample = [rng.randrange(m) for _ in idx]
        diffs.append(_mean([a[i] for i in sample]) - _mean([b[i] for i in sample]))
    diffs.sort()
    lo = (100 - ci) / 2
    hi = 100 - lo
    return diffs[int(lo / 100 * n_boot)], diffs[min(n_boot - 1, int(hi / 100 * n_boot))]


def paired_permutation_p(
    a: list[float], b: list[float], *, n_perm: int = 10000, seed: int = 7
) -> float:
    """Two-sided paired permutation test: randomly swap each item's (a,b) labels
    and see how often |mean diff| meets or exceeds the observed."""
    rng = random.Random(seed)
    obs = abs(_mean(a) - _mean(b))
    diffs = [ai - bi for ai, bi in zip(a, b)]
    count = 0
    for _ in range(n_perm):
        s = sum((d if rng.random() < 0.5 else -d) for d in diffs)
        if abs(s / len(diffs)) >= obs - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def contrast(name: str, a: list[float], b: list[float], **kw) -> Contrast:
    """Paired contrast a vs b (per-item aligned lists)."""
    assert len(a) == len(b) and a, f"{name}: need equal non-empty paired lists"
    lo, hi = paired_bootstrap_ci(a, b, **{k: v for k, v in kw.items() if k in ("n_boot", "ci", "seed")})
    p = paired_permutation_p(a, b, **{k: v for k, v in kw.items() if k in ("n_perm", "seed")})
    return Contrast(name, len(a), _mean(a), _mean(b), _mean(a) - _mean(b), lo, hi, p)


if __name__ == "__main__":
    rng = random.Random(0)
    # smoke: a is reliably higher than b -> positive diff, small p
    a = [1.0 if rng.random() < 0.8 else 0.0 for _ in range(40)]
    b = [1.0 if rng.random() < 0.4 else 0.0 for _ in range(40)]
    print(contrast("demo full-neutral", a, b).as_dict())
