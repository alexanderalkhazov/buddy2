"""Univariate signal analysis — the deepest-priority Phase 2 question: does ANY
individual feature contain real information, before touching a multivariate
model that might just be failing to extract signal that's actually there?

This is a whole-history DESCRIPTIVE scan (not a walk-forward OOS test — that's
what registry.run_experiment is for). Read it as "does this feature correlate
with the future at all, historically," not as a validated trading signal —
a feature can pass every check here and still fail to survive walk-forward
(regime-dependent, decayed, or simply noise that happened to correlate over
this exact sample). That distinction is the whole point of running both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from processing.ml.features import FEATURE_COLUMNS


def univariate_signal_table(dataset: pd.DataFrame, target_return: str = "fwd_ret_20d", target_direction: str = "fwd_up_20d") -> pd.DataFrame:
    rows = []
    for col in FEATURE_COLUMNS:
        sub = dataset.dropna(subset=[col, target_return, target_direction])
        if len(sub) < 100:
            rows.append({"feature": col, "n": len(sub), "note": "insufficient sample (<100)"})
            continue

        ic = spearmanr(sub[col], sub[target_return]).statistic
        try:
            auc = roc_auc_score(sub[target_direction], sub[col])
        except ValueError:
            auc = np.nan

        rows.append(
            {
                "feature": col,
                "n": len(sub),
                "spearman_ic_vs_fwd_ret": round(float(ic), 4) if ic == ic else None,
                "univariate_auc": round(float(auc), 4) if auc == auc else None,
                # AUC is direction-agnostic wrt the sign convention of the raw feature
                # (a feature that's predictive but inverted scores <0.5, not 0) — flag
                # that case explicitly rather than let it read as "no signal."
                "inverted_relationship": bool(auc == auc and auc < 0.48),
            }
        )
    out = pd.DataFrame(rows)
    if "univariate_auc" in out.columns:
        out["abs_edge_over_random"] = (out["univariate_auc"] - 0.5).abs()
        out = out.sort_values("abs_edge_over_random", ascending=False).drop(columns="abs_edge_over_random")
    return out.reset_index(drop=True)


def decile_conditional_returns(dataset: pd.DataFrame, feature: str, target_return: str = "fwd_ret_20d") -> pd.DataFrame:
    """Bucket the whole dataset into feature deciles and show conditional
    forward-return stats per bucket — the "RSI bottom 10% vs top 10%" style
    check the brief asks for, generalized to any feature."""
    sub = dataset.dropna(subset=[feature, target_return]).copy()
    if len(sub) < 100:
        return pd.DataFrame({"error": [f"insufficient sample for {feature} ({len(sub)} rows)"]})

    try:
        sub["bucket"] = pd.qcut(sub[feature], 10, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame({"error": [f"{feature} has too few unique values to decile"]})

    agg = sub.groupby("bucket").agg(
        n=(target_return, "size"),
        feature_range_low=(feature, "min"),
        feature_range_high=(feature, "max"),
        mean_fwd_return_pct=(target_return, "mean"),
        median_fwd_return_pct=(target_return, "median"),
        win_rate_pct=(target_return, lambda s: (s > 0).mean() * 100),
    ).round(3)
    return agg.reset_index()


def stability_by_year(dataset: pd.DataFrame, feature: str, target_return: str = "fwd_ret_20d") -> pd.DataFrame:
    """Does this feature's IC hold up year over year, or is it one lucky year
    driving the whole-sample number? A feature whose sign flips across years is
    not a reliable signal even if its pooled IC looks decent."""
    sub = dataset.dropna(subset=[feature, target_return]).copy()
    sub["year"] = pd.to_datetime(sub["date"]).dt.year
    rows = []
    for year, g in sub.groupby("year"):
        if len(g) < 30:
            rows.append({"year": int(year), "n": len(g), "ic": None, "note": "insufficient sample"})
            continue
        ic = spearmanr(g[feature], g[target_return]).statistic
        rows.append({"year": int(year), "n": len(g), "ic": round(float(ic), 4) if ic == ic else None})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from processing.ml.dataset import build_dataset

    ds = build_dataset()
    table = univariate_signal_table(ds)
    pd.set_option("display.width", 140)
    print(table.to_string(index=False))
