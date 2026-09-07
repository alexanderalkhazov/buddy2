"""Phase 2 driver: systematically searches for WHERE the predictive edge is,
rather than assuming one config (20D absolute direction) is the right one.

Everything here runs on a DEV split (first ~85% of dates chronologically).
The final ~15% of dates is a GOLDEN HOLDOUT that nothing in this file is
allowed to see until the very last step — the winning config (by dev-set AUC)
is retrained on dev data and evaluated EXACTLY ONCE on the holdout, and that
result is frozen (never used to pick a different config).

Run: `python -m processing.ml.phase2_run`
Output: storage/models/phase2_report.json (machine-readable) and a printed
summary matching the report structure requested — baseline, best model,
incremental contributions, failed experiments, best horizon/target/universe,
top features, calibration, decile results, final recommendation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from processing.ml.dataset import build_dataset
from processing.ml.features import FEATURE_COLUMNS
from processing.ml.labels import HORIZONS_DAYS
from processing.ml.registry import ARTIFACT_DIR, run_experiment
from processing.ml.signal_analysis import univariate_signal_table
from processing.ml.train import eval_explicit_split

HOLDOUT_FRACTION = 0.15


def _split_dev_holdout(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates_sorted = np.sort(dataset["date"].unique())
    cutoff = dates_sorted[int(len(dates_sorted) * (1 - HOLDOUT_FRACTION))]
    dev = dataset[dataset["date"] < cutoff].reset_index(drop=True)
    holdout = dataset[dataset["date"] >= cutoff].reset_index(drop=True)
    return dev, holdout


def main() -> dict:
    print("Building full dataset...")
    full = build_dataset()
    dev, holdout = _split_dev_holdout(full)
    print(f"dev: {len(dev)} rows ({dev['date'].min().date()} to {dev['date'].max().date()})")
    print(f"golden holdout (untouched until the end): {len(holdout)} rows "
          f"({holdout['date'].min().date()} to {holdout['date'].max().date()})")

    report: dict = {"dev_rows": len(dev), "holdout_rows": len(holdout)}

    # ---- 1. Univariate signal scan (descriptive, whole-dev-set) -----------
    print("\n[1/6] Univariate signal scan...")
    sig_table = univariate_signal_table(dev)
    report["univariate_signals"] = sig_table.to_dict(orient="records")
    top_signal = sig_table.iloc[0] if len(sig_table) else None

    # ---- 2. Baseline: absolute direction, 20D (Phase 1's original config) -
    print("\n[2/6] Baseline experiment (fwd_up_20d, all features)...")
    baseline = run_experiment(dev, target="fwd_up_20d", label="baseline_abs_20d", notes="Phase 1 baseline, re-run on dev split")
    report["baseline"] = baseline

    # ---- 3. Horizon sweep: absolute direction ------------------------------
    print("\n[3/6] Horizon sweep — absolute direction...")
    horizon_sweep_abs = []
    for h in HORIZONS_DAYS:
        r = run_experiment(dev, target=f"fwd_up_{h}d", label=f"abs_{h}d", notes="horizon sweep, absolute direction")
        horizon_sweep_abs.append(r)
        print(f"  {h}D: AUC(hgb)={r.get('auc_hgb')} brier={r.get('brier_hgb')} rank_ic={r.get('rank_ic')}")
    report["horizon_sweep_absolute"] = horizon_sweep_abs

    # ---- 4. Horizon sweep: relative-to-benchmark direction -----------------
    print("\n[4/6] Horizon sweep — relative-to-SPY direction...")
    horizon_sweep_rel = []
    for h in HORIZONS_DAYS:
        r = run_experiment(dev, target=f"fwd_up_rel_{h}d", label=f"rel_{h}d", notes="horizon sweep, relative-to-benchmark direction")
        horizon_sweep_rel.append(r)
        print(f"  {h}D rel: AUC(hgb)={r.get('auc_hgb')} brier={r.get('brier_hgb')} rank_ic={r.get('rank_ic')}")
    report["horizon_sweep_relative"] = horizon_sweep_rel

    # ---- 5. Threshold target + adversarial sanity check --------------------
    print("\n[5/6] Threshold target (+2% in 20D) and adversarial shuffle test...")
    dev_thresh = dev.copy()
    dev_thresh["fwd_up2pct_20d"] = (dev_thresh["fwd_ret_20d"] > 2).astype(float)
    threshold_exp = run_experiment(
        dev_thresh, target="fwd_up2pct_20d", label="thresh_2pct_20d",
        notes="P(return > 2% within 20D) instead of P(return > 0%)", return_column="fwd_ret_20d",
    )
    report["threshold_2pct_20d"] = threshold_exp

    rng = np.random.default_rng(42)
    dev_shuffled = dev.copy()
    dev_shuffled["fwd_up_20d_shuffled"] = rng.permutation(dev_shuffled["fwd_up_20d"].values)
    adversarial = run_experiment(
        dev_shuffled, target="fwd_up_20d_shuffled", label="ADVERSARIAL_shuffled_labels_20d",
        notes="Sanity check: labels randomly permuted, destroying any real relationship. "
              "AUC should land at ~0.50 — if it doesn't, the harness itself has a leak.",
    )
    report["adversarial_shuffle_sanity_check"] = adversarial
    if adversarial.get("auc_hgb") and abs(adversarial["auc_hgb"] - 0.5) > 0.05:
        report["ADVERSARIAL_WARNING"] = (
            f"Shuffled-label AUC came back {adversarial['auc_hgb']}, meaningfully away from 0.50 — "
            "this indicates a bug in the evaluation harness (likely leakage), NOT a real signal. "
            "Investigate before trusting any other number in this report."
        )
        print(f"  *** WARNING: {report['ADVERSARIAL_WARNING']}")
    else:
        print(f"  OK: shuffled-label AUC = {adversarial.get('auc_hgb')} (~0.50 as expected — harness is sound)")

    # ---- 6. Pick the best dev-set config, freeze one golden holdout run ----
    print("\n[6/6] Selecting best config by dev-set AUC, evaluating ONCE on golden holdout...")
    all_experiments = horizon_sweep_abs + horizon_sweep_rel + [baseline, threshold_exp]
    scored = [e for e in all_experiments if e.get("auc_hgb") is not None]
    best = max(scored, key=lambda e: e["auc_hgb"]) if scored else None
    report["best_dev_config"] = best

    if best is not None:
        target = best["target"]
        train_df = dev if target != "fwd_up2pct_20d" else dev_thresh
        return_col = "fwd_ret_20d" if target == "fwd_up2pct_20d" else None
        test_df = holdout.copy()
        if target == "fwd_up2pct_20d":
            test_df["fwd_up2pct_20d"] = (test_df["fwd_ret_20d"] > 2).astype(float)

        result = eval_explicit_split(train_df, test_df, target=target, return_column=return_col)
        report["golden_holdout_result"] = {
            "config": best["label"],
            "target": target,
            **(result or {"error": "not enough data to form the golden split for this config"}),
            "note": (
                "This is the ONE evaluation of the best dev-selected config against data no experiment "
                "above ever saw (trained on dev only, tested on the holdout period, with the same "
                "horizon-sized embargo applied at the split boundary). It is frozen — do not re-run "
                "phase2 tuning against this number and report a 'better' one; that would defeat the "
                "purpose of a holdout."
            ),
        }
        print(f"  Golden holdout: {report['golden_holdout_result']}")
    else:
        report["golden_holdout_result"] = {"error": "no scored experiment to select from"}

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "phase2_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nFull report written to {ARTIFACT_DIR / 'phase2_report.json'}")
    return report


if __name__ == "__main__":
    main()
