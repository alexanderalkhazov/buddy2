"""Phase 3 dataset: same point-in-time-safe feature/label machinery as
dataset.py, extended with sector labels and market-regime conditioning, built
over the broader EXPANDED_UNIVERSE_V2. Kept as a SEPARATE module (not an edit
to dataset.py) so Phase 1/2's dataset builder, and everything that reproduces
their exact numbers, is untouched.

sector is a STATIC, present-day GICS-ish label (see universe.py's
survivorship-bias disclosure) — a stock's sector rarely changes, so using
today's label for historical rows is a minor, disclosed impurity, not a
forward-return-relevant leak (sector membership is not the thing being
predicted).
"""

from __future__ import annotations

import pandas as pd

from processing.ml.dataset import MIN_HISTORY_BARS, SAMPLE_STRIDE
from processing.ml.features import FEATURE_COLUMNS, compute_features
from processing.ml.labels import HORIZONS_DAYS, compute_labels
from processing.ml.regime import compute_regime_features
from processing.ml.universe import BENCHMARK, EXPANDED_UNIVERSE, SECTOR_MAP
from storage.cache import get_ohlcv


def build_dataset_v2(
    universe: list[str] | None = None,
    horizons: tuple[int, ...] = HORIZONS_DAYS,
    refresh: bool = False,
) -> pd.DataFrame:
    universe = universe or EXPANDED_UNIVERSE
    bench_df = get_ohlcv(BENCHMARK, force=refresh)
    bench_labs = compute_labels(bench_df, horizons=horizons)
    regime = compute_regime_features(bench_df)

    rows, excluded = [], []
    for ticker in universe:
        try:
            df = get_ohlcv(ticker, force=refresh)
        except Exception as exc:
            excluded.append({"ticker": ticker, "reason": f"fetch failed: {exc}"})
            continue
        if len(df) < MIN_HISTORY_BARS:
            excluded.append({"ticker": ticker, "reason": f"only {len(df)} bars, need {MIN_HISTORY_BARS}+"})
            continue

        feats = compute_features(df, benchmark_df=bench_df)
        labs = compute_labels(df, horizons=horizons)
        merged = feats[FEATURE_COLUMNS].join(labs)

        bench_aligned = bench_labs.reindex(merged.index)
        for h in horizons:
            merged[f"fwd_rel_ret_{h}d"] = merged[f"fwd_ret_{h}d"] - bench_aligned[f"fwd_ret_{h}d"]
            merged[f"fwd_up_rel_{h}d"] = (merged[f"fwd_rel_ret_{h}d"] > 0).astype("float")
            merged.loc[bench_aligned[f"fwd_ret_{h}d"].isna(), [f"fwd_rel_ret_{h}d", f"fwd_up_rel_{h}d"]] = float("nan")

        merged = merged.join(regime.reindex(merged.index))
        merged = merged.iloc[MIN_HISTORY_BARS:]
        merged = merged.iloc[::SAMPLE_STRIDE]
        merged = merged.dropna(subset=FEATURE_COLUMNS)

        merged.insert(0, "ticker", ticker)
        merged.insert(1, "sector", SECTOR_MAP.get(ticker, "unknown"))
        merged.insert(2, "date", merged.index)
        rows.append(merged.reset_index(drop=True))

    if not rows:
        raise RuntimeError("no tickers in the expanded universe produced usable data")

    out = pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)
    return out, excluded


def universe_report(dataset: pd.DataFrame, excluded: list[dict]) -> dict:
    return {
        "n_tickers": int(dataset["ticker"].nunique()),
        "n_rows": int(len(dataset)),
        "date_range": [str(dataset["date"].min().date()), str(dataset["date"].max().date())],
        "sector_distribution": dataset.groupby("sector")["ticker"].nunique().to_dict(),
        "excluded_tickers": excluded,
        "survivorship_bias_note": (
            "Universe is TODAY's liquid large-cap constituent list per sector, not a point-in-time "
            "historical index membership snapshot — see processing/ml/universe.py for the full "
            "disclosure. Results here describe how today's well-known names behaved historically, "
            "not the true historical investable universe including delisted/acquired/bankrupt names."
        ),
        "market_cap_liquidity_note": (
            "Market-cap and liquidity bucketing (requested in the Phase 3 brief) was NOT built this "
            "round — it would require per-ticker point-in-time market cap/volume history beyond what "
            "this pass covers; flagged as a scope cut, not a silent omission."
        ),
    }


if __name__ == "__main__":
    ds, excl = build_dataset_v2()
    import json

    print(json.dumps(universe_report(ds, excl), indent=2, default=str))
