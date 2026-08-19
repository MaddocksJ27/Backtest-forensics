"""
02 -- Is a performance rating a measurement, or the result restated?

Racing Post Ratings (RPR) are the standard performance measure in British
racing. A natural idea is to reframe "which horse wins" as "which horse
most exceeds its official handicap mark", since that quantity is
continuous and gives a training signal for every runner rather than one
bit per race.

This script tests whether that reframing actually buys anything. It does
not: most of the target is the finishing position wearing different units.

Run:  python analysis/02_rpr_circularity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def top_ranked_wins(df, col, ascending=False):
    """How often does the top-ranked runner on `col` win its race?"""
    x = df.dropna(subset=[col]).copy()
    x["rank"] = x.groupby("race_id")[col].rank(ascending=ascending, method="min")
    per_race = x[x["rank"] == 1].groupby("race_id")["win"].max()
    return len(per_race), 100 * per_race.mean()


def main():
    df = pd.read_parquet(ROOT / "data/built.parquet")
    hcap = df[df.is_hcap].dropna(subset=["rpr_n", "or_n"]).copy()
    hcap["beat_mark"] = hcap.rpr_n - hcap.or_n
    other = df[~df.is_hcap].copy()

    print("=" * 68)
    print("RPR CIRCULARITY")
    print("=" * 68)

    # --- the seductive result --------------------------------------------
    print("\n[1] THE RESULT THAT LOOKS LIKE A DISCOVERY")
    n, p = top_ranked_wins(hcap, "beat_mark")
    print(f"  In handicaps, the horse that most exceeds its official mark")
    print(f"  wins {p:.2f}% of the time  ({n:,} races)")
    n2, p2 = top_ranked_wins(other, "rpr_n")
    print(f"  In non-handicaps, the top-rated horse on the day")
    print(f"  wins {p2:.2f}% of the time  ({n2:,} races)")

    print("\n  Ranked by margin over mark, win rate by rank:")
    h = hcap.copy()
    h["rank"] = h.groupby("race_id")["beat_mark"].rank(ascending=False, method="min")
    g = h.groupby(h["rank"].clip(upper=5))["win"].agg(["mean", "size"])
    g.columns = ["win rate", "n"]
    print(g.round(4).to_string())

    # --- why it is circular ----------------------------------------------
    print("\n[2] WHY THIS IS CIRCULAR")
    print("\n  Margin over mark by finishing position:")
    g = hcap[hcap.pos_n <= 10].groupby("pos_n")["beat_mark"].agg(
        ["mean", "std", "size"])
    print(g.round(2).to_string())

    h = hcap.dropna(subset=["pos_n", "ovr_btn"]).copy()
    h["btn_n"] = pd.to_numeric(h.ovr_btn, errors="coerce")
    h = h.dropna(subset=["btn_n"])
    print(f"\n  corr(margin over mark, finishing position) = "
          f"{hcap.beat_mark.corr(hcap.pos_n):+.4f}")

    sample = h.sample(min(200_000, len(h)), random_state=0)
    split = int(len(sample) * 0.75)
    tr, te = sample.iloc[:split], sample.iloc[split:]
    feats = ["pos_n", "btn_n", "ran_n"]
    model = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(
        tr[feats], tr.beat_mark)
    r2 = r2_score(te.beat_mark, model.predict(te[feats]))
    print(f"\n  R^2 predicting margin-over-mark from the RESULT ALONE")
    print(f"  (finishing position, beaten lengths, field size): {r2:.4f}")
    print(f"\n  -> ~{100 * r2:.0f}% of the quantity is the finishing order")
    print("     re-expressed in pounds. 'Predict the rating' is not an")
    print("     easier problem than 'predict the winner'; it is the same")
    print("     problem with the answer written into the target.")

    # --- what a winner needs ---------------------------------------------
    print("\n[3] WHAT DOES A WINNER ACTUALLY NEED?")
    w = hcap[hcap.win == 1]
    print(f"  winners exceed their mark by: mean {w.beat_mark.mean():+.2f} lb, "
          f"sd {w.beat_mark.std():.2f}")
    print(f"  10th pct {w.beat_mark.quantile(.1):+.1f}   "
          f"90th pct {w.beat_mark.quantile(.9):+.1f}")
    print(f"  all runners: mean {hcap.beat_mark.mean():+.2f} lb")
    print("\n  -> handicappers set marks so that most horses run below them.")
    print("     Read as a design fact about the handicapping system, this")
    print("     is interesting. Read as a prediction target, it is circular.")

    # --- the honest version ----------------------------------------------
    print("\n[4] THE HONEST, PRE-RACE VERSION")
    print("  Ratings only become predictors once lagged. Doing that:")
    for name, sub in [("handicaps", hcap), ("non-handicaps", other)]:
        _, a = top_ranked_wins(sub, "rpr_lto")
        _, b = top_ranked_wins(sub, "or_n")
        _, c = top_ranked_wins(sub, "sp_dec", ascending=True)
        print(f"\n  {name}:")
        print(f"    highest last-time-out RPR wins   {a:5.2f}%")
        print(f"    highest official rating wins     {b:5.2f}%")
        print(f"    market favourite wins            {c:5.2f}%   <-- best")
    print("\n  -> the best-rated horse on prior form loses to the market")
    print("     favourite in both race types, by a wide margin. Published")
    print("     ratings carry no information the price has not absorbed.")


if __name__ == "__main__":
    main()
