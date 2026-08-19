"""
03 -- How efficient is the market?

The central question for anyone building a betting model: how much is
there left to find once the price is accounted for?

This script measures the market's own discrimination and calibration,
then asks whether a substantial set of hand-built fundamental features
adds anything on top of it. The answer, on this data, is: very little.

Runtime is a few minutes -- it fits gradient-boosted trees several times.

Run:  python analysis/03_market_efficiency.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import log_loss, r2_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "2023-01-01"   # everything before is development, after is held out


def market_quality(df):
    print("\n" + "=" * 68)
    print("[1] HOW GOOD IS THE MARKET ON ITS OWN?")
    print("=" * 68)
    x = df.dropna(subset=["dv", "win"])
    base = x.win.mean()
    print(f"\n  AUC of de-vigged SP          {roc_auc_score(x.win, x.dv):.4f}")
    print(f"  log-loss, market             "
          f"{log_loss(x.win, x.dv.clip(1e-9, 1 - 1e-9)):.5f}")
    print(f"  log-loss, base rate only     "
          f"{log_loss(x.win, np.full(len(x), base)):.5f}")
    print("\n  -> the market moves log-loss ~15% off the base rate. Racing")
    print("     is mostly irreducible noise; a perfect model of ability")
    print("     would not get dramatically closer.")

    print("\n  Calibration of the de-vigged price (decile bins):")
    x = x.copy()
    x["bin"] = pd.qcut(x.dv, 10, duplicates="drop")
    cal = x.groupby("bin", observed=True).apply(
        lambda d: pd.Series({"n": len(d), "predicted": d.dv.mean(),
                             "actual": d.win.mean()}), include_groups=False)
    cal["actual/pred"] = cal.actual / cal.predicted
    print(cal.round(4).to_string())


def feature_groups():
    return {
        "BASE": ["age", "dist_f", "ran_n", "class_c", "draw_pct",
                 "course_c", "going_num", "surface_c"],
        "MARKET": ["log_dv"],
        "G1_form": ["rpr_lto", "rpr_mean3", "rpr_sd3", "rpr_best",
                    "career_runs"],
        "G2_mark": ["or_n", "or_lto", "or_change"],
        "G3_conditions": ["trip_change", "same_surface_i", "going_delta",
                          "days_since"],
        "G4_field": ["or_rank", "rpr_lto_rank", "field_mean_or"],
    }


def prepare(df):
    h = df[df.is_hcap].dropna(subset=["rpr_n", "or_n", "dv"]).copy()
    h["beat_mark"] = h.rpr_n - h.or_n
    h["log_dv"] = np.log(h.dv.clip(1e-6))
    h["class_c"] = h["class"].str.extract(r"(\d)").astype(float)
    h["course_c"] = h.course.astype("category").cat.codes
    h["surface_c"] = (h.surface == "aw").astype(int)
    h["same_surface_i"] = h.same_surface.astype(float)
    h["going_delta"] = h.going_num - h.going_num_lto
    grp = h.groupby("race_id")
    h["or_rank"] = grp["or_n"].rank(ascending=False, method="min")
    h["rpr_lto_rank"] = grp["rpr_lto"].rank(ascending=False, method="min")
    h["field_mean_or"] = grp["or_n"].transform("mean")
    return h


def fit_r2(tr, te, feats, target="beat_mark"):
    model = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.07, random_state=0, early_stopping=False)
    model.fit(tr[feats], tr[target])
    return r2_score(te[target], model.predict(te[feats]))


def main():
    df = pd.read_parquet(ROOT / "data/built.parquet")
    market_quality(df)

    h = prepare(df)
    tr, te = h[h.date < SPLIT], h[h.date >= SPLIT]
    G = feature_groups()
    print("\n" + "=" * 68)
    print("[2] CAN FUNDAMENTALS ADD TO THE PRICE?")
    print("=" * 68)
    print(f"\n  target: margin over official mark (a continuous stand-in")
    print(f"          for performance -- see analysis/02 for its limits)")
    print(f"  train {len(tr):,} runners before {SPLIT}")
    print(f"  test  {len(te):,} runners after  {SPLIT}")

    print("\n  Building up, out-of-sample R^2:")
    cumulative = G["BASE"] + G["MARKET"]
    r = fit_r2(tr, te, cumulative)
    print(f"    {'base + market price':<26} {len(cumulative):>2d} feats  R^2 = {r:.4f}")
    for key in ["G1_form", "G2_mark", "G3_conditions", "G4_field"]:
        cumulative = cumulative + G[key]
        r = fit_r2(tr, te, cumulative)
        print(f"    {'+ ' + key:<26} {len(cumulative):>2d} feats  R^2 = {r:.4f}")

    full = cumulative
    r_full = fit_r2(tr, te, full)
    print(f"\n  Ablation -- drop one group from the full model:")
    for key in ["G1_form", "G2_mark", "G3_conditions", "G4_field", "MARKET"]:
        feats = [f for f in full if f not in G[key]]
        r = fit_r2(tr, te, feats)
        tag = "  <-- the whole story" if key == "MARKET" else ""
        print(f"    without {key:<16} R^2 = {r:.4f}   "
              f"(change {r - r_full:+.4f}){tag}")

    print("\n  -> every fundamental feature combined moves out-of-sample R^2")
    print("     by a small fraction. Removing the single market-price column")
    print("     costs an order of magnitude more. The price already contains")
    print("     the fundamentals, plus fitness, stable intent and money that")
    print("     no public dataset carries.")

    print("\n" + "=" * 68)
    print("[3] WHAT THIS IMPLIES FOR MODEL DESIGN")
    print("=" * 68)
    print("""
  Modelling P(win) from fundamentals means re-deriving the market from a
  strict subset of its information. It cannot systematically win.

  The tractable question is narrower: where is the market's own price
  measurably wrong? That reframing is what makes A/E -- actual winners
  over market-implied winners -- the right yardstick, rather than
  accuracy, AUC or log-loss.

  It also sets the bar. At bookmaker SP a selection must clear a ~16%
  overround before it profits. On the exchange the bar is the commission
  rate. Which venue you can actually bet into changes what counts as an
  edge more than most modelling choices do.
""")


if __name__ == "__main__":
    main()
