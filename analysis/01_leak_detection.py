"""
01 -- Leak detection

Four kinds of lookahead show up in racing data. Each produces a result
that looks like an edge and is not one.

  A. Post-race columns used as predictors
  B. Rolling features whose window includes the current row
  C. Filtering on a price that is not known when the bet is placed
  D. Filtering on a statistic that is not determined until after the
     betting window closes

Run:  python analysis/01_leak_detection.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import loader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def leak_a(df):
    print("\n" + "=" * 68)
    print("A. POST-RACE COLUMNS  --  the most common and most costly")
    print("=" * 68)

    print("\nRacing Post Rating (rpr) by finishing position:")
    x = df.dropna(subset=["rpr_n", "pos_n"])
    g = x[x.pos_n <= 8].groupby("pos_n")["rpr_n"].agg(["mean", "size"])
    print(g.round(2).to_string())
    print(f"\n  corr(rpr, finishing position) = {x.rpr_n.corr(x.pos_n):+.4f}")
    print("  -> rpr is assigned AFTER the race, from the finishing order.")
    print("     It is a description of what happened, not a forecast.")

    print("\nIn-running style flag (pace_raw), same test:")
    s = df.dropna(subset=["dv", "pace_raw"])
    for v in sorted(s.pace_raw.unique()):
        x = s[s.pace_raw == v]
        print(f"  pace_raw={v:<5} n={len(x):>6d}  win={100 * x.win.mean():5.2f}%"
              f"  A/E={x.win.sum() / x.dv.sum():.4f}")
    print("  -> conditioning on today's running style gives A/E ~2.0.")
    print("     That is not an edge; it is the race result in disguise.")


def leak_b(df):
    print("\n" + "=" * 68)
    print("B. ROLLING-WINDOW LEAKAGE  --  verify by reconstruction")
    print("=" * 68)

    d = df.sort_values(["horse", "date", "race_id"]).copy()
    prior = d.groupby("horse", sort=False)["pace_raw"].shift(1)
    recon = (prior.groupby(d.horse, sort=False)
             .rolling(3, min_periods=1).mean()
             .reset_index(level=0, drop=True))
    match = np.isclose(d.pace3.astype(float), recon.astype(float), equal_nan=True)

    print(f"\n  rows reconstructed exactly: {match.sum():,} / {len(d):,} "
          f"({100 * match.mean():.4f}%)")
    first = d.groupby("horse", sort=False).cumcount() == 0
    print(f"  first-run rows holding a value (must be 0): "
          f"{d.loc[first, 'pace3'].notna().sum()}")

    sub = d[d.pace3.notna()]
    print(f"  corr(feature, TODAY's value)  {sub.pace3.corr(sub.pace_raw):+.4f}")
    print(f"  corr(feature, PRIOR value)    "
          f"{sub.pace3.corr(prior[sub.index]):+.4f}")
    print("  -> a leaking rolling feature correlates more with today than")
    print("     with the prior run. This one does not: it is clean.")


def leak_c(df):
    print("\n" + "=" * 68)
    print("C. FILTERING ON AN UNKNOWABLE PRICE")
    print("=" * 68)

    h = df[df.is_hcap].dropna(subset=["rpr_lto", "or_n", "dv", "BSP",
                                      "MORNINGWAP", "sp_dec", "days_since"]).copy()
    h["signal"] = h.rpr_lto - h.or_n
    sel = h[(h.signal >= 7) & (h.days_since <= 90)]

    print("\nSame selection, three price filters:")
    print("  " + loader.summarise(sel[sel.sp_dec <= 11],
                                  "filter on SP <= 11  [LEAK]", "MORNINGWAP"))
    print("  " + loader.summarise(sel[sel.MORNINGWAP <= 11],
                                  "filter on morning <= 11", "MORNINGWAP"))
    print("  " + loader.summarise(sel, "no price filter", "MORNINGWAP"))
    print("\n  -> starting price is not known when a morning bet is struck.")
    print("     Filtering on it silently drops horses that drifted, and")
    print("     drifters lose. The inflation is worth several ROI points.")


def leak_d(df):
    print("\n" + "=" * 68)
    print("D. FILTERING ON A NOT-YET-DETERMINED STATISTIC")
    print("=" * 68)
    print("\n  MORNINGWAP is a volume-weighted average of all bets matched")
    print("  up to roughly 11am. Two consequences:")
    print("    1. it is not a price anyone can take -- you cannot bet at")
    print("       an average;")
    print("    2. it is not determined until the window closes, so it")
    print("       cannot be used as a pre-bet filter either.")

    x = df.dropna(subset=["MORNINGWAP", "BSP", "sp_dec"])
    print(f"\n  median MORNINGWAP / BSP     {(x.MORNINGWAP / x.BSP).median():.4f}")
    print(f"  median MORNINGWAP / SP      {(x.MORNINGWAP / x.sp_dec).median():.4f}")
    print("  -> across all runners the morning average is well short of the")
    print("     eventual BSP, because most runners drift. It is a fair")
    print("     benchmark for early sentiment and a poor execution")
    print("     assumption -- the two are not the same thing.")

    print("\n  Related: a Betfair SP limit on a BACK bet is a MINIMUM price.")
    print("  There is no maximum (that exists only for lays). Any rule of")
    print("  the form 'skip if the price is too big' is unexecutable.")


def main():
    df = pd.read_parquet(ROOT / "data/built.parquet")
    print("=" * 68)
    print("LEAK DETECTION")
    print("=" * 68)
    leak_a(df)
    leak_b(df)
    leak_c(df)
    leak_d(df)
    print("\n" + "=" * 68)
    print("RULE OF THUMB: if a result looks too good, assume leakage and")
    print("go looking for it before trying to explain the result.")
    print("=" * 68)


if __name__ == "__main__":
    main()
