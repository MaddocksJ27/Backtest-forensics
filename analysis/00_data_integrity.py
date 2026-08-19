"""
00 -- Data integrity

Before any modelling, a betting dataset must pass a small number of
checks. The most important is that the market prices, de-vigged, produce
an A/E of 1.0 across the whole file. If they do not, either the prices,
the results, or the de-vigging is wrong, and every downstream number
will inherit the error.

Run:  python analysis/00_data_integrity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import loader  # noqa: E402


def main():
    df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data/built.parquet")

    print("=" * 68)
    print("DATA INTEGRITY")
    print("=" * 68)
    print(f"runners {len(df):,}   races {df.race_id.nunique():,}   "
          f"horses {df.horse.nunique():,}")
    print(f"dates   {df.date.min().date()} to {df.date.max().date()}")

    # --- 1. the headline check -------------------------------------------
    print("\n[1] OVERALL A/E  (must be ~1.0)")
    sub = df.dropna(subset=["dv"])
    ae_raw = sub.win.sum() / sub.dv.sum()

    # Dead heats: two horses can both be flagged as winners. Share the win.
    winners_per_race = sub.groupby("race_id")["win"].transform("sum")
    win_frac = np.where(winners_per_race > 0,
                        sub.win / winners_per_race.clip(lower=1), 0)
    ae_adj = win_frac.sum() / sub.dv.sum()

    print(f"  raw A/E                {ae_raw:.4f}")
    print(f"  dead-heat adjusted     {ae_adj:.4f}   <-- the number that matters")
    print(f"  excess winners         {int(sub.win.sum() - sub.race_id.nunique())}"
          f" (dead-heated places)")

    # --- 2. overround -----------------------------------------------------
    # Overround MUST be computed on complete books. Summing 1/price over a
    # filtered subset of runners omits part of the book and understates the
    # margin -- an easy and badly misleading mistake.
    print("\n[2] OVERROUND BY PRICE SOURCE  (complete books only)")
    for col, name in [("sp_dec", "bookmaker SP"), ("BSP", "Betfair SP"),
                      ("MORNINGWAP", "Betfair morning WAP")]:
        complete = df.groupby("race_id")[col].apply(lambda s: s.notna().all())
        full = df[df.race_id.isin(complete[complete].index)]
        orr = full.assign(inv=1 / full[col]).groupby("race_id")["inv"].sum()
        print(f"  {name:<22} mean {orr.mean():.4f}   median {orr.median():.4f}"
              f"   ({len(orr):,} races)")

    sub = df[df.is_hcap].dropna(subset=["rpr_lto", "or_n", "MORNINGWAP", "sp_dec"])
    bad = sub.assign(inv=1 / sub.MORNINGWAP).groupby("race_id")["inv"].sum()
    print(f"\n  same figure on a FILTERED subset:  {bad.mean():.4f}  <-- wrong")
    print("  -> dropping runners from a race silently removes part of the")
    print("     book. Always compute overround on complete fields.")
    print("  -> bookmaker SP carries ~16%; Betfair SP is margin-free;")
    print("     the morning WAP sits in between (thin early markets).")

    # --- 3. de-vigging ----------------------------------------------------
    print("\n[3] dv RECONSTRUCTION")
    print(f"  max |dv - recomputed|  {loader.devig_check(df):.2e}")
    s = df.dropna(subset=["dv"]).groupby("race_id")["dv"].sum()
    print(f"  races summing to 1.0   {100 * (abs(s - 1) < 1e-6).mean():.2f}%")

    # --- 4. structural consistency ---------------------------------------
    print("\n[4] STRUCTURAL CHECKS")
    print(f"  duplicate (race,horse) {df.duplicated(['race_id', 'horse']).sum()}")
    print(f"  win=1 but pos!=1       {((df.win == 1) & (df.pos_n != 1)).sum()}")
    print(f"  win=0 but pos==1       {((df.win == 0) & (df.pos_n == 1)).sum()}")
    actual = df.groupby("race_id").size()
    declared = df.groupby("race_id")["ran_n"].first()
    print(f"  field size matches     {100 * (actual == declared).mean():.2f}%")

    # --- 5. missingness ---------------------------------------------------
    print("\n[5] PRICE MISSINGNESS  (is it random?)")
    for col in ["sp_dec", "BSP", "MORNINGWAP"]:
        miss = df[col].isna()
        if miss.sum() == 0:
            print(f"  {col:<12} none missing")
            continue
        print(f"  {col:<12} {100 * miss.mean():5.2f}% missing | "
              f"win rate missing {df.loc[miss, 'win'].mean():.4f} "
              f"vs present {df.loc[~miss, 'win'].mean():.4f}")
    print("  -> a large gap here would mean prices are missing "
          "non-randomly, biasing any backtest that drops them.")

    # --- 6. favourite-longshot bias --------------------------------------
    print("\n[6] FAVOURITE-LONGSHOT BIAS  (a validity check, not a finding)")
    x = df.dropna(subset=["dv", "sp_dec"]).copy()
    x["band"] = pd.cut(x.sp_dec, [0, 2, 3, 5, 9, 17, 34, 1e9],
                       labels=["<2", "2-3", "3-5", "5-9", "9-17", "17-34", "34+"])
    g = x.groupby("band", observed=True).apply(
        lambda d: pd.Series({"n": len(d), "win%": 100 * d.win.mean(),
                             "A/E": d.win.sum() / d.dv.sum()}),
        include_groups=False)
    print(g.round(3).to_string())
    print("  -> A/E falling monotonically with price is the textbook pattern.")
    print("     Its presence is evidence the prices and results line up.")

    # --- 7. identity ------------------------------------------------------
    print("\n[7] IS HORSE NAME A SAFE KEY?")
    d = df.copy()
    d["foaled"] = d.date.dt.year - d.age
    spread = d.groupby("horse")["foaled"].agg(["min", "max", "size"])
    bad = spread[(spread["max"] - spread["min"]) > 1]
    print(f"  names with >1yr implied foaling spread: {len(bad)} "
          f"({100 * len(bad) / len(spread):.2f}% of names)")
    print(f"  runs affected: {d.horse.isin(bad.index).sum()} "
          f"({100 * d.horse.isin(bad.index).mean():.2f}% of runs)")
    print("  -> name collisions are rare enough to ignore, but should be")
    print("     measured rather than assumed.")


if __name__ == "__main__":
    main()
