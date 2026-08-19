"""
05 -- A ratings regime shift, 2022-2024

Any rule with a fixed numeric threshold assumes the quantity it thresholds
is stationary. This script shows a case where that assumption failed: the
relationship between Racing Post Ratings and official handicap marks
converged sharply from 2022, then partially reverted.

Nothing about the underlying racing changed. The number of runners clearing
any fixed rating-versus-mark threshold roughly quadrupled anyway.

Run:  python analysis/05_regime_shift.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 7   # illustrative only


def main():
    df = pd.read_parquet(ROOT / "data/built.parquet")
    h = df[df.is_hcap].copy()
    h["year"] = h.date.dt.year
    h["signal"] = h.rpr_lto - h.or_n     # last-time-out rating vs today's mark

    print("=" * 68)
    print("RATINGS REGIME SHIFT")
    print("=" * 68)

    print(f"\n[1] RUNNERS CLEARING A FIXED THRESHOLD (signal >= {THRESHOLD})")
    print(f"{'year':>6} {'hcap runs':>11} {'qualifiers':>11} {'% of runs':>11}")
    rates = {}
    for y in sorted(h.year.unique()):
        x = h[h.year == y]
        g = x.dropna(subset=["signal"])
        rate = 100 * (g.signal >= THRESHOLD).mean()
        rates[y] = rate
        print(f"{y:>6} {len(x):>11,} {int((g.signal >= THRESHOLD).sum()):>11,} "
              f"{rate:>10.2f}%")
    print("\n  (2020 was covid-shortened; the final year is partial)")

    baseline = np.mean([rates[y] for y in [2016, 2017, 2018, 2019]])
    peak = np.mean([rates[y] for y in [2023, 2024]])
    print(f"\n  2016-19 baseline  {baseline:.2f}%")
    print(f"  2023-24 peak      {peak:.2f}%   ->  {peak / baseline:.2f}x")
    if 2025 in rates:
        print(f"  2025              {rates[2025]:.2f}%   ->  "
              f"{rates[2025] / baseline:.2f}x baseline "
              f"(partial reversion, not a return)")

    print("\n[2] WHICH SIDE MOVED?")
    print(f"{'year':>6} {'mean RPR':>10} {'mean OR':>10} {'RPR - OR':>10}")
    for y in sorted(h.year.unique()):
        x = h[h.year == y]
        print(f"{y:>6} {x.rpr_n.mean():>10.2f} {x.or_n.mean():>10.2f} "
              f"{(x.rpr_n - x.or_n).mean():>10.2f}")

    base = h[h.year.isin([2016, 2017, 2018, 2019])]
    print("\n  Change versus the 2016-19 baseline:")
    print(f"  {'year':>6} {'d RPR':>9} {'d OR':>9} {'d (RPR-OR)':>12}")
    for y in [2022, 2023, 2024, 2025]:
        if y not in rates:
            continue
        x = h[h.year == y]
        print(f"  {y:>6} {x.rpr_n.mean() - base.rpr_n.mean():>+9.2f} "
              f"{x.or_n.mean() - base.or_n.mean():>+9.2f} "
              f"{(x.rpr_n - x.or_n).mean() - (base.rpr_n - base.or_n).mean():>+12.2f}")
    print("\n  -> in the peak years official marks fell by more than ratings")
    print("     rose. Describing this as 'a ratings change' points at the")
    print("     wrong side of the relationship: it is a convergence.")

    print("\n[3] IS IT COMPOSITIONAL?")
    print(f"{'year':>6} {'% AW':>8} {'% Cl6':>8} {'med field':>11} {'med days':>10}")
    h["is_aw"] = h.surface == "aw"
    h["cls"] = h["class"].str.extract(r"(\d)").astype(float)
    for y in [2018, 2019, 2021, 2022, 2023, 2024, 2025]:
        if y not in rates:
            continue
        x = h[h.year == y]
        print(f"{y:>6} {100 * x.is_aw.mean():>7.1f}% {100 * (x.cls == 6).mean():>7.1f}% "
              f"{x.ran_n.median():>11.0f} {x.days_since.median():>10.0f}")

    print("\n  Rating-versus-mark gap WITHIN a fixed class:")
    print(f"  {'year':>6} {'Class 4':>9} {'Class 5':>9} {'Class 6':>9}")
    for y in [2018, 2019, 2021, 2022, 2023, 2024, 2025]:
        if y not in rates:
            continue
        x = h[h.year == y]
        vals = [(x[x.cls == c].rpr_n - x[x.cls == c].or_n).mean() for c in (4, 5, 6)]
        print(f"  {y:>6} {vals[0]:>9.2f} {vals[1]:>9.2f} {vals[2]:>9.2f}")
    print("\n  -> race mix, field size and turnaround times are stable, and")
    print("     the shift appears inside every class. It is not a")
    print("     composition effect.")

    print("\n[4] WHY THIS MATTERS")
    print("""
  Two lessons, neither specific to racing:

  1. A fixed threshold on a non-stationary quantity silently changes
     what it selects. Here the population clearing it quadrupled while
     the threshold, and the underlying sport, stayed the same.

  2. Volume and edge move inversely. When a shift lets more of the
     population clear a bar, the marginal member is weaker, so any
     edge measured on the selected group is diluted. Tracking the
     selection RATE over time is therefore a cheap and effective
     early warning that a measured effect is drifting.

  The general practice: monitor the distribution your rule cuts, not
  just the performance of what it selects.
""")


if __name__ == "__main__":
    main()
