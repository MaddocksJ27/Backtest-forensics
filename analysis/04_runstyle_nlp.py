"""
04 -- Extracting running style from free-text race comments

Race comments are short, formulaic and written to a house style, which
makes them tractable without heavy NLP. The catch is that they describe
the whole race, so matching keywords anywhere in the string produces a
label that is only loosely about how the horse was ridden.

This script compares a loose whole-string match against a strict
first-positional-clause parse, and measures how much of today's running
style is predictable in advance at all.

Run:  python analysis/04_runstyle_nlp.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "2023-01-01"

LED = re.compile(
    r"^(led|made all|made virtually all|made every yard|soon led|led early|"
    r"led narrowly|disputed lead|set (?:a )?(?:strong |good |steady |slow )?"
    r"(?:pace|gallop))")
PROMINENT = re.compile(
    r"^(prominent|chased lead|pressed lead|pressed leader|close up|handy|"
    r"tracked lead|raced in second|with leader|in touch with lead)")
MIDFIELD = re.compile(
    r"^(midfield|mid-division|in touch|chased leaders|tracked leaders|"
    r"behind leaders|raced (?:in )?mid)")
HELD_UP = re.compile(
    r"^(held up|towards rear|in rear|slowly away|slowly into stride|dwelt|"
    r"steadied start|always (?:towards )?rear|last|missed break|"
    r"slow to start|reluctant to race|outpaced|started slowly)")


def classify(clause):
    if LED.match(clause):
        return 1.0
    if PROMINENT.match(clause):
        return 0.72
    if MIDFIELD.match(clause):
        return 0.40
    if HELD_UP.match(clause):
        return 0.10
    return np.nan


def strict_label(comments, max_clauses=3):
    """
    Take the first clause that actually describes a position.

    Openers like 'took keen hold' or 'raced wide' describe behaviour, not
    position, and push the positional description into a later clause.
    Scanning the first few clauses lifts coverage substantially.
    """
    parts = comments.fillna("").str.lower().str.split(" - ")
    out = pd.Series(np.nan, index=comments.index)
    for i in range(max_clauses):
        clause = parts.str[i].fillna("").str.strip()
        todo = out.isna()
        out.loc[todo] = clause[todo].map(classify)
    return out


def main():
    df = pd.read_parquet(ROOT / "data/built.parquet")
    print("=" * 68)
    print("RUNNING STYLE FROM RACE COMMENTS")
    print("=" * 68)

    d = df.copy()
    d["first_clause"] = d.comment.fillna("").str.split(" - ").str[0].str.strip().str.lower()
    d["strict"] = strict_label(d.comment)
    d["strict_first_only"] = d.first_clause.map(classify)

    # --- 1. how good is a loose match? -----------------------------------
    print("\n[1] LOOSE MATCHING IS NOT WHAT IT LOOKS LIKE")
    loose = d[d.pace_raw == 1.0]
    genuine = loose.first_clause.str.match(LED)
    print(f"\n  rows the shipped loose label calls 'led': {len(loose):,}")
    print(f"  of those, first clause genuinely leads:   "
          f"{genuine.sum():,} ({100 * genuine.mean():.1f}%)")

    print("\n  The same opening clause receives contradictory labels:")
    for phrase in ["took keen hold", "slowly into stride", "midfield", "held up"]:
        s = d[d.first_clause == phrase]
        if len(s) == 0:
            continue
        counts = s.pace_raw.value_counts().sort_index().to_dict()
        print(f"    {phrase:<20} n={len(s):>6,}  {counts}")
    print("\n  -> horses explicitly described as slow away are labelled")
    print("     'led' thousands of times. The label is matching keywords")
    print("     anywhere in a comment describing the whole race.")

    # --- 2. coverage ------------------------------------------------------
    print("\n[2] COVERAGE: FIRST CLAUSE vs FIRST POSITIONAL CLAUSE")
    print(f"  first clause only          "
          f"{100 * d.strict_first_only.notna().mean():5.1f}% of runners")
    print(f"  first positional clause    "
          f"{100 * d.strict.notna().mean():5.1f}% of runners")
    print("\n  label distribution (strict):")
    print(d.strict.value_counts(normalize=True).sort_index().round(4).to_string())

    # --- 3. predictability ------------------------------------------------
    print("\n[3] HOW PREDICTABLE IS RUNNING STYLE?")
    d = d.sort_values(["horse", "date", "race_id"])
    prior = d.groupby("horse", sort=False)["strict"].shift(1)
    grp = prior.groupby(d.horse, sort=False)
    d["style_lto"] = prior
    d["style_mean3"] = grp.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    d["style_career"] = grp.expanding().mean().reset_index(level=0, drop=True)
    d["style_runs"] = grp.expanding().count().reset_index(level=0, drop=True)

    first = d.groupby("horse", sort=False).cumcount() == 0
    print(f"\n  audit: first-run rows holding a lagged value (must be 0): "
          f"{d.loc[first, 'style_lto'].notna().sum()}")

    s = d.dropna(subset=["strict", "style_lto"])
    print("\n  correlation with today's style:")
    for col, name in [("pace3", "shipped loose rolling mean"),
                      ("style_lto", "strict, last run"),
                      ("style_mean3", "strict, mean of last 3"),
                      ("style_career", "strict, career mean")]:
        ss = s.dropna(subset=[col])
        print(f"    {name:<28} {ss[col].corr(ss.strict):+.4f}")

    print("\n  Out-of-sample R^2 predicting today's style:")
    m = s.dropna(subset=["style_mean3"]).copy()
    m["class_c"] = m["class"].str.extract(r"(\d)").astype(float)
    m["course_c"] = m.course.astype("category").cat.codes
    tr, te = m[m.date < SPLIT], m[m.date >= SPLIT]

    sets = {
        "loose feature only": ["pace3"],
        "strict form only": ["style_lto", "style_mean3", "style_career", "style_runs"],
        "+ race context": ["style_lto", "style_mean3", "style_career", "style_runs",
                           "draw_pct", "dist_f", "ran_n", "class_c", "course_c",
                           "going_num", "age", "trip_change", "days_since"],
    }
    for name, feats in sets.items():
        model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.07, random_state=0, early_stopping=False)
        model.fit(tr[feats], tr.strict)
        pred = model.predict(te[feats])
        print(f"    {name:<24} R^2 = {r2_score(te.strict, pred):.4f}   "
              f"corr = {np.corrcoef(pred, te.strict)[0, 1]:.4f}")

    print("\n  -> most of the gain comes from fixing the label, not from")
    print("     clever features. But the ceiling is low: running style is")
    print("     roughly 80% unpredictable. Any downstream feature built on")
    print("     it must be treated as a probability, never a hard label.")

    # --- 4. what drives it ------------------------------------------------
    print("\n[4] WHAT ACTUALLY DRIVES RUNNING STYLE")
    s = s.copy()
    s["led"] = (s.strict == 1.0).astype(int)
    print("\n  P(leads today) by field size:")
    s["fs"] = pd.cut(s.ran_n, [0, 6, 9, 12, 16, 40],
                     labels=["<=6", "7-9", "10-12", "13-16", "17+"])
    print(s.groupby("fs", observed=True)["led"].agg(["mean", "size"]).round(4).to_string())
    print("\n  P(leads today) by prior style:")
    t = s.dropna(subset=["style_mean3"]).copy()
    t["pb"] = pd.cut(t.style_mean3, [-.01, .2, .4, .6, .8, 1.01])
    print(t.groupby("pb", observed=True)["led"].agg(["mean", "size"]).round(4).to_string())
    print("\n  -> field size dominates, which is largely mechanical: there")
    print("     is only one lead to take. Prior style is the strongest")
    print("     horse-level signal.")


if __name__ == "__main__":
    main()
