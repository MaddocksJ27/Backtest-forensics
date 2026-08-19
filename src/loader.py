"""
Data loading and leak-free feature construction for GB flat racing.

Every derived feature here is built with an explicit shift(1) so that no
value is available before the race it describes has been run. The audit
functions in src/audit.py verify this holds.

Expected inputs (not distributed with this repo):
    flat_bsp.parquet        runner-level results + Betfair prices
    flat_2016_2026_csv.gz   raw source, used here only for race_name
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
PARQUET = DATA / "flat_bsp.parquet"
RAW_CSV = DATA / "flat_2016_2026_csv.gz"

# Turf going scale, firm (0) -> heavy (9.5). All-weather kept on its own
# scale because the two surfaces are not comparable.
TURF_GOING = {
    "Firm": 0.0, "Fast": 0.5, "Good To Firm": 1.5, "Good": 3.0,
    "Good To Yielding": 3.8, "Yielding": 4.5, "Good To Soft": 4.5,
    "Yielding To Soft": 5.5, "Sloppy": 6.0, "Soft": 6.5, "Muddy": 7.0,
    "Very Soft": 7.5, "Soft To Heavy": 8.0, "Heavy": 9.0, "Holding": 9.5,
    "Slow": 5.0, "Frozen": np.nan,
}
AW_GOING = {
    "Standard": 3.0, "Standard To Fast": 2.0, "Fast": 2.0,
    "Standard To Slow": 4.5, "Slow": 5.0,
}

HANDICAP_RE = r"handicap|h'cap|nursery"


def load_raw(parquet=PARQUET):
    """Load the runner-level results file."""
    df = pd.read_parquet(parquet)
    df["horse"] = df["horse"].astype(str)
    return df


def add_handicap_flag(df, csv=RAW_CSV):
    """
    Flag handicaps from the race name.

    Deriving this from weight structure does not work: jockey claims,
    sellers and nurseries break the weight-equals-mark relationship.
    The presence of an official rating is also unusable as a proxy --
    Racing Post publishes an OR for pattern-race runners too, which
    misclassifies ~66% of Group/Listed races as handicaps.
    """
    names = pd.read_csv(csv, usecols=["race_id", "horse", "race_name"], low_memory=False)
    names["horse"] = names["horse"].astype(str)
    names = names.drop_duplicates(["race_id", "horse"])

    out = df.merge(names, on=["race_id", "horse"], how="left", validate="m:1")
    rn = out["race_name"].fillna("").str.lower()
    out["is_hcap"] = rn.str.contains(HANDICAP_RE, regex=True)
    return out


def add_going_scale(df):
    """Map going descriptions onto a numeric scale, per surface."""
    out = df.copy()
    out["surface"] = np.where(
        out["course"].str.contains(r"\(AW\)", na=False), "aw", "turf"
    )
    out["going_num"] = np.where(
        out["surface"] == "aw",
        out["going"].map(AW_GOING),
        out["going"].map(TURF_GOING),
    )
    return out


def _shift_by_horse(df, col, periods=1):
    return df.groupby("horse", sort=False)[col].shift(periods)


def add_lagged_features(df):
    """
    Prior-run features. Sorted by horse then date so shift(1) means
    'the run immediately before this one'.

    NOTE: rpr and pace_raw are POST-race quantities (see analysis/01).
    Only their lagged forms may be used as predictors.
    """
    out = df.sort_values(["horse", "date", "race_id"]).copy()

    out["rpr_lto"] = _shift_by_horse(out, "rpr_n")
    out["or_lto"] = _shift_by_horse(out, "or_n")
    out["dist_lto"] = _shift_by_horse(out, "dist_f")
    out["going_lto"] = _shift_by_horse(out, "going")
    out["going_num_lto"] = _shift_by_horse(out, "going_num")
    out["surface_lto"] = _shift_by_horse(out, "surface")
    out["days_since"] = (out["date"] - _shift_by_horse(out, "date")).dt.days

    prior = out.groupby("horse", sort=False)["rpr_n"].shift(1)
    grp = prior.groupby(out["horse"], sort=False)
    out["rpr_mean3"] = grp.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    out["rpr_sd3"] = grp.rolling(3, min_periods=2).std().reset_index(level=0, drop=True)
    out["rpr_best"] = grp.expanding().max().reset_index(level=0, drop=True)
    out["career_runs"] = out.groupby("horse", sort=False).cumcount()

    out["or_change"] = out["or_n"] - out["or_lto"]
    out["trip_change"] = out["dist_f"] - out["dist_lto"]
    out["same_surface"] = out["surface"] == out["surface_lto"]
    return out


def build(parquet=PARQUET, csv=RAW_CSV):
    """Full pipeline: load, flag handicaps, add going scale and lags."""
    df = load_raw(parquet)
    df = add_handicap_flag(df, csv)
    df = add_going_scale(df)
    df = add_lagged_features(df)
    return df


# ---------------------------------------------------------------- metrics

def devig_check(df):
    """dv should be the overround-free SP probability, summing to 1 per race."""
    x = df.dropna(subset=["sp_dec", "dv"]).copy()
    x["inv"] = 1 / x["sp_dec"]
    x["recon"] = x["inv"] / x.groupby("race_id")["inv"].transform("sum")
    return (x["dv"] - x["recon"]).abs().max()


def actual_over_expected(df, prob_col="dv"):
    """
    A/E: actual winners divided by the sum of market-implied probabilities.
    1.0 means the market priced this group exactly right.
    """
    x = df.dropna(subset=[prob_col])
    expected = x[prob_col].sum()
    return np.nan if expected == 0 else x["win"].sum() / expected


def roi_at(df, price_col, commission=0.02):
    """Return on turnover for level stakes at a given price column."""
    x = df.dropna(subset=[price_col])
    ret = np.where(x["win"] == 1, (x[price_col] - 1) * (1 - commission), -1.0)
    return ret.mean(), ret.std(ddof=1) / np.sqrt(len(x))


def summarise(df, label, price_col="BSP"):
    """One-line summary used across the analysis scripts."""
    n = len(df)
    if n == 0:
        return f"{label:<34} n=0"
    ae = actual_over_expected(df)
    roi, se = roi_at(df, price_col)
    return (f"{label:<34} n={n:>6d}  win={100 * df['win'].mean():5.2f}%  "
            f"A/E={ae:.3f}  ROI={100 * roi:+6.2f}% (SE {100 * se:.2f})")
