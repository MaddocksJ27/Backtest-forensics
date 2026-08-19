"""Build and cache the feature table. Run once before the analysis scripts."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import loader

def main():
    out = Path(__file__).resolve().parents[1] / "data" / "built.parquet"
    for f in (loader.PARQUET, loader.RAW_CSV):
        if not f.exists():
            raise SystemExit(f"missing input: {f}\nSee README for expected files.")
    t = time.time()
    df = loader.build()
    df.to_parquet(out)
    print(f"built {df.shape[0]:,} rows x {df.shape[1]} cols in {time.time()-t:.0f}s")
    print(f"wrote {out}")
    print(f"handicaps {100*df.is_hcap.mean():.1f}%   "
          f"dv reconstruction error {loader.devig_check(df):.2e}")

if __name__ == "__main__":
    main()
