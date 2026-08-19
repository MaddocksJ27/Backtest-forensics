# Auditing a betting backtest

Five self-contained studies on 583,774 GB flat racing runners (2016–2026),
built around one question: **how do you tell a real edge from a leak?**

Most published betting analysis reports the winner. This reports the audit —
the checks a result has to survive, and the several that plausible-looking
findings did not.

I run a live strategy derived from this dataset. Its selection rules are not
in this repo. What is here is the methodology that produced it and, more
usefully, the methodology that killed most of what came before it.

---

## The studies

| | Study | Headline |
|---|---|---|
| **00** | [Data integrity](analysis/00_data_integrity.py) | De-vigged market prices give A/E 1.0000 after dead-heat adjustment. If they don't, nothing downstream is trustworthy. |
| **01** | [Leak detection](analysis/01_leak_detection.py) | Four kinds of lookahead, each with a test. Conditioning on a post-race column gives A/E 1.96 — a fabricated edge. |
| **02** | [RPR circularity](analysis/02_rpr_circularity.py) | The top-ranked horse on rating-minus-mark wins 98.4% of handicaps. R² 0.86 predicting that quantity from the finishing position alone. |
| **03** | [Market efficiency](analysis/03_market_efficiency.py) | 15 fundamental features add +0.015 R² over the price. Removing the price costs 0.068. |
| **04** | [Run-style NLP](analysis/04_runstyle_nlp.py) | Loose keyword matching is 33.8% precise. A strict clause parse reaches 92.7% coverage and lifts OOS R² from 0.070 to 0.191. |
| **05** | [Regime shift](analysis/05_regime_shift.py) | Runners clearing a fixed rating threshold quadrupled in 2022–24 with no change in the sport. |

---

## Four results worth the click

**A performance rating is not a measurement of performance.** Racing Post
Ratings correlate −0.88 with finishing position and are assigned after the
race. The natural reframing — predict how far a horse will exceed its
handicap mark, giving a continuous target instead of one bit per race —
turns out to be 86% recoverable from finishing position, beaten lengths and
field size alone. It is the result restated in pounds, not an easier problem.

**The market is very hard to beat, and it is worth quantifying how hard.**
De-vigged starting prices post AUC 0.780 and move log-loss ~15% off the base
rate. The best-rated horse on prior form wins 23.6% of handicaps; the
favourite wins 31.7%. Published ratings carry nothing the price has not
already absorbed.

**Fixed thresholds assume stationarity, and that assumption fails.** The gap
between ratings and official marks converged by ~2.3 lb from 2022 and
partially reverted. Runners clearing any fixed threshold roughly quadrupled.
Class mix, field size and turnaround times were flat throughout, and the
shift appears within every class — so it is not composition. Notably, the
larger movement was in official marks, not the ratings.

**Execution assumptions fail as often as modelling ones.** Two examples
that survived a long way into this project before being caught: a
volume-weighted average price used as though it were takeable (it is the
average of what others were matched at — you cannot bet at an average), and
a maximum-price rule on exchange starting-price bets (the limit is a
*minimum* for backs; no maximum exists). Both produced backtests that could
not have been executed.

---

## Running it

```bash
pip install -r requirements.txt

# place the source files in data/, then:
python src/build_cache.py            # ~40s, writes data/built.parquet
python analysis/00_data_integrity.py
python analysis/03_market_efficiency.py   # a few minutes, fits GBMs
```

Data is not distributed here (Racing Post ratings and Betfair price data are
not mine to redistribute). Expected in `data/`:

- `flat_bsp.parquet` — runner-level results with Betfair SP, morning WAP
- `flat_2016_2026_csv.gz` — source rows; used only for race names, which is
  the only reliable way to flag handicaps

Every derived feature is rebuilt from source in [`src/loader.py`](src/loader.py)
so that no number is inherited from an earlier session.

---

## The audit checklist

Reusable regardless of sport or market:

1. **De-vigged prices must give A/E ≈ 1.0 overall.** Adjust for dead heats.
2. **Compute overround on complete books.** Summing `1/price` over a filtered
   subset omits part of the book and understates the margin.
3. **Reconstruct every rolling feature from scratch and require an exact
   match.** Check that first-run rows are empty, and that the feature
   correlates more with the prior value than with today's.
4. **Ask of every column: was this knowable before the off?** Ratings,
   in-running comments and finishing positions are not.
5. **Ask the same of every filter.** Filtering on starting price is
   lookahead; it silently drops drifters, and drifters lose.
6. **Ask whether the assumed execution price is takeable.** Benchmarks and
   averages are neither takeable nor determined at bet time.
7. **Split by period, and report the out-of-sample number.** Then count how
   many variants you tried, and discount accordingly.
8. **Block-bootstrap by day** where bets cluster, rather than assuming
   independence.
9. **Monitor the selection rate over time,** not just the performance of what
   is selected. A rising rate signals a diluting edge.
10. **Favourite–longshot bias should be visible.** Its absence means the
    prices and results are not aligned.

---

## Caveats

Single sport, single market, one dataset. The 2016–2026 window has been used
for both development and validation across the wider project, so any
performance figure here should be read as an upper bound. Nothing in the
methodology is novel to quantitative finance — leak detection, walk-forward
validation and block bootstrapping are standard practice. The contribution,
such as it is, is applying them to a domain where they are often skipped, and
publishing what they killed.
