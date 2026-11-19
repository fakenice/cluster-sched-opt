# Changelog

All notable changes to this study are documented here. Dates are git commit dates
and map 1:1 to `git log`.

## [Unreleased]

- Planned: head-to-head against a published cluster-scheduler result under aligned
  metrics (Weng et al., NSDI '22 — see `bibliography.bib`)
- 2026-11-19 Delivered the held-out aging-threshold experiment that was previously
  listed here as planned. `scripts/10_aging_threshold.py` replays a 0.5 / 1 / 2 / 3 / 4 /
  6 h + no-aging grid over all 72 sweep instances (576 policy runs) and scores the 6
  committed tuning cells against the 66 holdout cells that no committed decision used;
  the replay's FIFO and priority-gpu rows reproduce `results/sweep_metrics.csv` on 144
  of 144 shared rows (max relative deviation 0.0). The result is a negative one: on
  holdout, mean wait is minimised by *no* aging (1 h carries +9.9 % median regret and
  loses to no-aging in 44 of 66 cells), the tail prefers ~1 h (beats no-aging's max wait
  in 42 of 66 cells, sign test p < 0.001, median max-wait regret 0.0 % vs 14.3 %),
  aggressive aging raises deadline misses (median 0.230 → 0.329 → 0.439 going from no
  aging to 1 h to 0.5 h), and the threshold the tuning set would have picked (3 h) is
  the per-cell holdout argmin in only 5 of 66 cells. README finding 13 records this and
  §6's "chosen by inspection" caveat is replaced by the measurement; finding 9 points
  forward. Added `--reuse-csv` (re-summarises a stored replay, refusing any table whose
  instance or policy coverage does not match the current grid) and a
  `run_all.py --stage aging` entry. New `tests/test_aging_reuse.py` (14 tests) pins the
  reuse guard, the merge that used to crash only after the whole replay had finished,
  the tie-break-to-smaller-threshold rule, and that the committed
  `results/aging_holdout.json` is re-derivable from the committed
  `results/aging_holdout.csv`
- 2026-11-18 Added `docs/external_baselines.md`: published backfill/EASY-vs-conservative
  and aging numbers re-checked against the original PDFs (TPDS'01 bidirectional gap and
  its "somewhat inconclusive" caveat, IPPS'99 ~15 % slack and near-identical defaults,
  JSSPP'03 load dependence ~7 % low load vs >20 % at 0.90, TPDS'07 backfill up
  to 28 % / 42 % / 33 % by metric and system, Shmueli & Feitelson's 27 % higher
  response-time vs 66 % lower slowdown, Feitelson '05 "minor variations", Tsafrir's estimation-noise
  warning), plus the five caliber rules that govern which of them this repo's deltas may be
  quoted against and the negative list (Lifka body text, LIFT, HypoThermIA, TALON and the
  response-ratio study could not be verified; deadline-metric gap stated as a gap). README
  gained section 7 "External baselines", which pushed How-to-cite to §8 and References to §9
  (`docs/DECISIONS.md` cross-reference updated accordingly)
- 2026-11-18 Added `docs/DECISIONS.md`: metric-denominator, solver-honesty and
  protocol decisions logged only where the repo carries written evidence (commit
  dates cited)

## [v1.0] - 2026-11-18

Release freeze: versioned, cited, and DOI-ready.

### Added

- `80b7931` verified bibliography (`bibliography.bib`), trace provenance doc
  (`docs/trace_provenance.md`: sha256, byte counts, schema verbatim, offline
  re-check recipe), README Quickstart + References sections, `CITATION.cff`

## 2026-11-07

- `376fcea` scheduler fixed at the source: no fabricated capacity, and the
  simulation refuses to hide missing capacity

## 2026-11-04

- `bebe7a7` fragmentation modelled as a host-level bill instead of assumed away
- `fe7dea6` ranking tested over 72 windows x seeds instead of the six that were picked

## 2026-11-03

- `3a2862f` request-only priority baselines; optimiser gain retested against them

## 2026-10-13/14

- `25157e7` initial trace-driven GPU scheduling study: MILP vs FIFO/EASY backfill,
  with a feasibility oracle
- `4e25729` public repo URL and one-page conclusion added to the README
