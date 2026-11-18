# Changelog

All notable changes to this study are documented here. Dates are git commit dates
and map 1:1 to `git log`.

## [Unreleased]

- Planned: re-derive the aging threshold on a held-out window (currently a constant
  of the setup; making it an experiment removes the "tuned once, never checked" smell)
- Planned: head-to-head against a published cluster-scheduler result under aligned
  metrics (Weng et al., NSDI '22 — see `bibliography.bib`)
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
