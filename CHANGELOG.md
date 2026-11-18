# Changelog

All notable changes to this study are documented here. Dates are git commit dates
and map 1:1 to `git log`.

## [Unreleased]

- Planned: re-derive the aging threshold on a held-out window (currently a constant
  of the setup; making it an experiment removes the "tuned once, never checked" smell)
- Planned: head-to-head against a published cluster-scheduler result under aligned
  metrics (Weng et al., NSDI '22 — see `bibliography.bib`)

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
