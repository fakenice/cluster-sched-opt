# External baselines (published numbers, each re-checked against the source)

Verified 2026-11-18 by downloading the papers themselves (HUJI-hosted PDFs,
Crossref for DOIs, PWA pages for archive availability). Old Feitelson-group
PDFs have ligature damage in their text layer (e.g. "backﬁlling" extracts as
"ba klling", "conclusive" as "on lusiv e"); quotes were minimally normalized
character-order-preservingly, and the archived PDFs are kept with this study's
research log. Rows not comparable with this repo are flagged in §5.

## 1. This study's own anchors (README findings, replay windows)

- EASY backfill vs FIFO: mean wait −2%…−29%; wait share −2.7…−20.8 pp;
  throughput 0…+2.8%; reclaimable idle −2%…−55%.
- Size-priority vs FIFO: mean wait −59%…−86%; captures 77%–114% of the
  rolling-window offline optimum across 72 instances (median 92%).
- Aging (1 h cap): capture stays 103% but deadline misses rise to 46%.
- Deadline-contract violations: 60.6–96.0% of jobs under FIFO; 1.6–30.0%
  under the offline optimum.

Provenance note that constrains every comparison below: this study replays
the **Alibaba PAI GPU trace** (Weng et al., NSDI '22 — see `bibliography.bib`),
in closed fixed-window mode with an observed-capacity rule. The published
backfill literature is mostly IBM SP2/BLUE-era logs in open-system simulation.
Same problem family, different machines, metrics and simulation frame — so
external numbers calibrate **order of magnitude and direction only**.

## 2. Backfill benefit measured in the published literature (SP2 era)

- **Mu'alem & Feitelson, "Utilization, Predictability, Workloads, and User
  Runtime Estimates in Scheduling the IBM SP2 with Backfilling", IEEE TPDS
  12(6), June 2001** (paper PDF verified 2026-11-18): EASY vs conservative
  per-month tables show differences in **both directions** — CTC response-time
  deltas of +0.6%…+7.8% (EASY worse), KTH October −8.5% wait-side and −25.0%
  bounded-slowdown (EASY better), and bounded-slowdown deltas up to +80.0%
  (EASY worse). Verbatim: "the simulation results are somewhat inconclusive,
  and depend on the workload and metric being used" and "the SP2 workloads
  favor the EASY backfilling over conservative backfilling. The only case in
  which conservative is a possible contender is when using the bounded
  slowdown metric and the KTH trace." (Bounded slowdown defined with a 10 s
  runtime floor.)
- **Talby & Feitelson, IPPS/SPDP 1999, LNCS 1666** (PDF verified): priority
  scheduling with slack factors cuts mean wait "by about 15% relative to EASY
  in an equal priorities scenario"; Table 1: "an average reduction of 16.5% in
  wait time over conservative scheduling (which has almost identical
  performance compared to that of the EASY scheduler)"; with slack factor 9
  the average improves to 19.25% — with an explicit starvation warning that
  grows with the slack factor. That warning is the published counterpart of
  this repo's aging trade-off (103% capture but deadline misses ↑).
- **Shmueli & Feitelson, JSSPP 2003** (HUJI TR version verified): lookahead
  backfill gains are strongly load-dependent (~7% at low load, >20% at 0.90
  load; bounded slowdown 30→8, "an improvement of more that 73%") — consistent
  with this repo's small backfill gains on non-saturated GPU windows.
- **Tsafrir, Etsion & Feitelson, IEEE TPDS 2007** (PDF verified):
  system-generated-estimate backfill reaches "up to 28%" (KTH slowdown),
  "typical improvements of 10–20%, and up to 42%" (BLUE bounded slowdown),
  "a 33% peak improvement in average wait (CTC and BLUE)". Also citable as
  background: "it has been estimated that 90-95% of the installations do not
  change this default configuration" (plain EASY in production).
- **EASY naming/origin**: Lifka, JSSPP'95 (LNCS 949, doi:10.1007/3-540-60153-8_35,
  in `bibliography.bib`) — title-level only; its body numbers could not be
  retrieved from any accessible route (see §6).

## 3. Methodological cautions published by the same community

These are why §5's rules are not merely this repo's house style:

- **Feitelson, IEEE TPDS 16(2):175–182, 2005** ("root causes" case study, PDF
  verified): "The complexity of modern computer systems may enable minor
  variations in performance evaluation procedures to actually determine the
  outcome" — and, verbatim on one such variation, "Surprisingly, for CTC
  conservative backfilling actually achieves somewhat higher backfilling
  levels!"
- **Shmueli & Feitelson, IEEE TPDS (user-aware design paper, PDF verified)**:
  metrics can move in opposite directions on the same data — "the average job
  response-time under CREASY is 27% higher compared to EASY, while the average
  slowdown is 66% lower"; "user productivity improves by up to 50% … while
  according to the conventional metrics, performance may actually degrade".
  This repo reports wait share **and** deadline miss **and** capture for the
  same reason.
- **Frachtenberg & Feitelson, JSSPP 2005 ("Pitfalls in Parallel Job Scheduling
  Evaluation", PDF verified)**: "many parallel workloads and machines do not
  even get close to 100% utilization … due to loss of resources to
  fragmentation (pitfall 26)" — the ancestor of this repo's host-level
  fragmentation bill and idle-with-backlog denominators.
- **Tsafrir, JSSPP 2010, LNCS 6253** (page-level note verified): deliberately
  inaccurate-estimate results are methodology artifacts — "researchers must
  stop using multiples of runtimes as estimates, or else their results would
  likely be invalid". Relevant to any future predictive/backfill extension of
  this study.

## 4. The metric with no external counterpart

This study's deadline-contract violation rate (60.6–96.0% under FIFO) has
**no published same-definition baseline** found in the literature above; the
nearest published constructs are bounded-slowdown tails and starvation
warnings (Talby & Feitelson 1999). The number is reported as a
trace-derived property of this workload, not benchmarked against the
literature — that gap is stated in the README rather than papered over.

## 5. Caliber rules for using any number above

1. **Baseline axis**: published deltas are mostly EASY-vs-conservative (two
   schedulers the literature itself calls "almost identical"); this study's
   deltas are vs **FIFO**. Same-looking percentages on different baselines are
   not interchangeable — always name the baseline.
2. **Simulation frame**: the SP2-era papers use open-system replay (future
   arrivals resampled); this study uses closed fixed windows. Backfill-style
   gains are systematically muted here; throughput percentages (0…+2.8% vs
   47→71 jobs/hour in the user-aware paper) are not comparable at all.
3. **Metric identity**: wait (queueing time) vs response (wait+run) vs
   bounded slowdown (runtime-floored at 10 s) vs stretch — the §2 numbers mix
   these; only same-name metrics may be compared, and even then across
   different workload families (rigid SP2 partitions vs elastic GPU groups).
4. **Era/technology**: 1996–2006 CPU MPP logs vs 2017–2018 GPU trace — orders
   of magnitude (tens of percent), direction, and sign-flip frequency are the
   honest read-outs; point-to-point equality is not.
5. **Window statistics**: TPDS'01's per-month tables show sign flips between
   months (85–15,646 jobs per month); this study's 72-instance capture-rate
   spread (77–114%) is the analogous dispersion statement. Do not merge the
   two into one sentence of "typical" numbers.

## 6. Negative results — what could not be verified (2026-11-18)

- **Lifka 1988/1995 body numbers** (backfill percentages, size-priority):
  Semantic Scholar routes returned persistent 429; OpenAlex hit a daily
  budget; DBLP anti-bot; dl.acm.org 403; HUJI pages have no Lifka entry; no
  accessible copy found — **title-level citation only**.
- **LIFT (IEEE Cluster 2014)**, **HypoThermIA (Banik et al.)**, **TALON /
  principal-agent ("what you wanna") scheduling**, **Feitelson's
  response-ratio-for-reservations note**: no accessible full text found after
  the routes recorded in the research log; **no numbers quoted from them**.
- **Anselm / Phoenix on the PWA archive**: absent from the current archive
  page (snapshot kept); the task-brief's "Anselm/BG/Phoenix/SDS2" framing does
  not match this repo's actual data (§1 note) and must not be reused in any
  derived document.
- Franke et al. SC'99 (ASCI Blue-Pacific): known only through a second-hand
  annotation ("Both are much better than FCFS") on Feitelson's commented
  bibliography page — citable as an annotation, not as a percentage.

## 7. Citation register

Citable (full text verified 2026-11-18): Mu'alem & Feitelson, TPDS 12(6)
2001; Talby & Feitelson, IPPS/SPDP 1999 (LNCS 1666); Shmueli & Feitelson,
JSSPP 2003 (HUJI TR CS-04-02 PDF) and TPDS (user-aware); Talby & Feitelson,
IPDPS 2005; Tsafrir, Etsion & Feitelson, TPDS 2007; Feitelson, TPDS 16(2)
2005; Frachtenberg & Feitelson, JSSPP 2005; Tsafrir, JSSPP 2010 (LNCS 6253,
note-level); Weng et al., NSDI '22 (trace provenance, already in
`bibliography.bib`).

Title-level only: Lifka JSSPP'95 doi:10.1007/3-540-60153-8_35; Nissimov &
Feitelson "Probabilistic Backfilling" doi:10.1007/978-3-540-78699-3_6; Lawson
& Smirni doi:10.1007/3-540-36180-4_5 / doi:10.1145/512840.512846; PWA index
paper doi:10.1016/j.jpdc.2014.06.013.

Not citable: any Lifka/LIFT/HypoThermIA/TALON percentage; "Anselm/Phoenix"
data claims; Franke SC'99 percentages.
