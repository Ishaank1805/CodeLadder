# Report plan

Outlines for the mid-submission report (6 Nov, 4 pages) and the final project report (30 Nov, 8 pages, ACM), with the evidence for each section. Numbers come from [RESULTS.md](RESULTS.md) (summary, hypotheses table and figure map at the top); figures from `python scripts/figures/report_figures.py results`, tables from `python scripts/report_tables.py results` (LaTeX in `results/tables/*.tex`). Positioning against prior work: [RELATED_WORK.md](RELATED_WORK.md). The framing for the final project report (research track or tool track) is still to be decided by the team; both outlines below work for either, the tool framing moving the demo and the policy export forward.

## Mid-submission (6 Nov, 4 pages)

*Status (6 Nov):* Prepared and verified. Full test suite passes (150 tests), report figures and tables generated in `results/`, interactive demo and offline replay ready.

What the proposal promised for this point: the generation pass, the sandboxed executor, the scorer, a first calibrated cascade, and preliminary results for H1 and H2. We are well past that, so the report can state outcomes for all four hypotheses and list the extensions as work in progress. Use the ACM `sigconf` template from the start so the final project report reuses the text.

| Section | Length | Content | Evidence |
| --- | --- | --- | --- |
| 1. Introduction | 0.5 page | Code that passes the visible tests can still be wrong (silent failures: 11 to 13% of accepted programs on HumanEval+/MBPP+ even for 32B). Free models of very different cost. Goal: answer only when the share of accepted-but-wrong answers is provably at most α, at the lowest cost; otherwise abstain. Contributions in 3 to 4 bullets | X17 f = 0 numbers; RELATED_WORK "Defensible contributions" |
| 2. CodeLadder | 1 page | The ladder: per rung draw, run visible tests and probes, score (agreement x self-confidence), accept, resample, repair, escalate or abstain. Learn-then-Test calibration of the thresholds (fixed sequence, Bonferroni paths), families (model selection, sub-ladders, two-level), price of abstaining κ. The guarantee in one sentence and its assumption (exchangeability) | System diagram (to draw, see below); `src/codeladder/controller.py`, `calibrate.py` |
| 3. Setup | 0.5 page | HumanEval+ and MBPP+ (542 tasks), LiveCodeBench release_v6 (1,054); Qwen2.5-Coder 1.5B to 32B and DeepSeek-Coder 1.3B to 33B on the cluster, gpt-oss-120b on a free API; 20 samples per task on HumanEval+/MBPP+ and 10 on LiveCodeBench, repair chains, 100 random calibration/test halves, δ = 10%; metrics: realised risk, share of splits above α, solved share, cost in tokens and energy | PLAN.md sections 1 and 3 |
| 4. Results | 1.5 pages | One paragraph per research question, each opening with the verdict from the hypotheses table: H1 confirmed (validity, published cascades fail), H2 refuted as stated (the combined score wins; P(True) interim), H3 half confirmed (repair only at 32B; learned policy no better), H4 refuted the other way (one large calibration beats sliding windows) | Table: hypotheses and outcomes (short form). Table `main` (HumanEval+/MBPP+ and LiveCodeBench rows only). Figure `fig_validity` and `fig_shift` side by side, or `fig_guaranteed` |
| 5. Status and plan | 0.5 page | Done beyond the proposal (X7, X9, X13, X14, X16, X17, demo); in progress: BigCodeBench, API on LiveCodeBench and BigCodeBench, 32B P(True); plan to the final project report; threats in two lines (benchmark oracles are finite, X17; one hardware setup) | PLAN.md, STATUS.md |

Figure and table budget for 4 pages: one diagram, one or two plots, two tables. Repository link and the commit history are part of the submission.

## Final project report (30 Nov, 8 pages plus references, ACM)

| Section | Content | Evidence |
| --- | --- | --- |
| 1. Introduction | Problem, the certified multi-action ladder, headline findings (guarantee holds; cheap-share law; guaranteed baselines; oracle sensitivity) | summary of findings |
| 2. Background and related work | Learn-then-Test and conformal risk control; cascades and routing (FrugalGPT, Model Cascading for Code, RouteLLM); guaranteed abstention (Task Abstention for code, Trust or Escalate); self-evaluation (P(True), verbalised confidence); repair versus resample | RELATED_WORK.md |
| 3. CodeLadder | Controller, scores (agreement, ratings, P(True), stress inputs for stdin), calibration families, κ, lower-tier budget, policy export | code; X15 |
| 4. Experimental setup | Four benchmarks (BigCodeBench added), seven local models plus the API tier, splits, metrics, energy measurement, compute | PLAN.md; X18 setup section |
| 5. Results | RQ1 validity (`fig_validity`, `fig_baselines`, table `main`); RQ2 signals (`fig_ptrue`, `fig_stdin`, table `signals`); RQ3 actions (`fig_repair`, X4); RQ4 shift (`fig_shift`) and conditional risk (`fig_conditional`); when ladders pay (`fig_economics`, table `economics`, `fig_router`); against guaranteed baselines (`fig_guaranteed`); significance (`fig_ci`, table `ci`); oracle sensitivity (`fig_oracle`, table `oracle`); energy (`fig_energy`) | figure map in RESULTS.md |
| 6. The tool | `export-policy`, `solve`, the interactive demo; how a practitioner certifies a ladder for their own tasks | X15; `codeladder demo` |
| 7. Threats to validity | Exchangeability and shift; oracle strength (X17); benchmark contamination; one hardware setup and free-tier API limits; prices as token counts | X17, E5, X7 |
| 8. Conclusion | | |
| AI-use disclosure | Required by the course; from `docs/AI_USAGE_LOG.md` (kept locally) | |

## To prepare before writing

- [x] System diagram of the ladder and the calibration (`scripts/figures/diagram.py`, `results/figures/fig_system.{pdf,png}`; for both reports and the demo video).
- [ ] Choose the template and set up the LaTeX project (Overleaf or `report/` in this repository); tables in `results/tables/*.tex` are ready to `\input`.
- [ ] Mid-submission figures: decide between `fig_guaranteed` and `fig_validity` + `fig_shift` for the single plot slot.
- [ ] After 40599 (X12 32B): regenerate figures and tables, update the X12 section and the hypotheses table (H2).
- [ ] After the BigCodeBench pools: `scripts/experiments/bigcodebench.sh`, then X18 results in RESULTS.md and the summary.
- [ ] Final project report only: the API tier on LiveCodeBench and BigCodeBench (E8 / X3 / X11).
