# Results log

Every number here can be regenerated from the cached pools with the command shown; the batches in `scripts/experiments/` rerun each group of commands (see README.md, "Reproducing results"), and `python scripts/show_results.py` prints the tables. The top of this file is for writing the reports: the summary of findings, the proposal's hypotheses with their outcome, and the map from each finding to its figure, table and batch. Below that is the running log, newest first.

Terms: tiers t1 to t4 are Qwen2.5-Coder 1.5B, 7B, 14B and 32B; d1 to d3 (DS) are DeepSeek-Coder 1.3B, 6.7B and 33B; api is gpt-oss-120b on a free API. Families of thresholds: model selection (`singles`: pick one certified model), sub-ladders (`subsets`: any ordered subset of tiers as a ladder), two-level (one escalation threshold for all lower rungs plus one for the top). κ (`k64`) is the price of abstaining when choosing among certified settings; "lean" (`lb1`) means one draw per lower rung. MCC is Model Cascading for Code, BSE Bayesian self-escalation. E and X numbers are the experiments and extensions listed in [PLAN.md](PLAN.md) (sections 3 and 7). Results are grouped by kind (`src/codeladder/results.py`): evaluate runs in `results/runs/<benchmarks>/`, X9 / X4 / X13 / X16 / X17 runs in `router/`, `actions/`, `economics/`, `guaranteed/`, `oracle/`, published-baseline replays in `baselines/`, analysis outputs in `analysis/`; every tool finds a result by its name, so the names quoted below work as they are. Results that no current batch produces (exploratory runs of 19 to 21 Oct and superseded runs) are kept in `results/archive/` (README inside); log sections before 26 Oct may quote LiveCodeBench numbers on 1,053 tasks, before the lcb/3566 refill, which differ from the current 1,054-task runs by under a point; the summary, the tables in `results/tables/` and sections from 26 Oct use the current runs.

Figures: `python scripts/figures/report_figures.py results` (writes `results/figures/fig_<name>.{pdf,png}`). Tables: `python scripts/report_tables.py results` (writes `results/tables/<name>.{md,tex}`, LaTeX with booktabs). Both pick up BigCodeBench when its results exist.

## Summary of findings (27 Oct)

1. **The guarantee holds.** Across 311 evaluation runs with 1,536 calibrations (3 benchmarks, 8 models, every family, score, ablation and router; every run the current batches produce, counted 27 Oct with `scripts/count_validity.py`; without the 18 runs built on the corrupt 32B P(True) values, kept in `results/archive/stale_ptrue32/` until job 40599 recreates them), the share of splits whose true risk exceeds α is at most 5% against δ = 10% (`fig_validity`). Under temporal shift on LiveCodeBench, thresholds calibrated once on the oldest problems keep the risk at or below α in every later window.
2. **Marginal is not conditional.** One pooled calibration leaves hard LiveCodeBench problems above α (32B at α = 0.10: 11.3% risk on hard problems, above α in 92% of splits); calibrating per difficulty restores the guarantee in every group for about 3 points of coverage (`fig_conditional`).
3. **When do ladders pay? The cheap-share law.** A ladder's saving is bounded by the share of tasks its cheap rung certifies at α; prices only set how fast that bound is reached (break-even at 1 to 4 times the cheap tier's price) (X13). On HumanEval+/MBPP+ at α = 0.20 a 14B to 32B ladder costs 33% of 32B alone when 32B is 4 times as expensive per token, falling to 3% as the price gap grows (14B certifies 93% of tasks), and lean sub-ladders cost 0.45 [0.27, 0.95] of calibrated model selection (0.21 with the free-API 120B top rung) at no significant loss (X14). On LiveCodeBench no cheap rung certifies more than 14% of problems, so ladders save nothing at any price, and at α ≤ 0.10 calibrated selection of one model is as good or significantly better (the price of searching more configurations, a finite-sample effect). Lean lower rungs, Pareto Testing, routing by difficulty, mixing model families and a learned up-front router (X9) do not change this.
4. **Published cascades break their targets.** Model Cascading for Code and Bayesian self-escalation never abstain, so they cannot operate below the models' error floor (infeasible at α ≤ 0.10 everywhere and at any α on LiveCodeBench); where tuned at α = 0.20 their true risk exceeds α in 51 to 100% of splits, in every tier set but one (Model Cascading for Code on 14B to 32B: 0%).
5. **A new label-free signal.** Model-written stress inputs, with "stress timeout = failed check" and agreement on stress outputs, lift AUROC on stdin problems by 0.08 to 0.14 on every tier, confirmed on five tiers not used to choose the rule. As models grow, their confidently wrong LiveCodeBench programs are increasingly correct but too slow (56% at 1.5B, 68% at 32B on stdin problems), which is what stress inputs catch.
6. **Signals and repair.** Agreement x self-confidence is the best score; agreement alone collapses for strong models at strict α. Verbalised self-confidence is a usable signal only from Qwen 14B up (rating AUROC 0.62 to 0.81; 1.5B rates everything 100, 7B 0.52 to 0.58) and for the 120B API model; DeepSeek-Coder's ratings carry almost no signal at any size (0.50 to 0.65; it mostly says 95 or declines), so DeepSeek 33B collapses the same way (4.9% solved at α = 0.05 on HumanEval+/MBPP+); size for size DeepSeek-Coder v1 is weaker than Qwen2.5-Coder, mostly on LiveCodeBench. Repair beats a fresh sample only at 32B on HumanEval+/MBPP+ (and for broken code at small sizes on LiveCodeBench) and adds at most about 2 points in the calibrated controller; a learned repair-or-resample policy (X4) finds those places but does no better than a fixed action, and nothing under model selection.
7. **Reliability is expensive.** A calibrated 32B answer on LiveCodeBench costs about ten times the energy of one greedy answer, which is wrong 58% of the time; 14B is the most energy-efficient certified model at α = 0.10. MBPP+ keeps a floor of about 15% confidently wrong answers at every model size up to 120B.
8. **Against guaranteed baselines** (X16): a Trust-or-Escalate cascade adapted to code solves 12 to 41 points less than CodeLadder on HumanEval+/MBPP+ and 10 to 26 points less on LiveCodeBench, at 2 to 6 times the cost; the best single model with calibrated abstention comes within about 4 points of CodeLadder's model selection, which gains on cost only where the cheap-share law allows.
9. **The certificate is only as strong as the oracle** (X17). Calibrated on the original HumanEval / MBPP tests the certificate is vacuous (claimed risk 0, true risk 11 to 13%, above α in every split); with 10 or 25% of EvalPlus's tests it breaks at the margin (32B at α = 0.10 above α in 28% / 14% of splits); half of the tests suffice.
10. **Interim: P(True)** (X12, 32B pending). The first-token probability of "Yes" to "is this solution correct?" separates right from wrong answers better than the verbalised rating for every tier but 14B on MBPP+ (AUROC 0.57 to 0.86), including the small models whose ratings carry no signal.

## Hypotheses from the proposal and their outcome

The proposal (registered-report style) stated four research questions with hypotheses. Outcomes so far, on HumanEval+, MBPP+ and LiveCodeBench (BigCodeBench to follow):

| Question and hypothesis | Outcome | Evidence |
| --- | --- | --- |
| RQ1 validity. H1: calibrated CodeLadder stays at or below α; uncalibrated cascades exceed α at matched cost | **Confirmed.** At most 5% of splits above α (δ = 10%) over 311 runs; Model Cascading for Code and Bayesian self-escalation cannot reach α ≤ 0.10 and, tuned at α = 0.20, exceed it in 51 to 100% of splits (one exception: Model Cascading for Code on 14B to 32B, 0%) | `fig_validity`, `fig_baselines`, table `main` |
| RQ2 efficiency. H2: execution-based scores reach α at lower cost than log-probability or self-reported confidence | **Refuted as stated; the combination wins.** Agreement alone collapses for strong models at strict α; agreement x self-confidence is best (32B: +40.8 points [29.6, 44.8] at α = 0.05). Verbalised self-ratings carry signal only from Qwen 14B up; P(True) carries signal at every size (X12). Execution-based stress inputs are the strongest single signal on stdin problems (X2) | `fig_ptrue`, `fig_stdin`, `fig_ci`, tables `signals`, `ci` |
| RQ3 actions. H3: resampling better on small tiers, repair on larger; an adaptive policy lowers cost | **First half confirmed, second refuted.** A fresh sample beats repair up to 14B; repair wins at 32B on HumanEval+/MBPP+ (and, the one exception below, for broken code at 7B on LiveCodeBench). A learned repair-or-resample policy (X4) does no better than a fixed action | `fig_repair`, X4 section |
| RQ4 shift. H4: calibrating on older LiveCodeBench problems breaks the guarantee on newer ones; sliding-window recalibration restores it | **Refuted: no shift effect found.** Calibrated once on the oldest window, risk stays at or below α in every later window (four tiers, five windows). Recalibrating on the previous window (same size) exceeds α in one of five windows at α = 0.10 (0.114) and at α = 0.20 (0.233), as often as δ = 10% allows for single calibrations; recalibration is not needed for this shift | `fig_shift` |

Beyond the proposal: marginal is not conditional (X7, `fig_conditional`); when ladders pay, the cheap-share law (X13, `fig_economics`, table `economics`); guaranteed baselines (X16, `fig_guaranteed`, table `main`); oracle sensitivity (X17, `fig_oracle`, table `oracle`); learned up-front routing (X9, `fig_router`); energy (E6, `fig_energy`).

## Figures and tables for the reports

| Finding | Figure (`results/figures/`) | Table (`results/tables/`) | Batch (`scripts/experiments/`) | Log section |
| --- | --- | --- | --- | --- |
| How CodeLadder works (system diagram) | `fig_system` (`scripts/figures/diagram.py`) | | | |
| Guarantee holds over every run | `fig_validity` | | all (`count_validity.py`) | summary |
| Main comparison: CodeLadder, best single model, Trust or Escalate, published cascades | `fig_guaranteed`, `fig_baselines` | `main` | `qwen_families.sh`, `guaranteed.sh`, `baselines.sh` | X16; published baselines |
| Headline differences with intervals | `fig_ci` | `ci` | `bootstrap.sh` | X14 |
| When ladders pay (cheap-share law) | `fig_economics` | `economics` | `economics.sh` | X13 |
| Confidence signals: rating against P(True) | `fig_ptrue` | `signals` | `ptrue.sh`, `deepseek.sh` | X12; X1 / X10 |
| Stress inputs on stdin problems | `fig_stdin` | | `signals.sh` | X2 |
| Conditional risk by difficulty | `fig_conditional` | | `difficulty.sh` | X7 |
| Temporal shift | `fig_shift` | | `calibration_data.sh` | E5 |
| Repair against resample | `fig_repair` | | `repair.sh` | E4; X4 |
| Learned up-front router | `fig_router` | | `router.sh` | X9 |
| Energy | `fig_energy` | | `energy.sh` | E6 |
| Oracle sensitivity | `fig_oracle` | `oracle` | `oracle.sh` | X17 |
| BigCodeBench (all of the above that apply) | same figures, extra panels | same tables, extra rows | `bigcodebench.sh` | X18 |

## 27 Oct: X12 P(True) against the verbalised rating (interim, 32B pending)

`codeladder annotate-ptrue` asks each tier "Is this solution correct on all valid inputs?" with the answer prefilled and reads the first-token probabilities, P(Yes) / (P(Yes) + P(No)), for every visible-passing candidate; `python scripts/show_results.py ratings <bench>/<tier>... --json results/analysis/signals_auroc.json` computes the AUROC of each signal for separating hidden-test-correct from wrong programs (`scripts/experiments/ptrue.sh`; table `signals`, `fig_ptrue`). The first 32B run used a corrupt copy of the model and was discarded; the rerun (job 40598) and the evaluations with P(True) as the score (40599) are queued.

| Benchmark, signal | 1.5B | 7B | 14B | 32B | DS 1.3B | DS 6.7B | DS 33B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HumanEval+, rating | 0.50 | 0.54 | 0.73 | 0.74 | 0.51 | 0.51 | 0.54 |
| HumanEval+, P(True) | 0.65 | 0.66 | 0.78 | pending | 0.57 | 0.66 | 0.74 |
| MBPP+, rating | 0.50 | 0.52 | 0.62 | 0.66 | 0.50 | 0.59 | 0.51 |
| MBPP+, P(True) | 0.65 | 0.55 | 0.60 | pending | 0.65 | 0.64 | 0.67 |
| LiveCodeBench, rating | 0.50 | 0.58 | 0.81 | 0.77 | 0.50 | 0.65 | 0.59 |
| LiveCodeBench, P(True) | 0.64 | 0.80 | 0.86 | pending | 0.60 | 0.80 | 0.80 |

Reading (interim): P(True) separates right from wrong better than the verbalised rating for every tier except 14B on MBPP+, and it carries signal where the rating carries none (Qwen 1.5B always says 100; DeepSeek-Coder mostly says 95). Whether this lets cheap rungs certify more at strict α, and so changes the cheap-share bound of X13, is what the queued evaluations answer.

## 27 Oct: X18 BigCodeBench setup (pools in progress)

BigCodeBench v0.1.4, complete prompts (docstring and imports, as in the official "complete" split), 1,140 tasks. The first test method of each task is shown to the model and used as the visible test; all test methods are the hidden tests. Execution uses the benchmark's own environment (`~/venvs/bcb`, Python 3.10 with the pinned libraries, every file verified against pypi.org) through `src/codeladder/bcb_runner.py`, matched to the official harness (one `__test__` module, unittest discovery, the environment's Python on PATH, TZ=UTC, NLTK data, 30 GB memory limits) and fenced so generated code cannot signal processes outside its own session. Six tasks are excluded because their reference solution fails in this environment: 220 (needs tkinter), 461 and 1104 (time out), 590 and 1005 (their tests fetch real URLs, refused from the cluster), 1127 (a file-hashing test fails with the reference). That leaves 1,134 tasks, 1,129 with model-written probe inputs for behavioural agreement; 148 form BigCodeBench-Hard, used as the difficulty group. Settings as LiveCodeBench: 10 samples, 2,048 tokens, repair chains from 10 parents to depth 2; ratings and P(True) for every tier. The API tier runs on a 400-task subset (`data/bcb/subset_api400.json`: the 148 hard tasks plus 252 sampled with seed 0). When the seven pools exist, `scripts/experiments/bigcodebench.sh` runs the core experiments and regenerates the figures and tables.

Label fix (27 Oct): the hidden test of a task is the `TestCases` class only, as in the official harness. Before the fix, BigCodeBench/790 (a second test class the benchmark never runs) made every candidate wrong, and two methods of /1028 with multi-line signatures were missed. Jobs started after the fix label correctly; `scripts/relabel_bcb.py` (first step of `bigcodebench.sh`) recomputes the labels of these two tasks in pools made before it.

## 26 Oct: X17 oracle sensitivity: how strong must the tests be for the certificate to hold?

`codeladder hidden-outcomes` ran every distinct visible-passing program of the four Qwen tiers on all HumanEval+/MBPP+ hidden tests without stopping at the first failure (37,312 candidates; the overall verdict agrees with the stored label for all but 53, 0.14%, all MBPP+, most likely time-limit sensitive), then `codeladder oracle-shift` calibrated on labels from a random fraction f of each task's hidden tests and measured the true risk on all of them (`scripts/experiments/oracle.sh`, job 39380; κ = 64, same 100 splits). f = 0 is the visible tests alone, which for HumanEval+/MBPP+ are the original HumanEval / MBPP tests. Claimed risk (on the weak labels) / true risk / share of splits with true risk above α / coverage:

| Setting | α | f = 0 (original tests) | f = 0.1 | f = 0.25 | f = 0.5 | f = 1 (all EvalPlus tests) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen 32B alone | 0.05 | 0.000 / 0.110 / **100%** / 0.94 | 0.013 / 0.017 / 0% / 0.43 | 0.013 / 0.017 / 0% | 0.016 / 0.017 / 0% | 0.016 / 0.017 / 0% / 0.42 |
| | 0.10 | 0.000 / 0.110 / **100%** / 0.94 | 0.069 / 0.086 / **28%** / 0.84 | 0.064 / 0.073 / **14%** / 0.77 | 0.059 / 0.060 / 9% / 0.70 | 0.048 / 0.046 / 4% / 0.61 |
| Qwen 1.5B alone | 0.10 | 0.000 / 0.127 / **100%** / 0.87 | 0.070 / 0.085 / 8% / 0.73 | 0.070 / 0.079 / 3% | 0.072 / 0.074 / 1% | 0.071 / 0.070 / 1% / 0.68 |
| Model selection, 4 tiers | 0.10 | 0.000 / 0.122 / **100%** / 0.95 | 0.070 / 0.083 / 10% / 0.84 | 0.072 / 0.077 / 5% | 0.068 / 0.069 / 1% | 0.067 / 0.064 / 1% / 0.76 |
| Sub-ladders, 4 tiers | 0.10 | 0.000 / 0.131 / **100%** / 0.96 | 0.063 / 0.071 / 4% / 0.81 | 0.059 / 0.065 / 1% | 0.057 / 0.055 / 0% | 0.055 / 0.052 / 0% / 0.73 |

(At α = 0.20 every setting stays below α: the models' silent-failure rate, 11 to 13%, is already under it. At α = 0.05 only f = 0 exceeds α in more than δ of the splits.)

Reading:

- **A certificate computed with the original HumanEval / MBPP tests is vacuous.** Every accepted program passes them, so the claimed risk is 0 while the true risk is 11 to 13%: above α = 0.05 and 0.10 in every split. This is the code version of "Certified Against Which Oracle?" (2026), and it is why CodeLadder labels with EvalPlus's extended tests.
- **Weaker but non-trivial oracles under-report risk by up to 1.7 points** (f = 0.1), enough to break the guarantee at the margin: 32B alone at α = 0.10 exceeds α in 28% of splits with 10% of the tests and 14% with a quarter (δ is 10%). With half of the tests or more, violations are back at or below δ everywhere.
- **Practical rule:** the calibration labels must come from a test suite much stronger than the visible tests; on HumanEval+/MBPP+ about half of EvalPlus's tests suffice. Since the hidden tests of every benchmark are themselves finite, the certified α should be read relative to the benchmark's oracle.

## 26 Oct: X14 paired bootstrap intervals for the headline comparisons

`codeladder compare A B --alphas 0.05 0.1 0.2 --out results/ci/<name>.json` (`src/codeladder/bootstrap.py`; batch `scripts/experiments/bootstrap.sh`, job 39266, after the 37 LiveCodeBench results that predated the lcb/3566 refill were recomputed on all 1,054 tasks). 200 resamples of the tasks with replacement; each resample reruns the whole calibration on both runs with the same 20 split draws. Point estimates use those 20 splits (so they differ slightly from the 100-split numbers above); 95% percentile intervals for "A minus B" in solved share and for the cost ratio A / B:

| Comparison (A minus B) | α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- |
| Lean sub-ladders minus model selection, HumanEval+/MBPP+ | -12.9 points [-24.2, -2.8] | -7.1 [-11.8, -3.3] | -1.3 [-2.0, +0.8]; cost **0.45 [0.27, 0.95]** |
| Lean sub-ladders minus model selection, LiveCodeBench | -2.1 [-3.9, -1.3] | -2.3 [-2.6, -0.7] | -0.9 [-1.8, -0.1]; cost 0.98 [0.91, 1.06] |
| Two-level ladder minus model selection, LiveCodeBench | -0.3 [-5.2, +0.2] | +0.1 [-0.6, +0.3] | -0.2 [-0.7, +0.4]; cost 1.12 [1.06, 1.27] |
| API ladders (lean sub-ladders) minus model selection, HumanEval+/MBPP+ | -11.0 [-29.7, -6.5] | -7.3 [-11.3, -2.5] | -0.5 [-1.8, +2.6]; cost **0.21 [0.14, 0.71]** |
| All 7 local models minus Qwen only, model selection, HumanEval+/MBPP+ | -6.0 [-12.3, +4.9] | -3.0 [-5.6, +2.3] | -0.6 [-1.8, +0.6] |
| All 7 minus Qwen only, LiveCodeBench | -2.5 [-3.2, +1.4] | -1.7 [-2.3, +0.8] | +0.5 [-0.1, +1.7] |
| Pareto Testing minus Bonferroni paths, LiveCodeBench sub-ladders | -1.7 [-3.1, +2.4] | -1.3 [-2.2, +1.1] | -0.5 [-1.2, +0.8] |
| Lean (1 draw) minus full lower rungs, LiveCodeBench sub-ladders | -1.8 [-3.2, -0.6]; cost 0.61 [0.41, 0.83] | -1.2 [-2.1, -0.2]; cost 0.66 [0.55, 0.82] | 0.0 [-1.1, +0.2]; cost 0.78 [0.66, 0.85] |
| 32B: agreement x self-confidence minus agreement alone, HumanEval+/MBPP+ | **+40.8 [+29.6, +44.8]** | +22.6 [+0.2, +37.8] | 0.0 [-0.3, +0.1] |

Reading: the headline claims survive the task-level uncertainty. Ladders cost significantly less than model selection only at α = 0.20 on HumanEval+/MBPP+ (and with the API top rung), at no significant loss; at strict α they solve significantly less (the price of searching more settings). On LiveCodeBench the ladder costs the same and solves slightly but significantly less. Adding DeepSeek tiers, Pareto Testing and full lower-rung budgets have no significant effect on the solved share; lean lower rungs cut cost by 22 to 39% for at most 2 points. Self-confidence is what lets 32B answer at strict α.

## 26 Oct: X16 guaranteed baselines (per-model abstention, Trust-or-Escalate) against CodeLadder

`codeladder guaranteed --benchmark <b> --tiers <...> --score agreement_selfconf --alphas 0.05 0.1 0.2 --tag=-k64` (`src/codeladder/guaranteed.py`; batch `scripts/experiments/guaranteed.sh`, job 39373). Every tier is replayed alone for every threshold; cascades are compositions of those replays (a task is answered by the first tier that accepts it and costs every tier it visits), on the same 100 calibration/test halves as CodeLadder. **Per-model abstention** is Learn-then-Test on one model's marginal risk (Task Abstention for LLMs in Code Generation, 2026; the "calibrated single model" of earlier sections). **Trust or Escalate** (ICLR 2025), adapted to code: rung by rung, on the calibration tasks that reach the rung, the most permissive threshold whose error among the tasks it accepts is certified at most α (exact binomial, fixed sequence, δ / rungs); unaccepted tasks escalate, and the cascade abstains after the last rung. "Lean": one draw per lower rung. Solved share / mean cost; violations 0 to 4% of splits for every method.

| Benchmarks, tiers | Method | α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- | --- |
| HumanEval+/MBPP+, Qwen 1.5B to 32B | CodeLadder model selection | **41.5%** / 15.3k | 69.0% / 8.7k | **82.9%** / 6.5k |
| | CodeLadder sub-ladders, lean | 25.5% / 14.8k | 62.3% / 8.5k | 82.2% / **3.5k** |
| | best single model (abstention) | 40.5% / 26.2k (32B) | **72.6%** / 9.7k (14B) | 83.7% / 7.6k (14B) |
| | Trust or Escalate | 0.8% / 59.6k | 28.3% / 51.3k | 71.4% / 12.7k |
| | Trust or Escalate, lean | 0.8% / 36.4k | 25.2% / 31.3k | 71.2% / 9.9k |
| LiveCodeBench, Qwen 1.5B to 32B | CodeLadder model selection | 32.6% / 54.3k | 41.4% / 60.6k | 46.9% / 55.6k |
| | CodeLadder sub-ladders, lean | 29.9% / 46.2k | 39.7% / 58.8k | 46.3% / 54.5k |
| | best single model (32B) | **34.4%** / 72.0k | **42.5%** / 65.5k | **48.2%** / 57.8k |
| | Trust or Escalate | 6.5% / 136.7k | 23.3% / 123.4k | 37.2% / 103.7k |
| | Trust or Escalate, lean | 2.9% / 91.5k | 24.4% / 80.1k | 29.7% / 72.6k |
| All three pooled, Qwen 1.5B to 32B | CodeLadder model selection | 40.9% / 26.4k | 54.2% / 41.2k | 60.2% / 41.2k |
| | best single model | 43.2% / 25.3k (14B) | 55.5% / 48.3k (32B) | 60.5% / 42.7k (32B) |
| | Trust or Escalate | 4.6% / 111.6k | 35.7% / 91.7k | 48.6% / 72.9k |
| HumanEval+/MBPP+, Qwen + 120B API | CodeLadder model selection, lean | 43.8% / 33.4k | 67.9% / 31.4k | 84.9% / 96.0k |
| | CodeLadder sub-ladders, lean | 26.6% / 109.0k | 61.7% / 58.1k | **85.0% / 21.5k** |
| | best single model (abstention) | **45.2%** / 190.9k (API) | **72.6%** / 9.7k (14B) | 85.1% / 100.3k (API) |
| | Trust or Escalate | 0.4% / 302.6k | 35.3% / 227.4k | 70.9% / 79.8k |

Reading:

- **Certifying each rung's conditional error (Trust or Escalate) is far more conservative than certifying the whole policy's marginal risk (CodeLadder).** At the same α and δ, Trust or Escalate solves 12 to 41 points less than CodeLadder's model selection on HumanEval+/MBPP+ and 10 to 26 points less on LiveCodeBench, at 2 to 6 times the cost; at α = 0.05 it barely answers anything. Each rung must show low error on the small, hard set of tasks that reach it, with only δ / rungs to spend.
- **Per-model abstention is a strong baseline**, as the earlier sections found: the best single model is within about 4 points of CodeLadder's model selection everywhere and sometimes above it (model selection pays a Bonferroni price over the models and must find the best model from calibration data, where the "best single model" row picks it after the fact from the test results). CodeLadder's gain over it is cost at loose α with a large price gap: at α = 0.20 the lean sub-ladders solve 82.2% at 3.5k against 83.7% at 7.6k for the best single model on HumanEval+/MBPP+, and 85.0% at 21.5k against 85.1% at 100.3k for the API model alone (about a fifth of its cost).
- The guarantee holds for all methods (violations at most 4% of splits).

## 25 Oct: X13 when do ladders pay? Price sweep and the cheap-share law

`codeladder economics --benchmark <b> --tiers <cheap> <top> --score agreement_selfconf --tag=-k64 --lower-budget 1` (`src/codeladder/economics.py`; batch `scripts/experiments/economics.sh`). Each tier is replayed once with its own token counts; the price of the top tier is then swept from 1x to 1024x the cheap tier's price per token without rerunning anything. At every price, the cheapest certified two-tier ladder (same 100 splits, κ = 64, one draw on the cheap rung) is compared at matched coverage (at most 1 point below) with the top model alone and with calibrated model selection between the two. "Cheap share" s_c is the share of tasks the cheap rung certifies at α. Cost of the ladder relative to the top model alone at price ratio 1x / 4x / 1024x (relative to model selection at 1024x in the last column):

| Benchmark, cheap to top | α | cheap share | vs top alone, 1x / 4x / 1024x | vs selection, 1024x |
| --- | --- | --- | --- | --- |
| HumanEval+/MBPP+, Qwen 7B to 32B | 0.10 | 0.34 | 0.83 / 0.53 / 0.43 | 1.00 |
| | 0.20 | 0.89 | 0.54 / 0.27 / **0.06** | 0.24 |
| HumanEval+/MBPP+, Qwen 14B to 32B | 0.10 | 0.46 | 0.93 / 0.42 / 0.23 | 1.00 |
| | 0.20 | 0.93 | 0.60 / 0.33 / **0.03** | 0.29 |
| HumanEval+/MBPP+, Qwen 1.5B to 32B | 0.10 | 0.38 | 0.78 / 0.39 / 0.26 | 1.00 |
| | 0.20 | 0.61 | 0.70 / 0.53 / 0.47 | 0.47 |
| HumanEval+/MBPP+, DeepSeek 1.3B to Qwen 32B | 0.20 | 0.54 | 1.00 / 0.67 / 0.53 | 0.53 |
| HumanEval+/MBPP+, Qwen 14B to API 120B | 0.20 | 0.84 | 0.46 / 0.31 / 0.26 | 0.26 |
| HumanEval+/MBPP+, any cheap tier | 0.05 | 0.00 to 0.12 | 0.69 to 1.00 at 1024x | 0.98 to 1.02 |
| LiveCodeBench, Qwen 14B to 32B | 0.10 / 0.20 | 0.06 / 0.14 | 0.98 / 0.92 at 1024x | 0.95 / 0.88 |
| LiveCodeBench, Qwen 1.5B or 7B or DeepSeek 1.3B to 32B | any | 0.00 to 0.02 | 0.99 to 1.00 | 0.98 to 1.00 |

Reading:

- **The cheap-share law.** The saving is bounded by the share of tasks the cheap rung certifies: once the top tier is much more expensive, the ladder's relative cost approaches roughly 1 - s_c (plus the cheap rung's own small cost), whatever the price. Up to 97% cheaper than 32B alone on HumanEval+/MBPP+ at α = 0.20 (14B certifies 93% of tasks there); nothing on LiveCodeBench at any price, because no cheap rung certifies more than 14% of its problems.
- **Prices only set how fast the bound is reached.** The break-even ratio r* = tok_c / (s_c · tok_T) predicts where the ladder starts to pay; measured break-even ratios are 1 to 4 wherever s_c is above about 0.3, so ladders pay as soon as the top tier is a few times more expensive.
- **Against calibrated model selection** the ladder only gains where selection must pick the top model to reach the coverage (α = 0.20 on HumanEval+/MBPP+); at α = 0.10 selection already picks the cheap model when it suffices and ties the ladder. The question "when do ladders pay?" is therefore a property of the models' certifiable share at α, not of prices.
- The same runs with P(True) in place of the verbalised rating will be in `results/economics/economics-*agreement_ptrue*`, recreated by job 40599 after the 32B rerun (X12).

## 24 Oct: X4 learned repair-or-resample policy under the same guarantee

`codeladder actions --benchmark <b> --tiers <...> [--family shared] --score agreement_selfconf --abstain-factor 64 --alphas 0.05 0.1 0.2 --tag=-k64` (`src/codeladder/actions.py`; batch `scripts/experiments/repair.sh`, job 36749). When the controller has not accepted, it repairs its most promising failing candidate or draws a fresh sample. The learned policy decides from the context of that choice: tier, error class of the candidate, its visible pass-rate bucket (0, below half, at least half) and its repair depth, backing off to (tier, error class) and then tier when a context has fewer than 30 training repairs. The pools hold both outcomes for every context, so it is fit with full information: repair where the cached repair child is more often useful than the other fresh samples of the same tier on the same problem. "Useful" is passing the checks and the hidden tests ("hidden", labels of the training problems only) or passing the checks alone ("visible", label-free). It is trained on a random 30% of the problems; all variants are compared on the other 70% with the same 100 calibration/test halves, mean of 3 seeds. Violations at most 4% of splits for every variant.

The fresh-sample rate has to be matched per problem: against the tier's overall rate, which includes problems where nothing fails, resampling looked better in every context (for example 32B wrong answers: repair 15.5% against 83.0% overall, but 14.4% on the same problems). The earlier label-free table (`controller.learn_action_table`) uses the overall visible-pass rate and so rarely repairs.

Solved share / mean cost (weighted tokens):

| Setting | Policy | α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- | --- |
| HumanEval+/MBPP+, 32B alone | always resample | 34.4% / 27.0k | 53.8% / 22.3k | 82.3% / 13.9k |
| | always repair | 34.4% / 29.0k | 53.7% / 24.2k | **84.0%** / 14.8k |
| | label-free table | 34.4% / 27.0k | 53.8% / 22.3k | 82.3% / 13.9k |
| | learned, label-free reward | 34.4% / 28.3k | 53.7% / 23.6k | 83.6% / 14.4k |
| | learned, hidden reward | 34.4% / 28.2k | 53.7% / 23.4k | 83.4% / 14.3k |
| LiveCodeBench, 7B alone | always resample | 20.2% / 12.3k | 27.1% / 11.7k | 35.0% / 10.7k |
| | always repair | **22.8%** / 13.1k | 28.0% / 12.4k | 33.9% / 11.5k |
| | label-free table | 21.9% / 12.4k | **29.0%** / 11.7k | 35.1% / 10.7k |
| | learned, label-free reward | 21.9% / 12.4k | 28.9% / 11.7k | 35.1% / 10.7k |
| | learned, hidden reward | 21.9% / 12.4k | 29.0% / 11.7k | 35.1% / 10.7k |
| HumanEval+/MBPP+, model selection over 4 tiers | always resample | 24.7% / 10.6k | 65.1% / 7.8k | 82.3% / 6.4k |
| | always repair | 24.5% / 11.0k | 64.5% / 8.3k | 83.1% / 11.5k |
| | learned, hidden reward | 24.7% / 10.8k | 65.1% / 8.0k | 82.6% / 8.8k |
| LiveCodeBench, model selection over 4 tiers | always resample | 31.1% / 52.3k | 40.0% / 55.2k | 46.5% / 54.7k |
| | always repair | 31.0% / 58.2k | 39.6% / 63.6k | 46.8% / 61.2k |
| | learned, hidden reward | 31.1% / 52.3k | 40.1% / 56.1k | 46.6% / 55.3k |

(The label-free table and the learned policy with label-free reward give the same numbers as the hidden-reward policy within 0.4 points in the model-selection rows.)

Reading:

- **The learned policy finds the right places to repair.** With matched fresh rates it repairs 32B wrong answers on HumanEval+/MBPP+ (15.5% against 14.4% for a fresh sample of the same problem), runtime errors at 1.5B to 14B in some contexts, and broken code at 1.5B and 7B on LiveCodeBench (7B: 37.5% against 12.4%); elsewhere it resamples. Label-free and hidden rewards choose almost the same contexts, so the labels add nothing.
- **It is worth at most about 2 points, and no more than a fixed action.** On the two settings where repair helps, the learned policy gains 1.1 to 1.9 points over resampling at the same or slightly higher cost, but a fixed choice does as well (always repair for 32B at α = 0.20: 84.0% against 83.4%; the label-free table for 7B on LiveCodeBench: identical). Under model selection over four tiers every policy is within 0.8 points of always resampling: the calibrated controller mostly accepts or escalates before a repair would matter, and the strongest tier, which repairs best, is also the one that needs it least.
- **Implication:** repair is a minor lever in a risk-controlled cascade. The guarantee holds for every policy (the action rule is fixed before calibration, so Learn-then-Test applies unchanged), and the per-step repair statistics of E4 overstate its value once acceptance is calibrated.

## 24 Oct: X1 / X10 corrected: DeepSeek-Coder (1.3B, 6.7B, 33B) and cross-family ladders

LiveCodeBench rows of Qwen tiers in this section were computed on 1,053 tasks, before the lcb/3566 refill; the current 1,054-task numbers are in table `main` and the X16 section (differences under a point).

**Why the earlier DeepSeek numbers were wrong.** DeepSeek-Coder mostly declines to rate its own code ("I'm sorry ... as an AI, I don't have the ability to execute code"). Share of visible-passing candidates with no parsable rating: d1 15 to 62%, d2 43 to 70%, d3 71 to 95% (LiveCodeBench d3: 3,473 of 3,671), against 0% for every Qwen tier. A missing rating counts as 0, so under agreement x self-confidence those candidates could never be accepted, and the sentence fallback read stray numbers ("a 3-character string" as 3%). The 21 Oct DeepSeek table and the first d3 / cross-family runs were biased against DeepSeek; everything below replaces them.

**Fix.** For DeepSeek tiers the answer is prefilled with "Confidence (0-100): " and the server continues it (`conf_prefill` in `configs/experiment.yaml`, vLLM `continue_final_message`; `annotate-confidence --redo`, old rating kept in `self_conf_v1`; jobs 36481 to 36483). Every reply now has a number.

**DeepSeek's self-confidence carries almost no signal, and in Qwen the signal appears only with scale.** DeepSeek answers 95 or 100 nearly always. AUROC of the rating for hidden correctness among visible-passing candidates, every tier (`python scripts/show_results.py ratings <benchmark>/<tier>`; 25 Oct; every rating parsed):

| Tier | HumanEval+ | MBPP+ | LiveCodeBench |
| --- | --- | --- | --- |
| Qwen 1.5B | 0.500 (always 100) | 0.500 | 0.500 |
| Qwen 7B | 0.544 | 0.518 | 0.582 |
| Qwen 14B | 0.734 | 0.618 | 0.806 |
| Qwen 32B | 0.744 | 0.655 | 0.769 |
| DeepSeek 1.3B | 0.511 | 0.504 | 0.495 |
| DeepSeek 6.7B | 0.513 | 0.588 | 0.651 |
| DeepSeek 33B | 0.544 | 0.510 | 0.590 |
| gpt-oss-120b (API) | 0.728 | 0.628 | (ratings pending) |

Verbalised confidence is a usable signal only from Qwen 14B up (and for the 120B API model); the small Qwen models rate everything the same, and DeepSeek-Coder never separates right from wrong much. This is why agreement x self-confidence helps the strong models at strict α and changes nothing for the small ones: on HumanEval+/MBPP+, Qwen 1.5B solves 27.6 / 60.7 / 74.2% at α = 0.05 / 0.10 / 0.20 with either score, while Qwen 32B goes from 0.0 / 32.9 / 83.1% with agreement alone to 40.5 / 56.4 / 83.1% with its ratings.

So for DeepSeek, agreement x self-confidence is close to agreement alone (the two scores give the same solved share within 2 points in every DeepSeek run below). On stdin problems, the stress rule with ratings: 0.844 (1.3B), 0.889 (6.7B), 0.865 (33B), against 0.845, 0.880, 0.863 for the rule alone.

**Held-out stress check on DeepSeek 33B** (`results/analysis/x2_stdin_signals.json`, 1,752 stdin candidates): AUROC 0.757 with visible tests, 0.863 with the stress rule (+0.106). The rule now holds on all five tiers not used to choose it (Qwen 14B and 32B, DeepSeek 1.3B, 6.7B and 33B).

**Each model alone**, calibrated, agreement x self-confidence, κ = 64 (`evaluate --benchmark <b> --tiers <t> --score agreement_selfconf --abstain-factor 64 --alphas 0.05 0.1 0.2`; batch `scripts/experiments/deepseek.sh`, job 36484). Solved share / mean cost; violations at most 5% of splits in every DeepSeek and cross-family run:

| Model | HumanEval+/MBPP+ α = 0.05 | α = 0.10 | α = 0.20 | LiveCodeBench α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek 1.3B | 33.5% / 2.2k | 59.2% / 1.7k | 70.8% / 1.4k | 8.7% / 4.3k | 12.4% / 4.2k | 12.4% / 4.2k |
| Qwen 1.5B | 27.6% / 1.1k | 60.7% / 0.8k | 74.2% / 0.6k | 10.8% / 2.5k | 16.4% / 2.5k | 16.4% / 2.5k |
| DeepSeek 6.7B | 23.0% / 15.9k | 54.3% / 12.4k | 74.7% / 9.2k | 14.4% / 23.5k | 20.0% / 21.9k | 22.1% / 21.6k |
| Qwen 7B | 6.1% / 6.6k | 68.6% / 3.7k | 81.5% / 2.8k | 22.8% / 12.1k | 29.4% / 11.3k | 35.8% / 10.6k |
| DeepSeek 33B | 4.9% / 47.2k | 67.3% / 29.0k | 78.7% / 21.9k | 20.3% / 100.9k | 23.3% / 98.8k | 31.1% / 88.4k |
| Qwen 14B | 30.2% / 15.1k | 72.6% / 9.7k | 83.8% / 7.6k | 33.0% / 32.0k | 39.2% / 29.9k | 43.4% / 26.9k |
| Qwen 32B | 40.5% / 26.2k | 56.4% / 21.7k | 83.1% / 13.9k | 35.4% / 71.2k | 42.8% / 65.0k | 48.4% / 57.4k |

(Qwen rows: the calibrated single-model baselines of the four-tier runs, same splits.) Uncalibrated, accepting the best visible-passing sample after 3 draws, DeepSeek 1.3B / 6.7B / 33B solve 65.1 / 65.8 / 76.3% of HumanEval+/MBPP+ and 9.0 / 17.3 / 26.0% of LiveCodeBench (Qwen 1.5B / 7B / 14B / 32B: 68.7 / 79.9 / 81.7 / 82.0% and 11.2 / 26.8 / 40.6 / 45.8%).

**Cross-family ladders and model selection** (`--family singles|subsets|two_level --lower-budget 1`):

| Setting | HumanEval+/MBPP+ α = 0.05 | α = 0.10 | α = 0.20 | LiveCodeBench α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen 4 tiers, model selection (reference) | 41.5% / 15.3k | 69.0% / 8.7k | 82.9% / 6.5k | 33.3% / 59.2k | 41.9% / 62.8k | 47.3% / 55.7k |
| DeepSeek 3 tiers, singles / subsets / two-level | 25.8 / 21.0 / 0.0% | 61.2 / 57.7 / 59.5% | 79.3 / 79.1 / 79.5% | 19.6 / 17.8 / 18.9% | 20.9 / 23.1 / 23.1% | 31.3 / 30.9 / 31.1% |
| DeepSeek 1.3B + Qwen 32B, singles / subsets / two-level | 40.1 / 38.3 / 35.3% | 58.4 / 54.9 / 52.1% | 82.8 / 82.4 / 82.4% | 32.5 / 31.9 / 31.9% | 42.4 / 41.7 / 41.9% | 47.9 / 47.0 / 47.2% |
| DeepSeek 1.3B + Qwen 14B + 32B, singles / subsets / two-level | 39.7 / 34.2 / 35.5% | 68.6 / 65.3 / 43.8% | 83.3 / 83.0 / 83.0% (8.0k / 4.2k / 3.9k) | 32.7 / 30.5 / 30.8% | 42.0 / 40.5 / 41.9% | 47.3 / 46.8 / 47.1% |
| Qwen 7B + DeepSeek 33B, singles / subsets / two-level | 6.5 / 5.6 / 0.0% | 65.2 / 61.6 / 57.7% | 81.5 / 81.0 / 80.6% | 20.6 / 20.4 / 19.4% | 26.9 / 28.2 / 27.4% | 35.6 / 35.3 / 33.7% |
| All 7 local models, model selection | 37.2% / 16.4k | 67.6% / 10.4k | 82.2% / 7.2k | 31.2% / 53.3k | 40.8% / 58.2k | 47.9% / 55.7k |

Reading:

- **DeepSeek-Coder (v1) is weaker than Qwen2.5-Coder at the same size, but less than the biased numbers suggested.** On HumanEval+/MBPP+ the 1.3B model matches Qwen 1.5B (59.2% against 60.7% at α = 0.10; more at α = 0.05), and the 33B model matches Qwen 7B (67.3% against 68.6%) at about eight times the cost. On LiveCodeBench the gap is large: DeepSeek 33B solves 23.3% at α = 0.10, below Qwen 7B (29.4%) and about half of Qwen 32B (42.8%). The earlier claim that DeepSeek 6.7B is not better than 1.3B holds only at α ≤ 0.10 on HumanEval+/MBPP+.
- **Without a usable self-confidence signal a strong model collapses at strict α.** DeepSeek 33B solves 4.9% of HumanEval+/MBPP+ at α = 0.05, the same collapse as Qwen 32B scored with agreement alone (0.0%, job 36476); with its own ratings Qwen 32B keeps 40.5%. The rating is what lets a strong model accept at strict risk levels.
- **Mixing families still does not change the economics.** The best cross-family ladders stay within about a point of Qwen-only model selection (for example DeepSeek 1.3B + Qwen 14B + 32B: 68.6% against 69.0% at α = 0.10 on HumanEval+/MBPP+; DeepSeek 1.3B + Qwen 32B: 42.4% against 41.9% on LiveCodeBench), and adding the three DeepSeek tiers to the candidate set lowers coverage slightly at strict α (all 7 models: 37.2% against 41.5% at α = 0.05 on HumanEval+/MBPP+), the price of testing more models. At α = 0.20 on HumanEval+/MBPP+ the cheap-rung ladders again cut cost by about 40% (DeepSeek 1.3B + Qwen 14B + 32B two-level: 83.0% / 3.9k against 82.9% / 6.5k), as with Qwen-only ladders.
- **Agreement-only versions** of every run above (`--score agreement`, job 36476, results `*-agreement-*`) match the agreement x self-confidence numbers for DeepSeek tiers within 2 points; for Qwen tiers agreement alone collapses at α = 0.05 (7B 0.6%, 14B 0.0%, 32B 0.0% on HumanEval+/MBPP+).

Energy of DeepSeek 33B (busy time; TP=2 x PP=2 on 2080 Ti): 3.9 J per output token on HumanEval+, 6.0 on MBPP+, 5.8 on LiveCodeBench.

## 23 Oct: X9 learned up-front router (RouteLLM style) under the same guarantee

`codeladder route --benchmark <b> --tiers <...> --score agreement_selfconf --abstain-factor 64 --alphas 0.05 0.1 0.2 --subsets --tag=-k64` (`src/codeladder/router.py`). The router sees only the problem: hashed tf-idf of the statement (unigrams and bigrams), benchmark, format (function, method, stdin), known difficulty and the largest stated constraint (for example 2 x 10^5). One L2-regularised logistic model per tier predicts the share of that tier's samples that pass the hidden tests (soft labels, L2 chosen by 3-fold cross-validation). It is trained on a random 30% of the problems, which are then left out; everything below is on the other 70%, with the same 100 calibration/test halves for every family. Routing rule: a problem goes to the cheapest tier whose prediction is at least τ (the last tier if none is), which then runs the single-tier acceptance rule with threshold λ. (τ, λ) is calibrated with Learn-then-Test, one fixed-sequence path per τ in {1.01, 0.9, ..., 0.1, 0} (11 paths, Bonferroni). The oracle router applies the same rule to the true pass shares: an upper bound for this rule, not for every router (it reads labels). Mean of 3 router seeds; violations 0% of splits for every family and α.

Solved share / mean cost (weighted tokens):

| Benchmarks, tiers | Family | α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- | --- |
| HumanEval+/MBPP+, t1 to t4 (router AUROC 0.54 to 0.60) | calibrated model selection | 26.8% / 13,686 | 64.7% / 9,633 | 82.6% / 7,843 |
| | sub-ladders | 7.7% / 7,324 | 63.4% / 16,468 | 82.3% / 7,674 |
| | learned router | 22.4% / 12,916 | 56.8% / 4,835 | 82.4% / 10,099 |
| | oracle router | 27.3% / 6,896 | 61.0% / 6,197 | 85.4% / 3,375 |
| LiveCodeBench, t1 to t4 (router AUROC 0.87 to 0.92) | calibrated model selection | 33.2% / 53,671 | 41.3% / 54,786 | 47.4% / 52,467 |
| | sub-ladders | 31.2% / 83,989 | 40.5% / 79,965 | 46.5% / 64,557 |
| | learned router | 27.3% / 74,131 | 40.2% / 66,658 | 46.9% / 56,927 |
| | oracle router | 29.0% / 72,202 | 41.8% / 61,305 | 47.7% / 51,486 |
| HumanEval+/MBPP+, t1 to t4 + api | calibrated model selection | 35.1% / 78,127 | 67.8% / 74,316 | 84.2% / 80,253 |
| | sub-ladders | 13.6% / 55,780 | 62.4% / 70,302 | 84.7% / 32,132 |
| | learned router | 24.5% / 123,605 | 62.2% / 96,510 | 83.7% / 90,332 |
| | oracle router | 43.8% / 50,452 | 83.7% / 40,481 | 88.5% / 18,161 |

Reading:

- **A learned up-front router never beats calibrated model selection.** It loses coverage at strict α (4 to 11 points at α = 0.05) and is never cheaper for the same coverage. Choosing among certified settings by cost + κ x abstention sometimes buys a cheaper setting with less coverage (HumanEval+/MBPP+ at α = 0.10: half the cost, 8 points fewer solved), which is a different point on the same frontier, not a gain.
- **Why it fails differs by benchmark.** On HumanEval+/MBPP+ the problem text barely predicts which model will succeed (AUROC 0.54 to 0.60). On LiveCodeBench it predicts well (AUROC about 0.9; the inputs include the difficulty label and constraint sizes), but what it predicts is that hard problems are hard for every model: the problems a small model solves, the 32B model solves too, and the router sends 92 to 98% of problems to the 32B model anyway.
- **The headroom is real only with a large price gap.** With the API top tier, perfect routing would solve 84% at α = 0.10 for 54% of the cost of model selection (68%), and 88.5% at α = 0.20 for 23% of the cost. On LiveCodeBench even the oracle router gains at most 0.5 points. One λ is shared by all routed tiers (a per-tier λ would multiply the paths); this costs the oracle router coverage at α = 0.05 on LiveCodeBench.
- **Implication for CodeLadder:** information that arrives after generation (agreement, self-confidence, stress checks) is what makes cheaper rungs safe; the problem statement alone is not enough. Where ladders pay (loose α, large price gap) sub-ladders match or beat the learned router (API ladder at α = 0.20: 84.7% / 32,132 against 83.7% / 90,332).

All three benchmarks pooled (1,596 problems; the router can also learn "HumanEval+/MBPP+ go to cheap tiers"; AUROC 0.85 to 0.90): model selection 39.5% / 26,576, 53.0% / 39,004, 59.6% / 41,000 at α = 0.05 / 0.10 / 0.20; learned router 33.6% / 48,014, 52.3% / 48,169, 59.3% / 40,516; oracle router 33.0% / 49,127, 55.9% / 42,200, 60.7% / 33,104; sub-ladders 40.2% / 55,574, 53.1% / 58,015, 59.5% / 51,232. Same conclusion: no gain from the learned router (job 36363 on a compute node; the login node's 2 GB memory cap killed it there). Figure: `fig_router`.

## 23 Oct: E8 / X3 free API top tier (gpt-oss-120b on Groq), HumanEval+ and MBPP+

Pools: 8 samples per task, no repairs, reasoning effort low, keys rotated across 4 free accounts (`scripts/run_api_tier.sh`). pass@1 89.3% (HumanEval+) and 79.8% (MBPP+); silent failures 8.0% and **15.6%** (the MBPP+ floor of confidently wrong answers persists even for a 120B model). Cost counts total parameters (params_b = 120; the model is a mixture of experts with about 5B active, so this is the conservative choice and makes the top rung expensive). Commands: `scripts/experiments/api.sh` (`evaluate --benchmark humaneval mbpp --tiers <...> --score agreement_selfconf --abstain-factor 64 [--family f --lower-budget 1]`).

| Setting | α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- |
| API alone, agreement x self-confidence | **45.2%** / 190.9k | 67.3% / 150.1k | 85.1% / 100.2k |
| API alone, agreement only | 0.0% | 33.8% / 166.2k | 85.0% / 86.9k |
| API alone, self-confidence only | 33.9% / 208.4k | 68.2% / 152.4k | 85.1% / 100.5k |
| Qwen 1.5B to 32B + API: singles | 43.8% / 33.4k | 67.9% / **31.4k** | 84.9% / 96.0k |
| Qwen 1.5B to 32B + API: subsets (lower budget 1) | 26.6% / 109.0k | 61.7% / 58.1k | 85.0% / **21.5k** |
| Qwen 1.5B to 32B + API: two-level (lower budget 1) | 31.9% / 219.1k | 63.4% / 164.7k | 84.2% / 35.4k |
| Qwen 14B + API: subsets / two-level | 35.0% / 32.0% | 66.4% / 67.2% | **85.9%** (28.8k) / 86.0% (27.9k) |

**Interpretation.** With a strong but expensive top rung, ladders finally pay clearly at loose α: at α = 0.20 a sub-ladder solves as much as the API model alone for about a fifth of the cost (21.5k against 100.2k), and 14B to API solves slightly more. At α = 0.10 single-model selection matches the API model's coverage for a fifth of its cost by choosing cheaper models. At α = 0.05 the API model alone is best. The pattern across the study: ladders pay when the price gap between rungs is large and α is loose; otherwise calibrated selection of one model is as good or better. Violations at most 2% of splits. LiveCodeBench for the API tier is running (400-problem subset first, then all 1,054).

## 21 Oct: X1 second model family (DeepSeek-Coder 1.3B, 6.7B) and X10 cross-family ladders (superseded)

Superseded by the 24 Oct section: most DeepSeek self-confidence ratings were missing and counted as 0, which biased these numbers against DeepSeek (the first fix, parsing ratings from sentences and 64 output tokens, recovered only some of them). Kept for the record: this run also fixed the ladder order, so `evaluate` now keeps the tier order given in `--tiers` (cheapest first). The original table is in the git history of this file.

## 21 Oct: X7 difficulty-aware calibration and conditional risk, X8 Pareto Testing at scale, silent-failure types

**X7** (`evaluate --benchmark livecodebench --tiers t1 t2 t3 t4 --group-by difficulty --score agreement_selfconf --family <f> --abstain-factor 64 [--lower-budget 1] --tag=-k64[-lb1]-diff`). "Pooled": one calibration over all 1,054 problems; "per difficulty": each of easy / medium / hard calibrated on its own at δ/3 (the cheapest certified setting per group, i.e. routing by known difficulty).

| Setting | pooled α = 0.05 | α = 0.10 | α = 0.20 | per difficulty α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- | --- | --- | --- |
| singles (4 tiers) | 32.6% / 54.3k | **41.4%** / 60.6k | 46.9% / 55.6k | 26.2% / 32.4k | 38.1% / 48.2k | 43.7% / 53.4k |
| subsets, lower budget 1 | 29.9% / 46.2k | 39.7% / 58.8k | 46.3% / 54.5k | 14.6% / 16.8k | 35.2% / 39.7k | 42.3% / 50.3k |
| 32B alone | 34.4% / 72.0k | 42.5% / 65.5k | 48.2% / 57.8k | 26.8% / 76.2k | 40.3% / 68.5k | 45.0% / 62.9k |

Routing by known difficulty is cheaper (about 20% at α = 0.10) but solves about 3 points less: each group gets a third of δ and a third of the calibration data. It is not a better operating point.

**Conditional risk of pooled calibration** (true risk within each difficulty group under the setting chosen for the whole pool, averaged over splits; share of splits where it exceeds α):

| Pooled setting | α | easy | medium | hard |
| --- | --- | --- | --- | --- |
| 32B alone | 0.10 | 0.020 (0%) | 0.090 (5%) | **0.113 (92%)** |
| 32B alone | 0.20 | 0.034 (0%) | 0.200 (74%) | **0.247 (100%)** |
| singles (4 tiers) | 0.10 | 0.017 (0%) | 0.085 (8%) | 0.099 (65%) |
| subsets, lower budget 1 | 0.20 | 0.041 (0%) | 0.190 (30%) | 0.219 (86%) |

The marginal guarantee holds because easy problems are almost never wrong when accepted, while hard problems run above α. Per-difficulty calibration restores the guarantee in every group (violations 0%) at the cost of about 3 points of coverage. Any deployment that cares about hard problems should calibrate per difficulty (or per user-defined group).

**X8** (`--method pareto`, lower budget 1, κ = 64): Pareto Testing matches Bonferroni paths within about 1 point everywhere (LiveCodeBench subsets at α = 0.10: 39.2% against 39.7%; pooled 1,596 tasks: 52.1% against 52.4%). Pooled over all three benchmarks the families are close: singles 54.2% / 41.2k, subsets 52.4% / 37.8k, two-level 54.1% / 54.5k at α = 0.10. The price of flexibility is not an artefact of the testing procedure.

**Silent-failure types** (`python scripts/analysis_silent_failures.py --benchmark <b> --tiers <t>`, first hidden failure of visible-passing wrong samples; a first version of the script reused freed tests across tiers and was fixed with `tasks.release_tests`; stored labels were re-checked):

| Tier | LiveCodeBench stdin: too slow / wrong output | LeetCode-style: too slow / wrong | MBPP+: wrong output / crash / too slow |
| --- | --- | --- | --- |
| 1.5B | 56% / 43% | 32% / 65% | 67% / 27% / 6% |
| 7B | 64% / 33% | 37% / 56% | 65% / 26% / 9% |
| 14B | 63% / 31% | 44% / 44% | 67% / 27% / 6% |
| 32B | **68%** / 26% | **53%** / 34% | 73% / 24% / 3% |

As models grow, their confidently wrong LiveCodeBench programs are increasingly correct but too slow (what stress inputs catch); the MBPP+ floor is wrong outputs at every size, consistent with under-specified tasks.

## 20 Oct: E6 energy cost of reliability

`evaluate ... --cost-metric energy_j --abstain-factor 64 --tag=-k64[-lb1]-energy`: cost is measured GPU energy (busy seconds only; the confidence request priced at 0.1 x the tier's joules per output token). Solved share / mean joules per task; violations at most 4% of splits.

| Setting | HumanEval+/MBPP+ α = 0.05 | α = 0.10 | α = 0.20 | LiveCodeBench α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- | --- | --- | --- |
| 1.5B alone | 27.6% / 0.3k | 60.7% / 0.2k | 74.2% / 0.2k | 11.2% / 0.6k | 16.3% / 0.5k | 16.3% / 0.5k |
| 7B alone | 6.1% / 0.8k | 68.6% / 0.4k | 81.5% / 0.3k | 22.5% / 1.4k | 29.0% / 1.3k | 35.7% / 1.2k |
| 14B alone | 30.2% / 1.6k | **72.6%** / 1.0k | **83.8%** / 0.8k | 32.5% / 4.3k | 39.0% / **4.0k** | 43.2% / 3.6k |
| 32B alone | **40.5%** / 2.9k | 56.4% / 2.4k | 83.1% / 1.5k | **34.4%** / 7.6k | **42.5%** / 6.9k | **48.2%** / 6.1k |
| singles (4 tiers) | 41.5% / 1.7k | 69.0% / 0.9k | 82.9% / 0.7k | 32.7% / 6.2k | 41.4% / 6.5k | 46.9% / 5.8k |
| subsets, lower budget 1 | 25.5% / 1.7k | 62.3% / 0.9k | 82.2% / **0.4k** | 29.9% / 5.5k | 39.7% / 6.4k | 46.3% / 5.8k |
| two-level, lower budget 1 | 35.7% / 3.4k | 47.5% / 3.1k | 82.0% / 0.5k | 31.1% / 8.6k | 41.5% / 7.8k | 46.7% / 6.7k |

Energy ranks the settings as token cost does. Per solved task on LiveCodeBench at α = 0.10: 14B about 10 kJ, 32B about 16 kJ; the reliable answer costs about ten times the energy of a single greedy 32B answer (which is wrong 58% of the time). The one regime where a ladder saves energy is again loose α on HumanEval+/MBPP+ (0.4 kJ against 0.7 kJ per task at α = 0.20).

## 20 Oct: E4 repair versus resample, all four tiers

`python scripts/analysis_repair.py --benchmarks <...> --json results/analysis/e4_repair_<...>.json`: success on hidden tests of one repair round against a fresh sample on the same tasks, by the failure shown to the model.

| Tier | HumanEval+/MBPP+: wrong answer (repair / fresh) | runtime error | LiveCodeBench: wrong answer | broken code |
| --- | --- | --- | --- | --- |
| 1.5B | 2.2% / 20.3% | 10.2% / 20.0% | 0.5% / 3.4% | 4.0% / 1.1% |
| 7B | 8.3% / 23.2% | 14.8% / 19.6% | 2.0% / 5.1% | **35.4% / 9.8%** |
| 14B | 12.3% / 17.0% | 13.5% / 17.2% | 2.8% / 6.2% | 5.1% / 13.7% |
| 32B | **16.9% / 9.8%** | **34.2% / 24.2%** | 3.9% / 5.1% | 9.0% / 8.4% |

Repair becomes worthwhile only at 32B on HumanEval+/MBPP+ (and for broken, mostly formatting, code at 7B on LiveCodeBench); elsewhere a fresh sample is better. In the calibrated controller this is worth little: 32B with repair instead of resampling solves 84.7% against 83.0% at α = 0.20 (+6% cost) and the same at α = 0.05 and 0.10; the label-free adaptive table matches resampling.

## 20 Oct: ladder design, lower-tier draw budget (κ = 64)

`evaluate ... --lower-budget b`: every tier below the last one used may draw at most b candidates (default 8). Solved share / mean cost; violations at most 1% of splits.

| Setting | b | α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- | --- |
| LiveCodeBench, 32B alone | (8) | 35.4% / 71.2k | **42.8%** / 65.0k | **48.4%** / 57.4k |
| LiveCodeBench, singles (4 tiers) | (8) | 33.3% / 59.2k | 41.9% / 62.8k | 47.3% / 55.7k |
| LiveCodeBench, two-level (4 tiers) | 8 | 33.0% / 122.4k | 42.0% / 111.7k | 46.9% / 95.9k |
| | 1 | 31.1% / 79.9k | 41.5% / 72.6k | 46.7% / 62.5k |
| LiveCodeBench, subsets (4 tiers) | 8 | 31.9% / 87.1k | 40.8% / 80.5k | 46.8% / 73.2k |
| | 1 | 29.9% / **46.2k** | 39.7% / **58.8k** | 46.3% / **54.5k** |
| HumanEval+/MBPP+, singles (4 tiers) | (8) | **41.5%** / 15.3k | **69.0%** / 8.7k | 82.9% / 6.5k |
| HumanEval+/MBPP+, subsets (4 tiers) | 8 | 25.7% / 15.0k | 67.4% / 15.4k | 83.2% / 5.6k |
| | 1 | 25.5% / 14.8k | 62.3% / 8.5k | 82.2% / **3.5k** |
| | 3 | 25.7% / 15.0k | 67.6% / 14.1k | 83.1% / 4.7k |

**Interpretation.** Letting cheap rungs give up after one draw cuts ladder cost by about 35% on LiveCodeBench, but the ladders still do not beat the best single model there (subsets: 39.7% at 58.8k against 42.8% at 65.0k for 32B alone at α = 0.10). On HumanEval+/MBPP+ at α = 0.20 a lean sub-ladder solves the same share as single-model selection for 46% less cost. Overall: under a finite-sample risk guarantee a ladder pays off only for cost at loose α on the easier benchmark; at strict α and on hard problems, calibrated selection of one model is as good or better.

## 20 Oct: LiveCodeBench with all four Qwen tiers (1,053 tasks)

Pools: 14B 1,054 (6.5M output tokens, 2.24 J/token busy), 32B 1,054 (stress signatures computed during generation). Commands as before with `--benchmark livecodebench --tiers t1 t2 t3 t4 --score agreement_selfconf --family <f> --abstain-factor <κ>`. Violations at most 5% of splits everywhere.

| Setting | α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- |
| 32B alone (κ = 4) | **35.4%** (71.2k) | **42.8%** (65.0k) | **48.4%** (57.4k) |
| 14B alone (κ = 4) | 33.0% (32.0k) | 39.2% (**29.9k**) | 43.4% (26.9k) |
| 7B alone (κ = 4) | 22.8% (12.1k) | 29.4% (11.3k) | 35.8% (10.6k) |
| singles, 4 tiers, κ = 64 | 33.3% (59.2k) | 41.9% (62.8k) | 47.3% (55.7k) |
| subsets, 4 tiers, κ = 64 | 31.9% (87.1k) | 40.8% (80.5k) | 46.8% (73.2k) |
| two-level, 4 tiers, κ = 64 | 33.0% (122.4k) | 42.0% (111.7k) | 46.9% (95.9k) |
| subsets, 1.5B/14B/32B, κ = 64 | 31.6% (81.1k) | 41.0% (76.6k) | 46.8% (72.5k) |
| two-level, 1.5B/14B/32B, κ = 64 | 32.4% (109.1k) | 41.9% (100.7k) | 46.9% (85.9k) |

**Interpretation.** With 1,053 tasks (526 calibration) the ladder still does not beat the best single model, and here it is also more expensive than 32B alone: on hard problems the lower tiers rarely reach a certifiable score, so they spend their full draw budget before escalating. 14B alone is the most cost-efficient point (39.2% at 29.9k against 42.8% at 65.0k for 32B at α = 0.10). A lower-tier draw budget (cheap rungs give up after 1 to 3 draws) was tested next (see "ladder design, lower-tier draw budget").

**E5 temporal shift, four tiers**: thresholds calibrated once on the oldest window stay at or below α in every later window (α = 0.10: risk 0.002 to 0.026; α = 0.20: 0.091 to 0.143); recalibrating on the previous window exceeds α once at α = 0.10 (0.114) and once at α = 0.20 (0.233).

**Stdin rule, held-out confirmation.** On tiers not used to choose it, the stress rule lifts AUROC for stdin candidates from 0.771 to 0.879 (14B, 2,446 candidates; 0.894 with self-confidence), 0.771 to 0.874 (32B, 2,746; 0.906 with self-confidence), 0.749 to 0.845 (DeepSeek 1.3B, 737) and 0.761 to 0.880 (DeepSeek 6.7B, 1,305) (`python scripts/analysis_stdin_signals.py`, `results/x2_stdin_signals.json`).

**Published baselines on LiveCodeBench** (7B to 14B): infeasible at every α (no setting reaches α ≤ 0.20 on any calibration half), since they always answer and the models are wrong on most LiveCodeBench problems.

## 20 Oct: published baselines against CodeLadder (HumanEval+ and MBPP+, 7B to 14B)

Commands: `codeladder make-selftests` / `make-traces` (generation against the 7B and 14B servers), `codeladder baseline-exec` (self-tests run against the first 8 pool samples; greedy traces labelled), `codeladder baselines --benchmark humaneval mbpp --tiers t2 t3 --alphas 0.05 0.1 0.2`; CodeLadder with `evaluate --tiers t2 t3 --score agreement_selfconf --family <f> --abstain-factor 4`. Same 100 random calibration/test halves, same weighted-token costs. The baselines' parameters (theta for Model Cascading for Code over k in {3, 5, 8}; tau for Bayesian self-escalation, with its logistic belief refit on every calibration half) are tuned as the original baselines do: the cheapest setting whose calibration error rate is at most α. Neither can abstain.

| Method | α = 0.05 | α = 0.10 | α = 0.20 | Splits with true risk above α |
| --- | --- | --- | --- | --- |
| CodeLadder, single-model selection | 29.0% (10.4k) | **69.1%** (6.3k) | 81.6% (**2.8k**) | 0% |
| CodeLadder, sub-ladders | 19.4% (9.0k) | 66.3% (7.0k) | 81.9% (3.2k) | 0% |
| CodeLadder, two-level ladder | 15.8% (23.4k) | 68.7% (12.4k) | **83.8%** (4.9k) | 0% |
| Model Cascading for Code (tuned theta) | infeasible | infeasible | 79.7% (6.3k) | **56%** of the 97 tuned splits |
| Bayesian self-escalation (tuned tau) | infeasible | infeasible | 75.4% (1.3k) | **100%** of the 24 tuned splits |

(Solved share and mean cost; "infeasible": no setting met α on any calibration half, because answering every task leaves the 7B/14B error floor of about 20% on MBPP+.)

All tier pairs and the full ladder (`codeladder baselines --tiers ...`; tuned at α = 0.20 where feasible; neither baseline is feasible at α = 0.05 or 0.10 on any tier set, nor at any α on LiveCodeBench):

| Benchmark, tiers | MCC at α = 0.20: splits tuned, true risk above α, solved, cost | BSE at α = 0.20 | Untuned reference |
| --- | --- | --- | --- |
| HumanEval+/MBPP+, 1.5B to 32B (4 tiers) | 74, **66%**, 78.7%, 15.3k | 74, **55%**, 78.6%, 2.5k | MCC most accurate: 80.4% solved, risk 0.196, 21.4k; greedy 32B: 80.6%, 3.3k |
| HumanEval+/MBPP+, 1.5B to 32B | 72, **81%**, 78.6%, 20.1k | 63, **51%**, 78.7%, 3.1k | |
| HumanEval+/MBPP+, 14B to 32B | 95, 0%, 80.1%, 11.5k | 87, **52%**, 79.3%, 2.4k | |
| LiveCodeBench, 4 tiers | infeasible | infeasible | MCC most accurate (444 LeetCode-style tasks): 56.4% solved, risk 0.436, 151.2k; greedy per tier 9.0% / 23.6% / 36.4% / 42.1% at 0.3k / 1.6k / 4.2k / 8.5k |

**The cost of reliability.** On LiveCodeBench the 32B greedy answer solves 42.1% of tasks for 8.5k per task but is wrong on 58% of them; CodeLadder's calibrated 32B solves 42.8% with accepted-but-wrong at most 10% (it abstains on the rest) for 65.0k per task, about 7.6 times the cost. On HumanEval+/MBPP+ the same comparison is 80.6% at 3.3k (19% wrong) against 82.9% at 6.5k (at most 20% wrong, α = 0.20) or 69.0% at 8.7k (α = 0.10). Without abstention neither published method can operate below the models' error floor, and where it can, tuning on calibration data without a finite-sample correction exceeds α in most splits. Bayesian self-escalation is the cheapest per task (one greedy generation, early aborts) but the least reliable.

## 19 Oct: first LiveCodeBench results (1.5B and 7B), temporal shift, transfer

Pools: 1.5B 1,054 tasks, 7B 1,053 (lcb/3566 failed: a repair prompt overflowed the 8k context; feedback is now clipped and the pool was regenerated). Agreement uses the stdin rule above. All runs `--alphas 0.05 0.1 0.2`, violations 0 to 6% of splits (δ = 10%).

| Setting (agreement x self-confidence, κ = 4) | α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- |
| LiveCodeBench, 7B alone | 22.8% (12.1k) | **29.4%** (11.3k) | 35.8% (10.6k) |
| LiveCodeBench, 1.5B alone | 10.8% (2.5k) | 16.4% (2.5k) | 16.4% (2.5k) |
| LiveCodeBench, 1.5B to 7B two-level | 19.6% (14.8k) | 23.9% (14.0k) | 34.8% (12.9k) |
| LiveCodeBench, singles / subsets | 10.2% / 10.4% | 16.8% / 16.7% | 35.1% / 35.1% |
| Pooled (1,595 tasks), 7B alone | **34.8%** (9.6k) | **44.7%** (8.8k) | 51.4% (8.0k) |
| Pooled, two-level | 32.6% (11.5k) | 41.4% (10.1k) | 50.8% (9.1k) |
| Pooled, singles / subsets | 25.8% / 25.1% | 34.3% / 33.7% | 51.1% / 51.1% |

With only 1.5B and 7B the ladder does not pay on LiveCodeBench: the 1.5B model solves 8% on the first try, so the cheap rung mostly adds cost. At κ = 4 the singles and subsets families often choose the cheap 1.5B setting with low coverage; on LiveCodeBench, where coverage is low, κ = 4 under-prices abstention (the κ sweep followed: "sub-ladders, calibrated model selection and the price of abstaining"). Learning curves (pooled) still rise between 300 and 500 calibration tasks at α = 0.05.

Signals, 7B alone on LiveCodeBench at α = 0.10: agreement x self-confidence 29.4%, agreement 27.5%, log-probability 23.7%, self-confidence alone 1.7%.

**E5 temporal shift** (`evaluate --benchmark livecodebench --tiers t1 t2 --shift 5`; problems ordered by release date, 6 windows): thresholds calibrated once on the oldest window keep risk at or below α in every later window (α = 0.10: risk 0.040 to 0.055; α = 0.05: 0.022 to 0.032; α = 0.20: 0.114 to 0.153). Recalibrating on the previous window only (same size, about 175 tasks) exceeds α in 2 of 5 windows at α = 0.10 (0.122, 0.112): the smaller calibration window costs more than the shift.

**X6 transfer** (`--calib-on humaneval mbpp`: calibrate on HumanEval+/MBPP+, test on LiveCodeBench): the target risk stays far below α (two-level: 0.2%, 1.6%, 14.4% at α = 0.05, 0.10, 0.20; no violations), but LiveCodeBench coverage collapses (13.3% solved at α = 0.10 against 23.9% when calibrated on LiveCodeBench). The guarantee transferred conservatively here; it is not guaranteed in general.

## 19 Oct: X2 stress inputs for stdin problems (LiveCodeBench, 1.5B and 7B)

`codeladder make-stress --tier t3` (Qwen 14B writes one input generator per stdin problem from its statement and constraints; one request to shrink an unusable output; first-line format check): **537 of 610 stdin problems (88%) get stress inputs** (258 one, 279 two; mean about 100k characters), for 1.94M generator tokens in total (about 3.2k per problem, a one-off harness cost). Candidates are run on them with the problem's own time limit (`add-probes --stress`).

Of the visible-passing candidates that time out on a stress input, 93 of 103 (1.5B) and 183 of 229 (7B) are hidden-wrong: a stress timeout is a strong, label-free failure signal. AUROC for separating hidden-correct from hidden-wrong visible-passing stdin candidates (10 samples per task):

| Score on stdin problems | 1.5B (876 cand.) | 7B (1,742 cand.) |
| --- | --- | --- |
| share passing the visible tests | 0.661 | 0.735 |
| agreement on mutated public inputs | 0.677 | 0.705 |
| agreement on mutated inputs + stress inputs | 0.693 | 0.730 |
| agreement on stress inputs only | 0.687 | 0.759 |
| visible share, stress timeout = fail | 0.771 | 0.841 |
| **agreement on stress inputs, stress timeout = fail** | **0.797** | **0.858** |

Adopted rule (`scoring.checks_ok`, `scoring.agreement_key`): on stdin problems a stress timeout counts as a failed check, agreement clusters on stress-input outputs, and mutated stdin probes are dropped (they split correct programs at 7B). The rule was chosen on the 1.5B and 7B pools; 14B, 32B and the DeepSeek tiers serve as its held-out confirmation.

## 19 Oct: X5 calibration learning curve (HumanEval+ and MBPP+, 1.5B / 14B / 32B, agreement x self-confidence, κ = 4)

Command: `codeladder evaluate --benchmark humaneval mbpp --tiers t1 t3 t4 --score agreement_selfconf --family <f> --abstain-factor 4 --calib-sizes 50 100 150 200 --alphas 0.05 0.1 0.2 --tag=-k4-lc` (same 100 splits and test halves; only the number of calibration tasks changes; 271 is the full calibration half). Solved share / mean cost; "none" means no setting was certified in any split.

| Family | α | 50 | 100 | 150 | 200 | 271 |
| --- | --- | --- | --- | --- | --- | --- |
| singles | 0.05 | none | 6.9% / 4.9k | 25.8% / 10.1k | 34.0% / 12.0k | 37.5% / 13.6k |
| subsets | 0.05 | none | 7.1% / 5.1k | 11.4% / 7.2k | 12.7% / 7.1k | 30.2% / 15.3k |
| two-level | 0.05 | none | 6.6% / 51.8k | 22.0% / 48.9k | 19.5% / 49.5k | 35.5% / 46.7k |
| singles | 0.10 | 25.7% / 8.2k | 58.2% / 7.5k | 65.1% / 7.1k | 66.7% / 7.5k | 67.1% / 7.2k |
| subsets | 0.10 | 27.5% / 8.8k | 43.7% / 13.0k | 60.8% / 8.6k | 62.5% / 8.4k | 68.1% / 8.8k |
| two-level | 0.10 | 18.6% / 48.7k | 53.6% / 34.3k | 58.0% / 30.3k | 61.7% / 26.9k | 67.9% / 21.1k |
| singles | 0.20 | 76.0% / 7.7k | 80.9% / 7.0k | 82.3% / 7.2k | 82.5% / 7.1k | 82.9% / 7.2k |
| subsets | 0.20 | 74.8% / 8.5k | 81.3% / 5.7k | 82.2% / 5.3k | 82.7% / 4.3k | 82.8% / 4.0k |
| two-level | 0.20 | 75.6% / 16.3k | 81.3% / 9.9k | 82.4% / 7.6k | 82.5% / 7.1k | 82.6% / 6.6k |

**Interpretation.** Single-model selection saturates by about 150 calibration tasks at α = 0.10 and 0.20. The richer families are still improving at 271: sub-ladders gain solved share at α = 0.10 (62.5% to 68.1% from 200 to 271) and keep cutting cost at α = 0.20 (5.3k to 4.0k from 150 to 271), and the two-level ladder's cost falls steadily. At α = 0.05 every family is still data-limited. The price of flexibility is a finite-sample effect, so pooled calibration with LiveCodeBench (about 800 calibration tasks) is the test of whether sub-ladders overtake model selection.

## 19 Oct: sub-ladders, calibrated model selection and the price of abstaining (HumanEval+ and MBPP+)

Two new calibrated families (`evaluate --family subsets|singles`): **subsets** contains every sub-ladder (any non-empty subset of the tiers, in order; one Bonferroni path per subset and escalation threshold), **singles** only one-tier ladders (calibrated model selection: the fair version of "pick the best single model", which pays the same Bonferroni price per model). Choosing the cheapest certified setting is degenerate once cheap sub-ladders exist (a 1.5B ladder that never accepts is certified and nearly free), so settings are now chosen by cost + c x abstention rate on the calibration tasks, with c = κ x the mean cost of the costliest setting (`--abstain-factor κ`; κ = 0 is the old rule). The choice among certified settings does not affect the guarantee.

Solved share and mean cost (violations 0 to 1% of splits everywhere). Agreement x self-confidence, tiers 1.5B, 14B, 32B (`--tiers t1 t3 t4 --score agreement_selfconf --family <f> --abstain-factor κ --tag=-kκ`):

| κ | Family | α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- | --- |
| 0 | singles | 11.4% (1.2k) | 56.1% (0.8k) | 74.1% (0.6k) |
| 0 | subsets | 4.0% (1.3k) | 53.8% (0.9k) | 73.5% (0.6k) |
| 2 | singles | 37.5% (13.6k) | 64.8% (5.3k) | 75.1% (1.4k) |
| 2 | subsets | 29.5% (14.4k) | 65.4% (6.6k) | 82.7% (3.8k) |
| 4 | singles | **37.5%** (13.6k) | 67.1% (7.2k) | **82.9%** (7.2k) |
| 4 | subsets | 30.2% (15.3k) | **68.1%** (8.8k) | 82.8% (**4.0k**) |
| any | two-level (all tiers) | 35.5% (46.7k) | 67.7% (20.6k) | 82.6% (6.6k) |

Agreement only, all four tiers (`--tiers t1 t2 t3 t4 --score agreement`):

| κ | Family | α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- | --- |
| 2 | singles | 11.4% (1.0k) | 66.3% (3.8k) | 81.5% (2.3k) |
| 2 | subsets | 1.1% (1.1k) | 59.4% (3.4k) | 81.6% (2.2k) |
| 4 | singles | 11.4% (1.0k) | 69.0% (6.0k) | 81.7% (2.5k) |
| 4 | subsets | 1.1% (1.1k) | 60.3% (3.9k) | 82.3% (3.0k) |

All four tiers, agreement x self-confidence (`--tiers t1 t2 t3 t4 --score agreement_selfconf`; 7B annotated on 19 Oct afternoon):

| κ | Family | α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- | --- |
| 2 | singles | 41.1% (12.9k) | 65.8% (4.4k) | 81.7% (2.8k) |
| 2 | subsets | 23.1% (12.3k) | 61.2% (5.2k) | 81.6% (2.5k) |
| 4 | singles | **41.4%** (13.4k) | **67.9%** (6.4k) | 81.8% (2.9k) |
| 4 | subsets | 25.5% (14.6k) | 64.6% (8.0k) | 82.3% (3.4k) |
| any | two-level | 35.7% (53.4k) | 51.8% (43.0k) | 83.2% (5.9k) |

Settings chosen by the sub-ladder family (κ = 4, 100 splits): at α = 0.05 single tiers (7B 44, 32B 36, 14B 16); at α = 0.10 mostly 1.5B to 14B (40) or 14B / 7B alone; at α = 0.20 1.5B to 7B to 14B (32) or 1.5B to 7B (32). 7B alone with the combined score: 6.1% / 68.6% / 81.4%. With four tiers the sub-ladder family tests 37 paths against 4 for single-model selection, and the flexibility costs more than it gains on 271 calibration tasks.

**Interpretation.** With a price on abstaining, sub-ladders match calibrated model selection at α = 0.10 and solve the same share at about half the cost at α = 0.20; the all-tier two-level ladder costs up to 3 times more than either. At α = 0.05 model selection wins: the sub-ladder family tests 15 paths instead of 3, and with 271 calibration tasks the stricter per-path level costs coverage. This is the price of flexibility, which should shrink with LiveCodeBench's 1,054 tasks. Headline settings from now on: κ = 4, families singles / subsets / two-level side by side.

## 19 Oct: 32B tier and four-tier ladders, HumanEval+ and MBPP+ pooled (542 tasks)

Pools (`scripts/report_pools.py`; energy is busy-time J per output token, see `scripts/recompute_energy.py`):

| Benchmark | Tier | pass@1 | Any of 20 correct | Visible-pass | Silent failures | J/token |
| --- | --- | --- | --- | --- | --- | --- |
| HumanEval+ | 14B | 86.3% | 93.3% | 90.1% | 4.2% | 1.71 |
| HumanEval+ | 32B | 85.6% | 92.1% | 89.1% | 3.9% | 5.15 |
| MBPP+ | 14B | 74.1% | 88.1% | 86.4% | 14.3% | 1.52 |
| MBPP+ | 32B | 77.1% | 85.2% | 89.9% | 14.3% | 3.46 |

32B is no better than 14B on HumanEval+, slightly better on MBPP+, keeps the MBPP+ floor of silent failures, and costs 2 to 3 times the energy per token (4 GPUs at low throughput).

Calibrated results (`codeladder evaluate --benchmark humaneval mbpp --tiers ... --score ... [--family two_level] --alphas 0.05 0.1 0.2`; solved share of all tasks, mean weighted-token cost per task; violations 0 to 4% of splits everywhere, below δ = 10%):

| Setting | α = 0.05 | α = 0.10 | α = 0.20 |
| --- | --- | --- | --- |
| 32B alone, agreement | 0.0% | 32.9% (21.1k) | 83.0% (11.5k) |
| 32B alone, agreement x self-confidence | **40.5%** (26.2k) | 56.4% (21.7k) | 83.0% (13.9k) |
| 14B alone, agreement x self-confidence | 30.2% (15.1k) | **72.6%** (9.7k) | 83.7% (7.6k) |
| 7B alone, agreement | 0.6% | 68.5% (3.1k) | 81.3% (**2.4k**) |
| 1.5B, 7B, 14B, 32B, agreement, two-level | 0.0% | 19.7% (42.5k) | 83.4% (5.1k) |
| 14B, 32B, agreement x self-confidence, two-level | 35.8% (45.0k) | 67.2% (21.9k) | 83.5% (11.2k) |
| 1.5B, 32B, agreement x self-confidence, two-level | 36.8% (26.4k) | 64.5% (14.4k) | 82.2% (6.2k) |
| 1.5B, 14B, 32B, agreement x self-confidence, two-level | 35.5% (46.7k) | 67.7% (20.6k) | 82.5% (6.6k) |

**Interpretation.** The guarantee holds for every ladder. But at each α the best certified result comes from a single model, and which one changes with α (32B at 0.05, 14B at 0.10, 7B for cost at 0.20). Ladders beat the top model on cost, not the best single model. The cause is the calibrated family: shared and two-level ladders always use every tier, so Learn-then-Test cannot choose "14B alone". Agreement alone is also too weak for the larger models (32B at α = 0.10 certifies 32.9% with agreement against 56.4% with the combined score). Next: the sub-ladder family below.

## 19 Oct: stdin probes for agreement (LiveCodeBench, 1.5B pools)

LiveCodeBench stdin problems (610 of 1,054) had no probe inputs, so agreement on them reduced to the share of samples passing the public tests. Probes now mutate the public inputs while keeping their format (`probes.make_stdin_probes`): 588 problems get 20 probes, 19 get fewer, 3 get none; no generator failures over 5 seeds of every input. Backfilled with `codeladder add-probes --tier t1 --benchmark livecodebench` (982 candidates, 2 minutes on 4 CPUs).

AUROC of the score for separating hidden-correct from hidden-wrong visible-passing candidates (10 samples per task):

| Problems | Candidates (correct) | Visible-pass share | Agreement with probes |
| --- | --- | --- | --- |
| stdin (192 tasks) | 876 (632) | 0.661 | 0.677 |
| LeetCode-style, argument probes (96 tasks) | 273 (205) | 0.670 | 0.756 |

Stdin probes help a little; they split 40 of the 192 tasks into more than one behaviour cluster. New LiveCodeBench pools (7B, 14B, 32B) get stdin probes at generation time.

Why the gap: first hidden-test failure of the 1.5B samples that pass the public tests but are wrong (all LiveCodeBench tasks):

| Problems | Silent failures | Too slow (timeout) | Wrong output | Crash |
| --- | --- | --- | --- | --- |
| stdin | 244 | 56% | 43% | 1% |
| LeetCode-style | 68 | 32% | 65% | 3% |

Most stdin silent failures are correct-but-too-slow programs. Probes derived from small public inputs cannot expose them, whatever the mutation. Simple size-scaling probes (single count line followed by an array, a string or N records, scaled up to the limit stated in the problem) would apply to only 91 of the 610 stdin problems and 21 of the 90 problems with 1.5B silent failures. A general fix needs per-problem input generators (for example written once per problem by a model from the statement); probes are recomputed from stored code on CPU, so no regeneration is needed.

## 19 Oct: E7 ablation, repair depth (2 versus 3 rounds), HumanEval+ and MBPP+ pooled

Command: `codeladder evaluate --benchmark humaneval mbpp --tiers <tiers> --action <repair|adaptive> --score agreement --alphas 0.1 0.2 --max-repair-depth <2|3> --tag=-depth<2|3>` (`--max-repair-depth` replays the pools with chains cut to that many rounds, which is equivalent to generating with that depth).

| Tiers, action | α | Solved, 2 rounds | Solved, 3 rounds | Cost change (3 vs 2 rounds) |
| --- | --- | --- | --- | --- |
| 7B, repair | 0.10 / 0.20 | 67.0% / 81.2% | 67.0% / 81.2% | +0.6% / +0.2% |
| 14B, repair | 0.10 / 0.20 | 67.3% / 83.1% | 67.3% / 83.0% | +0.3% / +0.1% |
| 1.5B, 7B, 14B, repair | 0.10 / 0.20 | 63.4% / 83.4% | 63.4% / 83.3% | +0.6% / +1.4% |
| 7B, 14B, ladder, adaptive | both | identical | identical | 0 |

The third round never changes the calibrated outcome: with at most 8 draws per tier, the controller rarely follows one chain to round 3. Raw counts agree it is a small source of fixes (6 to 19% of all repair fixes, with a higher share of silent failures). Decision: LiveCodeBench keeps 2 repair rounds; `generate --extend` can add a round later if the larger tiers show strong repair on LiveCodeBench. Repair-only and adaptive policies also do not beat resampling here (7B at α = 0.10: repair 67.0%, adaptive 68.5%, resample 68.5%).

## 19 Oct: 14B tier, confidence signals, per-benchmark calibration

**14B pools** (`report_pools.py`): HumanEval+ pass@1 86.3%, silent failures 4.2%; MBPP+ pass@1 74.1%, silent failures 14.3%; 1.77 / 1.53 J per output token. MBPP+'s silent-failure rate stays at about 14% for every model size (1.5B 13.6%, 7B 14.9%, 14B 14.3%): weak visible tests plus under-specified problems leave confidently wrong answers that larger models do not fix.

**The risk floor is MBPP+-specific.** 14B alone, calibrated, agreement score:

| Benchmark | α = 0.05 | α = 0.10 | α = 0.20 | Uncalibrated risk |
| --- | --- | --- | --- | --- |
| HumanEval+ | 36.8% | 88.7% | 90.1% | 4.4% |
| MBPP+ | 0% | 34.7% | 79.8% | 13.8% |
| Pooled | 0% | 68.3% | 83.6% | 10.9% |

**Bigger models make agreement a weaker signal.** At α = 0.05 (pooled, agreement), 1.5B alone certifies 27.6% solved but 7B and 14B certify almost nothing: large models are unanimous more often, including when they are consistently wrong.

**Confidence signals** (14B alone, solved at α = 0.05 / 0.10 / 0.20, pooled): agreement 0% / 68.3% / 83.6%; log-probability 1.3% / 2.9% / 84.4%; self-confidence 1.0% / 56.0% / 83.6%; **agreement x self-confidence 30.2% / 72.6% / 83.7%** (best).

**Ladder 1.5B to 14B, agreement x self-confidence, two thresholds** versus 14B alone with the same score:

| α | Ladder solved | Ladder cost | 14B solved | 14B cost |
| --- | --- | --- | --- | --- |
| 0.05 | 21.7% | 16,505 | 30.2% | 15,059 |
| 0.10 | 68.6% | 7,526 | 72.6% | 9,705 |
| 0.20 | 82.7% | 3,607 | 83.7% | 7,638 |

The ladder trades a few points of solved rate for 22 to 53% lower cost at α of 0.10 and above.

**Per-benchmark (Mondrian) calibration** (each benchmark calibrated separately at δ/2) lowers coverage at this sample size: MBPP+ alone cannot certify anything and abstains, e.g. 14B at α = 0.10 goes from 68.3% to 36.0% solved. Pooled calibration stays the default; group-level guarantees need more data per group.

The guarantee held in every run (true-risk violations 0 to 2% of splits, δ = 10%).

## 19 Oct: two-tier ladder (1.5B to 7B), HumanEval+ and MBPP+ pooled (542 tasks)

Command: `codeladder evaluate --benchmark humaneval mbpp --tiers t1 t2 [--family shared|two_level]`, 100 calibration/test splits (50/50), δ = 0.1, agreement score, resample policy.

**Guarantee.** The chosen setting's true risk (risk over the whole pool) exceeded α in at most 3% of splits in every configuration tried, below δ = 10%.

**Calibrated results at α = 0.10** (solved = accepted and correct; cost = weighted output tokens per task):

| Method | Coverage | Solved | Cost |
| --- | --- | --- | --- |
| Ladder, shared threshold, 8 draws per tier | 65.7% | 59.2% | 3,456 |
| Ladder, shared threshold, 3 draws per tier | 60.7% | 55.3% | 1,787 |
| Ladder, two thresholds, Bonferroni paths, 8 draws | 66.3% | 60.2% | 3,200 |
| Ladder, two thresholds, Pareto Testing, 8 draws | 56.1% | 51.1% | 3,619 |
| Calibrated 1.5B alone | 67.7% | 60.6% | 685 |
| Calibrated 7B alone | 75.4% | 68.5% | 3,134 |
| Uncalibrated 7B alone (risk 12.2%, violates α) | 92.0% | 79.8% | 2,046 |

At α = 0.20 the shared-threshold ladder matches 7B alone (81.3% solved) at half the cost (1,106 vs 2,352).

**Achievable frontier** (`scripts/analysis_frontier.py`, best setting by true risk, ignoring calibration):

| True risk cap | Best solved | Cost | Setting (1.5B threshold, 7B threshold) |
| --- | --- | --- | --- |
| 0.07 | 69.0% | 4,235 | (never, 0.7) |
| 0.085 | 72.5% | 2,648 | (0.7, 0.7) |
| 0.10 | 78.5% | 2,725 | (1.0, 0.5) |

**Interpretation.** With 271 calibration tasks, Learn-then-Test can only certify settings whose true risk is about 7% or lower at α = 0.10, so the certifiable frontier is roughly "7B alone". The ladder's advantage at this sample size is cost, and the gap between the achievable frontier (78.5%) and what is certified is the price of a finite-sample guarantee. Two things should shrink the gap: more calibration data (LiveCodeBench adds about 1,050 tasks) and stronger top tiers (14B, 32B) with fewer errors. Pareto Testing loses power here because half the calibration set only orders the configurations.

## 19 Oct: E4 repair versus resample (1.5B, 7B)

Command: `python scripts/analysis_repair.py`. Success on hidden tests of one repair round versus a fresh sample *on the same tasks*:

| Tier | Failure shown | Repair succeeds | Fresh sample succeeds |
| --- | --- | --- | --- |
| 1.5B | wrong answer | 2.7% | 23.6% |
| 1.5B | runtime error | 14.0% | 26.3% |
| 7B | wrong answer | 13.8% | 14.2% |

Resampling dominates at 1.5B; at 7B the two are level.

## 19 Oct: candidate pools (HumanEval+ and MBPP+)

Command: `python scripts/report_pools.py`.

| Benchmark | Tier | pass@1 | Any of 20 correct | Silent failures | J per output token |
| --- | --- | --- | --- | --- | --- |
| HumanEval+ | 1.5B | 57.5% | 86.6% | 8.6% | 0.56 |
| HumanEval+ | 7B | 81.7% | 92.7% | 5.9% | 1.11 |
| MBPP+ | 1.5B | 57.8% | 81.7% | 13.6% | 0.54 |
| MBPP+ | 7B | 70.0% | 87.0% | 14.9% | 0.97 |

Silent failures: share of visible-test-passing candidates that fail the hidden tests.
