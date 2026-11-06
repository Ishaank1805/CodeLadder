# CodeLadder

Risk-controlled code generation with model ladders. CodeLadder writes code with free models (Qwen2.5-Coder 1.5B, 7B, 14B and 32B; DeepSeek-Coder 1.3B, 6.7B and 33B; and gpt-oss-120b on a free API). At each step it decides whether to **accept**, **repair**, **resample**, **escalate** or **abstain**, and its thresholds are calibrated with Learn-then-Test so that, with probability at least 1 - δ, at most α of tasks end with code that was accepted but is wrong, at minimum cost.

Team Byte Me: Ekansh (2023114007), Ishaan (2023114016), Ayush (2023114001). Course project for Autonomous Software Engineering, Monsoon 2026, IIIT Hyderabad.  
Repository: [https://github.com/Ishaank1805/CodeLadder](https://github.com/Ishaank1805/CodeLadder)

## Main findings (6 Nov mid-submission)

E and X numbers are the experiments and extensions in [docs/PLAN.md](docs/PLAN.md); [docs/RESULTS.md](docs/RESULTS.md) defines the terms used here.

- The guarantee holds: over 311 evaluation runs (1,536 calibrations), at most 5% of splits exceed α against δ = 10%. It is marginal, not conditional: pooled calibration leaves hard LiveCodeBench problems above α unless each difficulty is calibrated on its own.
- Ladders pay only as far as the cheap rung certifies tasks at α (the cheap-share law, X13): on HumanEval+/MBPP+ at α = 0.20 a 14B to 32B ladder costs a third of 32B alone at a 4x price gap and 3% at large gaps, about half the cost of model selection there (a fifth with a 120B API top rung, bootstrap intervals in X14), nothing on LiveCodeBench at any price. Otherwise calibrated selection of one model is as good. A learned up-front router (X9), a learned repair-or-resample policy (X4), mixing model families (X10), Pareto Testing and lean lower rungs do not change this.
- Published cascades (Model Cascading for Code, Bayesian self-escalation) cannot abstain and exceed their risk target in 51 to 100% of splits (in all but one tier set).
- Against guaranteed baselines (X16), a Trust-or-Escalate cascade adapted to code solves 10 to 41 points less at 2 to 6 times the cost. The certificate is only as strong as the test oracle (X17): with the original HumanEval / MBPP tests it is vacuous (true risk 11 to 13% where 0 is claimed).
- Model-written stress inputs ("stress timeout = failed check") are a new label-free signal for stdin problems (AUROC +0.08 to 0.14 on every tier); verbalised self-confidence is what lets strong models accept at strict α, and DeepSeek-Coder's carries no signal.

Details, numbers and commands: [docs/RESULTS.md](docs/RESULTS.md) (summary at the top).

## Documents

| File | Contents |
| --- | --- |
| [docs/RESULTS.md](docs/RESULTS.md) | summary of findings, the proposal's hypotheses with their outcome, a map from each finding to its figure and table; then every result with the command that produced it, newest first |
| [docs/REPORT_PLAN.md](docs/REPORT_PLAN.md) | outlines of the mid-submission report and the final project report, with the evidence for each section |
| [docs/PLAN.md](docs/PLAN.md) | research questions, experiments E1 to E8 and extensions X1 to X18 with their status |
| [docs/STATUS.md](docs/STATUS.md) | operational handoff: cluster layout, running jobs, next actions, cluster lessons |
| [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | closest prior work, how CodeLadder differs, defensible contributions (working notes) |
| [docs/PREFLIGHT.md](docs/PREFLIGHT.md) | checks run before the main experiments |

> **Status (6 Nov 2026):** Mid-submission milestone complete. All 150 unit and integration tests passing, evaluation pipeline, offline replay, report figures/tables, and interactive demo verified.

## How it works

1. **Generate once.** For every task and model, draw candidates and build repair chains for failing ones. Run every candidate in a sandbox on the visible tests, on probe inputs (mutated arguments; for stdin problems, model-written stress inputs), and on the hidden tests (label only). Ask the model how confident it is in each visible-passing candidate. Everything is cached in `data/pools/`.
2. **Replay offline.** The controller replays decisions over the cached pools for any threshold, confidence score, action policy, ladder (any subset of models) or draw budget. This needs no GPU.
3. **Calibrate.** Learn-then-Test on random calibration halves: fixed-sequence testing for one threshold; Bonferroni paths or Pareto Testing for families of ladders; optionally per group (benchmark or difficulty). Among certified settings the one with the lowest cost plus a price for abstaining is chosen. Risk, cost, coverage and energy are measured on the test halves, against published baselines.

Benchmarks: HumanEval+ and MBPP+ (EvalPlus) and LiveCodeBench release_v6 (1,054 problems, LeetCode-style and stdin, with release dates and difficulty); BigCodeBench v0.1.4 (1,134 usable library-heavy tasks with unit tests, run in the benchmark's own pinned environment; pools in progress).

```
src/codeladder/
  types.py          Task, Candidate, TierPool and the other shared data types
  tasks.py          benchmark loaders and lazy task preparation (release_tests frees LiveCodeBench tests)
  executor.py       sandboxed runner (function, LeetCode method and stdin modes) and output comparison
  oracles.py        EvalPlus special-case checkers and LiveCodeBench graders
  probes.py         argument probes and format-preserving stdin probes
  stress.py         model-written stress inputs for stdin problems (make-stress)
  scoring.py        visible-test report, behaviour signatures, checks (stress timeout = fail), confidence scores
  prompts.py        initial, repair and confidence prompts; code extraction; confidence parsing
  llm.py            OpenAI-compatible client: SQLite cache, per-key rate limits, key pools, per-token statistics
  config.py         tiers and settings; API key pools from the environment or ~/.codeladder_keys
  generate.py       generation pass to pools (resumable, --extend top-up); confidence annotation; probe backfill
  controller.py     accept / repair / resample / escalate / abstain replay; tier skipping; lower-tier draw budget
  calibrate.py      Learn-then-Test, Pareto Testing, Bonferroni paths
  evaluate.py       threshold families (shared, two-level, sub-ladders, single-model selection), splits, groups,
                    conditional risk, learning curves, transfer, temporal shift, baselines
  baselines_gen.py  generation for published baselines (self-generated tests, greedy traces with token statistics)
  baselines.py      replay of Model Cascading for Code and Bayesian self-escalation on the same tasks
  router.py         learned up-front router (X9), calibrated with the same guarantee (route)
  actions.py        learned repair-or-resample policy (X4) against fixed actions (actions)
  economics.py      price sweeps and the break-even model for ladders (X13, economics)
  bootstrap.py      paired task-level bootstrap intervals between runs (X14, compare)
  guaranteed.py     guaranteed baselines: per-model abstention, Trust-or-Escalate adapted to code (X16)
  oracle.py         per-test outcomes and calibration on weaker test oracles (X17)
  bcb.py            BigCodeBench loader, unit-test execution, model-written probes (X18)
  bcb_runner.py     test runner executed by the BigCodeBench environment (signals fenced to its session)
  bcb_bin/          session-limited kill, pkill and killall put first on PATH for BigCodeBench tests
  live.py           export a calibrated policy; run it on one problem online or from pools (X15)
  demo.py           interactive demo server and page (demo_page.html), demo bundles
  energy.py         NVML energy logging, busy-time attribution from vLLM metrics
  cli.py            the `codeladder` command
configs/            models.yaml (tiers), experiment.yaml (all settings, per-benchmark and per-tier overrides)
scripts/experiments/ one batch per experiment group; regenerate results/ from the pools (all.sh runs them all)
results/             (git-ignored) runs/<benchmarks>/ evaluate runs; router/ actions/ economics/ guaranteed/ oracle/;
                    baselines/, analysis/, ci/, policies/, figures/, tables/; archive/ (old runs); layout in
                    src/codeladder/results.py, scripts/organise_results.py sorts a flat folder into it
scripts/ada/        cluster jobs: per-tier generation (run_tier.sbatch), CPU jobs (cpu_job.sbatch), waiters (after.sh)
scripts/figures/    report figures (report_figures.py), frontier plots, per-run plots; report tables: scripts/report_tables.py
scripts/            data preparation (prepare_lcb.py, prepare_bcb.py, make_subset.py, verify_wheels.py,
                    relabel_bcb.py), checks (preflight_evalplus.py, report_pools.py), analyses (analysis_*.py),
                    energy, result tables (show_results.py), guarantee count (count_validity.py), free API
                    runner (run_api_tier.sh), data packaging (package_data.sh)
tests/              unit tests, end-to-end runs with a fake model, dry runs of the experiment batches
examples/           a custom problem and exported policies for the demo
docs/               RESULTS, REPORT_PLAN, PLAN, STATUS, RELATED_WORK, PREFLIGHT
```

## Quick start (laptop, no GPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # editable: configs/ and data/ are read from this checkout
pytest                       # unit tests and fake-model end-to-end runs
```

## Generating pools (cluster)

One job per model serves it with vLLM, generates every benchmark, attaches energy, and asks for confidence ratings (see the header of `scripts/ada/run_tier.sbatch` and `docs/STATUS.md`):

```bash
sbatch --gres=gpu:1 -c 16 --export=ALL,TIER=t2,MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ,TP=1,BENCHES="humaneval mbpp livecodebench" scripts/ada/run_tier.sbatch
codeladder make-stress --tier t3                                   # stress inputs for stdin problems (once)
codeladder add-probes --tier t1 --benchmark livecodebench --stress # backfill probes for older pools
python scripts/recompute_energy.py                                 # busy-time energy for every run
TASK_IDS=data/lcb/subset_api400.json scripts/run_api_tier.sh livecodebench   # free API tier, quota-aware
codeladder annotate-confidence --tier d3 --benchmark mbpp --redo  # re-ask ratings after a prompt change (conf_prefill)
```

## Reproducing results

Every experiment replays the cached pools on CPU; each batch in `scripts/experiments/` regenerates the `results/` folders behind one part of RESULTS.md (`DRY=1` prints the commands; on the cluster run large batches as CPU jobs, `sbatch scripts/ada/cpu_job.sbatch scripts/experiments/<batch>.sh`):

| Batch | Experiments | RESULTS.md sections | Figures |
| --- | --- | --- | --- |
| `qwen_families.sh` | E1, E2, X5, X6, X8 with Qwen tiers: model selection, sub-ladders, two-level ladders, κ = 4 / 64, lower-tier budget, Pareto Testing, pooled benchmarks | four-tier ladders; LiveCodeBench with all four Qwen tiers; ladder design; X8 | `fig_validity`, `plot_frontier.py` |
| `kappa_sweep.sh` | price of abstaining κ | sub-ladders and the price of abstaining | `plot_frontier.py` |
| `calibration_data.sh` | X5 learning curves, E5 temporal shift, X6 transfer | X5; first LiveCodeBench results | `fig_shift` |
| `difficulty.sh` | X7 per-difficulty calibration and conditional risk | X7 | `fig_conditional` |
| `energy.sh` | E6 energy as the cost | E6 | `fig_energy` |
| `scores.sh` | E3 confidence scores, agreement-only runs | 14B tier, confidence signals; X1 / X10 | |
| `deepseek.sh` | X1 DeepSeek-Coder, X10 cross-family ladders | X1 / X10 corrected | `fig_ptrue` (ratings) |
| `api.sh` | E8 / X3 / X11 free API top tier | E8 / X3 | |
| `baselines.sh` | published baselines | published baselines against CodeLadder | `fig_baselines` |
| `repair.sh` | E4 repair statistics, E7 repair depth, X4 learned action policy | E4; E7; X4 | `fig_repair` |
| `router.sh` | X9 learned up-front router | X9 | `fig_router` |
| `economics.sh` | X13 price sweeps, cheap-share law | X13 | `fig_economics` |
| `bootstrap.sh` | X14 paired bootstrap intervals | X14 | `fig_ci` |
| `ptrue.sh` | X12 P(True) as the confidence signal | X12 (pending 32B) | `fig_ptrue` |
| `guaranteed.sh` | X16 guaranteed baselines | X16 | `fig_guaranteed` |
| `oracle.sh` | X17 oracle sensitivity | X17 | `fig_oracle` |
| `bigcodebench.sh` | X18: the core experiments on BigCodeBench once its pools exist (families, DeepSeek, P(True), X13, X14, X16, X9) | X18 | all, as extra panels |
| `signals.sh` | X2 stdin signals, silent-failure types, rating AUROC, guarantee count | X2; silent-failure types; summary | `fig_stdin` |

Inspecting results:

```bash
python scripts/show_results.py runs --baselines <run name>        # calibrated table, each model alone
python scripts/show_results.py route route-<...>                  # X9 (results are found by name)
python scripts/show_results.py actions actions-<...>              # X4
python scripts/show_results.py ratings livecodebench/d3           # self-confidence AUROC
python scripts/show_results.py economics economics-<...>          # X13
python scripts/show_results.py guaranteed guaranteed-<...>        # X16
python scripts/show_results.py ci results/ci/<name>.json          # X14
python scripts/count_validity.py results                          # guarantee over every run
```

Figures:

```bash
python scripts/figures/report_figures.py results     # 13 figures: see the map at the top of docs/RESULTS.md
python scripts/report_tables.py results              # report tables as Markdown and LaTeX in results/tables/
python scripts/figures/diagram.py results           # system diagram (fig_system)
python scripts/figures/plot_frontier.py results <run prefix>   # certified operating points across κ
python scripts/figures/plot_results.py results/runs/<benchmarks>/<run>   # one run: guarantee and cost against risk
```

## Interactive demo

```bash
codeladder demo-bundle --out data/demo      # once, where the benchmark data lives (the cluster)
codeladder demo                             # then open http://127.0.0.1:8765 (keys: R run, Space pause, arrow step)
```

1. **Certified policy.** *Build a ladder*: choose the benchmark, the models (Qwen, DeepSeek, mixed families, with the 120B API tier), how thresholds are set (one model, model selection, sub-ladders, two-level), the confidence score, the draws on cheaper rungs, and drag the risk level α; the server certifies the policy on the spot from the matching evaluate run in `results/` (165 settings) and shows its thresholds, what it answers and solves on its calibration tasks, and, at α = 0.05, 0.10 and 0.20, the held-out results over 100 random splits. *Saved policies* lists `export-policy` files.
2. **Problem.** Any benchmark task with cached candidates (search, difficulty filter, random pick), or your own function with visible tests.
3. **Run one problem.** **Replay** runs the controller on the cached pools; **live** asks the policy's models and is enabled only when every rung's model is reachable (the page says which are missing). Pace from narrated to instant, or step by step, with pause, step and skip to the end. The page shows each candidate (code, test results, behaviour group, ratings), the best score against the rung's certified threshold, each decision (accept, resample, repair, escalate, abstain), and finally the hidden-test verdict (ground truth the ladder never sees), the cost and the guarantee in plain words.
4. **Run many problems.** Replays the policy on 25 to all tasks and tallies answered, correct, accepted but wrong (with its running share against α), abstained, mean cost and the rung that answered; click a row to watch that problem.

Replay needs `data/demo`, `data/pools` and `results/` (`scripts/package_data.sh` bundles them). Live mode needs model servers: for local tiers `scripts/ada/serve_vllm.sbatch` on the cluster; for the API tier its keys, which stay on the cluster, so run the demo there and forward the port:

```bash
ssh -L 8765:127.0.0.1:8765 ada3 'cd codeladder && ~/venvs/codeladder/bin/codeladder demo'
```

## Using a calibrated ladder from the command line

A run's family can be certified on all of its tasks and saved as a policy; `solve` then runs it on one problem, either from the cached pools (no model needed) or online against the tiers' models (kept up with `scripts/ada/serve_vllm.sbatch`, or the API tier with keys):

```bash
codeladder export-policy humaneval+mbpp-t1-t2-t3-t4-resample-agreement_selfconf-subsets-k64-lb1 \
  --alpha 0.2 --out results/policies/evalplus-ladder-a0.2.json         # prints the guarantee
codeladder solve --policy examples/policies/evalplus-ladder-a0.2.json \
  --benchmark humaneval --task-id HumanEval/129 --replay                # decisions from the pools
codeladder solve --policy examples/policies/evalplus-ladder-a0.2.json \
  --problem examples/problem.json                                       # live, your own problem
```

`examples/policies/` holds five exported policies: on HumanEval+/MBPP+, the 1.5B to 7B to 14B ladder certified at α = 0.20 (82.5% solved at about a quarter of the cost of model selection), the ladder topped by the 120B API model at α = 0.20 (84.5% solved), model selection at α = 0.10 and the API model alone at α = 0.10; on LiveCodeBench, the 14B to 32B ladder at α = 0.20. A custom problem is a JSON file with `prompt`, `entry_point` and visible `tests` (`[{"input": [args], "output": value}]`); the guarantee only covers problems exchangeable with the calibration tasks.

## Free API tier

The `api` tier (gpt-oss-120b on Groq's free tier) reads a pool of keys from `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, ... in the environment or in `~/.codeladder_keys` (mode 600, never in git). Requests rotate across keys, a key that hits a rate limit rests for the time the server asks, and generation stops cleanly when every key's daily quota is spent; `scripts/run_api_tier.sh` resumes it. Responses are cached in `data/llm_cache/`. Only public benchmark prompts are sent.

## Reproducibility

Pools, LLM caches, energy logs, configs and the `summary.json` files under `results/` regenerate every number; `scripts/package_data.sh` bundles them for release. Large data stays out of git (see `.gitignore`).
