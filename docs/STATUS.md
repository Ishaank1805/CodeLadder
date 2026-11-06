# Operational status and handoff

Last updated 27 Oct 2026 (about 05:20 IST). Read this first when resuming work; the plan and every experiment's status are in [PLAN.md](PLAN.md), all results with their commands in [RESULTS.md](RESULTS.md) (summary of findings at the top), positioning against prior work in [RELATED_WORK.md](RELATED_WORK.md).

## Where things live

| What | Where |
| --- | --- |
| Code (source of truth) | this repository, `main` branch on GitHub (`Ekansh0301/Byte-Me`, private); CI (ruff on `src tests scripts`, pytest on Python 3.10 and 3.11, runner pinned to ubuntu-24.04) |
| Working copy on the cluster | `ada3:~/codeladder` (account `ayush.gupta`, home `/home2/ayush.gupta`, 50 GB quota, 32 GB used) |
| Python environments (cluster) | `~/venvs/codeladder` (harness), `~/venvs/vllm` (vLLM 0.6.6), `~/venvs/bcb` (BigCodeBench execution: Python 3.10 with the benchmark's pinned libraries; wheels in `~/bcb_wheels`, NLTK / TextBlob data in `~/nltk_data`) |
| Candidate pools | `ada3:~/codeladder/data/pools/<benchmark>/<tier>/*.json`; copy of HumanEval+, MBPP+, LiveCodeBench pools on the laptop for the demo |
| LLM response caches | `ada3:~/codeladder/data/llm_cache/<tier>-<benchmark>[-conf\|-ptrue\|-stress\|-selftests\|-traces\|-probes].sqlite`; the garbage 32B P(True) caches are kept in `data/llm_cache/broken/` |
| Benchmark data | LiveCodeBench `data/lcb/` (index, tests, stress inputs, `subset_api400.json`); BigCodeBench `data/bcb/` (`tasks.jsonl`, `hard_ids.json`, `subset_api400.json`, `probes.json`, `excluded.json` and reasons; `scripts/prepare_bcb.py` regenerates the first three) |
| Demo data | `data/demo/<benchmark>.pkl.gz` (`codeladder demo-bundle`), on the cluster and the laptop; policies in `examples/policies/` (five, tracked) and `results/policies/` |
| Per-test outcomes (X17) | `ada3:~/codeladder/data/oracle/{humaneval,mbpp}/t1..t4.json.gz` |
| Model weights | `~/hf_models/` (DeepSeek 1.3B, 6.7B; DeepSeek 33B is a symlink to `/share1/ayush.gupta/hf_models`), else per node in `/scratch/$USER/codeladder/models`, verified against the Hub once per node (`.verified`) |
| API keys | `ada3:~/.codeladder_keys` (`GROQ_API_KEY_1..10`, keys 1 to 6 set and verified, mode 600; never in git, never printed) |
| Job logs | `ada3:~/codeladder/logs/cl-tier-<job>.out` (GPU jobs), `logs/cl-cpu-<job>.out` (CPU jobs); vLLM logs and energy CSVs in `ada3:~/codeladder_logs/`; job chain record `~/codeladder_logs/chain.txt` |
| Results and figures | `ada3:~/codeladder/results/`, grouped by kind (`src/codeladder/results.py`; tools find a result by its name): `runs/<benchmarks>/` evaluate runs `<benchmarks>-<tiers>-<action>-<score>-<family><tag>/`; `router/` (X9), `actions/` (X4), `economics/` (X13), `guaranteed/` (X16), `oracle/` (X17); `baselines/` (published cascades); `analysis/` (signal, repair, silent-failure and preflight outputs, `signals_auroc*.json` for X12); `ci/` (X14); `policies/`; `figures/` (14 report figures); `tables/` (report tables, Markdown and LaTeX); `archive/` (runs no current batch produces, and the runs on the corrupt 32B P(True) under `archive/stale_ptrue32/`, to delete after 40599; README inside). Git-ignored; copy to the laptop with `rsync -az ada3:codeladder/results/ results/` from the repo folder (copy of 27 Oct on the laptop); `scripts/organise_results.py` sorts a flat copy into the layout |
| Experiment batches | `scripts/experiments/*.sh` (one per experiment group; `DRY=1` lists the commands); old one-off batches in `ada3:~/codeladder/scratch_archive/` (not in git) |

## Pools (27 Oct, 00:40)

| Tier | Model | HumanEval+ | MBPP+ | LiveCodeBench | BigCodeBench (1,134 tasks) |
| --- | --- | --- | --- | --- | --- |
| t1 | Qwen2.5-Coder 1.5B | 164 | 378 | 1,054 | 343 (job 40734, held, resumes) |
| t2 | Qwen2.5-Coder 7B AWQ | 164 | 378 | 1,054 | 738 (job 40079 running) |
| t3 | Qwen2.5-Coder 14B AWQ | 164 | 378 | 1,054 | queued (40082) |
| t4 | Qwen2.5-Coder 32B AWQ | 164 | 378 | 1,054 | queued (40083) |
| d1 | DeepSeek-Coder 1.3B | 164 | 378 | 1,054 | 637 (job 40080 running) |
| d2 | DeepSeek-Coder 6.7B AWQ | 164 | 378 | 1,054 | queued (40081) |
| d3 | DeepSeek-Coder 33B AWQ | 164 | 378 | 1,054 | queued (40084) |
| api | gpt-oss-120b (Groq free tier) | 164 | 378 | 400-problem subset done (ratings running); all 1,054 next | 400-task subset after the local pools |

Local tiers on HumanEval+, MBPP+ and LiveCodeBench are complete: verbalised self-confidence (DeepSeek with a prefilled answer), P(True) (all but 32B, which is being redone), stress signatures, energy, baseline generations.

## Running now

| What | Details | Expected |
| --- | --- | --- |
| X18 BigCodeBench pools | Memory decides parallelism: the partition allows 2.5 GB per CPU, so a 60 GB job takes 24 CPUs and a 40 GB job 16 (cap 40). Running: t2 40079 (60 GB) and d1 40080 (40 GB). Held until the 32B P(True) job starts (waiter pid 599250, `logs/x12-first-waiter.log`): t1 40734 (60 GB), d2 40081 (40 GB), t3 40082 (60 GB, 2 GPUs, now after d1 only). Then t4 40083 after t3, d3 40084 after t4 (4 GPUs each, run alone) Top-ups 41229 (t2) and 41230 (d1), 40 GB each, after 40598: the tasks whose long visible test overflowed the 8,192-token context (BigCodeBench/214, /242), now fitted by the client (27 Oct) | t2 about 05:00 and d1 about 08:00 on 27 Oct; t1 and d2 27 to 28 Oct; t3 about 28 Oct; t4 about 30 Oct; d3 about 31 Oct (big-model rates are estimates) |
| X12 32B P(True), then evaluation | 40598 (4 GPUs; runs once t2 and d1 finish, about 2.5 h), then `scripts/experiments/ptrue.sh` as 40599 | about midday 27 Oct |
| API LiveCodeBench subset | runner pid 785382 (all 400 pools at 00:40 on 27 Oct; now adding ratings) (`TASK_IDS=data/lcb/subset_api400.json scripts/run_api_tier.sh livecodebench`, log `logs/api-lcb.log`); 6 keys from the next generation pass | ratings about a day |
| Waiter pid 1976762 | when the subset runner exits: `sbatch ... scripts/ada/cpu_job.sbatch scripts/experiments/api.sh` (LiveCodeBench pinned to the subset with `--task-ids`, plus a Qwen-only reference on the same problems) | log `logs/api-eval-waiter.log` |
| Waiter pid 985269 | when the subset runner exits: `scripts/run_api_tier.sh livecodebench` (all 1,054; keeps the subset pools) | log `logs/api-lcb-full.log`; several days |

Check progress:

```bash
ssh ada3 'squeue -u $USER; cd ~/codeladder; for t in t1 t2 d1 d2 t3 t4 d3; do echo bcb $t $(ls data/pools/bigcodebench/$t 2>/dev/null | wc -l); done; ls data/pools/livecodebench/api | wc -l; pgrep -fa "^/bin/bash scripts|kill -0"'
ssh ada3 'tail -5 ~/codeladder_logs/chain.txt; cat ~/codeladder/logs/api-eval-waiter.log'
```

## Next actions, in order

1. **BigCodeBench pools.** Watch the chain (t1 resumes at 343, just before BigCodeBench/348, the task that once stopped the server: confirm the fence holds there); resubmit any job that stops (pools resume). For each finished tier check `grep -c "failed in a row\|server is down" logs/cl-tier-<job>.out`. When the seven tiers exist: `sbatch -c 8 --mem=40G scripts/ada/cpu_job.sbatch scripts/experiments/bigcodebench.sh` (all core experiments, then figures and tables), write up X18 under its setup section in RESULTS.md, refresh `demo-bundle` and copy the BigCodeBench pools and results to the laptop.
2. **API on BigCodeBench** after the local pools look sound: `TASK_IDS=data/bcb/subset_api400.json nohup scripts/run_api_tier.sh bigcodebench > logs/api-bcb.log 2>&1 &` on the login node (shares the keys with the LiveCodeBench run).
3. **X12:** when 40599 shows ALLDONE, copy results to the laptop, rerun `report_figures.py` and `report_tables.py` (40599 rewrites `results/analysis/signals_auroc.json` with 32B), complete the X12 section in RESULTS.md (tables with `python scripts/show_results.py runs results/*agreement_ptrue*`), including whether cheaper rungs certify more (economics runs in the same batch), and update H2 in the hypotheses table; then delete `results/archive/stale_ptrue32/` on both machines.
4. **API LiveCodeBench subset:** when the pinned evaluation finishes, record it (E8 / X3 / X11) against the Qwen-only reference on the same 400 problems. Never evaluate LiveCodeBench with the API tier without `--task-ids` while the full run is generating (its new pools are not rated yet).
5. **Demo:** upgraded 27 Oct (ladder builder over every evaluated setting, step-by-step playback, many-problems replay, live-mode check). After BigCodeBench, the API LiveCodeBench evaluation and 40599: copy their pools and `results/` to the laptop and rebuild `demo-bundle` so they appear in the builder (list in PLAN.md section 6). For the video's live part, run the demo on the cluster (`ssh -L 8765:127.0.0.1:8765 ada3 'cd codeladder && ~/venvs/codeladder/bin/codeladder demo'`, about 0.7 GB, fine on the login node) for the API tier, or start `serve_vllm.sbatch` jobs once the GPUs are free. On the laptop: `cd codeladder && PYTHONPATH=src python3 -m codeladder.cli demo`.
6. **Reports:** mid-submission (4 pages, 6 Nov) and final project report (8 pages ACM, 30 Nov); outlines and evidence in [REPORT_PLAN.md](REPORT_PLAN.md); framing (research or tool) still to be decided by the team. Start from the summary of findings and `docs/RELATED_WORK.md`.
7. **Release data** once pools are final: `scripts/package_data.sh`.

## Workflow for code changes

1. Edit and test on the laptop: `PYTHONPATH=src python3 -m pytest -q` (146 tests, including dry runs of every experiment batch with each command checked against the CLI parser, and the BigCodeBench runner) and `uvx ruff check src tests scripts --select E,F,W,I,B --ignore E501`.
2. Sync to the cluster from the laptop (never between cluster accounts), from `/home/ekansh-the-great/Music/agentic-ai`:
   `rsync -az --exclude .git --exclude .pytest_cache --exclude __pycache__ --exclude .ruff_cache --exclude data/ --exclude results/ --exclude logs/ --exclude docs/AI_USAGE_LOG.md --exclude '*.egg-info' -e "ssh -o BatchMode=yes" codeladder/ ada3:codeladder/`
3. Commit and push from the laptop. Each step of a running job starts a new Python process (and the BigCodeBench runner is read per test), so synced code affects running jobs: keep changes backward compatible and tested. SLURM copies a job script at submission, so `run_tier.sbatch` changes only affect later jobs.
4. New experiments: add a batch to `scripts/experiments/` (source `common.sh`, use `ev` / `cl` / `py`; check with `DRY=1`) and record each result in RESULTS.md with its command.
5. Large replays run as CPU jobs: `sbatch -c 4 --mem=24G scripts/ada/cpu_job.sbatch scripts/experiments/<batch>.sh`.

## Conventions

- Use only the `ada3` account for this project; never copy data to or from the `ada` account. Copies to the laptop are fine.
- Commits are authored by Ekansh only; no AI attribution lines or AI tool names in commits or tracked files.
- `docs/AI_USAGE_LOG.md` is git-ignored and kept locally for the course disclosure; keep it up to date.
- No em dashes or en dashes in documents.
- API keys are placed by a team member; never read, print or copy key values (check slots by name only).

## Cluster facts and lessons

**Resources and nodes**
- Account `research`, QoS `medium`: per user 4 GPUs, 40 CPUs, 125 GB, 4 days per job; partition `u22`, `--constraint=2080ti` (RTX 2080 Ti, 11 GB). The memory limit often binds before CPUs, and SLURM then reports `QOSMaxCpuPerUserLimit`.
- Tier resources: t1/t2/d1/d2 1 GPU; t3 2 GPUs (TP=2); t4 4 GPUs (TP=4); d3 4 GPUs as TP=2 x PP=2 with `NCCL_P2P_DISABLE=1 GPU_UTIL=0.82 VLLM_EXTRA="--max-num-seqs 16 --enforce-eager"`. Annotation-only jobs need only `-c 8`.
- Bad nodes: gnode069 (GPU fault), gnode068 (pipeline-parallel hang): exclude for 4-GPU jobs. `/share1` is not mounted on every node (gnode045 lacks it); user namespaces are disabled on compute nodes.
- Login node: 2 GB memory per user (the API runner takes about 750 MB), and it returns a 2 KB page instead of large downloads. Run replays and downloads as jobs.

**Models and servers**
- `stage_model` fails the job on a broken download and verifies each per-node copy against the Hub's SHA-256 once (`scripts/ada/verify_model.py`): a half-downloaded 32B copy on gnode072 once served garbage, and its replies were cached.
- vLLM 0.6.6 returns 500 when a token logprob is -inf; the client retries once without logprobs (SDK retries are off for local servers); `CODELADDER_NO_LOGPROBS=1` skips logprobs for known-bad reruns.
- `run_tier.sbatch` stops when the vLLM server dies, and `generate` stops after 12 failed tasks in a row; both are resumable by resubmitting.
- Models that decline to rate their own code (DeepSeek-Coder) need `conf_prefill: true`; after any annotation run check the share of unparsed ratings and, for P(True), that the values vary across prompts.

**BigCodeBench**
- Execution: `~/venvs/bcb` through `src/codeladder/bcb_runner.py`, matched to the official harness (one module `__test__`, unittest discovery, the environment's python on PATH, TZ=UTC, NLTK data, 30 GB limits). Six tasks are excluded because their reference fails here (network 403s, tkinter, a hash, two timeouts).
- Generated code can signal processes (BigCodeBench/348 stops every process with a given name, which once stopped the model server). The runner fences `os.kill` / `killpg` to the test's session and puts session-limited `pkill` / `killall` / `kill` first on PATH (`src/codeladder/bcb_bin`).
- Jobs need about 60 GB (test subprocesses load pandas / tensorflow next to the server); pass memory as `--mem-per-cpu` (`--mem` conflicts with run_tier.sbatch).

**Network and API**
- PyPI files download at 50 to 75 KB/s from the campus network (cluster and laptop); the Tsinghua mirror runs at 1 to 1.5 MB/s. Use the mirror and verify every file against pypi.org (`scripts/verify_wheels.py`).
- Groq free tier per key: about 200k tokens per day binds (1,000 requests per day, 8,000 tokens per minute); limits are per account. Estimate runs from tokens: LiveCodeBench about 1,200 tokens per request. Groq's front end rejects Python's default urllib user agent (error 1010); the OpenAI client is fine.

**Shell habits**
- Wait on a process ID (`while kill -0 PID`) or a job ID (`scripts/ada/after.sh JOB cmd`); never `pgrep -f` / `pkill -f` a pattern contained in the calling shell's own command line (it matches itself, on the cluster and the laptop). On the laptop `ls` is aliased and can stall in tool calls.

## History (condensed)

- 16 to 20 Oct: preflight, harness, pools for the Qwen tiers on HumanEval+, MBPP+ and LiveCodeBench, calibration families, stress inputs, published baselines, energy, first evaluations (families, κ, learning curves, transfer, shift, repair).
- 20 to 23 Oct: DeepSeek tiers (d3 after five attempts), X7 / X8, silent-failure types, API tier on HumanEval+/MBPP+, X9 router; API LiveCodeBench found token-limited.
- 24 Oct: DeepSeek ratings fixed with a prefilled answer and X1 / X10 rewritten; X4; repository organised (`scripts/experiments/`, `scripts/show_results.py`).
- 25 Oct: literature review; X12 P(True), X13 economics (cheap-share law), X14 bootstrap, X15 policy export and solve, X16 guaranteed baselines, X17 oracle sensitivity; BigCodeBench environment built from a verified mirror.
- 26 Oct: X14, X16, X17 written up; BigCodeBench runner matched to the official harness (44 exclusions down to 6), then fenced against process-killing tasks; corrupt 32B copy found and staging verified against the Hub; interactive demo (`codeladder demo`) built and rehearsed with real data; Groq keys 5 and 6 added.
- 27 Oct: report figures (13) and tables (`scripts/report_tables.py`), hypotheses table and figure map in RESULTS.md, REPORT_PLAN.md; BigCodeBench evaluation batch; repository review: BigCodeBench label fix (790, 1028; `scripts/relabel_bcb.py`), sandbox process-group kill, live-solve and script fixes, 95 exploratory runs moved to `results/archive/`; BigCodeBench jobs reordered (d1 and d2 at 40 GB, 32B P(True) first); API LiveCodeBench subset pools complete.

- 6 Nov: mid-submission milestone; all 150 unit/integration tests verified passing, report figures and LaTeX tables generated from evaluation results, interactive demonstration validated, and mid-submission documentation finalized.
