# Preflight checks (16 Oct 2026)

What was verified before starting, so there are no surprises later. Rerun the EvalPlus check on Ada with `sbatch scripts/ada/preflight.sbatch`.

## Passed

| Check | Result |
| --- | --- |
| Unit and end-to-end tests | 26 tests pass on Python 3.10 (oldest on Ada) and 3.14; lint clean |
| Models exist on Hugging Face | Qwen2.5-Coder 1.5B (3.1 GB), 7B-AWQ (5.6 GB), 14B-AWQ (10.0 GB), 32B-AWQ (19.3 GB); GGUF repos exist for the llama.cpp fallback. Total ~38 GB, fits the 100 GB `/share1` quota |
| vLLM image | `vllm/vllm-openai:v0.6.6.post1` exists (5.5 GB compressed). A CUDA 11.8 wheel of the same release also exists, as a fallback for old drivers |
| LiveCodeBench data | `livecodebench/code_generation_lite` is public on Hugging Face (not gated) |
| EvalPlus data | HumanEval+ 164 tasks (median 7 visible, ~970 hidden inputs per task); MBPP+ 378 tasks (median 3 visible, 105 hidden) |
| Execution cost | All 542 reference solutions run in our sandbox; median ~0.1 s per task for all visible + hidden inputs. ~225k candidates come to roughly 5 to 15 CPU-hours: not a bottleneck |
| **Labels match the official evaluator** | References + 988 AST mutants (1,530 programs) judged by us and by `evalplus.evaluate`. MBPP+: 971/971 agree. HumanEval+: 557/559 agree; the 2 differences are HumanEval/32, where EvalPlus 0.3.1 rejects its own reference solution (we accept any root within the task tolerance, which is correct) |

## Bugs found by the preflight and fixed

1. Benchmark inputs with complex numbers or sets could not be sent to the sandbox as JSON. Inputs now go in via pickle (trusted data only); outputs still come back as JSON.
2. Answers with more than 4,300 digits (HumanEval/83, /139) hit Python's int-to-string limit in the sandbox and in our comparison. Fixed in both places.
3. `python -I` silently ignored our fixed hash seed, so set ordering changed between runs and even reference solutions "failed" (MBPP+ 2, 579, 769). Fixed by running the sandbox with `-s -B` and a controlled environment.
4. EvalPlus's special checkers (set comparison, "a match was found", polynomial root, a few numeric oracles) and its default float tolerance (1e-6) are now mirrored in `oracles.py`, reusing EvalPlus's own helper functions.
5. Time limits are now per input, as in EvalPlus: max(2 s, 4 x the reference's time on that input). A flat limit wrongly failed slow-but-correct code (HumanEval/130).
6. Five tasks have huge outputs (up to 165 MB for one answer). Outputs over 1 MB now travel as a normalised fingerprint, and hidden-test runs get 4 GB. This also keeps parent memory small.
7. Candidates that fail a visible test skip the hidden tests (they are wrong by definition), and hidden tests run in chunks that stop at the first failure.

## Statistical feasibility of the guarantee

With δ = 0.1, the largest *calibration-set* error rate that still certifies a target α:

| Calibration tasks | α = 5% | α = 10% | α = 20% |
| --- | --- | --- | --- |
| 82 (half of HumanEval+) | 1.2% | 4.9% | 13.4% |
| 271 (half of HumanEval+ and MBPP+ pooled) | 3.0% | 7.4% | 16.6% |
| ~528 (half of LiveCodeBench) | 3.6% | 8.1% | 17.6% |

Consequence: headline results should use α = 10% on pooled benchmarks; α = 5% is a stretch that needs LiveCodeBench-scale data. HumanEval+ alone is too small for α = 5%.

## Still unknown (only Ada or a real model can answer)

| Unknown | How we find out | Fallback |
| --- | --- | --- |
| GPU driver version (decides vLLM 0.6.6 image vs fallback) | `scripts/ada/check_env.sh` on a GPU node | vLLM CUDA 11.8 wheel, or llama.cpp built with `u18/cuda/11.6` |
| Internet on compute nodes | `check_env.sh` | Download everything on the login node (already the plan) |
| Real throughput per tier on 2080 Ti | Week-1 throughput test | Fewer samples per task (`n_samples` 20 -> 10) |
| Queue wait times | First jobs | Spread jobs over three accounts; low-priority `sub` account for idle GPUs |
| Coverage at α = 10% with real models | Week-1 pilot: tiers t1 + t2 on HumanEval+ and MBPP+, then `codeladder evaluate` | Raise α or add tiers / samples |


## Verification summary (6 Nov)

- Environment: verified with Python 3.10 and all 150 unit/integration tests passing.
- Harness: sandboxed execution, process-group isolation, and timeout limits operating as designed.
- Benchmarks: EvalPlus (HumanEval+, MBPP+), LiveCodeBench release_v6, and BigCodeBench environments validated.
