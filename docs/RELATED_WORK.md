# Related work and positioning (working notes, 25 Oct 2026)

From a web literature search on 25 Oct 2026. **Verify every reference (title, authors, venue, arXiv ID) before citing**: many 2026 items are arXiv-only preprints, and a few IDs were recalled rather than opened (marked "check").

## Closest work and how CodeLadder differs

| Work | What it does | Difference from CodeLadder |
| --- | --- | --- |
| Task Abstention for LLMs in Code Generation (Zhou et al., arXiv 2605.17029, 2026, cs.SE) | Learn-then-Test with Bonferroni, execution clustering on LLM-generated tests; DeepSeek-Coder 33B, Qwen2.5-Coder 32B; HumanEval, MBPP | One model, accept or abstain. CodeLadder certifies a multi-model, multi-action policy under a cost objective; its "calibrated single model" baseline is this setting |
| Trust or Escalate (Jung, Brahman, Choi, ICLR 2025, arXiv 2407.18370) | Cascade of LLM judges with abstention; per-rung thresholds calibrated for provable human agreement | Judging preferences, not generating code; no execution signals, no repair or resample, no cost-optimal choice among certified settings |
| Selective generation for code with FDR control (Jeong, Kim, Park, ICML 2026, arXiv 2505.13553) | Selective code generation with generated unit tests (FuzzEval) | One model; no cascade or cost |
| RisCoSet (arXiv 2605.12201, 2026); PAC prediction sets for code (Khakhar et al., ICML 2023) | Risk-controlling sets of partial programs | Sets, not accept / escalate / abstain decisions |
| Conformal Language Modeling (Quach et al., ICLR 2024); CALM (NeurIPS 2022); Fast yet Safe (NeurIPS 2024, check ID) | LTT for sampling / stopping / early exit within one model | One model; text |
| Calibrate-Then-Delegate (2604.14251), edge-cloud conformal cascades (2510.17543), Conformal Cascade (2607.25018), RACER (2603.06616), Conformal LLM Routing (ACL SRW 2026) | Risk-controlled cascades or routers | Classification, MCQ, QA, safety; none on code or open-ended generation |
| FrugalGPT, AutoMix, Hybrid LLM, RouteLLM, Gupta et al. (ICLR 2024), cascade routing (Dekoninck et al., ICML 2025), Zellinger et al. (TMLR 2025; early abstention 2502.09054) | Cost-quality cascades and routers | No finite-sample guarantee on accepted-but-wrong outputs |
| Model Cascading for Code (Chen et al., 2405.15842); Bayesian self-escalation (Shaikh, 2608.24087) | Code cascades with self-tests / belief-based escalation | No abstention, no guarantee; we reimplement both and show they exceed α |
| Resample or Reroute? (Chen, 2607.08665, 2026) | Observable signals do not identify when to resample vs reroute (MBPP+, LiveCodeBench, BigCodeBench) | Overlaps our X4 negative result: cite and present X4 as confirming it under a risk guarantee |
| When Can Conformal Risk Control Certify LLM Outputs? (Kotte, 2606.29054, 2026) | Any distribution-free method must abstain on a share of inputs bounded below | Related to our cheap-share law (X13): present ours as the cost-side, code-specific instance |
| Certified Against Which Oracle? (Liu et al., 2609.25938, 2026) | Abstention certificates under-report risk when the test oracle changes (Text-to-SQL) | Motivates an oracle-sensitivity check: calibrate on HumanEval/MBPP base tests, evaluate on EvalPlus tests |
| CodeT, MBR-exec, LEVER, Coder-Reviewer, B4 (ASE 2024), AlphaCode clustering | Execution-based reranking | We reuse agreement as a calibrated score (incremental) |
| EvalPlus, ALGO (NeurIPS 2023), CodeContests+ (EMNLP Findings 2025), HardTests (NeurIPS 2025) | LLM-generated or mutated test inputs | We use stress inputs (timeout = failed check) as a certified acceptance signal |
| Spiess et al. (ICSE 2025), in-IDE calibration at scale (2510.22614), multicalibration for code (AI-SQE 2026), P(True) (Kadavath et al. 2022), verbalised confidence (Tian et al. 2023) | Code LLMs are poorly calibrated; post-hoc calibration helps little | LTT needs only a monotone score, not calibrated probabilities: use as motivation |

## Defensible contributions

1. Learn-then-Test / Pareto Testing certification of a multi-model, multi-action (accept, repair, resample, escalate, abstain) code-generation controller with a cost objective. No such work found.
2. Published code cascades violate risk targets (reimplemented, same tasks, 100 random splits).
3. Stress-input timeouts as a certified acceptance signal for stdin problems.
4. A certified-policy exporter that runs on new problems (demo).
5. Empirical laws with explanations: the cheap-share bound on savings (cite Kotte), verbalised self-ratings useful only from about 14B while P(True) carries signal at every size (X12), marginal risk hiding hard-problem risk (with Mondrian calibration as the remedy).

Incremental: the signals themselves, LTT itself, the router and repair-vs-resample negatives (cite Dekoninck 2025, Olausson et al. ICLR 2024, Resample or Reroute 2026).

## Additions suggested by the review

1. (Done as X16, except the conformal cascade.) Head-to-head with guaranteed baselines: per-rung LTT abstention (Task Abstention style; we already have calibrated single models), a Trust-or-Escalate cascade adapted to code, Conformal Cascade with execution clustering.
2. (X17 done; BigCodeBench is X18, in progress.) A realistic benchmark: BigCodeBench (Hard / Instruct), ClassEval or DevEval; an oracle-sensitivity analysis (base tests vs EvalPlus tests).
3. (Done: X7 and E5.) Conditional (Mondrian) calibration as a method, with the temporal-shift recipe (we have X7 and E5).
4. (Demo done, video to come; no Docker image yet.) Tool polish for a demo track: Docker image, report of the certified policy, real cost in dollars, latency and energy, video.

## Venues (check dates)

ICSE 2027 research (30 Jun 2026), FSE 2027 research (2 Oct 2026) and ICSE 2027 demonstrations (23 Oct 2026) deadlines have passed. Candidates: FSE 2027 tool demonstrations, ASE 2027, ICSE 2028. The course deadline (8-page ACM project report, 30 Nov 2026) is the binding one. ICSE demos: 4 pages IEEE, 3 to 5 minute video, tool runnable without building. ICSE research: 10 + 2 pages, double-anonymous, open science. FSE research: 18 + 4 pages, data availability section.
