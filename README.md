# CodeLadder

Risk-controlled code generation with model ladders. CodeLadder writes code with free models (Qwen2.5-Coder 1.5B, 7B, 14B and 32B; DeepSeek-Coder 1.3B, 6.7B and 33B; and gpt-oss-120b on a free API). At each step it decides whether to **accept**, **repair**, **resample**, **escalate** or **abstain**, and its thresholds are calibrated with Learn-then-Test so that, with probability at least 1 - δ, at most α of tasks end with code that was accepted but is wrong, at minimum cost.

Team Byte Me: Ekansh (2023114007), Ishaan (2023114016), Ayush (2023114001). Course project for Autonomous Software Engineering, Monsoon 2026, IIIT Hyderabad.

Development in progress.
