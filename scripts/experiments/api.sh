#!/bin/bash
# E8 / X3 / X11: the free API top tier (gpt-oss-120b on Groq) alone and as the top rung of
# ladders. LiveCodeBench runs are pinned to a fixed task list, by default the 400-problem
# subset (data/lcb/subset_api400.json), so pools of the full run that are not rated yet never
# enter; they get a Qwen-only reference on the same problems. When all 1,054 API pools are
# generated and rated: LCB_TASK_IDS= scripts/experiments/api.sh (empty: every problem).
# RESULTS.md: "E8 / X3 free API top tier".
source "$(dirname "$0")/common.sh"
LCB_TASK_IDS="${LCB_TASK_IDS-data/lcb/subset_api400.json}"
S="--score agreement_selfconf --abstain-factor 64"
ladders() {  # ladders <benchmark args...>: the API alone and as the top rung
  for s in agreement_selfconf agreement selfconf; do ev "$@" --tiers api --score $s --abstain-factor 64 --tag=-k64$SFX; done
  for tiers in "t1 t2 t3 t4 api" "t1 t3 api" "t2 api" "t3 api"; do
    for f in singles subsets two_level; do ev "$@" --tiers $tiers $S --family $f --lower-budget 1 --tag=-k64-lb1$SFX; done
  done
}
SFX="" ladders --benchmark humaneval mbpp
if [ -n "$LCB_TASK_IDS" ]; then
  SFX="-$(basename "$LCB_TASK_IDS" .json)"  # e.g. -subset_api400
  ladders --benchmark livecodebench --task-ids "$LCB_TASK_IDS"
  for f in singles subsets two_level; do  # Qwen-only reference on the same problems
    ev --benchmark livecodebench --task-ids "$LCB_TASK_IDS" --tiers t1 t2 t3 t4 $S --family $f --lower-budget 1 --tag=-k64-lb1$SFX
  done
else
  SFX="" ladders --benchmark livecodebench
fi
echo ALLDONE
