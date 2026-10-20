"""Busy-time energy attribution from NVML logs and vLLM metric lines."""

import time

from codeladder.energy import busy_energy_between, busy_times, energy_between


def test_busy_energy_counts_only_serving_seconds(tmp_path):
    t0 = time.mktime((2026, 10, 19, 9, 0, 0, 0, 0, -1))
    log = tmp_path / "energy.csv"
    rows = ["time,gpu,energy_j,power_w"]
    for s in range(0, 61):  # two GPUs, 100 W each, for 60 s
        rows += [f"{t0 + s},{g},{100.0 * s},100" for g in (0, 1)]
    log.write_text("\n".join(rows) + "\n")
    vllm = tmp_path / "vllm.log"
    vllm.write_text(
        "INFO 10-19 09:00:10 metrics.py:467] Avg prompt throughput: 1 tokens/s, Running: 3 reqs, Swapped: 0 reqs\n"
        "INFO 10-19 09:00:15 metrics.py:467] Avg prompt throughput: 1 tokens/s, Running: 1 reqs, Swapped: 0 reqs\n"
        "INFO 10-19 09:00:20 metrics.py:467] Avg prompt throughput: 0 tokens/s, Running: 0 reqs, Swapped: 0 reqs\n"
        "INFO 10-19 09:00:21 api_server.py:1] some other line\n"
    )
    busy = busy_times(vllm, 2026)
    assert busy == [t0 + 10, t0 + 15]
    assert energy_between(log, t0, t0 + 60) == 2 * 6000
    # samples at 6..15 s fall within 5 s before a busy line: 10 one-second intervals per GPU
    assert busy_energy_between(log, t0, t0 + 60, busy) == 2 * 10 * 100
    assert busy_energy_between(log, t0, t0 + 60, []) == 0
