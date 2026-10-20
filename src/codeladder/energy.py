"""GPU energy logging (NVML) and attribution to generated tokens.

The model server and the generation client usually run as separate processes,
often on different nodes. So the serve job runs ``codeladder energy-log`` in the
background on the GPU node, writing cumulative energy per GPU to a CSV. After a
generation run we compute joules per output token for that tier over the run's
time window, and attribute energy to each candidate as tokens_out x J/token.
"""

from __future__ import annotations

import bisect
import csv
import re
import time
from pathlib import Path


def log_energy(
    out_path: str | Path, interval_s: float = 1.0, duration_s: float | None = None
) -> None:
    """Append (unix_time, gpu_index, energy_j, power_w) rows until killed or duration elapses.

    Uses the total-energy counter where supported (Volta and newer, so RTX 2080 Ti),
    otherwise integrates instantaneous power (for example on GTX 1080 Ti).
    """
    import pynvml  # from the nvidia-ml-py package

    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    has_counter = []
    for h in handles:
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            has_counter.append(True)
        except pynvml.NVMLError:
            has_counter.append(False)
    integrated = [0.0] * len(handles)
    last_t = time.time()
    end = last_t + duration_s if duration_s else None
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "gpu", "energy_j", "power_w"])
        while end is None or time.time() < end:
            now = time.time()
            for i, h in enumerate(handles):
                power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                if has_counter[i]:
                    energy_j = pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
                else:
                    integrated[i] += power_w * (now - last_t)
                    energy_j = integrated[i]
                w.writerow([f"{now:.3f}", i, f"{energy_j:.3f}", f"{power_w:.2f}"])
            f.flush()
            last_t = now
            time.sleep(interval_s)


def energy_between(log_path: str | Path, t_start: float, t_end: float) -> float:
    """Total joules used by all logged GPUs between two unix times."""
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    with open(log_path) as f:
        for row in csv.DictReader(f):
            t = float(row["time"])
            if t_start <= t <= t_end:
                g = row["gpu"]
                first.setdefault(g, float(row["energy_j"]))
                last[g] = float(row["energy_j"])
    return sum(last[g] - first[g] for g in last)


def joules_per_token(log_path: str | Path, t_start: float, t_end: float, tokens_out: int) -> float:
    return energy_between(log_path, t_start, t_end) / max(tokens_out, 1)


_METRICS = re.compile(r"^INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) metrics\.py.*Running: (\d+) reqs")


def busy_times(vllm_log: str | Path, year: int) -> list[float]:
    """Unix times of vLLM metric lines reporting running requests. vLLM prints one line per
    ~5 s while it serves requests; times are in the node's local time zone."""
    out = []
    with open(vllm_log, errors="replace") as f:
        for line in f:
            m = _METRICS.match(line)
            if m and int(m.group(6)) > 0:
                mo, d, h, mi, s = (int(x) for x in m.groups()[:5])
                out.append(time.mktime((year, mo, d, h, mi, s, 0, 0, -1)))
    return sorted(out)


def busy_energy_between(
    log_path: str | Path, t_start: float, t_end: float, busy: list[float], window_s: float = 5.0
) -> float:
    """Joules used by all logged GPUs between two unix times, counting only energy samples
    that fall within ``window_s`` before a busy metric line (the server was generating).
    Excludes idle time, for example while the client runs tests."""
    last: dict[str, tuple[float, float]] = {}
    total = 0.0
    with open(log_path) as f:
        for row in csv.DictReader(f):
            t, g, e = float(row["time"]), row["gpu"], float(row["energy_j"])
            if not t_start <= t <= t_end:
                continue
            if g in last:
                k = bisect.bisect_left(busy, t)
                if k < len(busy) and busy[k] - t < window_s:
                    total += e - last[g][1]
            last[g] = (t, e)
    return total
