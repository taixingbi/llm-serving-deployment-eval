#!/usr/bin/env python3
"""Sample CPU/RAM and optional GPU metrics to a JSONL file until killed."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    print("ERROR: pip install psutil", file=sys.stderr)
    sys.exit(1)

_nvml = None
try:
    import pynvml

    pynvml.nvmlInit()
    _nvml = pynvml
except Exception:  # noqa: BLE001
    _nvml = None


def sample_gpu() -> list[dict]:
    if _nvml is None:
        return []
    out = []
    try:
        count = _nvml.nvmlDeviceGetCount()
    except Exception:  # noqa: BLE001
        return []
    for i in range(count):
        try:
            handle = _nvml.nvmlDeviceGetHandleByIndex(i)
            util = _nvml.nvmlDeviceGetUtilizationRates(handle)
            mem = _nvml.nvmlDeviceGetMemoryInfo(handle)
            row = {
                "index": i,
                "util_gpu_pct": float(util.gpu),
                "util_mem_pct": float(util.memory),
                "mem_used_mb": mem.used / (1024 * 1024),
                "mem_total_mb": mem.total / (1024 * 1024),
            }
            try:
                power = _nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                row["power_w"] = power
            except Exception:  # noqa: BLE001
                pass
            out.append(row)
        except Exception:  # noqa: BLE001
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="JSONL output path")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stop = False

    def _stop(*_args: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    with out_path.open("a") as f:
        while not stop:
            row = {
                "ts": time.time(),
                "cpu_pct": psutil.cpu_percent(interval=None),
                "ram_pct": psutil.virtual_memory().percent,
                "ram_used_mb": psutil.virtual_memory().used / (1024 * 1024),
                "gpus": sample_gpu(),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            time.sleep(args.interval)

    if _nvml is not None:
        try:
            _nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
