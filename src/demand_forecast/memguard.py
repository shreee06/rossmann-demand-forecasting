"""Stage-by-stage memory logging and a MemAvailable watchdog (Linux /proc).

Adapted from fashion_recsys.memguard. The watchdog exits the process immediately
(os._exit, so it also works from a thread) and writes no result files after that point.
"""
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from demand_forecast.config import MIN_AVAILABLE_KB


def _kb(path, key):
    with open(path) as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1])
    return -1


def snapshot(stage):
    return (f"{datetime.now():%H:%M:%S} {stage:<40} RSS {_kb('/proc/self/status', 'VmRSS:') / 1024:7.0f} MB  "
            f"peak {_kb('/proc/self/status', 'VmHWM:') / 1024:7.0f} MB  "
            f"MemAvailable {_kb('/proc/meminfo', 'MemAvailable:') / 1024:7.0f} MB")


class MemGuard:
    def __init__(self, log_path, min_kb=MIN_AVAILABLE_KB):
        self.log_path, self.min_kb = Path(log_path), min_kb
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("")

    def _write(self, line):
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def abort(self, msg):
        self._write(f"ABORT: {msg}")
        print(f"ABORT (no result files written): {msg}", flush=True)
        os._exit(3)

    def stage(self, name):
        import gc
        gc.collect()
        line = snapshot(name)
        self._write(line)
        print(line, flush=True)
        if _kb("/proc/meminfo", "MemAvailable:") < self.min_kb:
            self.abort(f"MemAvailable below 1 GB at stage '{name}'")

    def start_watchdog(self, every=2.0):
        def loop():
            while True:
                if _kb("/proc/meminfo", "MemAvailable:") < self.min_kb:
                    self.abort("MemAvailable dropped below 1 GB: " + snapshot("watchdog"))
                time.sleep(every)
        threading.Thread(target=loop, daemon=True).start()

    def peak_mb(self):
        return _kb("/proc/self/status", "VmHWM:") // 1024
