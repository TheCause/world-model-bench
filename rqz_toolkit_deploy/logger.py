"""
RQZ Logger — Structured JSONL logging for remote monitoring.

Writes one JSON object per line to a .jsonl file.
Readable via MCP compute-m4 get_results or direct file access.

Usage:
    log = RQZLogger("A5_dh_sigma", output_dir="results")
    log.info("Starting experiment", seed=0, total_seeds=5)
    log.progress(seed=0, step=50, total_steps=300, metrics={"rho": 0.42})
    log.seed_complete(seed=0, metrics={"rho_gap": 0.35}, status="PASS")
    log.experiment_complete(summary={...}, verdict="GO")
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class RQZLogger:
    def __init__(self, experiment: str, output_dir: str = "results",
                 log_every_n: int = 50, also_print: bool = True):
        self.experiment = experiment
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n = log_every_n
        self.also_print = also_print

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.output_dir / f"run_{experiment}_{ts}.jsonl"
        self.start_time = time.time()
        self._step_count = 0

    def _write(self, entry: Dict[str, Any]):
        entry["timestamp"] = datetime.now().isoformat()
        entry["experiment"] = self.experiment
        entry["elapsed_s"] = round(time.time() - self.start_time, 1)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        if self.also_print:
            _print_compact(entry)

    def info(self, message: str, **kwargs):
        self._write({"level": "INFO", "message": message, **kwargs})

    def error(self, message: str, **kwargs):
        self._write({"level": "ERROR", "message": message, **kwargs})

    def progress(self, seed: int, step: int, total_steps: int,
                 metrics: Optional[Dict] = None):
        """Log progress. Only writes every log_every_n steps."""
        self._step_count += 1
        if step % self.log_every_n != 0 and step != total_steps - 1:
            return
        entry = {
            "level": "PROGRESS",
            "seed": seed,
            "step": step,
            "total_steps": total_steps,
            "pct": round(100 * step / max(total_steps, 1), 1),
        }
        if metrics:
            entry["metrics"] = metrics
        self._write(entry)

    def seed_complete(self, seed: int, metrics: Dict, status: str = "OK"):
        self._write({
            "level": "SEED_DONE",
            "seed": seed,
            "metrics": metrics,
            "status": status,
        })

    def experiment_complete(self, summary: Dict, verdict: str):
        self._write({
            "level": "EXPERIMENT_DONE",
            "summary": summary,
            "verdict": verdict,
        })

    def get_log_path(self) -> Path:
        return self.log_path


def _print_compact(entry: Dict):
    level = entry.get("level", "?")
    elapsed = entry.get("elapsed_s", 0)

    if level == "PROGRESS":
        pct = entry.get("pct", 0)
        seed = entry.get("seed", "?")
        m = entry.get("metrics", {})
        metrics_str = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in m.items())
        print(f"  [{elapsed:>6.0f}s] seed {seed} {pct:5.1f}% {metrics_str}", flush=True)
    elif level == "SEED_DONE":
        seed = entry.get("seed", "?")
        status = entry.get("status", "?")
        print(f"  [{elapsed:>6.0f}s] seed {seed} DONE ({status})", flush=True)
    elif level == "EXPERIMENT_DONE":
        verdict = entry.get("verdict", "?")
        print(f"\n  [{elapsed:>6.0f}s] VERDICT: {verdict}", flush=True)
    elif level == "ERROR":
        print(f"  [{elapsed:>6.0f}s] ERROR: {entry.get('message', '?')}", file=sys.stderr, flush=True)
    else:
        msg = entry.get("message", "")
        print(f"  [{elapsed:>6.0f}s] {msg}", flush=True)
