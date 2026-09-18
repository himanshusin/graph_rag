"""Token and cost accounting for every model call.

Nothing in the app previously knew what it cost, which is why the corpus was
capped at four chunks per document by a slider rather than by a budget. The
ledger is what makes the frugality work verifiable: it records what each stage
actually spent, in dollars for hosted routes and in seconds for local ones.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class UsageLedger:
    """Accumulates usage for one run or one query."""

    def __init__(self):
        self._lock = threading.Lock()
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.started = time.perf_counter()

    def _bucket(self, stage: str) -> Dict[str, Any]:
        return self.stages.setdefault(stage, {
            "calls": 0, "cache_hits": 0, "templated": 0,
            "tokens_in": 0, "tokens_out": 0, "cost": 0.0,
            "provider": None, "model": None,
        })

    def record_call(self, stage: str, provider, response: Any) -> None:
        """Record one real model call. Usage comes from the provider's response."""
        usage = {}
        metadata = getattr(response, "response_metadata", None)
        if isinstance(metadata, dict):
            usage = metadata.get("token_usage") or {}
        if not usage:
            usage = getattr(response, "usage_metadata", None) or {}

        tokens_in = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)

        with self._lock:
            bucket = self._bucket(stage)
            bucket["calls"] += 1
            bucket["tokens_in"] += tokens_in
            bucket["tokens_out"] += tokens_out
            bucket["cost"] += provider.cost_of(tokens_in, tokens_out)
            bucket["provider"] = provider.key
            bucket["model"] = provider.model

    def record_cache_hit(self, stage: str) -> None:
        with self._lock:
            self._bucket(stage)["cache_hits"] += 1

    def record_templated(self, stage: str, count: int = 1) -> None:
        """Work that was answered deterministically instead of by a model."""
        with self._lock:
            self._bucket(stage)["templated"] += count

    # -- reporting -----------------------------------------------------------
    def totals(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "calls": sum(s["calls"] for s in self.stages.values()),
                "cache_hits": sum(s["cache_hits"] for s in self.stages.values()),
                "templated": sum(s["templated"] for s in self.stages.values()),
                "tokens_in": sum(s["tokens_in"] for s in self.stages.values()),
                "tokens_out": sum(s["tokens_out"] for s in self.stages.values()),
                "cost": round(sum(s["cost"] for s in self.stages.values()), 6),
                "elapsed_seconds": round(time.perf_counter() - self.started, 2),
            }

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            stages = {k: dict(v, cost=round(v["cost"], 6)) for k, v in self.stages.items()}
        return {"stages": stages, **self.totals()}

    def headline(self) -> str:
        """One line for a status row: honest about dollars vs seconds."""
        totals = self.totals()
        saved = totals["cache_hits"] + totals["templated"]
        parts = [f"{totals['calls']} calls"]
        if saved:
            parts.append(f"{saved} avoided")
        parts.append(f"{totals['tokens_in'] + totals['tokens_out']:,} tokens")
        parts.append(f"${totals['cost']:.4f}" if totals["cost"] else "free")
        parts.append(f"{totals['elapsed_seconds']:.0f}s")
        return " · ".join(parts)


def append_run(vault_dir: str | Path, entry: Dict[str, Any], limit: int = 200) -> None:
    """Append one completed run's usage to the durable ledger."""
    path = Path(vault_dir) / "usage_ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                rows = loaded
        except (ValueError, OSError):
            rows = []
    rows.insert(0, {"at": time.strftime("%Y-%m-%d %H:%M:%S"), **entry})
    path.write_text(json.dumps(rows[:limit], indent=2), encoding="utf-8")


def read_runs(vault_dir: str | Path, limit: int = 20) -> List[Dict[str, Any]]:
    path = Path(vault_dir) / "usage_ledger.json"
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return rows[:limit] if isinstance(rows, list) else []
    except (ValueError, OSError):
        return []


def lifetime_totals(vault_dir: str | Path) -> Dict[str, Any]:
    rows = read_runs(vault_dir, limit=10_000)
    return {
        "runs": len(rows),
        "cost": round(sum(float(r.get("cost") or 0) for r in rows), 4),
        "tokens": sum(int(r.get("tokens_in") or 0) + int(r.get("tokens_out") or 0) for r in rows),
        "calls_avoided": sum(int(r.get("cache_hits") or 0) + int(r.get("templated") or 0) for r in rows),
    }
