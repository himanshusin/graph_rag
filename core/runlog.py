"""Durable record of indexing runs.

A build runs inside a single Streamlit script run, so anything that restarts the
script — a nav click, a browser reconnect, a closed laptop — kills it with no
trace. This log writes the run to disk before the work starts and updates it as
stages complete, so an interrupted build is visible afterwards instead of
looking like nothing ever happened.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

RUNNING = "running"
COMPLETE = "complete"
FAILED = "failed"
INTERRUPTED = "interrupted"

# A run whose heartbeat is older than this and whose process is gone is dead.
STALE_AFTER_SECONDS = 60 * 60


class IndexRunLog:
    """Append-only-ish log of index runs, newest first."""

    def __init__(self, vault_dir: str = "./vault", limit: int = 20):
        self.path = Path(vault_dir) / "index_runs.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit = limit

    # -- storage -------------------------------------------------------------
    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save(self, runs: List[Dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(runs[: self.limit], indent=2), encoding="utf-8")

    # -- lifecycle -----------------------------------------------------------
    def start(self, documents: List[Dict[str, Any]], chunk_limit: int) -> str:
        """Record a run as started and return its id."""
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        runs = self._load()
        runs.insert(0, {
            "id": run_id,
            "status": RUNNING,
            "pid": os.getpid(),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "heartbeat": time.time(),
            "finished_at": None,
            "stage": "Starting",
            "progress": 0.0,
            "chunk_limit": chunk_limit,
            "documents": [
                {"id": d.get("id"), "title": d.get("title")} for d in documents
            ],
            "stats": {},
            "error": None,
        })
        self._save(runs)
        return run_id

    def update(self, run_id: str, stage: str, progress: float,
               stats: Optional[Dict[str, Any]] = None) -> None:
        """Record progress. Called on stage changes, not on every chunk."""
        runs = self._load()
        for run in runs:
            if run["id"] == run_id:
                run["stage"] = stage
                run["progress"] = round(float(progress), 3)
                run["heartbeat"] = time.time()
                if stats:
                    run["stats"].update(stats)
                break
        self._save(runs)

    def finish(self, run_id: str, stats: Optional[Dict[str, Any]] = None) -> None:
        runs = self._load()
        for run in runs:
            if run["id"] == run_id:
                run["status"] = COMPLETE
                run["stage"] = "Complete"
                run["progress"] = 1.0
                run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                run["heartbeat"] = time.time()
                if stats:
                    run["stats"].update(stats)
                break
        self._save(runs)

    def fail(self, run_id: str, error: str) -> None:
        runs = self._load()
        for run in runs:
            if run["id"] == run_id:
                run["status"] = FAILED
                run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                run["heartbeat"] = time.time()
                run["error"] = str(error)[:1000]
                break
        self._save(runs)

    # -- inspection ----------------------------------------------------------
    def all(self) -> List[Dict[str, Any]]:
        return self._load()

    def latest(self) -> Optional[Dict[str, Any]]:
        runs = self._load()
        return runs[0] if runs else None

    def active(self) -> Optional[Dict[str, Any]]:
        """The run this process is currently executing, if any."""
        for run in self._load():
            if run["status"] == RUNNING and run.get("pid") == os.getpid():
                return run
        return None

    def reap_stale(self) -> List[Dict[str, Any]]:
        """Mark runs left in ``running`` by a dead process as interrupted.

        Called once at app start. A run belonging to a live process is left
        alone so a build in another tab is not falsely condemned.
        """
        runs = self._load()
        reaped = []
        changed = False
        now = time.time()
        for run in runs:
            if run["status"] != RUNNING:
                continue
            pid = run.get("pid")
            age = now - float(run.get("heartbeat") or 0)
            if _process_alive(pid) and age < STALE_AFTER_SECONDS:
                continue
            run["status"] = INTERRUPTED
            run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            run["error"] = (
                f"The run stopped during '{run.get('stage', 'unknown')}' without finishing. "
                "A script restart (navigation, reload or a closed tab) ends an in-flight build."
            )
            reaped.append(run)
            changed = True
        if changed:
            self._save(runs)
        return reaped


def _process_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True
