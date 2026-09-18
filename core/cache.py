"""Content-addressed caches for LLM work.

Indexing is cumulative over the whole vault, so without a cache adding the fifth
document re-pays for the first four. Keys are content hashes that include the
prompt version and the provider's model, so a cached result is only ever reused
for the exact same text under the exact same model.

The cache is also the resumability primitive: a run interrupted half-way finds
every completed chunk already on disk, so resuming costs nothing.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class JSONCache:
    """One JSON file per key, under a directory. Safe for concurrent writers."""

    def __init__(self, root: str | Path, namespace: str):
        self.root = Path(root) / namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        # Shard by the first two characters so a big corpus does not put tens of
        # thousands of files in one directory.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(key)
        if not path.exists():
            with self._lock:
                self.misses += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        with self._lock:
            self.hits += 1
        return payload.get("value")

    def put(self, key: str, value: Dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": key, "stored_at": time.strftime("%Y-%m-%d %H:%M:%S"), "value": value}
        # Write-then-rename so a crash mid-write cannot leave a truncated entry
        # that would later parse as a valid empty extraction.
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            tmp.unlink(missing_ok=True)

    def stats(self) -> Dict[str, int]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(100 * self.hits / total) if total else 0,
        }

    def clear(self) -> int:
        removed = 0
        for path in self.root.rglob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def size(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json"))


class ExtractionCache(JSONCache):
    """Chunk text + prompt version + provider/model -> entities and relationships."""

    def __init__(self, vault_dir: str | Path = "./vault"):
        super().__init__(Path(vault_dir) / "cache", "extractions")


class ReportCache(JSONCache):
    """Community membership + prompt version + provider/model -> one report."""

    def __init__(self, vault_dir: str | Path = "./vault"):
        super().__init__(Path(vault_dir) / "cache", "reports")
