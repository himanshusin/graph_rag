import os
import json
import subprocess
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


class ChangeManagementAgent:
    """
    Automated Enterprise Change Management Agent.
    Manages semantic versioning, changelog generation, README synchronization,
    QA validation, and git maintenance.
    """

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()
        self.version_file = self.root_dir / "VERSION"
        self.changelog_file = self.root_dir / "CHANGELOG.md"
        self.readme_file = self.root_dir / "README.md"

    # -------------------------------------------------------------------------
    # 1. Semantic Versioning
    # -------------------------------------------------------------------------
    def get_version(self) -> str:
        """Read current version from VERSION file or fallback to 2.2.0."""
        if self.version_file.exists():
            return self.version_file.read_text(encoding="utf-8").strip()
        return "2.2.0"

    def bump_version(self, part: str = "patch") -> str:
        """Bump semver version (major, minor, or patch)."""
        curr = self.get_version()
        parts = curr.split(".")
        try:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        except Exception:
            major, minor, patch = 2, 2, 0

        if part == "major":
            major += 1
            minor = 0
            patch = 0
        elif part == "minor":
            minor += 1
            patch = 0
        else: # patch
            patch += 1

        new_version = f"{major}.{minor}.{patch}"
        self.version_file.write_text(new_version + "\n", encoding="utf-8")
        return new_version

    # -------------------------------------------------------------------------
    # 2. Automated Changelog Generator
    # -------------------------------------------------------------------------
    def update_changelog(self, release_version: Optional[str] = None) -> str:
        """Parse git history and generate structured CHANGELOG.md."""
        version = release_version or self.get_version()
        today = time.strftime("%Y-%m-%d")

        try:
            cmd = ["git", "log", "-n", "30", "--pretty=format:%h|||%s|||%ad", "--date=short"]
            result = subprocess.run(cmd, cwd=str(self.root_dir), capture_output=True, text=True)
            log_lines = result.stdout.strip().splitlines()
        except Exception:
            log_lines = []

        feats, fixes, refactors, docs, uis = [], [], [], [], []

        for line in log_lines:
            if "|||" not in line:
                continue
            commit_hash, subject, date = line.split("|||", 2)
            item = f"- [`{commit_hash}`] {subject}"

            sub_lower = subject.lower()
            if "feat" in sub_lower:
                feats.append(item)
            elif "fix" in sub_lower:
                fixes.append(item)
            elif "ui" in sub_lower or "style" in sub_lower:
                uis.append(item)
            elif "doc" in sub_lower or "readme" in sub_lower:
                docs.append(item)
            else:
                refactors.append(item)

        content = [
            "# Change log\n",
            "Release history for the GraphRAG knowledge workspace, generated from git history.\n",
            "---\n",
            f"## [v{version}] - {today}\n"
        ]

        if feats:
            content.append("### Features\n" + "\n".join(feats) + "\n")
        if uis:
            content.append("### UI and design system\n" + "\n".join(uis) + "\n")
        if fixes:
            content.append("### Fixes\n" + "\n".join(fixes) + "\n")
        if refactors:
            content.append("### Architecture\n" + "\n".join(refactors) + "\n")
        if docs:
            content.append("### Documentation\n" + "\n".join(docs) + "\n")

        changelog_text = "\n".join(content)
        self.changelog_file.write_text(changelog_text, encoding="utf-8")
        return changelog_text

    # -------------------------------------------------------------------------
    # 3. Synchronize README.md
    # -------------------------------------------------------------------------
    def sync_readme(self) -> str:
        """Update README.md with live version badges, sub-tree guide, and semantic versioning while preserving rich conceptual sections."""
        version = self.get_version()
        if self.readme_file.exists():
            content = self.readme_file.read_text(encoding="utf-8")
            # Update Version badge: [![Version](https://img.shields.io/badge/Version-X.Y.Z-blue.svg...
            content = re.sub(
                r'img\.shields\.io/badge/Version-[\d\.]+-blue\.svg',
                f'img.shields.io/badge/Version-{version}-blue.svg',
                content
            )
            # Update VERSION line in ASCII tree: ├── VERSION    # Semantic Versioning (X.Y.Z)
            content = re.sub(
                r'├── VERSION\s+# Semantic Versioning \([\d\.]+\)',
                f'├── VERSION                         # Semantic Versioning ({version})',
                content
            )
            self.readme_file.write_text(content, encoding="utf-8")
            return content
        else:
            readme_content = f"""# GraphRAG knowledge workspace

[![Version](https://img.shields.io/badge/Version-{version}-blue.svg?style=flat-square)](#)
[![Python](https://img.shields.io/badge/Python-3.13+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](#)

Knowledge extraction, community graph reasoning and hybrid vector retrieval over a
document vault, built on GraphRAG concepts, ChromaDB and Streamlit.

---

## Architecture

```
GraphRAG-Breakdown/
├── app.py                          # Entry point: sidebar, routing
├── CHANGELOG.md                    # Generated by the change management agent
├── VERSION                         # Semantic versioning ({version})
├── requirements.txt                # Dependencies
│
├── core/                           # Engines
│   ├── pipeline.py                 # Chunking, extraction, Louvain, reports, exports
│   ├── rag.py                      # Retrieval and generation for the five modes
│   ├── vault.py                    # Document retention, tables, index state
│   └── change_manager.py           # Versioning, changelog, QA checks
│
├── ui/                             # Presentation
│   ├── theme.css                   # Design tokens and components
│   ├── tokens.py                   # Token values and text helpers
│   ├── components.py               # HTML fragments
│   └── data.py                     # Cached data access
│
├── screens/                        # Search, Vault, Concept map, Catalog, Governance
├── vault/                          # Retained documents, extracted tables, catalog.json
├── scripts/                        # QA suite and pre-push orchestration
├── ragtest/output/                 # Parquet artifacts (entities, relationships, nodes, reports)
├── notebook/                       # Concept map HTML and the ChromaDB store
└── media/                          # Diagrams
```
"""
            self.readme_file.write_text(readme_content, encoding="utf-8")
            return readme_content

    # -------------------------------------------------------------------------
    # 4. QA Validation Suite
    # -------------------------------------------------------------------------
    PARQUET_SCHEMA = {
        "entities": ["title", "type", "description"],
        "relationships": ["source", "target", "description", "weight"],
        "nodes": ["title", "community", "degree"],
        "community_reports": ["community", "title", "summary", "rank", "full_content"],
    }

    def run_qa_checks(self) -> Dict[str, Any]:
        """Run every health check against the live artifacts and time each one.

        Returns a report whose ``checks`` list is what the Governance screen
        renders: ``label``, ``ok``, ``detail``, ``error`` and ``duration``.
        """
        started_all = time.perf_counter()
        checks: List[Dict[str, Any]] = []

        def record(label: str, fn):
            started = time.perf_counter()
            try:
                ok, detail = fn()
                error = ""
            except Exception as exc:
                ok, detail, error = False, "check raised", f"{type(exc).__name__}: {exc}"
            checks.append({
                "label": label,
                "ok": bool(ok),
                "detail": detail,
                "error": error,
                "duration": round(time.perf_counter() - started, 2),
            })

        record("Python syntax · app.py, core, ui, screens", self._check_syntax)
        for name in self.PARQUET_SCHEMA:
            record(f"Parquet · create_final_{name}", lambda n=name: self._check_parquet(n))
        record("ChromaDB · paper_collection reachable", self._check_chroma)
        record("Vector provenance · document_id on chunks", self._check_chroma_provenance)
        record("Vault catalog integrity · SHA-256", self._check_vault_integrity)
        record("Structured tables extracted", self._check_structured_tables)
        record("Concept map artifact", self._check_concept_map)
        record("OPENAI_API_KEY present", self._check_api_key)

        failed = [c for c in checks if not c["ok"]]
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checks": checks,
            "passed_count": len(checks) - len(failed),
            "total_count": len(checks),
            "duration": round(time.perf_counter() - started_all, 2),
            "errors": [f"{c['label']}: {c['error'] or c['detail']}" for c in failed],
            "passed": not failed,
        }
        # Summary keys kept for the command-line QA runner.
        results["syntax_check"] = "PASSED" if checks[0]["ok"] else "FAILED"
        results["parquet_tables"] = (
            "PASSED" if all(c["ok"] for c in checks[1:5]) else "WARNING (Missing or empty)"
        )
        results["chromadb_vault"] = "PASSED" if checks[5]["ok"] else "FAILED"
        results["vault_catalog"] = "PASSED" if checks[7]["ok"] else "FAILED"
        results["structured_tables"] = checks[8]["detail"]
        return results

    # -- individual checks ---------------------------------------------------
    def _check_syntax(self) -> Tuple[bool, str]:
        failures = []
        count = 0
        for py_file in self.root_dir.glob("**/*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
            count += 1
            try:
                compile(py_file.read_text(encoding="utf-8"), str(py_file), "exec")
            except SyntaxError as exc:
                failures.append(f"{py_file.name}:{exc.lineno} {exc.msg}")
        if failures:
            raise SyntaxError("; ".join(failures))
        return True, f"{count} files"

    def _check_parquet(self, name: str) -> Tuple[bool, str]:
        path = self.root_dir / f"ragtest/output/create_final_{name}.parquet"
        if not path.exists():
            return False, "not built yet"
        import pandas as pd
        frame = pd.read_parquet(path)
        missing = [c for c in self.PARQUET_SCHEMA[name] if c not in frame.columns]
        if missing:
            return False, f"missing columns: {', '.join(missing)}"
        if frame.empty:
            return False, "0 rows"
        return True, f"{len(frame)} rows"

    def _chroma(self):
        import chromadb
        client = chromadb.PersistentClient(path=str(self.root_dir / "notebook/chromadb"))
        return client.get_or_create_collection(name="paper_collection")

    def _check_chroma(self) -> Tuple[bool, str]:
        collection = self._chroma()
        count = collection.count()
        return count > 0, f"{count:,} docs" if count else "0 docs"

    def _check_chroma_provenance(self) -> Tuple[bool, str]:
        """Passages must carry document_id or citations cannot link back to the vault."""
        collection = self._chroma()
        count = collection.count()
        if not count:
            return False, "no chunks indexed"
        sample = collection.get(limit=min(count, 200), include=["metadatas"])
        metadatas = [m or {} for m in (sample.get("metadatas") or [])]
        tagged = sum(1 for m in metadatas if m.get("document_id"))
        if tagged == len(metadatas):
            return True, f"{tagged}/{len(metadatas)} tagged"
        return False, f"only {tagged}/{len(metadatas)} chunks tagged"

    def _check_vault_integrity(self) -> Tuple[bool, str]:
        catalog_path = self.root_dir / "vault/catalog.json"
        if not catalog_path.exists():
            return False, "catalog.json missing"
        from core.vault import DocumentVault
        vault = DocumentVault(vault_dir=str(self.root_dir / "vault"))
        documents = vault.get_catalog()
        issues = vault.verify_integrity()
        if issues:
            return False, "; ".join(f"{i['id']}: {i['problem']}" for i in issues[:3])
        return True, f"{len(documents)} docs"

    def _check_structured_tables(self) -> Tuple[bool, str]:
        tables_dir = self.root_dir / "vault/tables"
        if not tables_dir.exists():
            return True, "0 tables"
        total = 0
        for path in tables_dir.glob("*_tables.json"):
            try:
                total += len(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                return False, f"unreadable: {path.name}"
        return True, f"{total} tables"

    def _check_concept_map(self) -> Tuple[bool, str]:
        path = self.root_dir / "notebook/interactive_graph.html"
        nodes = self.root_dir / "ragtest/output/create_final_nodes.parquet"
        if not nodes.exists():
            # A map left over from a previous corpus must not read as healthy.
            return False, "stale: no graph to draw" if path.exists() else "no graph yet"
        if not path.exists():
            return False, "not generated yet"
        text = path.read_text(encoding="utf-8")
        if "#FFFFFF" not in text.upper():
            return False, "not on the light palette"
        return True, f"{path.stat().st_size // 1024} KB"

    def _check_api_key(self) -> Tuple[bool, str]:
        if os.getenv("OPENAI_API_KEY"):
            return True, ".env"
        return False, "not set"

    # -------------------------------------------------------------------------
    # 5. Master Synchronization
    # -------------------------------------------------------------------------
    def run_full_sync(self, bump: Optional[str] = None) -> Dict[str, Any]:
        """Execute full change management cycle."""
        if bump:
            new_v = self.bump_version(bump)
        else:
            new_v = self.get_version()

        qa = self.run_qa_checks()
        changelog = self.update_changelog(new_v)
        readme = self.sync_readme()

        return {
            "version": new_v,
            "qa_results": qa,
            "changelog_updated": bool(changelog),
            "readme_updated": bool(readme)
        }


if __name__ == "__main__":
    agent = ChangeManagementAgent()
    res = agent.run_full_sync()
    qa = res["qa_results"]
    print(f"Change management cycle finished for v{res['version']}")
    print(f"QA: {qa['passed_count']} / {qa['total_count']} checks passed in {qa['duration']}s")
    for issue in qa["errors"]:
        print(f"  - {issue}")
