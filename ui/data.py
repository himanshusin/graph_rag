"""Cached data access shared by every screen."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from core.change_manager import ChangeManagementAgent
from core.rag import GraphData, suggested_prompts as build_suggested_prompts
from core.vault import DocumentVault

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "ragtest" / "output"
CHROMA_DIR = ROOT / "notebook" / "chromadb"
CONCEPT_MAP_PATH = ROOT / "notebook" / "interactive_graph.html"
VAULT_DIR = ROOT / "vault"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"

MODEL_NAME = "gpt-4o-mini"

PARQUET_FILES = {
    "entities": "create_final_entities.parquet",
    "relationships": "create_final_relationships.parquet",
    "nodes": "create_final_nodes.parquet",
    "reports": "create_final_community_reports.parquet",
}


@st.cache_data(show_spinner=False)
def load_graph_frames(signature: str) -> Dict[str, pd.DataFrame]:
    """Read the four parquet artifacts. ``signature`` busts the cache on change."""
    frames: Dict[str, pd.DataFrame] = {}
    for key, filename in PARQUET_FILES.items():
        path = OUTPUT_DIR / filename
        try:
            frames[key] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        except Exception:
            frames[key] = pd.DataFrame()
    return frames


def output_signature() -> str:
    """Fingerprint of the parquet set, so a rebuild invalidates the cache."""
    parts = []
    for filename in PARQUET_FILES.values():
        path = OUTPUT_DIR / filename
        parts.append(f"{filename}:{path.stat().st_mtime_ns if path.exists() else 0}")
    return "|".join(parts)


def graph_data() -> GraphData:
    frames = load_graph_frames(output_signature())
    return GraphData(
        entities=frames["entities"],
        relationships=frames["relationships"],
        nodes=frames["nodes"],
        reports=frames["reports"],
    )


@st.cache_resource(show_spinner=False)
def chroma_collection():
    """The passage collection. Returns None when the store cannot be opened."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_or_create_collection(name="paper_collection")
    except Exception:
        return None


def passage_count() -> int:
    collection = chroma_collection()
    try:
        return collection.count() if collection is not None else 0
    except Exception:
        return 0


@st.cache_resource(show_spinner=False)
def get_llm(temperature: float = 0.2, model_name: str = MODEL_NAME):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )


@st.cache_resource(show_spinner=False)
def get_vault() -> DocumentVault:
    return DocumentVault(vault_dir=str(VAULT_DIR))


@st.cache_resource(show_spinner=False)
def get_change_agent() -> ChangeManagementAgent:
    return ChangeManagementAgent(root_dir=str(ROOT))


@st.cache_data(show_spinner=False, ttl=120)
def qa_report(signature: str) -> Dict[str, Any]:
    """Run the QA suite. Cached briefly so the sidebar badge is cheap."""
    return get_change_agent().run_qa_checks()


def current_qa_report() -> Dict[str, Any]:
    return qa_report(output_signature())


@st.cache_data(show_spinner=False)
def suggested_prompts(signature: str) -> List[Dict[str, str]]:
    return build_suggested_prompts(graph_data())


def current_suggested_prompts() -> List[Dict[str, str]]:
    return suggested_prompts(output_signature())


def app_version() -> str:
    return get_change_agent().get_version()


@st.cache_data(show_spinner=False)
def local_user_initials() -> str:
    """Initials for the message avatar, taken from the local git identity."""
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=5,
        )
        name = result.stdout.strip()
    except Exception:
        name = ""
    if not name:
        return ""
    parts = [p for p in re.split(r"\s+", name) if p]
    return "".join(p[0] for p in parts[:2]).upper()


def parse_changelog(limit: int = 4) -> List[Dict[str, Any]]:
    """Parse CHANGELOG.md into releases with their commit rows."""
    if not CHANGELOG_PATH.exists():
        return []
    releases: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in CHANGELOG_PATH.read_text(encoding="utf-8").splitlines():
        header = re.match(r"^##\s*\[?v?([0-9][^\]\s]*)\]?\s*-\s*(.+?)\s*$", line)
        if header:
            if current:
                releases.append(current)
            current = {"version": header.group(1), "date": header.group(2), "commits": []}
            continue
        commit = re.match(r"^-\s*\[`([0-9a-f]{6,})`\]\s*(.+?)\s*$", line)
        if commit and current is not None:
            message = re.sub(
                r"^(feat|fix|docs|chore|refactor|test|style|ui|perf)(\([^)]*\))?:\s*",
                "", commit.group(2),
            )
            current["commits"].append({"sha": commit.group(1)[:7], "message": message})
    if current:
        releases.append(current)
    return releases[:limit]


def clear_caches() -> None:
    """Drop every cache after an ingest, a delete or a rebuild."""
    st.cache_data.clear()
    st.cache_resource.clear()
