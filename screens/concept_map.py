"""Concept map screen: PyVis network with a selection inspector (mockup 1e)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from core.pipeline import write_concept_map
from ui import components as c
from ui import data
from ui.tokens import esc, query_string

GRAPH_HEIGHT = 620
MAX_DRAWN_NODES = 220


def cell_length(value: Any) -> int:
    """Length of a parquet list cell.

    Arrow list columns come back as numpy arrays, so ``value or []`` raises
    "truth value of an array is ambiguous" rather than falling back.
    """
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def render(state: Dict[str, Any]) -> None:
    graph = state["graph"]
    size_by = st.query_params.get("size", "degree")
    if size_by not in ("degree", "rank"):
        size_by = "degree"
    community_filter = _community_filter(state)

    titles = (
        sorted(graph.entities["title"].astype(str).tolist())
        if not graph.entities.empty else []
    )
    requested = st.query_params.get("node") or st.query_params.get("find")
    default_index = titles.index(requested) if requested in titles else (0 if titles else None)

    split = st.container(key="split")
    graph_col, rail_col = split.columns([1, 0.42], gap=None)

    selected_title: Optional[str] = None
    with graph_col:
        with st.container(key="topbar_concepts", horizontal=True, vertical_alignment="center"):
            drawn = _visible_nodes(graph, community_filter)
            st.markdown(
                '<div class="k-topbar__title">Concept map</div>'
                f'<div class="k-topbar__note">{len(drawn)} nodes · '
                f'{_visible_edge_count(graph, drawn)} edges · Louvain</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                c.segmented(
                    [
                        ("degree", "Degree", query_string(
                            screen="concepts", size="degree", node=requested,
                            community=st.query_params.get("community"))),
                        ("rank", "Rank", query_string(
                            screen="concepts", size="rank", node=requested,
                            community=st.query_params.get("community"))),
                    ],
                    size_by,
                ),
                unsafe_allow_html=True,
            )
            with st.container(key="find_node"):
                selected_title = st.selectbox(
                    "Find node",
                    titles,
                    index=default_index,
                    label_visibility="collapsed",
                    placeholder="Find node",
                    key="node_picker",
                )

        with st.container(key="graph"):
            if graph.nodes.empty:
                st.markdown(
                    c.empty_state("No graph yet — add a document.",
                                  "Ingest a document in the Vault to build the concept map."),
                    unsafe_allow_html=True,
                )
            else:
                _render_network(graph, community_filter, size_by, selected_title)

    with rail_col:
        _render_inspector(graph, selected_title, state)


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------
def _community_filter(state: Dict[str, Any]) -> Optional[List[int]]:
    raw = st.query_params.get("community")
    if raw in (None, "", "all"):
        return None
    try:
        return [int(raw)]
    except (TypeError, ValueError):
        return None


def _visible_nodes(graph, community_filter: Optional[List[int]]) -> pd.DataFrame:
    if graph.nodes.empty:
        return graph.nodes
    frame = graph.nodes
    if community_filter:
        frame = frame[frame["community"].astype(int).isin(community_filter)]
    if len(frame) > MAX_DRAWN_NODES:
        frame = frame.sort_values("degree", ascending=False).head(MAX_DRAWN_NODES)
    return frame


def _visible_edge_count(graph, drawn: pd.DataFrame) -> int:
    if graph.relationships.empty or drawn.empty:
        return 0
    titles = set(drawn["title"].astype(str))
    return int(
        graph.relationships.apply(
            lambda r: str(r["source"]) in titles and str(r["target"]) in titles, axis=1
        ).sum()
    )


def _render_network(graph, community_filter, size_by: str, selected: Optional[str]) -> None:
    """Regenerate the map for the current filter, then embed it."""
    drawn = _visible_nodes(graph, community_filter)
    signature = hashlib.sha1(
        "|".join([
            data.output_signature(),
            str(community_filter),
            size_by,
            str(selected),
            str(len(drawn)),
        ]).encode("utf-8")
    ).hexdigest()[:12]

    target = data.CONCEPT_MAP_PATH.parent / f"concept_map_{signature}.html"
    if not target.exists():
        write_concept_map(
            drawn,
            graph.relationships,
            target,
            size_by=size_by,
            selected=selected,
            height=GRAPH_HEIGHT,
        )
        # Keep the canonical artifact in step for the QA check and the notebook.
        write_concept_map(
            graph.nodes, graph.relationships, data.CONCEPT_MAP_PATH,
            size_by=size_by, height=GRAPH_HEIGHT,
        )
        _prune_cached_maps(target)

    # Served from a path rather than inline HTML so the frame keeps the app's
    # origin, which is what lets the node-click handler reach the host page.
    st.iframe(target, height=GRAPH_HEIGHT)
    note = "Embedded network · physics on · drag to explore · click a node to inspect it"
    if community_filter:
        note += f" · filtered to community {community_filter[0]}"
    if len(drawn) < len(graph.nodes):
        note += f" · showing the {len(drawn)} most connected of {len(graph.nodes)}"
    st.markdown(f'<div class="k-graph__note">{esc(note)}</div>', unsafe_allow_html=True)


def _prune_cached_maps(keep: Path, limit: int = 6) -> None:
    """Keep only the most recent filtered maps on disk."""
    maps = sorted(
        keep.parent.glob("concept_map_*.html"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in maps[limit:]:
        try:
            stale.unlink()
        except OSError:
            pass


# -----------------------------------------------------------------------------
# Inspector
# -----------------------------------------------------------------------------
def _render_inspector(graph, title: Optional[str], state: Dict[str, Any]) -> None:
    st.markdown(
        '<div class="k-rail__head">Selected node</div>',
        unsafe_allow_html=True,
    )
    if not title or graph.entities.empty:
        st.markdown(
            '<div class="k-rail__body"><div class="k-ev__more">'
            "No concepts indexed yet.</div></div>",
            unsafe_allow_html=True,
        )
        return

    matches = graph.entities[graph.entities["title"].astype(str) == title]
    if matches.empty:
        st.markdown(
            '<div class="k-rail__body"><div class="k-ev__more">'
            "That concept is no longer in the graph.</div></div>",
            unsafe_allow_html=True,
        )
        return

    entity = matches.iloc[0]
    community = graph.community_of(title)
    degree = graph.degree_of(title)
    passages = cell_length(entity.get("text_unit_ids"))

    relationships = []
    if not graph.relationships.empty:
        touching = graph.relationships[
            (graph.relationships["source"].astype(str) == title)
            | (graph.relationships["target"].astype(str) == title)
        ].sort_values("weight", ascending=False).head(6)
        for _index, row in touching.iterrows():
            other = row["target"] if str(row["source"]) == title else row["source"]
            relationships.append({
                "title": other,
                "kind": row.get("type") or "related",
                "weight": f"{float(row.get('weight', 0)):.0f}",
            })

    st.markdown(
        f'<div class="k-rail__body k-rail__body--node">'
        + c.node_inspector(
            title=title,
            entity_type=entity.get("type", "CONCEPT"),
            community=community if community is not None else "—",
            community_label=graph.community_title(community) if community is not None else "—",
            degree=degree,
            passages=passages,
            description=entity.get("description", ""),
            relationships=relationships,
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    with st.container(key="link_node_actions", horizontal=True):
        if st.button(f"Ask about {title[:18]}", key="ask_about_node", type="primary"):
            st.session_state.messages.append({
                "role": "user",
                "content": f"What is {title} and how does it relate to other concepts?",
            })
            st.session_state.running = {
                "query": f"What is {title} and how does it relate to other concepts?",
                "mode": "local",
            }
            st.query_params.clear()
            st.query_params["screen"] = "search"
            st.query_params["mode"] = "local"
            st.rerun()
        if st.button("Open in catalog", key="open_in_catalog"):
            st.query_params.clear()
            st.query_params["screen"] = "catalog"
            st.query_params["tab"] = "concepts"
            st.query_params["term"] = title
            st.rerun()
