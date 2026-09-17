"""Catalog screen: analyst tables over the graph artifacts (mockup 1f)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from ui import components as c
from ui import data
from ui.tokens import clip, community_color, esc, highlight, query_string

PAGE_SIZES = [25, 50, 100]
DATAFRAME_THRESHOLD = 500  # above this, fall back to st.dataframe


def render(state: Dict[str, Any]) -> None:
    graph = state["graph"]
    counts = {
        "concepts": len(graph.entities),
        "relationships": len(graph.relationships),
        "nodes": len(graph.nodes),
        "briefs": len(graph.reports),
    }
    tab = st.query_params.get("tab", "concepts").lower()
    if tab not in counts:
        tab = "concepts"

    term = st.query_params.get("term", "")

    if tab == "briefs":
        split = st.container(key="split")
        main_col, rail_col = split.columns([1, 0.42], gap=None)
    else:
        main_col, rail_col = st.container(), None

    with main_col:
        with st.container(key="topbar_catalog", horizontal=True, vertical_alignment="center"):
            st.markdown(
                '<div style="display:flex;align-items:center;gap:22px;height:48px">'
                '<div class="k-topbar__title">Catalog</div>'
                f"{c.catalog_tabs(tab, counts)}</div>",
                unsafe_allow_html=True,
            )
            with st.container(key="search_catalog"):
                typed = st.text_input(
                    "Search",
                    value=term,
                    placeholder="Search this tab",
                    label_visibility="collapsed",
                    key=f"catalog_term_{tab}",
                )

        if typed != term:
            if typed:
                st.query_params["term"] = typed
            else:
                st.query_params.pop("term", None)
            st.rerun()

        with st.container(key="pad_catalog"):
            if tab == "concepts":
                _concepts(graph, typed)
            elif tab == "relationships":
                _relationships(graph, typed)
            elif tab == "nodes":
                _nodes(graph, typed)
            else:
                _briefs(graph, typed, rail_col)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def _filter(frame: pd.DataFrame, term: str, columns: Sequence[str]) -> pd.DataFrame:
    """Case-insensitive contains across every named text column."""
    if not term or frame.empty:
        return frame
    mask = pd.Series(False, index=frame.index)
    for column in columns:
        if column in frame.columns:
            mask |= frame[column].astype(str).str.contains(term, case=False, na=False, regex=False)
    return frame[mask]


def _page_size(key: str) -> int:
    raw = st.query_params.get("rows")
    try:
        size = int(raw)
        return size if size in PAGE_SIZES else PAGE_SIZES[0]
    except (TypeError, ValueError):
        return PAGE_SIZES[0]


def _page(frame: pd.DataFrame, key: str, size: int) -> pd.DataFrame:
    total_pages = max(1, (len(frame) + size - 1) // size)
    page = min(st.session_state.get(key, 0), total_pages - 1)
    st.session_state[key] = page
    return frame.iloc[page * size:(page + 1) * size]


def _footer(frame: pd.DataFrame, key: str, term: str, size: int, tab: str) -> None:
    total_pages = max(1, (len(frame) + size - 1) // size)
    page = st.session_state.get(key, 0)
    shown = min(size, max(0, len(frame) - page * size))

    with st.container(key=f"link_pager_{key}", horizontal=True, vertical_alignment="center"):
        st.markdown(c.table_footer(shown, len(frame), term), unsafe_allow_html=True)
        st.markdown(
            c.segmented(
                [
                    (str(option), f"Rows {option}",
                     query_string(screen="catalog", tab=tab, term=term or None, rows=option))
                    for option in PAGE_SIZES
                ],
                str(size),
            ),
            unsafe_allow_html=True,
        )
        if st.button("Previous", key=f"prev_{key}", disabled=page == 0):
            st.session_state[key] = page - 1
            st.rerun()
        st.markdown(
            f'<span class="k-foot__page k-mono">{page + 1} / {total_pages}</span>',
            unsafe_allow_html=True,
        )
        if st.button("Next", key=f"next_{key}", disabled=page + 1 >= total_pages):
            st.session_state[key] = page + 1
            st.rerun()
        st.download_button(
            "Export CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{key}.csv",
            mime="text/csv",
            key=f"csv_{key}",
            width="content",
        )


def _community_cell(community: Any) -> str:
    return (
        '<span style="display:flex;align-items:center;gap:6px">'
        f'<span class="k-swatch" style="background:{community_color(community)}"></span>'
        f"{esc(community)}</span>"
    )


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
def _concepts(graph, term: str) -> None:
    if graph.entities.empty:
        st.markdown(
            c.empty_card("No concepts indexed yet. Ingest a document in the Vault to build the graph."),
            unsafe_allow_html=True,
        )
        return

    frame = _filter(graph.entities, term, ["title", "type", "description"])
    if frame.empty:
        st.markdown(c.empty_card(f"No concepts match “{term}”."), unsafe_allow_html=True)
        return

    degrees: Dict[str, Dict[str, Any]] = {}
    if not graph.nodes.empty and "title" in graph.nodes.columns:
        deduped = graph.nodes.drop_duplicates(subset=["title"])
        degrees = {
            str(row["title"]): {"degree": row.get("degree", "—"), "community": row.get("community", "—")}
            for row in deduped.to_dict("records")
        }

    size = _page_size("concepts")
    if len(frame) > DATAFRAME_THRESHOLD:
        _dataframe_fallback(frame, ["title", "type", "description"])
    else:
        grid = "1.4fr .8fr 3fr .6fr .8fr"
        rows = []
        for row in _page(frame, "catalog_concepts", size).to_dict("records"):
            stats = degrees.get(str(row["title"]), {})
            description = str(row.get("description", ""))
            rows.append([
                f'<span class="k-grid__name">{highlight(row["title"], term)}</span>',
                f'<span class="k-mono k-muted" style="font-size:11px">{esc(row.get("type", ""))}</span>',
                f'<span class="k-muted k-clip" title="{esc(description)}">'
                f"{highlight(clip(description, 160), term)}</span>",
                f'<span class="k-mono" style="text-align:right">{esc(stats.get("degree", "—"))}</span>',
                _community_cell(stats.get("community", "—")),
            ])
        st.markdown(
            c.data_table(
                [("Title", ""), ("Type", ""), ("Description", ""),
                 ("Degree", "text-align:right"), ("Community", "")],
                rows,
                grid,
                row_hrefs=[
                    query_string(screen="concepts", node=row["title"])
                    for row in _page(frame, "catalog_concepts", size).to_dict("records")
                ],
            ),
            unsafe_allow_html=True,
        )
    _footer(frame, "catalog_concepts", term, size, "concepts")


def _relationships(graph, term: str) -> None:
    if graph.relationships.empty:
        st.markdown(c.empty_card("No relationships indexed yet."), unsafe_allow_html=True)
        return

    frame = _filter(graph.relationships, term, ["source", "target", "type", "description"])
    if frame.empty:
        st.markdown(c.empty_card(f"No relationships match “{term}”."), unsafe_allow_html=True)
        return

    size = _page_size("relationships")
    if len(frame) > DATAFRAME_THRESHOLD:
        _dataframe_fallback(frame, ["source", "target", "type", "weight", "description"])
    else:
        grid = "1.6fr .9fr .6fr 2.6fr"
        rows = []
        for row in _page(frame, "catalog_relationships", size).to_dict("records"):
            description = str(row.get("description", ""))
            rows.append([
                '<span class="k-grid__name k-clip">'
                f'{highlight(row["source"], term)} <span class="k-faint">→</span> '
                f'{highlight(row["target"], term)}</span>',
                f'<span class="k-mono k-muted" style="font-size:11px">'
                f'{highlight(row.get("type", "related"), term)}</span>',
                f'<span class="k-mono" style="text-align:right">'
                f'{float(row.get("weight", 0)):.0f}/10</span>',
                f'<span class="k-muted k-clip" title="{esc(description)}">'
                f"{highlight(clip(description, 170), term)}</span>",
            ])
        st.markdown(
            c.data_table(
                [("Source → Target", ""), ("Type", ""),
                 ("Weight", "text-align:right"), ("Description", "")],
                rows,
                grid,
            ),
            unsafe_allow_html=True,
        )
    _footer(frame, "catalog_relationships", term, size, "relationships")


def _nodes(graph, term: str) -> None:
    if graph.nodes.empty:
        st.markdown(c.empty_card("No graph nodes indexed yet."), unsafe_allow_html=True)
        return

    frame = _filter(graph.nodes, term, ["title"])
    if frame.empty:
        st.markdown(c.empty_card(f"No nodes match “{term}”."), unsafe_allow_html=True)
        return

    size = _page_size("nodes")
    if len(frame) > DATAFRAME_THRESHOLD:
        _dataframe_fallback(frame, ["title", "community", "degree"])
    else:
        grid = "2fr .8fr .8fr"
        page = _page(frame, "catalog_nodes", size).to_dict("records")
        rows = [
            [
                f'<span class="k-grid__name">{highlight(row["title"], term)}</span>',
                _community_cell(row.get("community", "—")),
                f'<span class="k-mono" style="text-align:right">{esc(row.get("degree", "—"))}</span>',
            ]
            for row in page
        ]
        st.markdown(
            c.data_table(
                [("Title", ""), ("Community", ""), ("Degree", "text-align:right")],
                rows,
                grid,
                row_hrefs=[query_string(screen="concepts", node=row["title"]) for row in page],
            ),
            unsafe_allow_html=True,
        )
    _footer(frame, "catalog_nodes", term, size, "nodes")


def _briefs(graph, term: str, rail_col) -> None:
    if graph.reports.empty:
        st.markdown(c.empty_card("No domain briefs generated yet."), unsafe_allow_html=True)
        return

    frame = _filter(graph.reports, term, ["title", "summary", "full_content"])
    if frame.empty:
        st.markdown(c.empty_card(f"No briefs match “{term}”."), unsafe_allow_html=True)
        return

    frame = frame.sort_values("rank", ascending=False)
    selected = st.query_params.get("community") or str(frame.iloc[0]["community"])

    grid = "2.4fr 3.4fr .7fr .6fr"
    rows = []
    hrefs = []
    active_index = None
    for position, row in enumerate(frame.to_dict("records")):
        if str(row["community"]) == str(selected):
            active_index = position
        rank = float(row.get("rank", 0) or 0)
        rank_class = "k-tag" if rank >= 7 else ("k-tag k-tag--warn" if rank >= 4 else "k-tag k-tag--plain")
        summary = str(row.get("summary", ""))
        rows.append([
            f'<span class="k-grid__name k-clip" title="{esc(row["title"])}">'
            f'{highlight(row["title"], term)}</span>',
            f'<span class="k-muted k-clip" title="{esc(summary)}">'
            f"{highlight(clip(summary, 150), term)}</span>",
            f'<span class="{rank_class}" style="justify-self:start">rank {rank:.1f}</span>',
            f'<span class="k-mono k-faint" style="text-align:right">{esc(row.get("size", "—"))}</span>',
        ])
        hrefs.append(query_string(screen="catalog", tab="briefs", term=term or None,
                                  community=row["community"]))

    st.markdown(
        c.data_table(
            [("Domain", ""), ("Summary", ""), ("Importance", ""), ("Concepts", "text-align:right")],
            rows,
            grid,
            row_hrefs=hrefs,
            active_index=active_index,
        ),
        unsafe_allow_html=True,
    )
    with st.container(key="link_pager_briefs", horizontal=True, vertical_alignment="center"):
        st.markdown(c.table_footer(len(frame), len(graph.reports), term), unsafe_allow_html=True)
        st.download_button(
            "Export CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="domain_briefs.csv",
            mime="text/csv",
            key="csv_briefs",
            width="content",
        )

    if rail_col is not None:
        match = frame[frame["community"].astype(str) == str(selected)]
        with rail_col:
            st.markdown('<div class="k-rail__head">Domain brief</div>', unsafe_allow_html=True)
            if match.empty:
                st.markdown(
                    '<div class="k-rail__body"><div class="k-ev__more">'
                    "Select a domain to read its brief.</div></div>",
                    unsafe_allow_html=True,
                )
                return
            report = match.iloc[0]
            st.markdown(
                '<div class="k-rail__body" style="gap:10px">'
                f'<div style="font:500 15px/1.3 var(--k-sans)">{esc(report["title"])}</div>'
                '<div style="display:flex;gap:6px;align-items:center;font:11px var(--k-mono);'
                'color:var(--k-faint)">'
                f'<span>community {esc(report["community"])}</span><span>·</span>'
                f'<span>{esc(report.get("size", "—"))} concepts</span><span>·</span>'
                f'<span>rank {float(report.get("rank", 0)):.1f}</span></div>'
                f'<div class="k-muted" style="line-height:1.55;font-size:12.5px">'
                f'{esc(report.get("rank_explanation", ""))}</div></div>',
                unsafe_allow_html=True,
            )
            with st.container(key="pad_brief"):
                st.markdown(str(report.get("full_content", "")))


def _dataframe_fallback(frame: pd.DataFrame, columns: List[str]) -> None:
    """Large filtered sets keep st.dataframe, per the guide's §8 note."""
    present = [column for column in columns if column in frame.columns]
    st.dataframe(frame[present], width="stretch", hide_index=True, height=520)
