"""Vault screen: document registry, ingest progress and the inspector (mockup 1d)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from core.pipeline import GraphRAGEngine
from ui import components as c
from ui import data
from ui.tokens import esc, query_string

REGISTRY_GRID = "2.2fr .7fr .7fr .8fr .8fr 1.2fr 1fr"
STEPS = ["Persist", "Extract text", "Extract tables", "Build graph", "Sync vectors"]

# Maps the engine's progress steps onto the registry's five ingest steps.
STEP_ALIASES = {
    "Chunk documents": "Extract text",
    "Extract concepts": "Build graph",
    "Cluster communities": "Build graph",
    "Write reports": "Build graph",
    "Sync vectors": "Sync vectors",
    "Draw concept map": "Sync vectors",
}


def render(state: Dict[str, Any]) -> None:
    vault = state["vault"]

    with st.container(key="topbar_vault", horizontal=True, vertical_alignment="center"):
        st.markdown(
            '<div class="k-topbar__title">Vault</div>'
            '<div class="k-topbar__note">Retained permanently · SHA-256 deduplicated</div>',
            unsafe_allow_html=True,
        )
        if st.button("Add document", type="primary", key="vault_add"):
            st.session_state.vault_add_open = not st.session_state.get("vault_add_open", False)
            st.rerun()

    with st.container(key="pad_vault"):
        catalog = sorted(
            vault.get_catalog(),
            key=lambda d: str(d.get("uploaded_at", "")),
            reverse=True,
        )

        if st.session_state.get("vault_add_open"):
            _uploader(vault, catalog)

        if not catalog:
            st.markdown(
                c.empty_state("No documents yet.", "Use Add document to ingest a PDF, text or markdown file."),
                unsafe_allow_html=True,
            )
            return

        selected_id = st.query_params.get("doc") or catalog[0]["id"]
        known = {d["id"] for d in catalog}
        if selected_id not in known:
            selected_id = catalog[0]["id"]

        st.markdown(_registry_html(catalog, selected_id), unsafe_allow_html=True)

        document = vault.get_document(selected_id)
        if document:
            _inspector(vault, document, state)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
def _registry_html(
    catalog: List[Dict[str, Any]],
    selected_id: Optional[str],
    pending: Optional[Dict[str, Any]] = None,
) -> str:
    head = (
        f'<div class="k-grid__head" style="grid-template-columns:{REGISTRY_GRID}">'
        "<span>Document</span><span>Format</span><span>Size</span><span>Pages</span>"
        "<span>Tables</span><span>Uploaded</span><span>Status</span></div>"
    )

    rows = []
    if pending:
        rows.append(_pending_row(pending))
    for document in catalog:
        if pending and document["id"] == pending.get("id"):
            continue
        rows.append(_registry_row(document, selected_id))
    return f'<div class="k-card">{head}{"".join(rows)}</div>'


def _registry_row(document: Dict[str, Any], selected_id: Optional[str]) -> str:
    status = _status_cell(document)
    classes = "k-grid__row"
    if document["id"] == selected_id:
        classes += " k-grid__row--on"
    return (
        f'<a class="{classes}" href="{query_string(screen="vault", doc=document["id"])}" '
        f'target="_self" style="grid-template-columns:{REGISTRY_GRID}">'
        f'<span><span class="k-grid__name">{esc(document.get("title", ""))}</span><br>'
        f'<span class="k-grid__id">{esc(document["id"])}</span></span>'
        f'<span>{esc(document.get("source_type", "—"))}</span>'
        f'<span class="k-num">{esc(format_size(document.get("file_size_kb")))}</span>'
        f'<span class="k-num">{esc(_or_dash(document.get("page_count")))}</span>'
        f'<span class="k-num">{esc(_or_dash(document.get("table_count")))}</span>'
        f'<span class="k-muted">{esc(format_uploaded(document.get("uploaded_at", "")))}</span>'
        f"{status}</a>"
    )


def _pending_row(pending: Dict[str, Any]) -> str:
    """The in-progress ingest, shown as a registry row with a 2px progress bar."""
    label = pending.get("label", "Working")
    fraction = float(pending.get("fraction", 0.0) or 0.0)
    return (
        f'<div class="k-grid__row" style="grid-template-columns:{REGISTRY_GRID}">'
        f'<span><span class="k-grid__name">{esc(pending.get("title", ""))}</span><br>'
        f'<span class="k-grid__id">{esc(pending.get("id", "…"))}</span></span>'
        f'<span>{esc(pending.get("format", "—"))}</span>'
        f'<span class="k-num">{esc(pending.get("size", "—"))}</span>'
        f'<span class="k-num">{esc(_or_dash(pending.get("pages")))}</span>'
        f'<span class="k-num">{esc(_or_dash(pending.get("tables")))}</span>'
        '<span class="k-muted">Just now</span>'
        f'<span>{c.status_dot("progress", label)}{c.progress_bar(fraction)}</span></div>'
    )


def _status_cell(document: Dict[str, Any]) -> str:
    status = str(document.get("status", "")).lower()
    if "needs" in status:
        return f'<span>{c.status_dot("warn", "Needs re-index")}</span>'
    if document.get("chunks_indexed"):
        return f'<span>{c.status_dot("ok", "Indexed")}</span>'
    return f'<span>{c.status_dot("idle", "Not indexed")}</span>'


def format_size(size_kb: Any) -> str:
    try:
        size_kb = float(size_kb)
    except (TypeError, ValueError):
        return "—"
    return f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.0f} KB"


def format_uploaded(value: Any) -> str:
    """Relative for the last 24 hours, absolute before that."""
    try:
        stamp = pd.to_datetime(value)
    except (ValueError, TypeError):
        return str(value)[:16] or "—"
    if pd.isna(stamp):
        return "—"
    delta = pd.Timestamp.now() - stamp
    seconds = delta.total_seconds()
    if 0 <= seconds < 60:
        return "Just now"
    if 0 <= seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if 0 <= seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return stamp.strftime("%b %d, %H:%M")


def _or_dash(value: Any) -> Any:
    if value in (None, 0, "", "0"):
        return "—"
    return value


# -----------------------------------------------------------------------------
# Inspector
# -----------------------------------------------------------------------------
def _inspector(vault, document: Dict[str, Any], state: Dict[str, Any]) -> None:
    doc_id = document["id"]
    tables = vault.get_tables(doc_id)
    left, right = st.columns(2, gap="medium")

    with left:
        indexed = document.get("chunks_indexed") or 0
        total = document.get("chunks_total") or 0
        chunk_text = f"{indexed} of {total}" if total else "not indexed"
        st.markdown(
            '<div class="k-card"><div class="k-card__body">'
            '<div style="display:flex;align-items:center;margin-bottom:12px">'
            f'<span class="k-card__title">{esc(document.get("title", ""))}</span></div>'
            '<div class="k-kv">'
            f'<span>SHA-256</span><span class="k-mono" style="font-size:11px;word-break:break-all" '
            f'title="{esc(document.get("sha256", ""))}">{esc(document.get("sha256", ""))}</span>'
            f'<span>Storage</span><span class="k-mono" style="font-size:11.5px;word-break:break-all">'
            f'{esc(Path(document.get("storage_path", "")).name)}</span>'
            f'<span>Characters</span><span class="k-mono" style="font-size:11.5px">'
            f'{document.get("char_count", 0):,}</span>'
            f'<span>Chunks in graph</span><span class="k-mono" style="font-size:11.5px">'
            f'{esc(chunk_text)}</span>'
            f'<span>Origin</span><span>{esc(document.get("metadata", {}).get("origin", "User upload"))}</span>'
            "</div></div></div>",
            unsafe_allow_html=True,
        )

        with st.container(key="link_actions_vault", horizontal=True):
            if st.button("Re-index graph", key=f"reindex_{doc_id}"):
                _rebuild(vault, state, chunk_limit=state["chunk_limit"])
            with st.container(key="link_danger", horizontal=True):
                _delete_controls(vault, doc_id)

    with right:
        _tables_panel(tables, doc_id)

    selected_table_id = st.query_params.get("table")
    if selected_table_id:
        _table_view(tables, selected_table_id)


def _delete_controls(vault, doc_id: str) -> None:
    confirm_key = f"confirm_del_{doc_id}"
    if st.session_state.get(confirm_key):
        st.markdown(
            f'<span class="k-faint" style="font-size:12px">Delete {esc(doc_id[:12])}…?</span>',
            unsafe_allow_html=True,
        )
        if st.button("Delete", key=f"do_delete_{doc_id}"):
            vault.delete_document(doc_id, chroma_dir=str(data.CHROMA_DIR))
            vault.mark_needs_reindex()
            st.session_state[confirm_key] = False
            st.query_params.pop("doc", None)
            data.clear_caches()
            st.rerun()
        if st.button("Cancel", key=f"cancel_delete_{doc_id}"):
            st.session_state[confirm_key] = False
            st.rerun()
    else:
        if st.button("Delete", key=f"delete_{doc_id}"):
            st.session_state[confirm_key] = True
            st.rerun()


def _tables_panel(tables: List[Dict[str, Any]], doc_id: str) -> None:
    if not tables:
        st.markdown(
            '<div class="k-card"><div class="k-card__body">'
            '<div style="display:flex;align-items:center;margin-bottom:12px">'
            '<span class="k-card__title">Extracted tables</span></div>'
            '<div class="k-faint">No structured tables were found in this document.</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    rows = "".join(
        f'<a href="{query_string(screen="vault", doc=doc_id, table=table.get("table_id"))}" '
        'target="_self" style="display:flex;gap:10px;align-items:center;color:var(--k-text);'
        'padding:3px 0">'
        f'<span class="k-mono k-faint" style="font-size:11px;width:34px;flex:none">'
        f'p. {esc(table.get("page", 1))}</span>'
        f'<span class="k-clip">{esc(table.get("title", "Table"))}</span>'
        f'<span class="k-mono k-faint" style="margin-left:auto;font-size:11px;flex:none">'
        f'{table.get("rows_count", 0)} × {len(table.get("columns", []))}</span></a>'
        for table in tables[:5]
    )
    more = (
        f'<div class="k-faint" style="font-size:11.5px;padding-top:4px">{len(tables) - 5} more</div>'
        if len(tables) > 5 else ""
    )
    st.markdown(
        '<div class="k-card"><div class="k-card__body">'
        '<div style="display:flex;align-items:center;margin-bottom:12px">'
        '<span class="k-card__title">Extracted tables</span>'
        f'<span class="k-mono k-faint" style="margin-left:6px;font-size:11px">{len(tables)}</span>'
        '<span class="k-faint" style="margin-left:auto;font-size:11.5px">rows × columns</span></div>'
        f'<div style="display:grid;gap:4px">{rows}{more}</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def _table_view(tables: List[Dict[str, Any]], table_id: str) -> None:
    """Full-width view of one extracted table plus its row facts."""
    table = next((t for t in tables if t.get("table_id") == table_id), None)
    if not table:
        return
    st.markdown(
        '<div class="k-card"><div class="k-card__head">'
        f'<span class="k-card__title">{esc(table.get("title", "Table"))}</span>'
        f'<span class="k-mono k-faint" style="font-size:11px">page {esc(table.get("page", 1))} · '
        f'{table.get("rows_count", 0)} × {len(table.get("columns", []))}</span></div></div>',
        unsafe_allow_html=True,
    )
    records = table.get("records") or []
    if records:
        st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
    elif table.get("markdown"):
        st.markdown(table["markdown"])

    facts = table.get("row_facts") or []
    if facts:
        with st.expander(f"Row facts ({len(facts)})"):
            for fact in facts[:40]:
                st.markdown(f"- `{fact}`")


# -----------------------------------------------------------------------------
# Ingest
# -----------------------------------------------------------------------------
def _uploader(vault, catalog: List[Dict[str, Any]]) -> None:
    with st.container(key="card_upload"):
        uploaded = st.file_uploader(
            "Add a PDF, text or markdown document",
            type=["pdf", "txt", "md"],
            key="vault_file",
        )
        index_now = st.checkbox("Re-index graph now", value=True, key="vault_index_now")
        chunk_limit = st.slider(
            "Chunk limit per document", min_value=1, max_value=40, value=8,
            help="Caps how many chunks of each document are sent for extraction.",
            key="vault_chunk_limit",
        )
        if not uploaded:
            return
        if st.button("Upload & retain", type="primary", key="vault_upload_go"):
            _ingest(vault, uploaded, catalog, index_now, chunk_limit)


def _ingest(vault, uploaded, catalog, index_now: bool, chunk_limit: int) -> None:
    """Persist the upload, then rebuild the graph over the whole vault."""
    registry_slot = st.empty()
    pending = {
        "title": uploaded.name,
        "id": "hashing…",
        "format": Path(uploaded.name).suffix.lstrip(".").upper() or "TXT",
        "size": format_size(len(uploaded.getvalue()) / 1024),
        "label": "Persist 1 / 5",
        "fraction": 0.05,
    }

    def paint():
        registry_slot.markdown(
            _registry_html(catalog, None, pending), unsafe_allow_html=True
        )

    paint()
    with st.status("Adding the document", expanded=True) as status:
        status.write("Persisting to the vault and extracting text and tables")
        stored = vault.store_document(
            filename=uploaded.name,
            content_bytes=uploaded.getvalue(),
            source_type="pdf" if uploaded.name.lower().endswith(".pdf") else "txt",
            metadata={"origin": "User upload"},
        )
        pending.update({
            "id": stored["id"],
            "pages": stored.get("page_count"),
            "tables": stored.get("table_count"),
            "label": "Extract tables 3 / 5",
            "fraction": 0.3,
        })
        paint()

        if not index_now:
            vault.mark_needs_reindex()
            status.update(label=f"Stored {stored['filename']}", state="complete")
            data.clear_caches()
            st.session_state.vault_add_open = False
            st.rerun()

        last_paint = [0.0]

        def on_progress(fraction: float, message: str, stats: Dict[str, Any]):
            step = STEP_ALIASES.get(stats.get("step", ""), "Build graph")
            index = STEPS.index(step) + 1 if step in STEPS else 4
            label = message
            if stats.get("chunk_total"):
                label = f"Extracting {stats.get('chunk_current', 0)} / {stats['chunk_total']}"
            pending.update({
                "label": f"{label}" if stats.get("chunk_total") else f"{step} {index} / 5",
                "fraction": 0.3 + 0.7 * max(0.0, min(1.0, fraction)),
            })
            now = time.perf_counter()
            if now - last_paint[0] > 0.4:
                last_paint[0] = now
                paint()
                status.write(message)

        try:
            result = _run_engine(vault, chunk_limit, on_progress)
        except Exception as exc:
            status.update(label="Indexing failed", state="error")
            st.markdown(
                c.error_block("Couldn't index this document.", f"{type(exc).__name__}: {exc}"),
                unsafe_allow_html=True,
            )
            return

        status.update(
            label=(
                f"Indexed {result['entities_count']} concepts, "
                f"{result['relationships_count']} relationships, "
                f"{result['communities_count']} communities in {result['elapsed_seconds']}s"
            ),
            state="complete",
        )

    data.clear_caches()
    st.session_state.vault_add_open = False
    st.rerun()


def ingest_files(vault, files, chunk_limit: int = 8, index_now: bool = True) -> List[str]:
    """Store uploaded files and re-index the vault. Returns the stored ids.

    Shared with the composer's Attach control so both paths behave identically.
    """
    stored_ids: List[str] = []
    with st.status("Adding to the vault", expanded=True) as status:
        for upload in files:
            status.write(f"Persisting {upload.name}")
            stored = vault.store_document(
                filename=upload.name,
                content_bytes=upload.getvalue(),
                source_type="pdf" if upload.name.lower().endswith(".pdf") else "txt",
                metadata={"origin": "Composer attachment"},
            )
            stored_ids.append(stored["id"])
            status.write(
                f"Stored {stored['id']} · {stored.get('page_count') or '—'} pages · "
                f"{stored.get('table_count', 0)} tables"
            )

        if not index_now:
            vault.mark_needs_reindex()
            status.update(label=f"Stored {len(stored_ids)} document(s)", state="complete")
            return stored_ids

        try:
            result = _run_engine(vault, chunk_limit, lambda f, m, s: status.write(m))
        except Exception as exc:
            status.update(label="Indexing failed", state="error")
            st.markdown(
                c.error_block("Couldn't index the attachment.", f"{type(exc).__name__}: {exc}"),
                unsafe_allow_html=True,
            )
            return stored_ids
        status.update(
            label=(f"Indexed {result['entities_count']} concepts from "
                   f"{len(result['documents'])} document(s) in {result['elapsed_seconds']}s"),
            state="complete",
        )
    data.clear_caches()
    return stored_ids


def _run_engine(vault, chunk_limit: int, on_progress=None) -> Dict[str, Any]:
    """Rebuild the graph from every retained document and record the chunk counts."""
    documents = []
    for entry in vault.get_catalog():
        text = vault.get_document_text(entry["id"])
        if text.strip():
            documents.append({"id": entry["id"], "title": entry["title"], "text": text})
    if not documents:
        raise ValueError("No readable documents in the vault.")

    engine = GraphRAGEngine(
        output_dir=str(data.OUTPUT_DIR),
        chroma_dir=str(data.CHROMA_DIR),
        notebook_dir=str(data.CONCEPT_MAP_PATH.parent),
        model_name=data.MODEL_NAME,
        temperature=0.0,
    )
    result = engine.build_from_documents(
        documents,
        max_chunks_per_doc=chunk_limit,
        progress_callback=on_progress,
    )
    for entry in result.get("documents", []):
        vault.mark_indexed(
            entry["document_id"], entry["chunks_indexed"], entry["chunks_total"]
        )
    return result


def _rebuild(vault, state: Dict[str, Any], chunk_limit: int) -> None:
    """Re-index every retained document from the inspector's action."""
    with st.status("Re-indexing the knowledge graph", expanded=True) as status:
        def on_progress(fraction: float, message: str, stats: Dict[str, Any]):
            status.write(message)

        try:
            result = _run_engine(vault, chunk_limit, on_progress)
        except Exception as exc:
            status.update(label="Re-index failed", state="error")
            st.markdown(
                c.error_block("Couldn't re-index the graph.", f"{type(exc).__name__}: {exc}"),
                unsafe_allow_html=True,
            )
            return
        status.update(
            label=f"Re-indexed {result['entities_count']} concepts in {result['elapsed_seconds']}s",
            state="complete",
        )
    data.clear_caches()
    st.rerun()
