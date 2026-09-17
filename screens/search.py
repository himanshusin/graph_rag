"""Search screen: thread in the middle, evidence rail on the right (mockup 1a).

Compare answers use the two-column block from 1b; the empty thread uses the
title and numbered prompts from 1c.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import streamlit as st

from core import rag
from ui import components as c
from ui import data
from ui.tokens import ACCENT, FAINT, esc, query_string

META_REFRESH_SECONDS = 0.25


def render(state: Dict[str, Any]) -> None:
    graph = state["graph"]
    mode = state["mode"]
    source = state["source"]
    _consume_prompt_link()
    _consume_pending_after_ingest()

    has_corpus = (not graph.is_empty) or data.passage_count() > 0
    messages: List[Dict[str, Any]] = st.session_state.messages

    split = st.container(key="split")
    thread_col, rail_col = split.columns([1, 0.42], gap=None)

    with thread_col:
        _top_bar(messages, mode, source)
        with st.container(key="thread"):
            if not has_corpus:
                st.markdown(
                    c.empty_state(
                        "No documents yet.",
                        "Add a document in the Vault and index it to start asking questions.",
                    ),
                    unsafe_allow_html=True,
                )
            elif not messages:
                st.markdown(
                    c.suggested_prompts(data.current_suggested_prompts()),
                    unsafe_allow_html=True,
                )

            selected_index = _selected_index(messages)
            for position, message in enumerate(messages):
                _render_message(position, message, selected=position == selected_index)

            running = st.session_state.get("running")
            if running:
                _run_and_stream(running, state)

    with rail_col:
        _render_rail(messages, source)

    _composer(mode, source, has_corpus, state["vault"], state["chunk_limit"])


# -----------------------------------------------------------------------------
# Chrome
# -----------------------------------------------------------------------------
def _top_bar(messages: List[Dict[str, Any]], mode: str, source: str) -> None:
    last_question = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    crumb = f"Thread · {last_question[:46]}" if last_question else "New thread"
    st.markdown(
        '<div style="height:48px;display:flex;align-items:center;gap:6px;padding:0 24px;'
        'border-bottom:1px solid var(--k-border);background:var(--k-panel)">'
        '<div class="k-topbar__title">Search</div><div class="k-topbar__sep">/</div>'
        f'<div class="k-topbar__crumb">{esc(crumb)}</div>'
        f'<div style="margin-left:auto">{c.source_pills(source, mode)}</div></div>',
        unsafe_allow_html=True,
    )


def _composer(mode: str, source: str, has_corpus: bool, vault, chunk_limit: int) -> None:
    """Mode chips, then the composer. Attachments go straight into the vault."""
    st.markdown(c.composer_modes(mode, source), unsafe_allow_html=True)

    placeholder = "Ask the knowledge base…" if has_corpus else "Add a document to start"
    submitted = st.chat_input(
        placeholder,
        key="composer",
        accept_file="multiple",
        file_type=["pdf", "txt", "md"],
    )
    if not submitted:
        return

    # accept_file makes the return value an object with .text and .files.
    files = getattr(submitted, "files", None) or []
    text = getattr(submitted, "text", None) if hasattr(submitted, "text") else str(submitted)

    if files:
        from screens import vault as vault_screen
        vault_screen.ingest_files(vault, files, chunk_limit=chunk_limit)
        if text:
            st.session_state.pending_after_ingest = text
        st.rerun()

    if text and has_corpus:
        _start(text, mode)


# -----------------------------------------------------------------------------
# Messages
# -----------------------------------------------------------------------------
def _selected_index(messages: List[Dict[str, Any]]) -> Optional[int]:
    """The assistant message whose evidence fills the rail; defaults to the last."""
    requested = st.query_params.get("msg")
    if requested is not None:
        try:
            index = int(requested)
            if 0 <= index < len(messages) and messages[index]["role"] == "assistant":
                return index
        except (TypeError, ValueError):
            pass
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "assistant":
            return index
    return None


def _render_message(position: int, message: Dict[str, Any], selected: bool) -> None:
    if message["role"] == "user":
        st.markdown(
            c.user_message(message["content"], data.local_user_initials()),
            unsafe_allow_html=True,
        )
        return

    if message.get("error"):
        st.markdown(
            c.assistant_meta(message.get("mode", ""), 0, message.get("elapsed"), None, selected),
            unsafe_allow_html=True,
        )
        with st.container(key=f"err_{position}"):
            st.markdown(
                c.error_block("Couldn't complete this search.", message["error"]),
                unsafe_allow_html=True,
            )
            if st.button("Retry", key=f"retry_{position}"):
                _start(message["query"], message.get("mode_key", "global"))
        return

    st.markdown(
        c.assistant_meta(
            message.get("mode", ""),
            len(message.get("citations", [])),
            message.get("elapsed"),
            message.get("grounding"),
            selected,
            select_href=query_string(screen="search", msg=position),
        ),
        unsafe_allow_html=True,
    )

    if message.get("comparison"):
        _render_compare(message["comparison"], key=str(position))
    else:
        _render_answer(message["content"], key=f"ans_msg_{position}",
                       citations=message.get("citations"))

    _render_actions(position, message)


def _render_actions(position: int, message: Dict[str, Any]) -> None:
    with st.container(key=f"link_actions_{position}", horizontal=True):
        if st.button("Copy", key=f"copy_{position}"):
            st.session_state[f"show_copy_{position}"] = not st.session_state.get(
                f"show_copy_{position}", False
            )
        if not message.get("comparison") and message.get("mode_key") != "vector":
            if st.button("Compare with vector search", key=f"cmp_btn_{position}"):
                _start(message["query"], "compare")
        if st.button("Rerun", key=f"rerun_btn_{position}"):
            _start(message["query"], message.get("mode_key", "global"))
        st.markdown(
            f'<span class="k-actions__model k-mono">{esc(message.get("model", data.MODEL_NAME))}</span>',
            unsafe_allow_html=True,
        )

    if st.session_state.get(f"show_copy_{position}"):
        body = message.get("content") or _comparison_text(message.get("comparison", {}))
        st.code(body, language="markdown")


def _comparison_text(comparison: Dict[str, Any]) -> str:
    if not comparison:
        return ""
    return (
        f"## Graph · {comparison.get('graph_mode', '')}\n\n{comparison.get('graph_answer', '')}\n\n"
        f"## Vector · passages\n\n{comparison.get('vector_answer', '')}"
    )


def _render_answer(text: str, key: str, citations: Optional[List[Dict[str, Any]]] = None) -> None:
    """Render markdown with [n] turned into citation links, when n resolves."""
    with st.container(key=key):
        st.markdown(c.mark_citations(text, citations), unsafe_allow_html=True)


def _render_compare(comparison: Dict[str, Any], key: str) -> None:
    synced = st.session_state.get("compare_synced", False)
    with st.container(key=f"cmprow_{key}"):
        with st.container(key=f"cmptoggles_{key}", horizontal=True):
            st.toggle("Synced scroll", key=f"synced_{key}",
                      value=synced, on_change=_flip_synced)
        if st.session_state.get(f"synced_{key}"):
            st.markdown(
                '<style>[class*="st-key-cmpcard_"]{max-height:420px;overflow-y:auto}</style>',
                unsafe_allow_html=True,
            )
        left, right = st.columns(2, gap="small")
        with left:
            with st.container(key=f"cmpcard_{key}_graph"):
                st.markdown(
                    c.compare_head(
                        f"Graph · {comparison.get('graph_mode', 'DRIFT')}",
                        comparison.get("graph_meta", ""),
                        comparison.get("graph_grounding"),
                        ACCENT,
                    ),
                    unsafe_allow_html=True,
                )
                _render_answer(comparison["graph_answer"], key=f"ans_cmp_{key}_graph",
                               citations=comparison.get("graph_citations"))
                st.markdown(c.compare_footnotes(comparison.get("graph_citations", [])),
                            unsafe_allow_html=True)
        with right:
            with st.container(key=f"cmpcard_{key}_vector"):
                st.markdown(
                    c.compare_head(
                        "Vector · passages",
                        comparison.get("vector_meta", ""),
                        comparison.get("vector_grounding"),
                        FAINT,
                    ),
                    unsafe_allow_html=True,
                )
                _render_answer(comparison["vector_answer"], key=f"ans_cmp_{key}_vector",
                               citations=comparison.get("vector_citations"))
                st.markdown(c.compare_footnotes(comparison.get("vector_citations", [])),
                            unsafe_allow_html=True)


def _flip_synced() -> None:
    st.session_state.compare_synced = not st.session_state.get("compare_synced", False)


# -----------------------------------------------------------------------------
# Evidence rail
# -----------------------------------------------------------------------------
def _render_rail(messages: List[Dict[str, Any]], source: str) -> None:
    selected_index = _selected_index(messages)
    message = messages[selected_index] if selected_index is not None else None
    citations = message.get("citations", []) if message else []
    allowed = rag.SOURCE_KINDS.get(source, rag.SOURCE_KINDS["all"])

    header_left, header_right = st.columns([1, 1], vertical_alignment="center")
    with header_left:
        st.markdown(
            '<div class="k-rail__head" style="border-bottom:none;height:auto;padding:14px 0 0 16px">'
            f'Evidence <span class="k-rail__count">{len(citations)}</span></div>',
            unsafe_allow_html=True,
        )
    with header_right:
        show_context = st.toggle(
            "Show context",
            value=st.session_state.get("show_raw_context", False),
            key="toggle_raw_context",
        )
    st.session_state.show_raw_context = show_context

    if st.session_state.get("running"):
        st.markdown(c.evidence_skeleton(2), unsafe_allow_html=True)
        return

    if show_context:
        st.markdown(c.raw_context(message.get("raw_context", "") if message else ""),
                    unsafe_allow_html=True)

    st.markdown(
        c.evidence_rail(citations, allowed_kinds=allowed),
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Running a query
# -----------------------------------------------------------------------------
def _start(question: str, mode: str) -> None:
    """Queue a run: record the question, then rerun so the answer streams in."""
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.running = {"query": question, "mode": mode}
    st.query_params.pop("msg", None)
    st.rerun()


def _consume_pending_after_ingest() -> None:
    """Run the question that accompanied an attachment, once indexing finished."""
    pending = st.session_state.pop("pending_after_ingest", None)
    if pending:
        _start(pending, st.query_params.get("mode", "global"))


def _consume_prompt_link() -> None:
    """Handle ?prompt=n from the suggested-prompt list."""
    raw = st.query_params.get("prompt")
    if raw is None:
        return
    st.query_params.pop("prompt", None)
    prompts = data.current_suggested_prompts()
    try:
        index = int(raw) - 1
    except (TypeError, ValueError):
        return
    if 0 <= index < len(prompts):
        _start(prompts[index]["text"], prompts[index]["mode"])


def _run_and_stream(running: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Retrieve, stream the answer, then store the finished message."""
    question = running["query"]
    mode = running["mode"]
    graph = state["graph"]
    vault = state["vault"]
    source = state["source"]
    collection = data.chroma_collection()
    llm = data.get_llm(temperature=state["temperature"])

    meta_slot = st.empty()
    meta_slot.markdown(
        c.assistant_meta(rag.MODE_LABELS[mode], 0, 0.0),
        unsafe_allow_html=True,
    )
    body_slot = st.empty()
    body_slot.markdown(c.answer_skeleton(), unsafe_allow_html=True)

    started = time.perf_counter()
    try:
        if mode == "compare":
            message = _run_compare(question, graph, collection, vault, source,
                                   state["citations_k"], llm, body_slot, meta_slot, started)
        else:
            message = _run_single(question, mode, graph, collection, vault, source,
                                  state["citations_k"], llm, body_slot, meta_slot, started)
    except Exception as exc:
        message = {
            "role": "assistant",
            "content": "",
            "mode": rag.MODE_LABELS[mode],
            "mode_key": mode,
            "query": question,
            "citations": [],
            "elapsed": round(time.perf_counter() - started, 2),
            "model": data.MODEL_NAME,
            "error": f"{type(exc).__name__}: {exc}",
        }

    st.session_state.messages.append(message)
    st.session_state.running = None
    meta_slot.empty()
    body_slot.empty()
    st.rerun()


def _run_single(question, mode, graph, collection, vault, source, citations_k,
                llm, body_slot, meta_slot, started) -> Dict[str, Any]:
    retrieval = rag.retrieve(
        mode, question, data=graph, collection=collection, vault=vault,
        source=source, top_k=citations_k,
    )
    meta_slot.markdown(
        c.assistant_meta(rag.MODE_LABELS[mode], len(retrieval.citations),
                         time.perf_counter() - started),
        unsafe_allow_html=True,
    )
    text = _stream_into(
        body_slot,
        rag.answer_stream(retrieval, llm),
        lambda elapsed: c.assistant_meta(
            rag.MODE_LABELS[mode], len(retrieval.citations), elapsed
        ),
        meta_slot,
        started,
    )
    return {
        "role": "assistant",
        "content": text,
        "mode": rag.MODE_LABELS[mode],
        "mode_key": mode,
        "query": question,
        "citations": retrieval.citations,
        "notes": retrieval.notes,
        "elapsed": round(time.perf_counter() - started, 2),
        "grounding": rag.compute_grounding(text),
        "raw_context": retrieval.context,
        "model": data.MODEL_NAME,
        "source_filter": source,
    }


def _run_compare(question, graph, collection, vault, source, citations_k,
                 llm, body_slot, meta_slot, started) -> Dict[str, Any]:
    """Run the graph and vector engines concurrently, then render both columns."""
    retrievals = rag.retrieve_compare(
        question, data=graph, collection=collection, vault=vault,
        source=source, top_k=citations_k,
    )
    graph_retrieval, vector_retrieval = retrievals["graph"], retrievals["vector"]

    merged = rag.renumber(graph_retrieval, vector_retrieval)
    meta_slot.markdown(
        c.assistant_meta(rag.MODE_LABELS["compare"], len(merged),
                         time.perf_counter() - started),
        unsafe_allow_html=True,
    )
    # Both columns show their own skeleton while the two answers generate.
    body_slot.markdown(
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'
        f'<div class="k-card"><div class="k-card__body">{c.answer_skeleton()}</div></div>'
        f'<div class="k-card"><div class="k-card__body">{c.answer_skeleton()}</div></div></div>',
        unsafe_allow_html=True,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        graph_future = pool.submit(rag.answer, graph_retrieval, llm)
        vector_future = pool.submit(rag.answer, vector_retrieval, llm)
        graph_answer = graph_future.result()
        vector_answer = vector_future.result()

    graph_grounding = rag.compute_grounding(graph_answer)
    vector_grounding = rag.compute_grounding(vector_answer)

    return {
        "role": "assistant",
        "content": "",
        "mode": rag.MODE_LABELS["compare"],
        "mode_key": "compare",
        "query": question,
        "citations": merged,
        "elapsed": round(time.perf_counter() - started, 2),
        "grounding": max(graph_grounding, vector_grounding),
        "model": data.MODEL_NAME,
        "source_filter": source,
        "raw_context": (
            f"### Graph context ({graph_retrieval.mode})\n{graph_retrieval.context}\n\n"
            f"### Vector context\n{vector_retrieval.context}"
        ),
        "comparison": {
            "graph_answer": graph_answer,
            "vector_answer": vector_answer,
            "graph_mode": rag.MODE_LABELS[graph_retrieval.mode].split(" ")[0].upper(),
            "graph_meta": graph_retrieval.summary(),
            "vector_meta": vector_retrieval.summary() or "no passages above threshold",
            "graph_grounding": graph_grounding,
            "vector_grounding": vector_grounding,
            "graph_citations": graph_retrieval.citations,
            "vector_citations": vector_retrieval.citations,
        },
    }


def _stream_into(body_slot, chunks, meta_builder, meta_slot, started) -> str:
    """Write streamed fragments into a slot, refreshing the elapsed timer as it goes."""
    collected: List[str] = []
    last_meta = 0.0
    for chunk in chunks:
        collected.append(chunk)
        body_slot.markdown("".join(collected))
        now = time.perf_counter()
        if now - started - last_meta >= META_REFRESH_SECONDS:
            last_meta = now - started
            meta_slot.markdown(meta_builder(last_meta), unsafe_allow_html=True)
    return "".join(collected)
