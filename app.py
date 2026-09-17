import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Tuple, List, Optional

import streamlit as st
import pandas as pd
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
import chromadb
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import importlib
import core.vault
import core.pipeline
import core.change_manager
import core.design
importlib.reload(core.vault)
importlib.reload(core.pipeline)
importlib.reload(core.change_manager)
importlib.reload(core.design)

from core.vault import DocumentVault
from core.pipeline import DocumentIngestor, GraphRAGEngine
from core.change_manager import ChangeManagementAgent
from core import design

MODEL_NAME = "gpt-4o-mini"

# -----------------------------------------------------------------------------
# 1. Page configuration & design system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Knowledge · GraphRAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")

vault = DocumentVault(vault_dir="./vault")
change_agent = ChangeManagementAgent()
app_version = change_agent.get_version()

# -----------------------------------------------------------------------------
# 2. Data & client caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_graph_data(root_path: str = "./ragtest") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = Path(root_path) / "output"
    entities_path = out_dir / "create_final_entities.parquet"
    relationships_path = out_dir / "create_final_relationships.parquet"
    nodes_path = out_dir / "create_final_nodes.parquet"
    reports_path = out_dir / "create_final_community_reports.parquet"

    entities = pd.read_parquet(entities_path) if entities_path.exists() else pd.DataFrame()
    relationships = pd.read_parquet(relationships_path) if relationships_path.exists() else pd.DataFrame()
    nodes = pd.read_parquet(nodes_path) if nodes_path.exists() else pd.DataFrame()
    reports = pd.read_parquet(reports_path) if reports_path.exists() else pd.DataFrame()

    return entities, relationships, nodes, reports

@st.cache_resource(show_spinner=False)
def load_chroma_db(path: str = "./notebook/chromadb"):
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name="paper_collection")
    return client, collection

@st.cache_resource(show_spinner=False)
def get_llm(temperature: float = 0.2, model_name: str = MODEL_NAME):
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

entities_df, relationships_df, nodes_df, community_df = load_graph_data()
chroma_client, paper_collection = load_chroma_db()
catalog_docs = vault.get_catalog()
all_vault_tables = vault.get_tables() if hasattr(vault, "get_tables") else []


def append_table_context(query: str, context_lines: List[str], citations: List[Dict[str, Any]], top_k: int = 2):
    """Retrieve matched structured tables and inject markdown tables & row facts into LLM context."""
    matched_tables = vault.find_relevant_tables(query, top_k=top_k)
    if matched_tables:
        context_lines.append("\n### Verified structured tables & quantitative metric data:")
        for t in matched_tables:
            context_lines.append(f"#### Table: {t.get('title', 'Extracted Table')} (Page {t.get('page', 1)})")
            context_lines.append(t.get("markdown", ""))
            if t.get("row_facts"):
                context_lines.append("**Row Observations & Numerical Facts:**\n" + "\n".join(t["row_facts"][:8]))

            citations.append({
                "index": len(citations) + 1,
                "kind": "Table",
                "name": t.get("title", "Extracted table"),
                "meta": f"p. {t.get('page', 1)}",
                "source": "tables",
                "document_title": f"Table: {t.get('title', 'Table')}",
                "chunk_id": t.get("table_id", "tab_ref"),
                "excerpt": (
                    f"{len(t.get('columns', []))} columns · {t.get('rows_count', 0)} rows: "
                    f"{', '.join(t.get('columns', [])[:4])}"
                ),
                "relevance": "Verified quantitative table",
                "is_table": True,
                "table_markdown": t.get("markdown", "")
            })

# -----------------------------------------------------------------------------
# 3. Query reasoning engines with source citations & numerical grounding
# -----------------------------------------------------------------------------
def query_graphrag_global(query: str, reports: pd.DataFrame, llm: ChatOpenAI) -> Tuple[str, str, List[Dict[str, Any]]]:
    context_lines = ["### Executive Thematic Domain Summaries:"]
    citations = []
    for idx, row in reports.iterrows():
        context_lines.append(f"#### [{idx + 1}] {row['title']}\n{row['summary']}\n")
        citations.append({
            "index": idx + 1,
            "kind": "Domain",
            "name": row["title"],
            "meta": f"rank {row['rank']}",
            "source": "graph",
            "document_title": f"Domain: {row['title']}",
            "chunk_id": f"domain_{row['community']}",
            "excerpt": row["summary"],
            "relevance": f"Rank: {row['rank']}",
            "is_table": False
        })

    append_table_context(query, context_lines, citations, top_k=2)
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an Executive AI Strategic Advisor performing a Macro-Level Strategic Synthesis across business domains.\n"
        "Synthesize an executive-level summary answering the inquiry across the domain reports and structured tables below.\n"
        "When the inquiry involves numbers, metrics, evaluations, benchmark scores, or quantitative facts, quote the exact numerical values, percentages, units, and comparative results directly from the verified domain reports and structured tables.\n"
        "Include bracketed citation footnotes like [1], [2] referencing specific supporting sources.\n\n"
        "Thematic Domain Reports & Quantitative Data:\n{context}\n\n"
        "Executive Inquiry: {query}\n\n"
        "Structure: Executive Summary, Strategic & Quantitative Analysis, Core Trade-offs, and Actionable Recommendations."
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text, citations

def query_graphrag_local(query: str, entities: pd.DataFrame, rels: pd.DataFrame, llm: ChatOpenAI, top_k: int = 15) -> Tuple[str, str, List[Dict[str, Any]]]:
    context_lines = ["### Identified Concepts & Metrics:"]
    citations = []
    for idx, row in entities.head(top_k).iterrows():
        context_lines.append(f"- [{idx + 1}] **{row['title']}** ({row['type']}): {row['description']}")
        citations.append({
            "index": idx + 1,
            "kind": "Concept",
            "name": row["title"],
            "meta": str(row["type"]),
            "source": "graph",
            "document_title": f"Concept: {row['title']} ({row['type']})",
            "chunk_id": f"entity_{row['human_readable_id']}",
            "excerpt": row["description"],
            "relevance": "Direct graph node",
            "is_table": False
        })

    context_lines.append("\n### Direct Relational Dependencies & Quantitative Measurements:")
    for _, row in rels.head(top_k).iterrows():
        context_lines.append(f"- **{row['source']}** ➔ **{row['target']}** (Strength: {row['weight']}/10): {row['description']}")

    append_table_context(query, context_lines, citations, top_k=2)
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an Executive AI Strategic Advisor performing Targeted Fact & Concept Lookup.\n"
        "Provide a crisp, actionable business response grounded in the verified concept network and structured tables below.\n"
        "When the inquiry involves numbers, metrics, evaluations, benchmark scores, or quantitative facts, quote the exact numerical values, percentages, units, and comparative results directly from the verified concept network and structured tables.\n"
        "Include citation footnotes like [1], [2] where relevant.\n\n"
        "Verified Knowledge Network & Tables:\n{context}\n\n"
        "Executive Question: {query}\n\n"
        "Format with clear bullet points, quantitative facts, strategic implications, and concise takeaways."
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text, citations

def query_graphrag_drift(query: str, reports: pd.DataFrame, rels: pd.DataFrame, llm: ChatOpenAI) -> Tuple[str, str, List[Dict[str, Any]]]:
    context_lines = ["### Thematic Strategy Briefs:"]
    citations = []
    for idx, row in reports.head(6).iterrows():
        context_lines.append(f"- [{idx + 1}] **{row['title']}**: {row['summary']}")
        citations.append({
            "index": idx + 1,
            "kind": "Domain",
            "name": row["title"],
            "meta": f"rank {row['rank']}" if "rank" in row else "",
            "source": "graph",
            "document_title": row["title"],
            "chunk_id": f"report_{row['community']}",
            "excerpt": row["summary"],
            "relevance": "Strategic domain",
            "is_table": False
        })

    context_lines.append("\n### Granular Cross-Functional Dependencies & Metrics:")
    for _, row in rels.head(20).iterrows():
        context_lines.append(f"- **{row['source']}** <-> **{row['target']}**: {row['description']}")

    append_table_context(query, context_lines, citations, top_k=2)
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an Executive AI Strategic Advisor performing a Deep-Dive Cross-Functional Analysis.\n"
        "Answer the business inquiry by connecting macro business strategy with granular operational and technological dependencies.\n"
        "When the inquiry involves numbers, metrics, evaluations, benchmark scores, or quantitative facts, quote the exact numerical values, percentages, units, and comparative results directly from the verified data.\n"
        "Include citation footnotes like [1], [2] referencing supporting strategic briefs and tables.\n\n"
        "Knowledge Context & Tables:\n{context}\n\n"
        "Executive Question: {query}\n\n"
        "Provide a comprehensive, high-impact business analysis with clear structure, exact numbers, risk mitigation insights, and execution steps."
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text, citations

def query_chroma_rag(query: str, collection, llm: ChatOpenAI, num_results: int = 4) -> Tuple[str, str, List[Dict[str, Any]]]:
    results = collection.query(query_texts=[query], n_results=num_results)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    context_lines = []
    citations = []
    for idx, (doc, meta, c_id) in enumerate(zip(docs, metas, ids), 1):
        doc_name = meta.get("document_title", "Knowledge vault") if meta else "Knowledge vault"
        context_lines.append(f"--- Citation [{idx}] from {doc_name} ({c_id}) ---\n{doc.strip()}\n")
        citations.append({
            "index": idx,
            "kind": "Passage",
            "name": doc_name,
            "meta": str(c_id),
            "source": "documents",
            "document_title": doc_name,
            "chunk_id": c_id,
            "excerpt": doc.strip(),
            "quote": doc.strip(),
            "relevance": "Direct vector match",
            "is_table": False
        })

    append_table_context(query, context_lines, citations, top_k=2)
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an AI Analyst performing Standard Document Passage Retrieval.\n"
        "Answer the business question strictly using the retrieved source passages and structured tables below.\n"
        "Quote exact numerical figures and measurements accurately.\n"
        "Include citation footnotes like [1], [2] directly in your text.\n\n"
        "Source Passages & Tables:\n{context}\n\n"
        "Business Question: {query}\n\n"
        "Answer:"
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text, citations

# -----------------------------------------------------------------------------
# 4. Governance helpers
# -----------------------------------------------------------------------------
def build_qa_checks(report: Dict[str, Any]) -> List[Dict[str, str]]:
    """Flatten the change agent's QA report into display rows."""
    parquet_ok = report.get("parquet_tables") == "PASSED"
    checks = [
        {
            "label": "Python syntax · app.py, core/*",
            "ok": report.get("syntax_check") == "PASSED",
            "meta": report.get("syntax_check", ""),
        },
        {
            "label": "Parquet · create_final_entities",
            "ok": parquet_ok and not entities_df.empty,
            "meta": f"{len(entities_df)} rows",
        },
        {
            "label": "Parquet · create_final_relationships",
            "ok": parquet_ok and not relationships_df.empty,
            "meta": f"{len(relationships_df)} rows",
        },
        {
            "label": "Parquet · create_final_nodes",
            "ok": parquet_ok and not nodes_df.empty,
            "meta": f"{len(nodes_df)} rows",
        },
        {
            "label": "Parquet · create_final_community_reports",
            "ok": parquet_ok and not community_df.empty,
            "meta": f"{len(community_df)} rows",
        },
        {
            "label": "ChromaDB · paper_collection reachable",
            "ok": report.get("chromadb_vault") == "PASSED",
            "meta": f"{paper_collection.count()} docs",
        },
        {
            "label": "Vault catalog integrity · SHA-256",
            "ok": report.get("vault_catalog") == "PASSED",
            "meta": f"{len(catalog_docs)} docs",
        },
        {
            "label": "Structured tables extracted",
            "ok": str(report.get("structured_tables", "")).startswith("PASSED"),
            "meta": f"{len(all_vault_tables)} tables",
        },
        {
            "label": "OPENAI_API_KEY present",
            "ok": bool(api_key),
            "meta": ".env",
        },
    ]
    return checks


def parse_changelog(path: Path, limit: int = 4) -> List[Dict[str, Any]]:
    """Parse CHANGELOG.md into release entries with their commit lines."""
    if not path.exists():
        return []
    releases: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        header = re.match(r"^##\s*\[?v?([0-9][^\]\s]*)\]?\s*-\s*(.+?)\s*$", line)
        if header:
            if current:
                releases.append(current)
            current = {"version": header.group(1), "date": header.group(2), "commits": []}
            continue
        commit = re.match(r"^-\s*\[`([0-9a-f]{6,})`\]\s*(.+?)\s*$", line)
        if commit and current:
            message = re.sub(r"^(feat|fix|docs|chore|refactor|test|style)(\([^)]*\))?:\s*", "", commit.group(2))
            current["commits"].append({"sha": commit.group(1)[:7], "message": message})
    if current:
        releases.append(current)
    return releases[:limit]


qa_report = change_agent.run_qa_checks()
qa_checks = build_qa_checks(qa_report)
qa_passed_count = sum(1 for c in qa_checks if c["ok"])

# -----------------------------------------------------------------------------
# 5. Sidebar — navigation, reasoning mode, precision controls
# -----------------------------------------------------------------------------
nav_counts = {"vault": len(catalog_docs)}
st.markdown(design.stylesheet(nav_counts), unsafe_allow_html=True)

MODE_LABELS = [label for _key, label in design.MODES]
MODE_KEY_BY_LABEL = {label: key for key, label in design.MODES}

with st.sidebar:
    st.markdown(design.brand(app_version), unsafe_allow_html=True)

    nav_keys = [key for key, _label in design.NAV]
    requested_screen = st.query_params.get("screen", "search")
    if requested_screen not in nav_keys:
        requested_screen = "search"
    screen = requested_screen

    st.markdown(design.nav_bar(screen, nav_counts), unsafe_allow_html=True)

    if screen == "search":
        valid_modes = [k for k, _ in design.MODES]
        requested_mode = st.query_params.get("mode", st.session_state.get("reasoning_mode", "global"))
        if requested_mode not in valid_modes:
            requested_mode = "global"
        mode_key = requested_mode
        st.session_state["reasoning_mode"] = mode_key
        st.markdown(design.reasoning_modes(mode_key), unsafe_allow_html=True)

        with st.container(key="controls"):
            temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            top_k_passages = st.slider("Citations", min_value=2, max_value=8, value=4)
    else:
        mode_key = "global"
        temp, top_k_passages = 0.2, 4

    if screen == "concepts" and not nodes_df.empty and "community" in nodes_df.columns:
        community_sizes = nodes_df["community"].value_counts().head(5)
        rows = ['<div class="k-side-label">Communities</div>']
        for community, size in community_sizes.items():
            title = community
            if not community_df.empty and "community" in community_df.columns:
                match = community_df[community_df["community"].astype(str) == str(community)]
                if not match.empty:
                    title = match.iloc[0]["title"]
            rows.append(
                '<div style="display:flex;align-items:center;gap:8px;padding:5px 8px;font-size:12px">'
                f'<span style="width:8px;height:8px;border-radius:2px;background:{design.community_color(community)}"></span>'
                f'<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{design.esc(title)}</span>'
                f'<span style="margin-left:auto;font:11px var(--k-mono);color:var(--k-faint)">{size}</span></div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

    st.markdown(
        design.sidebar_status(qa_passed_count == len(qa_checks), len(qa_checks)),
        unsafe_allow_html=True,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# -----------------------------------------------------------------------------
# 6. Screen: Search
# -----------------------------------------------------------------------------
CITATION_RE = re.compile(r"\[(\d+)\](?!\()")


def render_answer(text: str, key: str, citations: Optional[List[Dict[str, Any]]] = None):
    """Render an answer body with citation markers [n] styled as accent superscripts linking to evidence cards."""
    max_c = len(citations) if citations else 999

    def _repl(match):
        idx = int(match.group(1))
        if 1 <= idx <= max_c:
            return f'<sup class="k-cite"><a href="#cite-{idx}" aria-label="Source {idx}">[{idx}]</a></sup>'
        return match.group(0)

    marked = CITATION_RE.sub(_repl, text)
    with st.container(key=key):
        st.markdown(marked, unsafe_allow_html=True)


def render_compare(comparison: Dict[str, Any], key: str):
    with st.container(key=f"cmprow_{key}"):
        left, right = st.columns(2, gap="small")
        with left:
            with st.container(key=f"cmpcard_{key}_graph"):
                g_ground = design.grounding_tag(comparison.get("graph_grounding", 0.95))
                st.markdown(
                    '<div class="k-compare__head"><span class="k-swatch" '
                    f'style="background:{design.ACCENT}"></span><span class="k-card__title">Graph · '
                    f'{design.esc(comparison.get("graph_mode", "DRIFT deep-dive"))}</span>'
                    f'<span class="k-compare__meta">{comparison.get("graph_sources", 0)} sources</span>'
                    f'<span style="margin-left:auto">{g_ground}</span></div>',
                    unsafe_allow_html=True,
                )
                render_answer(comparison["graph_answer"], key=f"ans_cmp_{key}_graph", citations=comparison.get("graph_citations"))
                if comparison.get("graph_citations"):
                    footnotes = "".join(
                        f'<div style="font-size:11.5px;color:var(--k-muted);padding:2px 0">'
                        f'<span class="k-cite">[{c.get("index", idx+1)}]</span> {design.esc(c.get("kind", "Source"))} · {design.esc(c.get("name", ""))}</div>'
                        for idx, c in enumerate(comparison["graph_citations"][:4])
                    )
                    st.markdown(f'<div style="padding:8px 16px 12px;border-top:1px solid var(--k-border)">{footnotes}</div>', unsafe_allow_html=True)
        with right:
            with st.container(key=f"cmpcard_{key}_vector"):
                v_ground = design.grounding_tag(comparison.get("vector_grounding", 0.82))
                st.markdown(
                    '<div class="k-compare__head"><span class="k-swatch" '
                    f'style="background:{design.FAINT}"></span><span class="k-card__title">Vector · top-'
                    f'{comparison.get("vector_sources", 0)} passages</span>'
                    '<span class="k-compare__meta">ChromaDB · paper_collection</span>'
                    f'<span style="margin-left:auto">{v_ground}</span></div>',
                    unsafe_allow_html=True,
                )
                render_answer(comparison["vector_answer"], key=f"ans_cmp_{key}_vector", citations=comparison.get("vector_citations"))
                if comparison.get("vector_citations"):
                    footnotes = "".join(
                        f'<div style="font-size:11.5px;color:var(--k-muted);padding:2px 0">'
                        f'<span class="k-cite">[{c.get("index", idx+1)}]</span> {design.esc(c.get("kind", "Source"))} · {design.esc(c.get("name", ""))}</div>'
                        for idx, c in enumerate(comparison["vector_citations"][:4])
                    )
                    st.markdown(f'<div style="padding:8px 16px 12px;border-top:1px solid var(--k-border)">{footnotes}</div>', unsafe_allow_html=True)


def run_query(question: str, mode: str, temperature: float, citations_k: int) -> Dict[str, Any]:
    """Execute one reasoning run and return the assistant message."""
    llm = get_llm(temperature=temperature)
    started = time.perf_counter()

    if mode == "global":
        answer, ctx, citations = query_graphrag_global(question, community_df, llm)
    elif mode == "local":
        answer, ctx, citations = query_graphrag_local(question, entities_df, relationships_df, llm)
    elif mode == "drift":
        answer, ctx, citations = query_graphrag_drift(question, community_df, relationships_df, llm)
    elif mode == "vector":
        answer, ctx, citations = query_chroma_rag(question, paper_collection, llm, num_results=citations_k)
    else:
        # Concurrent execution of Graph and Vector reasoning engines for Compare mode
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_graph = executor.submit(query_graphrag_drift, question, community_df, relationships_df, llm)
            fut_vector = executor.submit(query_chroma_rag, question, paper_collection, llm, citations_k)
            graph_answer, gctx, graph_citations = fut_graph.result()
            vector_answer, vctx, vector_citations = fut_vector.result()

        citations = graph_citations + vector_citations
        for position, citation in enumerate(citations, 1):
            citation["index"] = position

        graph_grounding = design.compute_grounding(graph_answer)
        vector_grounding = design.compute_grounding(vector_answer)

        return {
            "role": "assistant",
            "content": "",
            "mode": dict(design.MODES)["compare"],
            "mode_key": "compare",
            "query": question,
            "citations": citations,
            "elapsed": time.perf_counter() - started,
            "grounding": max(graph_grounding, vector_grounding),
            "raw_context": f"### Graph Context (DRIFT):\n{gctx}\n\n### Vector Context (ChromaDB):\n{vctx}",
            "comparison": {
                "graph_answer": graph_answer,
                "vector_answer": vector_answer,
                "graph_mode": "DRIFT deep-dive",
                "graph_sources": len(graph_citations),
                "vector_sources": len(vector_citations),
                "graph_grounding": graph_grounding,
                "vector_grounding": vector_grounding,
                "graph_citations": graph_citations,
                "vector_citations": vector_citations,
            },
        }

    grounding = design.compute_grounding(answer)
    return {
        "role": "assistant",
        "content": answer,
        "mode": dict(design.MODES)[mode],
        "mode_key": mode,
        "query": question,
        "citations": citations,
        "elapsed": time.perf_counter() - started,
        "grounding": grounding,
        "raw_context": ctx,
    }


def screen_search():
    if "prompt" in st.query_params:
        try:
            p_val = st.query_params.pop("prompt")
            p_idx = int(p_val) - 1
            if 0 <= p_idx < len(design.SUGGESTED_PROMPTS):
                p_text, p_mode = design.SUGGESTED_PROMPTS[p_idx]
                st.session_state.pending_query = (p_text, p_mode)
                st.rerun()
        except Exception:
            pass

    last_query = next(
        (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"),
        "",
    )
    source_filter_key = st.query_params.get("source", "all").lower()

    split = st.container(key="split")
    thread_col, rail_col = split.columns([1, 0.42], gap=None)

    with thread_col:
        crumb = f"Thread · {last_query[:46]}" if last_query else "New thread"
        st.markdown(
            f'<div style="height:48px;display:flex;align-items:center;gap:6px;padding:0 24px;'
            f'border-bottom:1px solid var(--k-border);background:#FFFFFF">'
            f'<div class="k-topbar__title">Search</div><div class="k-topbar__sep">/</div>'
            f'<div class="k-topbar__crumb">{design.esc(crumb)}</div>'
            f'<div style="margin-left:auto">{design.source_pills(source_filter_key)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.container(key="thread"):
            if not st.session_state.messages:
                st.markdown(design.suggested_prompts_box(), unsafe_allow_html=True)

            for position, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    st.markdown(
                        '<div class="k-msg"><div class="k-msg__av">HS</div>'
                        f'<div class="k-msg__q">{design.esc(message["content"])}</div></div>',
                        unsafe_allow_html=True,
                    )
                    continue

                grounding_score = message.get("grounding", 0.0)
                grounding_pill = f'<span>·</span>{design.grounding_tag(grounding_score)}' if grounding_score > 0 else ""
                st.markdown(
                    '<div class="k-msg k-msg--ai"><div class="k-msg__av k-msg__av--ai">G</div><div class="k-msg__meta">'
                    f'<span class="k-msg__mode">{design.esc(message.get("mode", ""))}</span><span>·</span>'
                    f'<span>{len(message.get("citations", []))} sources</span><span>·</span>'
                    f'<span class="k-mono">{message.get("elapsed", 0):.1f}s</span>'
                    f'{grounding_pill}</div></div>',
                    unsafe_allow_html=True,
                )
                if message.get("comparison"):
                    render_compare(message["comparison"], key=f"cmp_{position}")
                else:
                    render_answer(message["content"], key=f"ans_msg_{position}", citations=message.get("citations"))

                with st.container(key=f"link_actions_{position}", horizontal=True):
                    if st.button("Compare with vector search", key=f"cmp_btn_{position}"):
                        st.session_state.pending_query = (message["query"], "compare")
                        st.rerun()
                    if st.button("Rerun", key=f"rerun_btn_{position}"):
                        st.session_state.pending_query = (message["query"], message["mode_key"])
                        st.rerun()
                    st.markdown(
                        f'<span class="k-actions__model k-mono">{MODEL_NAME}</span>',
                        unsafe_allow_html=True,
                    )

    last_citations = next(
        (m.get("citations", []) for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
        [],
    )
    source_filter = {"documents": "documents", "graph": "graph", "tables": "tables"}.get(source_filter_key)
    shown = [c for c in last_citations if not source_filter or c.get("source") == source_filter]

    if shown:
        cards = [design.evidence_card(c, focus=c.get("is_table", False)) for c in shown[:6]]
        if len(shown) > 6:
            cards.append(f'<div class="k-ev__more">{len(shown) - 6} more</div>')
        body = "".join(cards)
    else:
        body = '<div class="k-ev__more">Citations appear here once you run a query.</div>'

    with rail_col:
        r_head_col1, r_head_col2 = st.columns([1, 1], vertical_alignment="center")
        with r_head_col1:
            st.markdown(
                f'<div class="k-rail__head" style="border-bottom:none;height:auto;padding:14px 0 0 16px">Evidence '
                f'<span class="k-rail__count">{len(shown)}</span></div>',
                unsafe_allow_html=True,
            )
        with r_head_col2:
            show_context = st.toggle("Show context", value=st.session_state.get("show_raw_context", False), key="toggle_raw_context")
            st.session_state.show_raw_context = show_context

        if show_context:
            last_assistant_msg = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
            raw_ctx = last_assistant_msg.get("raw_context", "") if last_assistant_msg else ""
            if raw_ctx:
                st.markdown(
                    f'<div style="padding:8px 12px;font:11px/1.4 var(--k-mono);color:var(--k-muted);'
                    f'background:var(--k-canvas);border:1px solid var(--k-border);border-radius:6px;'
                    f'margin:8px 12px;max-height:220px;overflow-y:auto;white-space:pre-wrap;">'
                    f'{design.esc(raw_ctx)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="k-faint" style="padding:6px 12px;font-size:11.5px">No raw context captured.</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="k-rail__body">{body}</div>',
            unsafe_allow_html=True,
        )

    typed = st.chat_input("Ask the knowledge base…")

    question, question_mode = (typed, mode_key) if typed else (st.session_state.pending_query or (None, None))
    st.session_state.pending_query = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Searching the knowledge base…"):
                st.session_state.messages.append(run_query(question, question_mode, temp, top_k_passages))
        except Exception as exc:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"The query could not be completed: {exc}",
                "mode": dict(design.MODES)[question_mode],
                "mode_key": question_mode,
                "query": question,
                "citations": [],
                "elapsed": 0.0,
            })
        st.rerun()


# -----------------------------------------------------------------------------
# 7. Screen: Vault
# -----------------------------------------------------------------------------
VAULT_COLUMNS = "2.2fr .7fr .7fr .8fr .8fr 1.2fr 1fr"


def format_size(size_kb) -> str:
    try:
        size_kb = float(size_kb)
    except (TypeError, ValueError):
        return "—"
    return f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.0f} KB"


def format_uploaded(value: str) -> str:
    try:
        return pd.to_datetime(value).strftime("%b %d, %H:%M")
    except (ValueError, TypeError):
        return str(value)[:16]


def screen_vault():
    with st.container(key="topbar_vault", horizontal=True, vertical_alignment="center"):
        st.markdown(
            '<div class="k-topbar__title">Vault</div>'
            '<div class="k-topbar__note">Retained permanently · SHA-256 deduplicated</div>',
            unsafe_allow_html=True,
        )
        if st.button("Add document", type="primary", key="vault_add"):
            st.session_state.vault_add_open = not st.session_state.get("vault_add_open", False)

    with st.container(key="pad_vault"):
        if st.session_state.get("vault_add_open"):
            render_vault_uploader()

        catalog = vault.get_catalog()
        if not catalog:
            st.markdown(
                '<div class="k-card"><div class="k-card__body k-faint">'
                "No documents in the vault yet. Use Add document to ingest a PDF or text file.</div></div>",
                unsafe_allow_html=True,
            )
            return

        selected_id = st.query_params.get("doc") or catalog[0]["id"]
        if selected_id not in {d["id"] for d in catalog}:
            selected_id = catalog[0]["id"]

        head = (
            f'<div class="k-grid__head" style="grid-template-columns:{VAULT_COLUMNS}">'
            "<span>Document</span><span>Format</span><span>Size</span><span>Pages</span>"
            "<span>Tables</span><span>Uploaded</span><span>Status</span></div>"
        )
        rows = []
        for document in catalog:
            indexed = "index" in str(document.get("status", "")).lower()
            status = (
                f'<span style="display:flex;align-items:center;gap:6px;color:{design.SUCCESS}">'
                f'<span class="k-dot" style="background:{design.SUCCESS}"></span>Indexed</span>'
                if indexed
                else f'<span class="k-muted">{design.esc(document.get("status", "Pending"))}</span>'
            )
            rows.append(
                f'<a class="k-grid__row{" k-grid__row--on" if document["id"] == selected_id else ""}" '
                f'href="?screen=vault&doc={design.esc(document["id"])}" target="_self" '
                f'style="grid-template-columns:{VAULT_COLUMNS}">'
                f'<span><span class="k-grid__name">{design.esc(document["title"])}</span><br>'
                f'<span class="k-grid__id">{design.esc(document["id"])}</span></span>'
                f'<span>{design.esc(document.get("source_type", "—"))}</span>'
                f'<span class="k-num">{design.esc(format_size(document.get("file_size_kb")))}</span>'
                f'<span class="k-num">{design.esc(document.get("page_count", "—"))}</span>'
                f'<span class="k-num">{design.esc(document.get("table_count", 0) or "—")}</span>'
                f'<span class="k-muted">{design.esc(format_uploaded(document.get("uploaded_at", "")))}</span>'
                f"{status}</a>"
            )
        st.markdown(f'<div class="k-card">{head}{"".join(rows)}</div>', unsafe_allow_html=True)

        document = vault.get_document(selected_id)
        if not document:
            return

        tables = vault.get_tables(selected_id) if hasattr(vault, "get_tables") else []
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown(
                '<div class="k-card"><div class="k-card__body">'
                f'<div style="display:flex;align-items:center;margin-bottom:12px">'
                f'<span class="k-card__title">{design.esc(document["title"])}</span></div>'
                '<div class="k-kv">'
                f'<span>SHA-256</span><span class="k-mono" style="font-size:11px;word-break:break-all">'
                f'{design.esc(document["sha256"])}</span>'
                f'<span>Storage</span><span class="k-mono" style="font-size:11.5px;word-break:break-all">'
                f'{design.esc(Path(document["storage_path"]).name)}</span>'
                f'<span>Characters</span><span class="k-mono" style="font-size:11.5px">'
                f'{document.get("char_count", 0):,}</span>'
                f'<span>Extracted tables</span><span class="k-mono" style="font-size:11.5px">'
                f'{document.get("table_count", 0)}</span>'
                f'<span>Origin</span><span>{design.esc(document.get("metadata", {}).get("origin", "User upload"))}</span>'
                "</div></div></div>",
                unsafe_allow_html=True,
            )
            with st.container(key="link_delete", horizontal=True):
                confirm_key = f"confirm_del_{selected_id}"
                if st.session_state.get(confirm_key, False):
                    st.markdown(
                        f'<span class="k-faint" style="font-size:12px">Delete {design.esc(selected_id[:12])}…?</span>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Delete", key=f"do_delete_{selected_id}", type="primary"):
                        vault.delete_document(selected_id)
                        st.session_state[confirm_key] = False
                        st.query_params.pop("doc", None)
                        st.cache_data.clear()
                        st.rerun()
                    if st.button("Cancel", key=f"cancel_delete_{selected_id}"):
                        st.session_state[confirm_key] = False
                        st.rerun()
                else:
                    if st.button("Delete document", key=f"delete_{selected_id}"):
                        st.session_state[confirm_key] = True
                        st.rerun()

        with right:
            table_rows = "".join(
                '<div style="display:flex;gap:10px;align-items:center">'
                f'<span class="k-mono k-faint" style="font-size:11px;width:34px">p. {design.esc(t.get("page", 1))}</span>'
                f'<span>{design.esc(t.get("title", "Table"))}</span>'
                f'<span class="k-mono k-faint" style="margin-left:auto;font-size:11px">'
                f'{len(t.get("columns", []))} × {t.get("rows_count", 0)}</span></div>'
                for t in tables[:5]
            ) or '<div class="k-faint">No structured tables extracted from this document.</div>'
            more = (
                f'<div class="k-faint" style="font-size:11.5px;padding-top:4px">{len(tables) - 5} more</div>'
                if len(tables) > 5
                else ""
            )
            st.markdown(
                '<div class="k-card"><div class="k-card__body">'
                '<div style="display:flex;align-items:center;margin-bottom:12px">'
                '<span class="k-card__title">Extracted tables</span>'
                f'<span class="k-mono k-faint" style="margin-left:6px;font-size:11px">{len(tables)}</span>'
                '<span class="k-faint" style="margin-left:auto;font-size:11.5px">Row facts</span></div>'
                f'<div style="display:grid;gap:6px;color:var(--k-muted)">{table_rows}{more}</div>'
                "</div></div>",
                unsafe_allow_html=True,
            )
            for table in tables[:5]:
                with st.expander(f"{table.get('title', 'Table')} · page {table.get('page', 1)}"):
                    st.markdown(table.get("markdown", ""))
                    for fact in table.get("row_facts", [])[:8]:
                        st.markdown(f"- `{fact}`")


def render_vault_uploader():
    with st.container(key="card_upload"):
        uploaded = st.file_uploader(
            "Add a PDF, text or markdown document to the vault",
            type=["pdf", "txt", "md"],
        )
        if not uploaded:
            return
        index_now = st.checkbox("Re-index the knowledge graph with this document", value=True)
        chunk_depth = st.slider("Chunk limit for graph building", min_value=1, max_value=30, value=5)
        if st.button("Upload & retain", type="primary", key="vault_upload_go"):
            with st.spinner("Persisting document to the vault…"):
                stored = vault.store_document(
                    filename=uploaded.name,
                    content_bytes=uploaded.read(),
                    source_type="pdf" if uploaded.name.endswith(".pdf") else "txt",
                    metadata={"origin": "User upload"},
                )
            st.success(f"Stored {stored['filename']} · {stored['id']}")
            if index_now:
                with st.status("Building the knowledge graph…", expanded=True) as status:
                    engine = GraphRAGEngine(model_name=MODEL_NAME, temperature=0.0)
                    engine.build_from_text(
                        document_text=stored["extracted_text"],
                        document_title=stored["filename"],
                        max_chunks=chunk_depth,
                    )
                    status.update(label="Indexing complete", state="complete")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.vault_add_open = False
            st.rerun()


# -----------------------------------------------------------------------------
# 8. Screen: Concept map
# -----------------------------------------------------------------------------
def screen_concepts():
    titles = sorted(entities_df["title"].astype(str).tolist()) if not entities_df.empty else []
    find_param = st.query_params.get("find") or st.query_params.get("node")
    default_idx = titles.index(find_param) if (find_param and find_param in titles) else (0 if titles else None)

    split = st.container(key="split")
    graph_col, rail_col = split.columns([1, 0.42], gap=None)

    with graph_col:
        with st.container(key="topbar_concepts", horizontal=True, vertical_alignment="center"):
            st.markdown(
                '<div class="k-topbar__title">Concept map</div>'
                f'<div class="k-topbar__note">{len(entities_df)} nodes · {len(relationships_df)} edges · Louvain</div>',
                unsafe_allow_html=True,
            )
            with st.container(key="find_node"):
                selected_title = st.selectbox(
                    "Find node",
                    titles,
                    index=default_idx,
                    label_visibility="collapsed",
                    placeholder="Find node",
                )

        with st.container(key="graph"):
            graph_file = Path("./notebook/interactive_graph.html")
            if graph_file.exists():
                st.components.v1.html(graph_file.read_text(encoding="utf-8"), height=640, scrolling=False)
                st.markdown(
                    '<div class="k-graph__note">Embedded PyVis network · physics on · drag to explore</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="k-empty"><div class="k-empty__title">No concept map yet.</div>'
                    '<div class="k-empty__text">Ingest a document in the Vault to build the graph.</div></div>',
                    unsafe_allow_html=True,
                )

    body = (
        render_node_inspector(selected_title)
        if selected_title
        else '<div class="k-ev__more">No concepts indexed yet.</div>'
    )
    with rail_col:
        st.markdown(
            '<div class="k-rail"><div class="k-rail__head">Selected node</div>'
            f'<div class="k-rail__body k-rail__body--node">{body}</div></div>',
            unsafe_allow_html=True,
        )
        if selected_title:
            with st.container(key="link_node_actions", horizontal=True):
                if st.button(f"Ask about {selected_title[:16]}", key="ask_about_node"):
                    st.session_state.pending_query = (
                        f"What is {selected_title} and how does it relate to other concepts in the corpus?",
                        "local",
                    )
                    st.query_params["screen"] = "search"
                    st.rerun()
                if st.button("Open in catalog", key="open_in_catalog"):
                    st.query_params["screen"] = "catalog"
                    st.query_params["term"] = selected_title
                    st.rerun()


def render_node_inspector(title: str) -> str:
    entity = entities_df[entities_df["title"].astype(str) == title].iloc[0]
    node_row = None
    if not nodes_df.empty and "title" in nodes_df.columns:
        matches = nodes_df[nodes_df["title"].astype(str) == title]
        node_row = matches.iloc[0] if not matches.empty else None

    community = node_row["community"] if node_row is not None and "community" in node_row else "—"
    degree = node_row["degree"] if node_row is not None and "degree" in node_row else "—"
    community_title = ""
    if not community_df.empty and community != "—":
        match = community_df[community_df["community"].astype(str) == str(community)]
        if not match.empty:
            community_title = match.iloc[0]["title"]

    related = pd.DataFrame()
    if not relationships_df.empty:
        related = relationships_df[
            (relationships_df["source"].astype(str) == title)
            | (relationships_df["target"].astype(str) == title)
        ].head(4)

    relationship_rows = "".join(
        '<div style="display:flex;gap:8px;align-items:center">'
        f'<span>{design.esc(r["target"] if str(r["source"]) == title else r["source"])}</span>'
        f'<span class="k-mono k-faint" style="margin-left:auto;font-size:11px">{r.get("weight", "—")}/10</span></div>'
        for _i, r in related.iterrows()
    ) or '<div class="k-faint">No relationships recorded.</div>'

    return (
        f'<div><div style="font:500 17px/1.2 var(--k-sans);letter-spacing:-.01em">{design.esc(title)}</div>'
        '<div style="margin-top:4px;display:flex;gap:6px;font:11px var(--k-mono);color:var(--k-faint)">'
        f'<span>{design.esc(entity.get("type", "CONCEPT"))}</span><span>·</span>'
        f'<span style="color:{design.ACCENT}">{design.esc(community_title or f"community {community}")}</span></div></div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">'
        f'{design.stat_box(degree, "degree")}{design.stat_box(community, "community")}'
        f'{design.stat_box(len(related), "links")}</div>'
        f'<div class="k-muted" style="line-height:1.55">{design.esc(entity.get("description", ""))}</div>'
        '<div><div class="k-side-label" style="margin:0 0 8px;padding:0">Relationships</div>'
        f'<div style="display:grid;gap:6px;color:var(--k-muted)">{relationship_rows}</div></div>'
    )


# -----------------------------------------------------------------------------
# 9. Screen: Catalog
# -----------------------------------------------------------------------------
PAGE_SIZE = 25


def screen_catalog():
    tab_counts = {
        "concepts": len(entities_df),
        "relationships": len(relationships_df),
        "nodes": len(nodes_df),
        "briefs": len(community_df),
    }
    current_tab = st.query_params.get("tab", "concepts").lower()
    if current_tab not in tab_counts:
        current_tab = "concepts"

    initial_term = st.query_params.get("term", "")
    with st.container(key="topbar_catalog", horizontal=True, vertical_alignment="center"):
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:22px;height:48px">'
            f'<div class="k-topbar__title">Catalog</div>'
            f'{design.catalog_tabs(current_tab, tab_counts)}'
            f'</div>',
            unsafe_allow_html=True,
        )
        with st.container(key="search_catalog"):
            term = st.text_input("Search", value=initial_term, placeholder="Search catalog", label_visibility="collapsed")

    with st.container(key="pad_catalog"):
        if current_tab == "concepts":
            render_catalog_concepts(term)
        elif current_tab == "relationships":
            render_catalog_relationships(term)
        elif current_tab == "nodes":
            render_catalog_nodes(term)
        else:
            render_catalog_briefs(term)


def _paginate(frame: pd.DataFrame, state_key: str) -> pd.DataFrame:
    page = st.session_state.get(state_key, 0)
    total_pages = max(1, (len(frame) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, total_pages - 1)
    st.session_state[state_key] = page
    return frame.iloc[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]


def _pager(frame: pd.DataFrame, state_key: str, term: str):
    total_pages = max(1, (len(frame) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.session_state.get(state_key, 0)
    shown = min(PAGE_SIZE, max(0, len(frame) - page * PAGE_SIZE))
    with st.container(key=f"link_pager_{state_key}", horizontal=True, vertical_alignment="center"):
        suffix = f" · filtered by “{term}”" if term else ""
        st.markdown(
            f'<span class="k-foot__label">Showing {shown} of {len(frame)}{design.esc(suffix)}</span>',
            unsafe_allow_html=True,
        )
        if st.button("Previous", key=f"prev_{state_key}", disabled=page == 0):
            st.session_state[state_key] = page - 1
            st.rerun()
        st.markdown(
            f'<span class="k-foot__page k-mono">{page + 1} / {total_pages}</span>',
            unsafe_allow_html=True,
        )
        if st.button("Next", key=f"next_{state_key}", disabled=page + 1 >= total_pages):
            st.session_state[state_key] = page + 1
            st.rerun()
        st.download_button(
            "Export CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{state_key}.csv",
            mime="text/csv",
            key=f"csv_{state_key}",
            width="content",
        )


def _empty_card(message: str):
    st.markdown(
        f'<div class="k-card"><div class="k-card__body k-faint">{design.esc(message)}</div></div>',
        unsafe_allow_html=True,
    )


def render_catalog_concepts(term: str):
    if entities_df.empty:
        _empty_card("No concepts indexed yet. Ingest a document in the Vault to build the graph.")
        return
    frame = entities_df
    if term:
        frame = frame[
            frame["title"].str.contains(term, case=False, na=False)
            | frame["description"].str.contains(term, case=False, na=False)
        ]
    columns = "1.4fr .8fr 3fr .6fr .8fr"
    rows = [
        f'<div class="k-grid__head" style="grid-template-columns:{columns}">'
        "<span>Title</span><span>Type</span><span>Description</span>"
        '<span style="text-align:right">Degree</span><span>Community</span></div>'
    ]
    degrees = {}
    if not nodes_df.empty and "title" in nodes_df.columns:
        dedup_nodes = nodes_df.drop_duplicates(subset=["title"])
        degrees = dedup_nodes.set_index(dedup_nodes["title"].astype(str))[["degree", "community"]].to_dict("index")
    for _i, row in _paginate(frame, "catalog_concepts").iterrows():
        stats = degrees.get(str(row["title"]), {})
        community = stats.get("community", "—")
        rows.append(
            f'<div class="k-grid__row" style="grid-template-columns:{columns}">'
            f'<span class="k-grid__name">{design.highlight(row["title"], term)}</span>'
            f'<span class="k-mono k-muted" style="font-size:11px">{design.esc(row["type"])}</span>'
            f'<span class="k-muted k-clip">{design.highlight(row["description"], term)}</span>'
            f'<span class="k-mono" style="text-align:right">{stats.get("degree", "—")}</span>'
            '<span style="display:flex;align-items:center;gap:6px">'
            f'<span style="width:8px;height:8px;border-radius:2px;background:{design.community_color(community)}"></span>'
            f"{design.esc(community)}</span></div>"
        )
    st.markdown(f'<div class="k-card">{"".join(rows)}</div>', unsafe_allow_html=True)
    _pager(frame, "catalog_concepts", term)


def render_catalog_relationships(term: str):
    if relationships_df.empty:
        _empty_card("No relationships indexed yet.")
        return
    frame = relationships_df
    if term:
        frame = frame[
            frame["source"].str.contains(term, case=False, na=False)
            | frame["target"].str.contains(term, case=False, na=False)
            | frame["description"].str.contains(term, case=False, na=False)
        ]
    columns = "1.2fr 1.2fr 3fr .6fr"
    rows = [
        f'<div class="k-grid__head" style="grid-template-columns:{columns}">'
        "<span>Source</span><span>Target</span><span>Description</span>"
        '<span style="text-align:right">Weight</span></div>'
    ]
    for _i, row in _paginate(frame, "catalog_relationships").iterrows():
        rows.append(
            f'<div class="k-grid__row" style="grid-template-columns:{columns}">'
            f'<span class="k-grid__name">{design.highlight(row["source"], term)}</span>'
            f'<span class="k-grid__name">{design.highlight(row["target"], term)}</span>'
            f'<span class="k-muted k-clip">{design.highlight(row["description"], term)}</span>'
            f'<span class="k-mono" style="text-align:right">{design.esc(row["weight"])}</span></div>'
        )
    st.markdown(f'<div class="k-card">{"".join(rows)}</div>', unsafe_allow_html=True)
    _pager(frame, "catalog_relationships", term)


def render_catalog_nodes(term: str):
    if nodes_df.empty:
        _empty_card("No graph nodes indexed yet.")
        return
    frame = nodes_df
    if term:
        frame = frame[frame["title"].str.contains(term, case=False, na=False)]
    columns = "2fr .8fr .8fr"
    rows = [
        f'<div class="k-grid__head" style="grid-template-columns:{columns}">'
        "<span>Title</span><span>Community</span>"
        '<span style="text-align:right">Degree</span></div>'
    ]
    for _i, row in _paginate(frame, "catalog_nodes").iterrows():
        rows.append(
            f'<div class="k-grid__row" style="grid-template-columns:{columns}">'
            f'<span class="k-grid__name">{design.highlight(row["title"], term)}</span>'
            '<span style="display:flex;align-items:center;gap:6px">'
            f'<span style="width:8px;height:8px;border-radius:2px;background:{design.community_color(row["community"])}"></span>'
            f'{design.esc(row["community"])}</span>'
            f'<span class="k-mono" style="text-align:right">{design.esc(row["degree"])}</span></div>'
        )
    st.markdown(f'<div class="k-card">{"".join(rows)}</div>', unsafe_allow_html=True)
    _pager(frame, "catalog_nodes", term)


def render_catalog_briefs(term: str):
    if community_df.empty:
        _empty_card("No domain briefs generated yet.")
        return
    frame = community_df
    if term:
        frame = frame[
            frame["title"].str.contains(term, case=False, na=False)
            | frame["summary"].str.contains(term, case=False, na=False)
        ]
    for _i, row in frame.head(PAGE_SIZE).iterrows():
        st.markdown(
            '<div class="k-card" style="margin-bottom:10px"><div class="k-card__body">'
            '<div style="display:flex;align-items:baseline;gap:8px">'
            f'<span class="k-card__title">{design.highlight(row["title"], term)}</span>'
            f'<span class="k-mono k-faint" style="font-size:11px">community {design.esc(row["community"])}</span>'
            f'<span class="k-tag k-tag--accent" style="margin-left:auto">rank {design.esc(row["rank"])}</span></div>'
            f'<div class="k-muted" style="margin-top:8px;line-height:1.55">{design.highlight(row["summary"], term)}</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
        with st.expander("Full brief"):
            st.markdown(row["full_content"])


# -----------------------------------------------------------------------------
# 10. Screen: Governance
# -----------------------------------------------------------------------------
def screen_governance():
    with st.container(key="topbar_governance", horizontal=True, vertical_alignment="center"):
        st.markdown(
            '<div class="k-topbar__title">Governance</div>'
            f'<div class="k-topbar__note">Change Management Agent · last run {design.esc(qa_report.get("timestamp", ""))}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Re-run QA & sync", key="gov_sync"):
            with st.spinner("Running the change management cycle…"):
                result = change_agent.run_full_sync()
            st.success(f"Synchronized v{result['version']} with the changelog and README.")
            st.rerun()

    with st.container(key="pad_governance"):
        left, right = st.columns([1, 1.2], gap="medium")

        with left:
            all_passed = qa_passed_count == len(qa_checks)
            sorted_checks = sorted(qa_checks, key=lambda c: 0 if not c["ok"] else 1)
            checks_html = "".join(
                f'<div class="k-check"><span class="{"k-check__ok" if check["ok"] else "k-check__bad"}">'
                f'{"✓" if check["ok"] else "✕"}</span><span>{design.esc(check["label"])}</span>'
                f'<span class="k-check__meta">{design.esc(check["meta"])}</span></div>'
                + (
                    f'<div style="padding:4px 16px 8px;font:11px var(--k-mono);color:var(--k-danger);background:#FFF5F5">'
                    f'{design.esc(check.get("error", ""))}</div>'
                    if not check["ok"] and check.get("error")
                    else ""
                )
                for check in sorted_checks
            )
            st.markdown(
                '<div class="k-card"><div class="k-card__head">'
                '<span class="k-card__title">QA health</span>'
                f'<span class="k-tag{"" if all_passed else " k-tag--warn"}" '
                'style="display:flex;align-items:center;gap:6px;border-radius:999px;padding:2px 8px">'
                f'<span class="k-dot" style="background:{design.SUCCESS if all_passed else design.WARN}"></span>'
                f'{qa_passed_count} / {len(qa_checks)} passed</span>'
                f'<span class="k-check__meta">v{design.esc(app_version)}</span></div>'
                f"{checks_html}</div>",
                unsafe_allow_html=True,
            )
            if qa_report.get("errors"):
                with st.expander(f"Reported issues ({len(qa_report['errors'])})"):
                    for issue in qa_report["errors"]:
                        st.markdown(f"- {issue}")

        with right:
            releases = parse_changelog(Path("./CHANGELOG.md"))
            if not releases:
                _empty_card("The changelog is generated on the first sync.")
                return
            entries = []
            for position, release in enumerate(releases):
                is_current = position == 0
                commits = "".join(
                    f'<div class="k-tl__commit"><span class="k-tl__sha">{design.esc(c["sha"])}</span>'
                    f'<span>{design.esc(c["message"])}</span></div>'
                    for c in release["commits"][:4]
                )
                extra = (
                    f'<div class="k-faint" style="font-size:11.5px;margin-top:6px">'
                    f'{len(release["commits"]) - 4} more commits</div>'
                    if len(release["commits"]) > 4
                    else ""
                )
                connector = '<span class="k-tl__line"></span>' if position < len(releases) - 1 else ""
                entries.append(
                    '<div class="k-tl"><div class="k-tl__rail">'
                    f'<span class="k-tl__dot{"" if is_current else " k-tl__dot--past"}"></span>'
                    f"{connector}</div>"
                    '<div class="k-tl__body"><div style="display:flex;align-items:baseline;gap:8px">'
                    f'<span class="k-tl__ver">v{design.esc(release["version"])}</span>'
                    f'<span class="k-faint" style="font-size:11.5px">{design.esc(release["date"])}</span>'
                    + (
                        '<span class="k-tag k-tag--accent" style="margin-left:auto">current</span>'
                        if is_current
                        else ""
                    )
                    + f"</div>{commits}{extra}</div></div>"
                )
            st.markdown(
                '<div class="k-card"><div class="k-card__head">'
                '<span class="k-card__title">Changelog</span>'
                '<span class="k-mono k-faint" style="font-size:11px">CHANGELOG.md</span>'
                '<span class="k-muted" style="margin-left:auto;font-size:11.5px">'
                "Semantic versioning · patch on sync</span></div>"
                f'<div style="padding:16px 16px 6px">{"".join(entries)}</div></div>',
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# 11. Router
# -----------------------------------------------------------------------------
if screen == "search":
    screen_search()
elif screen == "vault":
    screen_vault()
elif screen == "concepts":
    screen_concepts()
elif screen == "catalog":
    screen_catalog()
else:
    screen_governance()
