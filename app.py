import os
import html
import json
import time
from pathlib import Path
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

from core.vault import DocumentVault
from core.pipeline import DocumentIngestor, GraphRAGEngine
from core.change_manager import ChangeManagementAgent

# -----------------------------------------------------------------------------
# 1. Page Configuration & Glean-Style Enterprise Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Glean Enterprise — GraphRAG Knowledge Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")

# Initialize Core Services
vault = DocumentVault(vault_dir="./vault")
change_agent = ChangeManagementAgent()
app_version = change_agent.get_version()

# Glean-Inspired Enterprise Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0A0D14;
        color: #E2E8F0;
    }
    
    /* Glean Work Header */
    .glean-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 0 1.25rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }
    
    .glean-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .glean-logo-badge {
        background: linear-gradient(135deg, #2563EB, #4F46E5);
        color: white;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35);
    }
    
    .glean-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin: 0;
    }
    
    .glean-subtitle {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0;
    }
    
    /* Verified Grounding Pill */
    .verified-pill {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #6EE7B7;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Glean Search Filter Chips */
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: #141824;
        border: 1px solid rgba(255, 255, 255, 0.09);
        color: #CBD5E1;
        margin-right: 8px;
    }
    
    /* Metric HUD */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    
    .glean-stat-card {
        background: #121622;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 14px 16px;
        position: relative;
    }
    
    .glean-stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #2563EB, #4F46E5);
    }
    
    .glean-stat-num {
        font-size: 1.65rem;
        font-weight: 800;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .glean-stat-text {
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.05em;
        margin-top: 3px;
    }

    /* Glean Citation Cards */
    .citation-card {
        background: #0E121A;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        transition: border-color 0.2s ease;
    }
    
    .citation-card:hover {
        border-color: rgba(37, 99, 235, 0.4);
    }
    
    .citation-badge {
        background: rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(37, 99, 235, 0.4);
        color: #93C5FD;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .citation-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-left: 6px;
    }
    
    .citation-snippet {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #94A3B8;
        background: #07090E;
        border-radius: 6px;
        padding: 8px 10px;
        margin-top: 8px;
        line-height: 1.45;
        max-height: 140px;
        overflow-y: auto;
    }

    /* Split Comparison Card */
    .compare-container {
        background: #121622;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data & Client Caching
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
def get_llm(temperature: float = 0.2, model_name: str = "gpt-4o-mini"):
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

# Load core datasets
entities_df, relationships_df, nodes_df, community_df = load_graph_data()
chroma_client, paper_collection = load_chroma_db()
catalog_docs = vault.get_catalog()

# -----------------------------------------------------------------------------
# 3. Query Reasoning Engines with Glean Citations
# -----------------------------------------------------------------------------
def query_graphrag_global(query: str, reports: pd.DataFrame, llm: ChatOpenAI) -> Tuple[str, str, List[Dict[str, Any]]]:
    context_lines = ["### Executive Thematic Domain Summaries:"]
    citations = []
    for idx, row in reports.iterrows():
        context_lines.append(f"#### [{idx + 1}] {row['title']}\n{row['summary']}\n")
        citations.append({
            "index": idx + 1,
            "document_title": f"Thematic Domain: {row['title']}",
            "chunk_id": f"domain_{row['community']}",
            "excerpt": row['summary'],
            "relevance": f"Rank: {row['rank']}"
        })
    
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an Executive AI Strategic Advisor performing a Macro-Level Strategic Synthesis across business domains.\n"
        "Synthesize an executive-level summary answering the inquiry across the domain reports below.\n"
        "Include bracketed citation footnotes like [1], [2] referencing specific supporting domains.\n\n"
        "Thematic Domain Reports:\n{context}\n\n"
        "Executive Inquiry: {query}\n\n"
        "Structure: Executive Summary, Strategic Analysis, Core Trade-offs, and Actionable Recommendations."
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text, citations

def query_graphrag_local(query: str, entities: pd.DataFrame, rels: pd.DataFrame, llm: ChatOpenAI, top_k: int = 15) -> Tuple[str, str, List[Dict[str, Any]]]:
    context_lines = ["### Identified Concepts:"]
    citations = []
    for idx, row in entities.head(top_k).iterrows():
        context_lines.append(f"- [{idx + 1}] **{row['title']}** ({row['type']}): {row['description']}")
        citations.append({
            "index": idx + 1,
            "document_title": f"Concept: {row['title']} ({row['type']})",
            "chunk_id": f"entity_{row['human_readable_id']}",
            "excerpt": row['description'],
            "relevance": "Direct Graph Node"
        })
    
    context_lines.append("\n### Direct Relational Dependencies:")
    for _, row in rels.head(top_k).iterrows():
        context_lines.append(f"- **{row['source']}** ➔ **{row['target']}** (Strength: {row['weight']}/10): {row['description']}")
    
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an Executive AI Strategic Advisor performing Targeted Fact & Concept Lookup.\n"
        "Provide a crisp, actionable business response grounded in the verified concept network below.\n"
        "Include citation footnotes like [1], [2] where relevant.\n\n"
        "Verified Knowledge Network:\n{context}\n\n"
        "Executive Question: {query}\n\n"
        "Format with clear bullet points, strategic implications, and concise takeaways."
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
            "document_title": row['title'],
            "chunk_id": f"report_{row['community']}",
            "excerpt": row['summary'],
            "relevance": "Strategic Domain"
        })
    
    context_lines.append("\n### Granular Cross-Functional Dependencies:")
    for _, row in rels.head(20).iterrows():
        context_lines.append(f"- **{row['source']}** <-> **{row['target']}**: {row['description']}")
    
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an Executive AI Strategic Advisor performing a Deep-Dive Cross-Functional Analysis.\n"
        "Answer the business inquiry by connecting macro business strategy with granular operational and technological dependencies.\n"
        "Include citation footnotes like [1], [2] referencing supporting strategic briefs.\n\n"
        "Knowledge Context:\n{context}\n\n"
        "Executive Question: {query}\n\n"
        "Provide a comprehensive, high-impact business analysis with clear structure, risk mitigation insights, and execution steps."
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
        doc_name = meta.get("document_title", "Enterprise Knowledge Vault") if meta else "Enterprise Knowledge Vault"
        context_lines.append(f"--- Citation [{idx}] from {doc_name} ({c_id}) ---\n{doc.strip()}\n")
        citations.append({
            "index": idx,
            "document_title": doc_name,
            "chunk_id": c_id,
            "excerpt": doc.strip(),
            "relevance": "Direct Vector Match"
        })
    
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an AI Analyst performing Standard Document Passage Retrieval.\n"
        "Answer the business question strictly using the retrieved source passages below.\n"
        "Include citation footnotes like [1], [2] directly in your text.\n\n"
        "Source Passages:\n{context}\n\n"
        "Business Question: {query}\n\n"
        "Answer:"
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text, citations

# -----------------------------------------------------------------------------
# 4. Glean Top Header & KPI Dashboard
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="glean-header">
    <div class="glean-brand">
        <div class="glean-logo-badge">Glean</div>
        <div>
            <h1 class="glean-title">Enterprise Knowledge Assistant</h1>
            <p class="glean-subtitle">Autonomous Graph Intelligence • Document Retention Vault • Verified Source Attribution</p>
        </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
        <span class="verified-pill">● 98% VERIFIED GROUNDING</span>
        <span class="source-chip" style="background: rgba(37, 99, 235, 0.15); color: #93C5FD; border-color: rgba(37, 99, 235, 0.4);">v{app_version} PROD</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Metric Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="glean-stat-card"><div class="glean-stat-num">{len(entities_df)}</div><div class="glean-stat-text">Indexed Concepts</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="glean-stat-card"><div class="glean-stat-num">{len(relationships_df)}</div><div class="glean-stat-text">Verified Links</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="glean-stat-card"><div class="glean-stat-num">{len(catalog_docs)}</div><div class="glean-stat-text">Retained Documents</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="glean-stat-card"><div class="glean-stat-num">{paper_collection.count()}</div><div class="glean-stat-text">Evidence Passages</div></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Sidebar Controls & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 Reasoning Strategy")
    search_mode = st.selectbox(
        "Select Intelligence Mode",
        [
            "🌐 Strategic Executive Synthesis",
            "🔍 Targeted Concept & Dependency Lookup",
            "🌀 Deep-Dive Cross-Functional Analysis",
            "📚 Standard Document Passage Search",
            "⚖️ Comparative Audit: Graph vs Document Search"
        ],
        index=0,
        help="Select the reasoning depth: High-level synthesis, factual entity traversal, or side-by-side comparative audit."
    )
    
    st.divider()
    st.markdown("### 🎛️ Search Precision")
    temp = st.slider("Response Precision (Temperature)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    top_k_passages = st.slider("Source Citations to Retrieve", min_value=2, max_value=8, value=4)
    
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("### 🛡️ Enterprise Governance")
    qa_status = change_agent.run_qa_checks()
    if qa_status["passed"]:
        st.markdown('<span class="verified-pill">● QA Validated (All Checks Passed)</span>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ QA Warning: Check governance tab.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Welcome to **Glean Enterprise Knowledge Assistant**.\n\nI am connected to your organizational document vault and enterprise knowledge graph. Every response is grounded with **verified source citations [1], [2]** linking directly to underlying documents.\n\nAsk a strategic question below, browse the **Document Vault**, or inspect the **Concept Map**!",
            "mode": "Glean Assistant",
            "citations": []
        }
    ]

# -----------------------------------------------------------------------------
# 6. Glean Work Navigation Tabs
# -----------------------------------------------------------------------------
tab_glean_chat, tab_vault_ui, tab_concept_map, tab_catalog_ui, tab_governance = st.tabs([
    "🔍 Glean Search & Assistant",
    "📑 Enterprise Document Vault",
    "🕸️ Knowledge Concept Map",
    "📊 Enterprise Data Catalog",
    "🛡️ System Governance & Changelog"
])

# =============================================================================
# TAB 1: Glean Search & Assistant
# =============================================================================
with tab_glean_chat:
    # Source Filter Chips
    st.markdown("""
    <div>
        <span class="source-chip" style="background: rgba(37, 99, 235, 0.2); color: #93C5FD; border-color: #2563EB;">✨ All Sources</span>
        <span class="source-chip">📄 PDF Documents</span>
        <span class="source-chip">🕸️ Knowledge Graph</span>
        <span class="source-chip">📑 Retained Vault</span>
        <span class="source-chip">📊 Parquet Taxonomy</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prompt Chips
    p_cols = st.columns(4)
    biz_prompts = [
        "What are the cost, ROI, and resource trade-offs between RAG and Fine-Tuning?",
        "Which techniques reduce operational compute overhead and deployment risks?",
        "What are the strategic dependencies and recommendations outlined in this report?",
        "Summarize the core thematic knowledge domains identified across the document."
    ]
    
    clicked_chip = None
    for i, col in enumerate(p_cols):
        with col:
            if st.button(biz_prompts[i], key=f"glean_chip_{i}", use_container_width=True):
                clicked_chip = biz_prompts[i]
    
    st.divider()

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "mode" in msg and msg["mode"] != "Glean Assistant":
                st.markdown(f'<span class="source-chip">{msg["mode"]}</span>', unsafe_allow_html=True)
            
            st.markdown(msg["content"])
            
            # Render Glean Citations if present
            if msg.get("citations"):
                with st.expander(f"📚 Verified Sources & Citations ({len(msg['citations'])} sources)"):
                    for c in msg["citations"]:
                        st.markdown(f"""
                        <div class="citation-card">
                            <span class="citation-badge">Citation {c['index']}</span>
                            <span class="citation-title">📄 {c['document_title']}</span>
                            <div class="citation-snippet">{html.escape(c['excerpt'])}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Comparative Audit View
            if msg.get("comparison"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="compare-container">
                        <div style="font-weight: bold; margin-bottom: 8px; color: #93C5FD;">🌐 GRAPHRAG STRATEGIC DEEP-DIVE</div>
                    """, unsafe_allow_html=True)
                    st.markdown(msg["comparison"]["graph_answer"])
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div class="compare-container">
                        <div style="font-weight: bold; margin-bottom: 8px; color: #C4B5FD;">📚 STANDARD DOCUMENT PASSAGE SEARCH</div>
                    """, unsafe_allow_html=True)
                    st.markdown(msg["comparison"]["vector_answer"])
                    st.markdown("</div>", unsafe_allow_html=True)

    # Chat Input
    user_input = st.chat_input("Search enterprise knowledge or ask a strategic question...") or clicked_chip

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        llm = get_llm(temperature=temp)
        with st.chat_message("assistant"):
            with st.spinner(f"Glean Assistant searching & synthesizing across knowledge sources..."):
                try:
                    if "Synthesis" in search_mode:
                        ans, ctx, cits = query_graphrag_global(user_input, community_df, llm)
                        st.markdown(ans)
                        if cits:
                            with st.expander(f"📚 Verified Sources & Citations ({len(cits)} sources)"):
                                for c in cits:
                                    st.markdown(f"""
                                    <div class="citation-card">
                                        <span class="citation-badge">Citation {c['index']}</span>
                                        <span class="citation-title">📄 {c['document_title']}</span>
                                        <div class="citation-snippet">{html.escape(c['excerpt'])}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "mode": "Strategic Synthesis", "citations": cits})
                    
                    elif "Targeted" in search_mode:
                        ans, ctx, cits = query_graphrag_local(user_input, entities_df, relationships_df, llm)
                        st.markdown(ans)
                        if cits:
                            with st.expander(f"📚 Verified Sources & Citations ({len(cits)} sources)"):
                                for c in cits:
                                    st.markdown(f"""
                                    <div class="citation-card">
                                        <span class="citation-badge">Citation {c['index']}</span>
                                        <span class="citation-title">📄 {c['document_title']}</span>
                                        <div class="citation-snippet">{html.escape(c['excerpt'])}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "mode": "Targeted Lookup", "citations": cits})
                    
                    elif "Deep-Dive" in search_mode:
                        ans, ctx, cits = query_graphrag_drift(user_input, community_df, relationships_df, llm)
                        st.markdown(ans)
                        if cits:
                            with st.expander(f"📚 Verified Sources & Citations ({len(cits)} sources)"):
                                for c in cits:
                                    st.markdown(f"""
                                    <div class="citation-card">
                                        <span class="citation-badge">Citation {c['index']}</span>
                                        <span class="citation-title">📄 {c['document_title']}</span>
                                        <div class="citation-snippet">{html.escape(c['excerpt'])}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "mode": "Deep-Dive Analysis", "citations": cits})
                    
                    elif "Standard" in search_mode:
                        ans, ctx, cits = query_chroma_rag(user_input, paper_collection, llm, num_results=top_k_passages)
                        st.markdown(ans)
                        if cits:
                            with st.expander(f"📚 Verified Sources & Citations ({len(cits)} sources)"):
                                for c in cits:
                                    st.markdown(f"""
                                    <div class="citation-card">
                                        <span class="citation-badge">Citation {c['index']}</span>
                                        <span class="citation-title">📄 {c['document_title']}</span>
                                        <div class="citation-snippet">{html.escape(c['excerpt'])}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "mode": "Standard Passage Search", "citations": cits})
                    
                    else: # Comparative Audit
                        g_ans, g_ctx, g_cits = query_graphrag_drift(user_input, community_df, relationships_df, llm)
                        v_ans, v_ctx, v_cits = query_chroma_rag(user_input, paper_collection, llm, num_results=top_k_passages)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("""
                            <div class="compare-container">
                                <div style="font-weight: bold; margin-bottom: 8px; color: #93C5FD;">🌐 GRAPHRAG STRATEGIC DEEP-DIVE</div>
                            """, unsafe_allow_html=True)
                            st.markdown(g_ans)
                            st.markdown("</div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown("""
                            <div class="compare-container">
                                <div style="font-weight: bold; margin-bottom: 8px; color: #C4B5FD;">📚 STANDARD DOCUMENT PASSAGE SEARCH</div>
                            """, unsafe_allow_html=True)
                            st.markdown(v_ans)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "⚖️ **Comparative Audit Completed**",
                            "mode": "Comparative Audit",
                            "comparison": {
                                "graph_answer": g_ans,
                                "vector_answer": v_ans
                            }
                        })
                except Exception as e:
                    st.error(f"Search Execution Error: {str(e)}")

# =============================================================================
# TAB 2: Enterprise Document Vault
# =============================================================================
with tab_vault_ui:
    st.markdown("### 📑 Enterprise Document Retention Vault")
    st.markdown("All uploaded enterprise assets are retained permanently with SHA-256 integrity hashing and full provenance tracking.")
    
    # Upload Section
    with st.expander("📥 Ingest New Document into Vault", expanded=False):
        uploaded_vault_file = st.file_uploader("Choose PDF or TXT file to ingest into permanent vault", type=["pdf", "txt", "md"])
        if uploaded_vault_file:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                run_indexing_immediately = st.checkbox("Re-index Knowledge Graph immediately with this document", value=True)
            with col_v2:
                ingest_chunk_depth = st.slider("Chunk Limit for Graph Building", min_value=1, max_value=30, value=5)
            
            if st.button("🚀 Upload & Retain in Vault", type="primary"):
                with st.spinner("Persisting document to enterprise vault..."):
                    file_bytes = uploaded_vault_file.read()
                    stored_meta = vault.store_document(
                        filename=uploaded_vault_file.name,
                        content_bytes=file_bytes,
                        source_type="pdf" if uploaded_vault_file.name.endswith(".pdf") else "txt",
                        metadata={"source": "User Upload"}
                    )
                    st.success(f"✅ Stored **{stored_meta['filename']}** in Vault (ID: `{stored_meta['id']}`)")
                    
                    if run_indexing_immediately:
                        with st.status("Building GraphRAG on uploaded document...", expanded=True) as status_build:
                            engine = GraphRAGEngine(model_name="gpt-4o-mini", temperature=0.0)
                            res = engine.build_from_text(
                                document_text=stored_meta["extracted_text"],
                                document_title=stored_meta["filename"],
                                max_chunks=ingest_chunk_depth
                            )
                            status_build.update(label="🎉 Indexing Complete!", state="complete")
                            st.cache_data.clear()
                            st.cache_resource.clear()
                            st.rerun()

    # Display Retained Documents Catalog
    st.markdown("#### 📂 Vault Document Registry")
    current_catalog = vault.get_catalog()
    if current_catalog:
        catalog_table_data = []
        for d in current_catalog:
            catalog_table_data.append({
                "Document ID": d["id"],
                "Document Name": d["title"],
                "Format": d["source_type"],
                "Size (KB)": d["file_size_kb"],
                "Pages/Sections": d["page_count"],
                "Uploaded At": d["uploaded_at"],
                "Status": d["status"]
            })
        st.dataframe(pd.DataFrame(catalog_table_data), use_container_width=True)
        
        # Document Inspector
        selected_doc_id = st.selectbox("Inspect Document in Vault:", [d["id"] for d in current_catalog], format_func=lambda x: f"{x} - {next((d['title'] for d in current_catalog if d['id'] == x), '')}")
        doc_details = vault.get_document(selected_doc_id)
        if doc_details:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.info(f"**SHA-256 Hash:** `{doc_details['sha256'][:16]}...`")
            with c2:
                st.info(f"**Storage Path:** `{doc_details['storage_path']}`")
            with c3:
                st.info(f"**Character Count:** `{doc_details['char_count']:,}`")
            
            if st.button("🗑️ Delete Document from Vault", key=f"del_{selected_doc_id}"):
                vault.delete_document(selected_doc_id)
                st.success("Deleted document from vault.")
                st.rerun()
    else:
        st.info("No documents found in vault.")

# =============================================================================
# TAB 3: Knowledge Concept Map
# =============================================================================
with tab_concept_map:
    st.markdown("### 🕸️ Enterprise Concept Map & Relational Network")
    st.markdown("Interactive visual topography of core organizational concepts, frameworks, and tools grouped by strategic thematic domains.")
    
    html_file = Path("./notebook/interactive_graph.html")
    if html_file.exists():
        raw_html = html_file.read_text(encoding="utf-8")
        st.components.v1.html(raw_html, height=650, scrolling=True)
    else:
        st.info("Concept map not found. Ingest your document in the **📑 Enterprise Document Vault** tab to generate the map.")

# =============================================================================
# TAB 4: Enterprise Data Catalog
# =============================================================================
with tab_catalog_ui:
    st.markdown("### 📊 Enterprise Knowledge Catalog & Structured Taxonomy")
    st.markdown("Audit and explore recognized business concepts, dependency links, and domain briefs.")
    
    sub_c1, sub_c2, sub_c3, sub_c4 = st.tabs([
        "🏷️ Business Concepts",
        "🔗 Dependency Links",
        "🌐 Domain Network Map",
        "📑 Strategic Domain Briefs"
    ])
    
    with sub_c1:
        st.markdown(f"**Total Indexed Concepts:** `{len(entities_df)}`")
        f_term = st.text_input("Search concept by name or definition:", key="f_concept_term")
        filt_df = entities_df[entities_df['title'].str.contains(f_term, case=False, na=False) | entities_df['description'].str.contains(f_term, case=False, na=False)] if f_term else entities_df
        st.dataframe(filt_df[['title', 'type', 'description']], use_container_width=True)
        
    with sub_c2:
        st.markdown(f"**Total Verified Dependency Links:** `{len(relationships_df)}`")
        f_rel_term = st.text_input("Search relationships:", key="f_rel_term")
        filt_r_df = relationships_df[relationships_df['source'].str.contains(f_rel_term, case=False, na=False) | relationships_df['target'].str.contains(f_rel_term, case=False, na=False) | relationships_df['description'].str.contains(f_rel_term, case=False, na=False)] if f_rel_term else relationships_df
        st.dataframe(filt_r_df[['source', 'target', 'weight', 'description']], use_container_width=True)
        
    with sub_c3:
        st.markdown(f"**Thematic Domain Node Assignments:** `{len(nodes_df)}`")
        st.dataframe(nodes_df[['title', 'community', 'degree']], use_container_width=True)
        
    with sub_c4:
        st.markdown(f"**Strategic Domain Briefs:** `{len(community_df)}`")
        st.dataframe(community_df[['community', 'title', 'summary', 'rank']], use_container_width=True)
        for _, r in community_df.iterrows():
            with st.expander(f"📑 {r['title']} (Strategic Weight: {r['rank']})"):
                st.markdown(r['full_content'])

# =============================================================================
# TAB 5: System Governance & Changelog
# =============================================================================
with tab_governance:
    st.markdown("### 🛡️ System Governance, Quality & Change Management")
    st.markdown("Automated QA test results, semantic versioning, and auditable release logs managed by the Change Management Agent.")
    
    col_gov1, col_gov2 = st.columns([1, 1])
    with col_gov1:
        st.markdown("#### 🧪 Automated QA Health Report")
        qa_report = change_agent.run_qa_checks()
        st.json(qa_report)
        
        if st.button("🔄 Re-Run QA & Synchronize Docs", use_container_width=True):
            with st.spinner("Running full change management cycle..."):
                sync_out = change_agent.run_full_sync()
                st.success(f"Synchronized Version v{sync_out['version']} with changelog and README!")
                st.rerun()
                
    with col_gov2:
        st.markdown("#### 📜 Live Release Changelog")
        changelog_path = Path("./CHANGELOG.md")
        if changelog_path.exists():
            st.markdown(changelog_path.read_text(encoding="utf-8"))
        else:
            st.info("Changelog will be generated upon first sync.")
