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

from graphrag_pipeline import DocumentIngestor, GraphRAGEngine

# -----------------------------------------------------------------------------
# 1. Enterprise Page Configuration & Executive Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Knowledge Intelligence Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")

# Executive Enterprise SaaS Styling (Inspired by Stripe, Palantir, McKinsey, Linear)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0A0D14;
        color: #E2E8F0;
    }
    
    /* Executive Header */
    .executive-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }
    
    .header-branding {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    
    .header-icon {
        background: linear-gradient(135deg, #3B82F6, #6366F1);
        color: white;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 1.35rem;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
    }
    
    .platform-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
    }
    
    .platform-subtitle {
        color: #94A3B8;
        font-size: 0.88rem;
        font-weight: 400;
        margin: 2px 0 0 0;
    }
    
    .compliance-pill {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6EE7B7;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* KPI Dashboard Cards */
    .kpi-card {
        background: #121620;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 14px 18px;
        position: relative;
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #3B82F6, #6366F1);
    }
    
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #94A3B8;
        letter-spacing: 0.05em;
        margin-top: 3px;
    }
    
    /* Executive Badges */
    .badge-exec {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    
    .badge-primary {
        background: rgba(59, 130, 246, 0.15);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }
    
    .badge-secondary {
        background: rgba(139, 92, 246, 0.15);
        color: #C4B5FD;
        border: 1px solid rgba(139, 92, 246, 0.35);
    }

    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    /* Audit Trail & Citation Box */
    .audit-terminal {
        background: #0B0E14;
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 8px;
        padding: 14px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #CBD5E1;
        line-height: 1.55;
        max-height: 300px;
        overflow-y: auto;
    }

    /* Executive Split Card */
    .exec-card {
        background: #121620;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 18px;
        height: 100%;
    }
    
    .exec-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data & Client Caching
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_graph_data(root_path: str = "./ragtest") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the enterprise knowledge datasets."""
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
    """Load ChromaDB document vault."""
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name="paper_collection")
    return client, collection

@st.cache_resource(show_spinner=False)
def get_llm(temperature: float = 0.2, model_name: str = "gpt-4o-mini"):
    """Initialize Enterprise LLM Reasoning Engine."""
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

# Load core datasets
entities_df, relationships_df, nodes_df, community_df = load_graph_data()
chroma_client, paper_collection = load_chroma_db()

# -----------------------------------------------------------------------------
# 3. Executive AI Reasoning & Search Strategies
# -----------------------------------------------------------------------------
def query_graphrag_local(query: str, entities: pd.DataFrame, rels: pd.DataFrame, llm: ChatOpenAI, top_k: int = 15) -> Tuple[str, str]:
    context_lines = ["### Verified Business & Technical Entities:"]
    for _, row in entities.head(top_k).iterrows():
        context_lines.append(f"- **{row['title']}** ({row['type']}): {row['description']}")
    
    context_lines.append("\n### Direct Relational Dependencies:")
    for _, row in rels.head(top_k).iterrows():
        context_lines.append(f"- **{row['source']}** ➔ **{row['target']}** (Confidence: {row['weight']}/10): {row['description']}")
    
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an Executive AI Strategic Advisor performing Targeted Fact & Concept Lookup.\n"
        "Provide a crisp, actionable business response grounded in the verified concept network below.\n\n"
        "Verified Knowledge Network:\n{context}\n\n"
        "Executive Ingestion/Question: {query}\n\n"
        "Format your answer with clear bullet points, strategic implications, and concise takeaways."
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text

def query_graphrag_global(query: str, reports: pd.DataFrame, llm: ChatOpenAI) -> Tuple[str, str]:
    context_lines = ["### Executive Thematic Domain Summaries:"]
    for _, row in reports.iterrows():
        context_lines.append(f"#### {row['title']}\n{row['summary']}\n")
    
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an Executive AI Strategic Advisor performing a Macro-Level Strategic Synthesis across business domains.\n"
        "Synthesize an executive-level summary answering the inquiry across the domain reports below.\n\n"
        "Thematic Domain Reports:\n{context}\n\n"
        "Executive Inquiry: {query}\n\n"
        "Structure the response for senior leadership: Executive Summary, Strategic Analysis, Core Trade-offs, and Actionable Recommendations."
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text

def query_graphrag_drift(query: str, reports: pd.DataFrame, rels: pd.DataFrame, llm: ChatOpenAI) -> Tuple[str, str]:
    context_lines = ["### Thematic Strategy Briefs:"]
    for _, row in reports.head(6).iterrows():
        context_lines.append(f"- **{row['title']}**: {row['summary']}")
    
    context_lines.append("\n### Granular Cross-Functional Dependencies:")
    for _, row in rels.head(20).iterrows():
        context_lines.append(f"- **{row['source']}** <-> **{row['target']}**: {row['description']}")
    
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an Executive AI Strategic Advisor performing a Deep-Dive Cross-Functional Analysis.\n"
        "Answer the business inquiry by connecting macro business strategy with granular operational and technological dependencies.\n\n"
        "Knowledge Context:\n{context}\n\n"
        "Executive Question: {query}\n\n"
        "Provide a comprehensive, high-impact business analysis with clear structure, risk mitigation insights, and execution steps."
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text

def query_chroma_rag(query: str, collection, llm: ChatOpenAI, num_results: int = 4) -> Tuple[str, str]:
    results = collection.query(query_texts=[query], n_results=num_results)
    docs = results.get("documents", [[]])[0]
    
    context_lines = [f"--- Document Citation #{idx} ---\n{doc.strip()}\n" for idx, doc in enumerate(docs, 1)]
    context_text = "\n".join(context_lines)
    
    prompt = ChatPromptTemplate.from_template(
        "You are an AI Analyst performing Standard Document Passage Retrieval.\n"
        "Answer the business question strictly using the retrieved source passages below.\n"
        "If evidence is insufficient, state so clearly.\n\n"
        "Source Passages:\n{context}\n\n"
        "Business Question: {query}\n\n"
        "Answer:"
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text

# -----------------------------------------------------------------------------
# 4. Top Executive Header & KPI Dashboard
# -----------------------------------------------------------------------------
st.markdown("""
<div class="executive-header">
    <div class="header-branding">
        <div class="header-icon">🏛️</div>
        <div>
            <h1 class="platform-title">Enterprise Knowledge Intelligence</h1>
            <p class="platform-subtitle">Autonomous Graph-Augmented Decision Support & Strategic Research Platform</p>
        </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
        <span class="compliance-pill">● SECURE AUDIT TRAIL</span>
        <span class="badge-exec badge-primary">ENTERPRISE v2.0</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Executive KPI Dashboard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(entities_df)}</div><div class="kpi-label">Indexed Business Concepts</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(relationships_df)}</div><div class="kpi-label">Verified Relationship Links</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(community_df)}</div><div class="kpi-label">Strategic Thematic Domains</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-value">{paper_collection.count()}</div><div class="kpi-label">Source Evidence Passages</div></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Sidebar Controls & Analysis Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 Strategic Analysis Mode")
    search_mode = st.selectbox(
        "Select Reasoning Strategy",
        [
            "🌐 Strategic Executive Synthesis",
            "🔍 Targeted Concept & Dependency Lookup",
            "🌀 Deep-Dive Cross-Functional Analysis",
            "📚 Standard Document Passage Search",
            "⚖️ Comparative Audit: Graph vs Document Search"
        ],
        index=0,
        help="Select the intelligence depth: High-level executive synthesis, direct fact lookup, or side-by-side comparative audit."
    )
    
    st.divider()
    st.markdown("### 🎛️ Analysis Profile")
    reasoning_tone = st.select_slider(
        "Response Tone & Focus",
        options=["Strictly Factual (0.0)", "Balanced Executive Summary (0.2)", "Comprehensive & Exploratory (0.5)"],
        value="Balanced Executive Summary (0.2)"
    )
    temp = 0.0 if "0.0" in reasoning_tone else (0.2 if "0.2" in reasoning_tone else 0.5)
    
    top_k_passages = st.slider("Source Citations to Retrieve", min_value=2, max_value=10, value=4)
    
    st.divider()
    if st.button("🗑️ Clear Session & History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("### 🔒 Infrastructure Status")
    if api_key:
        st.markdown('<span class="badge-exec badge-emerald">● Enterprise LLM Connected</span>', unsafe_allow_html=True)
    else:
        st.error("Missing `OPENAI_API_KEY` in environment.")

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Welcome to the **Enterprise Knowledge Intelligence Platform**.\n\nI am your **Executive AI Strategic Advisor**, connected to your organizational knowledge base built from *The Ultimate Guide to Fine-Tuning LLMs & Enterprise AI Operations*. \n\nHow may I assist you today? You can choose a strategic inquiry below, compare **Graph Intelligence vs Document Search**, or use the **📄 Ingestion & Knowledge Extraction** tab to index new enterprise documents on demand.",
            "mode": "System Advisor",
            "context": None
        }
    ]

# -----------------------------------------------------------------------------
# 6. Business Navigation Flow
# -----------------------------------------------------------------------------
tab_ingest, tab_advisor, tab_map, tab_catalog, tab_vault = st.tabs([
    "📄 Ingestion & Knowledge Extraction",
    "💬 Executive AI Advisor & Analysis",
    "🕸️ Business Concept Map",
    "📊 Enterprise Knowledge Catalog",
    "📑 Document Vault & Source Citations"
])

# =============================================================================
# TAB 1: Ingestion & Knowledge Extraction
# =============================================================================
with tab_ingest:
    st.markdown("### 📄 Automated Enterprise Document Ingestion")
    st.markdown("Ingest corporate PDF reports, whitepapers, contracts, or strategy memos to automatically construct a structured, audit-ready enterprise knowledge network.")
    
    input_type = st.radio(
        "Select Document Source:",
        ["📁 Upload Corporate Document (PDF / TXT / Word / MD)", "✍️ Paste Executive Brief / Text", "📚 Load Benchmark Enterprise Research"],
        horizontal=True
    )
    
    doc_text_payload = ""
    source_title = "Enterprise Document Ingestion"

    if input_type == "📁 Upload Corporate Document (PDF / TXT / Word / MD)":
        uploaded_doc = st.file_uploader("Upload business report, research paper, or operational manual", type=["pdf", "txt", "md"])
        if uploaded_doc is not None:
            source_title = uploaded_doc.name
            if uploaded_doc.name.endswith(".pdf"):
                with st.spinner("Extracting text from PDF document..."):
                    doc_text_payload = DocumentIngestor.extract_from_pdf_bytes(uploaded_doc.read())
            else:
                doc_text_payload = uploaded_doc.read().decode("utf-8")
            st.success(f"📄 Successfully loaded **{source_title}** ({len(doc_text_payload):,} characters)")
            
    elif input_type == "✍️ Paste Executive Brief / Text":
        pasted_text = st.text_area(
            "Paste executive memo, strategy document, or meeting transcript:",
            height=200,
            placeholder="Paste your source text here..."
        )
        if pasted_text.strip():
            doc_text_payload = pasted_text.strip()
            source_title = "Executive Brief Ingestion"
            
    else: # Preset Benchmark Reports
        preset_doc = st.selectbox(
            "Select a pre-packaged enterprise research brief:",
            [
                "📖 The Ultimate Guide to Fine-Tuning LLMs (CeADAR Research - 115 Pages)",
                "🏗️ Enterprise RAG & Semantic Architecture Standard",
                "🤖 Multi-Agent Systems & Mixture-of-Agents Operational Framework"
            ]
        )
        if "Fine-Tuning" in preset_doc:
            p_file = Path("./ragtest/input/ft_guide.txt")
            if p_file.exists():
                doc_text_payload = p_file.read_text(encoding="utf-8")
                source_title = "The Ultimate Guide to Fine-Tuning LLMs"
                st.info(f"Loaded benchmark report: **{len(doc_text_payload):,}** characters")
        elif "Architecture" in preset_doc:
            doc_text_payload = """
            Retrieval-Augmented Generation (RAG) combines dense vector search with large language models to provide factual grounding.
            Vector databases like ChromaDB index high-dimensional embeddings generated by transformer models.
            Knowledge Graphs capture explicit relationships, entity types, and hierarchical community clusters that vector embeddings alone miss.
            GraphRAG unifies structured graph reasoning with unstructured dense vector search to solve multi-hop enterprise inquiries.
            """
            source_title = "Enterprise RAG Architecture Standard"
        else:
            doc_text_payload = """
            Mixture of Agents (MoA) harnesses layered architectures of specialized LLMs collaborating to synthesize high-quality reasoning.
            Direct Preference Optimization (DPO) and Proximal Policy Optimization (PPO) align these agent clusters with human preferences.
            Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT/LoRA) adapt individual specialist agents with minimal compute.
            """
            source_title = "Multi-Agent Systems Operational Framework"

    st.divider()
    st.markdown("#### ⚙️ Extraction Depth & Governance Settings")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        ingestion_depth = st.select_slider(
            "Ingestion Scope & Processing Depth:",
            options=["Fast Review (5 Chunks)", "Standard Audit (15 Chunks)", "Comprehensive Ingestion (Full Document)"],
            value="Fast Review (5 Chunks)"
        )
        max_chunks_limit = 5 if "5" in ingestion_depth else (15 if "15" in ingestion_depth else 50)
    with col_g2:
        st.markdown("**Compliance & Governance Protocol:**")
        st.markdown("- Automatic Entity Deduplication & Cross-Referencing\n- Modularity-based Thematic Domain Clustering\n- Verifiable Citation Passages Embedded")

    # Ingestion Button
    if st.button("🚀 Ingest & Build Enterprise Knowledge Network", type="primary", use_container_width=True, disabled=not bool(doc_text_payload.strip())):
        progress_bar = st.progress(0.0)
        status_box = st.status("Initializing Enterprise Knowledge Extraction...", expanded=True)
        
        engine = GraphRAGEngine(model_name="gpt-4o-mini", temperature=0.0)
        
        def handle_exec_progress(pct: float, message: str, stats: Dict[str, Any]):
            progress_bar.progress(pct)
            status_box.write(f"**[{int(pct * 100)}%]** {message}")
            if stats:
                status_box.json(stats)
        
        try:
            results = engine.build_from_text(
                document_text=doc_text_payload,
                max_chunks=max_chunks_limit,
                chunk_size=1200,
                chunk_overlap=100,
                progress_callback=handle_exec_progress
            )
            
            status_box.update(label=f"🎉 Enterprise Knowledge Ingestion Complete in {results['elapsed_seconds']}s!", state="complete", expanded=False)
            
            # Clear caches so data propagates across all tabs
            st.cache_data.clear()
            st.cache_resource.clear()
            
            st.balloons()
            st.success(f"""
            ✅ **Knowledge Extraction Successfully Completed for: {source_title}**
            - **Business Concepts Identified**: {results['entities_count']}
            - **Strategic Relationship Links**: {results['relationships_count']}
            - **Thematic Strategic Domains**: {results['communities_count']}
            - **Verified Source Passages**: {results['chunks_count']}
            """)
            st.rerun()
            
        except Exception as e:
            status_box.update(label="❌ Knowledge Ingestion Failed", state="error", expanded=True)
            st.error(f"Ingestion Error: {str(e)}")

# =============================================================================
# TAB 2: Executive AI Advisor & Analysis
# =============================================================================
with tab_advisor:
    # Strategic Business Questions
    st.markdown("**💡 Strategic Business Inquiries:**")
    p_cols = st.columns(4)
    business_prompts = [
        "What are the cost, ROI, and resource trade-offs between RAG and Fine-Tuning?",
        "Which techniques reduce operational compute overhead and deployment risks?",
        "What are the strategic dependencies and recommendations outlined in this report?",
        "Summarize the core thematic knowledge domains identified across the document."
    ]
    
    clicked_prompt = None
    for i, col in enumerate(p_cols):
        with col:
            if st.button(business_prompts[i], key=f"biz_chip_{i}", use_container_width=True):
                clicked_prompt = business_prompts[i]
    
    st.divider()

    # Conversation Timeline
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "mode" in msg and msg["mode"] != "System Advisor":
                badge_style = "badge-secondary" if "Passage" in msg["mode"] else "badge-primary"
                st.markdown(f'<span class="badge-exec {badge_style}">{msg["mode"]}</span>', unsafe_allow_html=True)
            
            st.markdown(msg["content"])
            
            if msg.get("context"):
                with st.expander("🔍 Audit Trail & Source Evidence"):
                    st.markdown(f'<div class="audit-terminal">{html.escape(msg["context"])}</div>', unsafe_allow_html=True)
            
            if msg.get("comparison"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="exec-card">
                        <div class="exec-card-title"><span class="badge-exec badge-primary">🌐 GRAPHRAG STRATEGIC DEEP-DIVE</span></div>
                    """, unsafe_allow_html=True)
                    st.markdown(msg["comparison"]["graph_answer"])
                    with st.expander("Audit Trail (Relational Proof)"):
                        st.markdown(f'<div class="audit-terminal">{html.escape(msg["comparison"]["graph_context"])}</div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div class="exec-card">
                        <div class="exec-card-title"><span class="badge-exec badge-secondary">📚 STANDARD DOCUMENT PASSAGE SEARCH</span></div>
                    """, unsafe_allow_html=True)
                    st.markdown(msg["comparison"]["vector_answer"])
                    with st.expander("Audit Trail (Direct Passages)"):
                        st.markdown(f'<div class="audit-terminal">{html.escape(msg["comparison"]["vector_context"])}</div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    # Question Input
    user_query = st.chat_input("Enter strategic question, architectural inquiry, or policy question...") or clicked_prompt

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        llm = get_llm(temperature=temp)
        with st.chat_message("assistant"):
            with st.spinner(f"Synthesizing response using {search_mode}..."):
                try:
                    if "Synthesis" in search_mode:
                        ans, ctx = query_graphrag_global(user_query, community_df, llm)
                        st.markdown('<span class="badge-exec badge-primary">Strategic Synthesis</span>', unsafe_allow_html=True)
                        st.markdown(ans)
                        with st.expander("🔍 Audit Trail & Source Evidence"):
                            st.markdown(f'<div class="audit-terminal">{html.escape(ctx)}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "mode": "Strategic Synthesis", "context": ctx})
                    
                    elif "Targeted" in search_mode:
                        ans, ctx = query_graphrag_local(user_query, entities_df, relationships_df, llm)
                        st.markdown('<span class="badge-exec badge-primary">Targeted Concept Lookup</span>', unsafe_allow_html=True)
                        st.markdown(ans)
                        with st.expander("🔍 Audit Trail & Source Evidence"):
                            st.markdown(f'<div class="audit-terminal">{html.escape(ctx)}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "mode": "Targeted Concept Lookup", "context": ctx})
                    
                    elif "Deep-Dive" in search_mode:
                        ans, ctx = query_graphrag_drift(user_query, community_df, relationships_df, llm)
                        st.markdown('<span class="badge-exec badge-primary">Deep-Dive Analysis</span>', unsafe_allow_html=True)
                        st.markdown(ans)
                        with st.expander("🔍 Audit Trail & Source Evidence"):
                            st.markdown(f'<div class="audit-terminal">{html.escape(ctx)}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "mode": "Deep-Dive Analysis", "context": ctx})
                    
                    elif "Standard" in search_mode:
                        ans, ctx = query_chroma_rag(user_query, paper_collection, llm, num_results=top_k_passages)
                        st.markdown('<span class="badge-exec badge-secondary">Standard Passage Retrieval</span>', unsafe_allow_html=True)
                        st.markdown(ans)
                        with st.expander("🔍 Audit Trail & Source Evidence"):
                            st.markdown(f'<div class="audit-terminal">{html.escape(ctx)}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "mode": "Standard Passage Search", "context": ctx})
                    
                    else: # Comparative Audit
                        g_ans, g_ctx = query_graphrag_drift(user_query, community_df, relationships_df, llm)
                        v_ans, v_ctx = query_chroma_rag(user_query, paper_collection, llm, num_results=top_k_passages)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("""
                            <div class="exec-card">
                                <div class="exec-card-title"><span class="badge-exec badge-primary">🌐 GRAPHRAG STRATEGIC DEEP-DIVE</span></div>
                            """, unsafe_allow_html=True)
                            st.markdown(g_ans)
                            with st.expander("Audit Trail (Relational Proof)"):
                                st.markdown(f'<div class="audit-terminal">{html.escape(g_ctx)}</div>', unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown("""
                            <div class="exec-card">
                                <div class="exec-card-title"><span class="badge-exec badge-secondary">📚 STANDARD DOCUMENT PASSAGE SEARCH</span></div>
                            """, unsafe_allow_html=True)
                            st.markdown(v_ans)
                            with st.expander("Audit Trail (Direct Passages)"):
                                st.markdown(f'<div class="audit-terminal">{html.escape(v_ctx)}</div>', unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "⚖️ **Comparative Audit Complete**",
                            "mode": "Comparative Audit",
                            "comparison": {
                                "graph_answer": g_ans,
                                "graph_context": g_ctx,
                                "vector_answer": v_ans,
                                "vector_context": v_ctx
                            }
                        })
                except Exception as e:
                    st.error(f"Advisory Engine Error: {str(e)}")

# =============================================================================
# TAB 3: Business Concept Map
# =============================================================================
with tab_map:
    st.markdown("### 🕸️ Enterprise Concept Map & Relational Network")
    st.markdown("Interactive visual topography of core organizational concepts, frameworks, and tools grouped by strategic thematic domains.")
    
    html_file = Path("./notebook/interactive_graph.html")
    if html_file.exists():
        raw_html = html_file.read_text(encoding="utf-8")
        st.components.v1.html(raw_html, height=650, scrolling=True)
    else:
        st.info("Concept map not found. Ingest your document in the **📄 Ingestion & Knowledge Extraction** tab to generate the map.")

# =============================================================================
# TAB 4: Enterprise Knowledge Catalog
# =============================================================================
with tab_catalog:
    st.markdown("### 📊 Enterprise Knowledge Catalog & Structured Taxonomy")
    st.markdown("Audit and explore the recognized business concepts, dependencies, and domain summaries.")
    
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
# TAB 5: Document Vault & Source Citations
# =============================================================================
with tab_vault:
    st.markdown("### 📑 Document Vault & Source Citations")
    st.markdown("Direct inspection of underlying document passages with semantic relevance verification.")
    
    st.info(f"Connected Vault: **`paper_collection`** | Total Indexed Passages: **{paper_collection.count()}**")
    
    test_vault_q = st.text_input("Verify Semantic Source Passage Lookup:", value="Parameter-efficient fine tuning LoRA vs Full Fine Tuning")
    k_vault = st.slider("Passages to Inspect", min_value=1, max_value=8, value=3)
    
    if st.button("Query Document Vault", use_container_width=True):
        res = paper_collection.query(query_texts=[test_vault_q], n_results=k_vault)
        docs = res.get("documents", [[]])[0]
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0] if "distances" in res and res["distances"] else [None] * len(docs)
        
        for idx, (doc_id, doc, dist) in enumerate(zip(ids, docs, distances), 1):
            dist_label = f'<span class="badge-exec badge-primary">Match Distance: {round(dist, 4)}</span>' if dist is not None else ""
            st.markdown(f"#### 📄 Source Evidence #{idx}: Citation `{doc_id}` {dist_label}", unsafe_allow_html=True)
            st.markdown(f'<div class="audit-terminal">{html.escape(doc)}</div>', unsafe_allow_html=True)
