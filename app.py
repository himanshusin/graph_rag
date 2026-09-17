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
# 1. Page Configuration & Ultra-Modern Minimalist Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GraphRAG Studio — Autonomous Knowledge Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")

# Million-Dollar Startup Minimalist Dark Theme (Linear / Vercel Aesthetic)
st.markdown("""
<style>
    /* Global Base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #08090C;
        color: #E2E8F0;
    }
    
    /* Sleek Startup Header */
    .brand-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 1.5rem;
    }
    
    .brand-logo-area {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .brand-icon {
        background: linear-gradient(135deg, #6366F1, #06B6D4);
        padding: 8px 12px;
        border-radius: 10px;
        font-size: 1.3rem;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.35);
    }
    
    .brand-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(180deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .brand-subtitle {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0;
    }
    
    .version-pill {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #818CF8;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Minimalist Glass Card */
    .glass-card {
        background: #11131A;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.25);
    }
    
    /* HUD Metric Cards */
    .hud-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    
    .hud-card {
        background: #11131A;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 14px 16px;
        position: relative;
        overflow: hidden;
    }
    
    .hud-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #6366F1, #06B6D4);
        opacity: 0.7;
    }
    
    .hud-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.03em;
    }
    
    .hud-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.06em;
        margin-top: 4px;
    }
    
    /* Pill Badges */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .pill-indigo {
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.35);
    }
    
    .pill-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #67E8F9;
        border: 1px solid rgba(6, 182, 212, 0.35);
    }
    
    .pill-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    /* Code & Context Box */
    .context-terminal {
        background: #0B0C10;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 14px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #94A3B8;
        line-height: 1.5;
        max-height: 280px;
        overflow-y: auto;
    }

    /* Comparison Split Container */
    .compare-card {
        background: #11131A;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 16px;
        height: 100%;
    }
    
    .compare-header {
        font-size: 0.95rem;
        font-weight: 700;
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
    """Load the GraphRAG parquet datasets."""
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
    """Load ChromaDB client and paper_collection."""
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name="paper_collection")
    return client, collection

@st.cache_resource(show_spinner=False)
def get_llm(temperature: float = 0.2, model_name: str = "gpt-4o-mini"):
    """Initialize ChatOpenAI."""
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

# Load core datasets
entities_df, relationships_df, nodes_df, community_df = load_graph_data()
chroma_client, paper_collection = load_chroma_db()

# -----------------------------------------------------------------------------
# 3. GraphRAG & Vector Retrieval Query Functions
# -----------------------------------------------------------------------------
def query_graphrag_local(query: str, entities: pd.DataFrame, rels: pd.DataFrame, llm: ChatOpenAI, top_k: int = 15) -> Tuple[str, str]:
    context_lines = ["### Identified Graph Entities:"]
    for _, row in entities.head(top_k).iterrows():
        context_lines.append(f"- **{row['title']}** ({row['type']}): {row['description']}")
    
    context_lines.append("\n### Key Graph Relationships:")
    for _, row in rels.head(top_k).iterrows():
        context_lines.append(f"- **{row['source']}** ➔ **{row['target']}** (Weight: {row['weight']}): {row['description']}")
    
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an expert AI assistant performing GraphRAG Local Search.\n"
        "Answer the question using the local entity subgraph and relationships below.\n\n"
        "Graph Context:\n{context}\n\n"
        "User Question: {query}\n\n"
        "Provide a clear, well-structured response grounded in the provided entity and relationship context."
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text

def query_graphrag_global(query: str, reports: pd.DataFrame, llm: ChatOpenAI) -> Tuple[str, str]:
    context_lines = ["### Hierarchical Community Reports:"]
    for _, row in reports.iterrows():
        context_lines.append(f"#### {row['title']}\n{row['summary']}\n")
    
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an expert AI assistant performing GraphRAG Global Search.\n"
        "Synthesize a comprehensive, high-level answer across the Leiden community reports below.\n\n"
        "Community Reports:\n{context}\n\n"
        "User Question: {query}\n\n"
        "Synthesize key themes, architectural trade-offs, and high-level insights."
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text

def query_graphrag_drift(query: str, reports: pd.DataFrame, rels: pd.DataFrame, llm: ChatOpenAI) -> Tuple[str, str]:
    context_lines = ["### High-Level Community Context:"]
    for _, row in reports.head(6).iterrows():
        context_lines.append(f"- **{row['title']}**: {row['summary']}")
    
    context_lines.append("\n### Granular Multi-Hop Traversal Relationships:")
    for _, row in rels.head(20).iterrows():
        context_lines.append(f"- **{row['source']}** <-> **{row['target']}**: {row['description']}")
    
    context_text = "\n".join(context_lines)
    prompt = ChatPromptTemplate.from_template(
        "You are an expert AI assistant performing GraphRAG DRIFT Search (Dynamic Reasoning & Flexible Traversal).\n"
        "Answer the question combining both macro community reports and granular multi-hop relationship traversals.\n\n"
        "Context:\n{context}\n\n"
        "User Question: {query}\n\n"
        "Provide a deep, nuanced answer connecting high-level architectures with granular implementation details."
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text

def query_chroma_rag(query: str, collection, llm: ChatOpenAI, num_results: int = 4) -> Tuple[str, str]:
    results = collection.query(query_texts=[query], n_results=num_results)
    docs = results.get("documents", [[]])[0]
    
    context_lines = [f"--- Chunk #{idx} ---\n{doc.strip()}\n" for idx, doc in enumerate(docs, 1)]
    context_text = "\n".join(context_lines)
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert AI assistant performing Vector RAG.\n"
        "Answer the user question using only the retrieved document chunks below.\n"
        "If you don't know the answer, state so honestly.\n\n"
        "Retrieved Chunks:\n{context}\n\n"
        "User Question: {query}\n\n"
        "Answer:"
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "query": query})
    return answer, context_text

# -----------------------------------------------------------------------------
# 4. Minimalist Brand Header & Top-Level HUD
# -----------------------------------------------------------------------------
st.markdown("""
<div class="brand-container">
    <div class="brand-logo-area">
        <div class="brand-icon">⚡</div>
        <div>
            <h1 class="brand-title">GraphRAG Studio</h1>
            <p class="brand-subtitle">Autonomous Knowledge Graph & Hybrid Vector Intelligence Engine</p>
        </div>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
        <span class="version-pill">v2.0 • PROD</span>
        <span class="pill pill-emerald">● SYSTEM ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Telemetry HUD
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="hud-card"><div class="hud-val">{len(entities_df)}</div><div class="hud-label">Graph Entities</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="hud-card"><div class="hud-val">{len(relationships_df)}</div><div class="hud-label">Relationship Edges</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="hud-card"><div class="hud-val">{len(community_df)}</div><div class="hud-label">Leiden Communities</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="hud-card"><div class="hud-val">{paper_collection.count()}</div><div class="hud-label">Vector Chunks</div></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    search_mode = st.selectbox(
        "🎯 Search Strategy",
        [
            "🌐 GraphRAG: Global Search",
            "🔍 GraphRAG: Local Search",
            "🌀 GraphRAG: DRIFT Search",
            "📚 ChromaDB: Vector RAG",
            "⚔️ Side-by-Side: Graph vs Vector RAG"
        ],
        index=0,
        help="Choose between macro community synthesis, local subgraph traversal, or dense vector similarity."
    )
    
    st.divider()
    st.markdown("### 🎛️ Inference Hyperparameters")
    temp = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    top_k_chunks = st.slider("Vector Chunks (Top-K)", min_value=1, max_value=10, value=4)
    
    st.divider()
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("### 🔐 Credentials")
    if api_key:
        st.markdown(f'<span class="pill pill-indigo">OpenAI Key: {api_key[:7]}...</span>', unsafe_allow_html=True)
    else:
        st.error("Missing `OPENAI_API_KEY` in `.env`")

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Welcome to **GraphRAG Studio**.\n\nYour autonomous knowledge base is initialized and ready. You can query using **GraphRAG (Global/Local/DRIFT)**, compare with **Vector RAG**, inspect the **Interactive Graph**, or use the **⚡ On-Demand Builder** tab to index any new document on demand!",
            "mode": "System",
            "context": None
        }
    ]

# -----------------------------------------------------------------------------
# 6. Tab Navigation
# -----------------------------------------------------------------------------
tab_builder, tab_chat, tab_graph, tab_parquet, tab_chroma = st.tabs([
    "⚡ On-Demand Builder",
    "💬 AI Chatbot",
    "🕸️ Interactive Graph",
    "📊 Parquet Explorer",
    "🗄️ ChromaDB Store"
])

# =============================================================================
# TAB 1: On-Demand GraphRAG Builder
# =============================================================================
with tab_builder:
    st.markdown("### ⚡ Autonomous Knowledge Ingestion & Graph Builder")
    st.markdown("Ingest documents (PDF, TXT, Raw Text) to automatically extract entities, build the knowledge graph, cluster communities, and embed vectors on demand.")
    
    input_method = st.radio(
        "Select Document Input Method:",
        ["📁 Upload File (PDF / TXT / MD)", "✍️ Direct Text Input", "📚 Load Preset Dataset"],
        horizontal=True
    )
    
    document_text_to_process = ""
    source_name = "Custom Ingestion"

    if input_method == "📁 Upload File (PDF / TXT / MD)":
        uploaded_file = st.file_uploader("Choose a document", type=["pdf", "txt", "md"])
        if uploaded_file is not None:
            source_name = uploaded_file.name
            if uploaded_file.name.endswith(".pdf"):
                with st.spinner("Extracting text from PDF..."):
                    document_text_to_process = DocumentIngestor.extract_from_pdf_bytes(uploaded_file.read())
            else:
                document_text_to_process = uploaded_file.read().decode("utf-8")
            st.success(f"📄 Loaded **{source_name}** ({len(document_text_to_process):,} characters)")
            
    elif input_method == "✍️ Direct Text Input":
        raw_text_input = st.text_area(
            "Paste text, article, documentation, or transcript here:",
            height=200,
            placeholder="Paste your source text here..."
        )
        if raw_text_input.strip():
            document_text_to_process = raw_text_input.strip()
            source_name = "Raw Text Input"
            
    else: # Preset Dataset
        preset_choice = st.selectbox(
            "Choose a pre-packaged benchmark dataset:",
            [
                "📖 The Ultimate Guide to Fine-Tuning LLMs (CeADAR Research)",
                "🏗️ Modern Enterprise RAG & Vector Search Architecture",
                "🤖 Multi-Agent Systems & Mixture-of-Agents (MoA) Overview"
            ]
        )
        if "Fine-Tuning" in preset_choice:
            preset_path = Path("./ragtest/input/ft_guide.txt")
            if preset_path.exists():
                document_text_to_process = preset_path.read_text(encoding="utf-8")
                source_name = "The Ultimate Guide to Fine-Tuning LLMs"
                st.info(f"Loaded preset guide: {len(document_text_to_process):,} characters")
        elif "Architecture" in preset_choice:
            document_text_to_process = """
            Retrieval-Augmented Generation (RAG) combines dense vector search with large language models to provide factual grounding.
            Vector databases like ChromaDB and Pinecone index high-dimensional embeddings generated by transformer models.
            Knowledge Graphs capture explicit relationships, entity types, and hierarchical community clusters that vector embeddings alone miss.
            GraphRAG unifies structured graph reasoning with unstructured dense vector search to solve multi-hop question answering.
            """
            source_name = "Enterprise RAG Architecture Spec"
        else:
            document_text_to_process = """
            Mixture of Agents (MoA) harnesses layered architectures of specialized LLMs collaborating to synthesize high-quality reasoning.
            Direct Preference Optimization (DPO) and Proximal Policy Optimization (PPO) align these agent clusters with human preferences.
            Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT/LoRA) adapt individual specialist agents with minimal compute.
            """
            source_name = "Multi-Agent Systems Overview"

    st.divider()
    st.markdown("#### ⚙️ Ingestion & Chunking Parameters")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        chunk_size_param = st.slider("Token Chunk Size", min_value=300, max_value=2500, value=1200, step=100)
    with col_c2:
        overlap_param = st.slider("Chunk Overlap", min_value=0, max_value=300, value=100, step=20)
    with col_c3:
        max_chunks_limit = st.slider("Max Chunks to Process", min_value=1, max_value=30, value=5, help="Controls number of chunks for extraction to manage API execution time.")

    # Execution Action Button
    if st.button("🚀 Build GraphRAG Knowledge Engine", type="primary", use_container_width=True, disabled=not bool(document_text_to_process.strip())):
        progress_bar = st.progress(0.0)
        status_box = st.status("Initializing GraphRAG Pipeline...", expanded=True)
        
        engine = GraphRAGEngine(model_name="gpt-4o-mini", temperature=0.0)
        
        def handle_progress(pct: float, message: str, stats: Dict[str, Any]):
            progress_bar.progress(pct)
            status_box.write(f"**[{int(pct * 100)}%]** {message}")
            if stats:
                status_box.json(stats)
        
        try:
            results = engine.build_from_text(
                document_text=document_text_to_process,
                max_chunks=max_chunks_limit,
                chunk_size=chunk_size_param,
                chunk_overlap=overlap_param,
                progress_callback=handle_progress
            )
            
            status_box.update(label=f"🎉 GraphRAG Engine Built Successfully in {results['elapsed_seconds']}s!", state="complete", expanded=False)
            
            # Clear caches so the new data propagates to all tabs immediately
            st.cache_data.clear()
            st.cache_resource.clear()
            
            st.balloons()
            st.success(f"""
            ✅ **Indexing Complete for {source_name}!**
            - **Entities Extracted**: {results['entities_count']}
            - **Relationships Created**: {results['relationships_count']}
            - **Leiden Communities**: {results['communities_count']}
            - **Vector Chunks Synced**: {results['chunks_count']}
            """)
            st.rerun()
            
        except Exception as e:
            status_box.update(label="❌ Pipeline Execution Failed", state="error", expanded=True)
            st.error(f"Error during graph building: {str(e)}")

# =============================================================================
# TAB 2: AI Chatbot
# =============================================================================
with tab_chat:
    # Example Prompt Chips
    st.markdown("**💡 Quick Prompt Chips:**")
    p_cols = st.columns(4)
    sample_queries = [
        "How does a company choose between RAG, fine-tuning, and PEFT?",
        "What are the core trade-offs between LoRA, QLoRA, and Full Fine-Tuning?",
        "Explain the role of Supervised Fine-Tuning and DPO.",
        "What are the major entity communities identified in this guide?"
    ]
    
    clicked_query = None
    for i, col in enumerate(p_cols):
        with col:
            if st.button(sample_queries[i], key=f"chat_chip_{i}", use_container_width=True):
                clicked_query = sample_queries[i]
    
    st.divider()

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "mode" in msg and msg["mode"] != "System":
                pill_class = "pill-cyan" if "Vector" in msg["mode"] else "pill-indigo"
                st.markdown(f'<span class="pill {pill_class}">{msg["mode"]}</span>', unsafe_allow_html=True)
            
            st.markdown(msg["content"])
            
            if msg.get("context"):
                with st.expander("🔍 Inspect Retrieved Grounding Context"):
                    st.markdown(f'<div class="context-terminal">{html.escape(msg["context"])}</div>', unsafe_allow_html=True)
            
            if msg.get("comparison"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="compare-card">
                        <div class="compare-header"><span class="pill pill-indigo">🌐 GRAPHRAG DRIFT</span></div>
                    """, unsafe_allow_html=True)
                    st.markdown(msg["comparison"]["graph_answer"])
                    with st.expander("Graph Context"):
                        st.markdown(f'<div class="context-terminal">{html.escape(msg["comparison"]["graph_context"])}</div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div class="compare-card">
                        <div class="compare-header"><span class="pill pill-cyan">📚 CHROMADB VECTOR RAG</span></div>
                    """, unsafe_allow_html=True)
                    st.markdown(msg["comparison"]["vector_answer"])
                    with st.expander("Vector Context"):
                        st.markdown(f'<div class="context-terminal">{html.escape(msg["comparison"]["vector_context"])}</div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    # Chat Input
    user_input = st.chat_input("Ask a question about Fine-Tuning, PEFT, RAG, or Graph Communities...") or clicked_query

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        llm = get_llm(temperature=temp)
        with st.chat_message("assistant"):
            with st.spinner(f"Reasoning with {search_mode}..."):
                try:
                    if "Global" in search_mode:
                        ans, ctx = query_graphrag_global(user_input, community_df, llm)
                        st.markdown('<span class="pill pill-indigo">GraphRAG: Global</span>', unsafe_allow_html=True)
                        st.markdown(ans)
                        with st.expander("🔍 Inspect Retrieved Grounding Context"):
                            st.markdown(f'<div class="context-terminal">{html.escape(ctx)}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "mode": "GraphRAG: Global", "context": ctx})
                    
                    elif "Local" in search_mode:
                        ans, ctx = query_graphrag_local(user_input, entities_df, relationships_df, llm)
                        st.markdown('<span class="pill pill-indigo">GraphRAG: Local</span>', unsafe_allow_html=True)
                        st.markdown(ans)
                        with st.expander("🔍 Inspect Retrieved Grounding Context"):
                            st.markdown(f'<div class="context-terminal">{html.escape(ctx)}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "mode": "GraphRAG: Local", "context": ctx})
                    
                    elif "DRIFT" in search_mode:
                        ans, ctx = query_graphrag_drift(user_input, community_df, relationships_df, llm)
                        st.markdown('<span class="pill pill-indigo">GraphRAG: DRIFT</span>', unsafe_allow_html=True)
                        st.markdown(ans)
                        with st.expander("🔍 Inspect Retrieved Grounding Context"):
                            st.markdown(f'<div class="context-terminal">{html.escape(ctx)}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "mode": "GraphRAG: DRIFT", "context": ctx})
                    
                    elif "Vector RAG" in search_mode:
                        ans, ctx = query_chroma_rag(user_input, paper_collection, llm, num_results=top_k_chunks)
                        st.markdown('<span class="pill pill-cyan">ChromaDB: Vector RAG</span>', unsafe_allow_html=True)
                        st.markdown(ans)
                        with st.expander("🔍 Inspect Retrieved Grounding Context"):
                            st.markdown(f'<div class="context-terminal">{html.escape(ctx)}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": ans, "mode": "ChromaDB: Vector RAG", "context": ctx})
                    
                    else: # Side-by-Side Comparison
                        g_ans, g_ctx = query_graphrag_drift(user_input, community_df, relationships_df, llm)
                        v_ans, v_ctx = query_chroma_rag(user_input, paper_collection, llm, num_results=top_k_chunks)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("""
                            <div class="compare-card">
                                <div class="compare-header"><span class="pill pill-indigo">🌐 GRAPHRAG DRIFT</span></div>
                            """, unsafe_allow_html=True)
                            st.markdown(g_ans)
                            with st.expander("Graph Context"):
                                st.markdown(f'<div class="context-terminal">{html.escape(g_ctx)}</div>', unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown("""
                            <div class="compare-card">
                                <div class="compare-header"><span class="pill pill-cyan">📚 CHROMADB VECTOR RAG</span></div>
                            """, unsafe_allow_html=True)
                            st.markdown(v_ans)
                            with st.expander("Vector Context"):
                                st.markdown(f'<div class="context-terminal">{html.escape(v_ctx)}</div>', unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "⚔️ **Side-by-Side Comparison Complete**",
                            "mode": "Comparison",
                            "comparison": {
                                "graph_answer": g_ans,
                                "graph_context": g_ctx,
                                "vector_answer": v_ans,
                                "vector_context": v_ctx
                            }
                        })
                except Exception as e:
                    st.error(f"Inference Error: {str(e)}")

# =============================================================================
# TAB 3: Interactive Knowledge Graph
# =============================================================================
with tab_graph:
    st.markdown("### 🕸️ Interactive Knowledge Graph Visualization")
    st.markdown("Full physics simulation. Drag nodes, zoom in/out, and explore entity relationships clustered by Leiden communities.")
    
    html_file = Path("./notebook/interactive_graph.html")
    if html_file.exists():
        raw_html = html_file.read_text(encoding="utf-8")
        st.components.v1.html(raw_html, height=650, scrolling=True)
    else:
        st.info("Interactive graph not found. Go to **⚡ On-Demand Builder** tab and build your graph.")

# =============================================================================
# TAB 4: GraphRAG Parquet Explorer
# =============================================================================
with tab_parquet:
    st.markdown("### 📊 GraphRAG Parquet Schema & Table Viewer")
    st.markdown("Inspect and search the structured knowledge tables generated by the pipeline.")
    
    sub1, sub2, sub3, sub4 = st.tabs([
        "🏷️ Entities",
        "🔗 Relationships",
        "🌐 Nodes",
        "📑 Community Reports"
    ])
    
    with sub1:
        st.markdown(f"**Total Entities:** `{len(entities_df)}`")
        f_ent = st.text_input("Filter entities:", key="fe_input")
        filt_e = entities_df[entities_df['title'].str.contains(f_ent, case=False, na=False) | entities_df['description'].str.contains(f_ent, case=False, na=False)] if f_ent else entities_df
        st.dataframe(filt_e, use_container_width=True)
        
    with sub2:
        st.markdown(f"**Total Relationships:** `{len(relationships_df)}`")
        f_rel = st.text_input("Filter relationships:", key="fr_input")
        filt_r = relationships_df[relationships_df['source'].str.contains(f_rel, case=False, na=False) | relationships_df['target'].str.contains(f_rel, case=False, na=False) | relationships_df['description'].str.contains(f_rel, case=False, na=False)] if f_rel else relationships_df
        st.dataframe(filt_r, use_container_width=True)
        
    with sub3:
        st.markdown(f"**Total Nodes:** `{len(nodes_df)}`")
        st.dataframe(nodes_df, use_container_width=True)
        
    with sub4:
        st.markdown(f"**Total Community Reports:** `{len(community_df)}`")
        st.dataframe(community_df[['human_readable_id', 'community', 'title', 'summary', 'rank']], use_container_width=True)
        for _, r in community_df.iterrows():
            with st.expander(f"📑 {r['title']} (Rank: {r['rank']})"):
                st.markdown(r['full_content'])

# =============================================================================
# TAB 5: ChromaDB Store
# =============================================================================
with tab_chroma:
    st.markdown("### 🗄️ ChromaDB Dense Vector Store")
    st.markdown("Inspect embedded chunks and execute live vector semantic queries with distance metrics.")
    
    st.info(f"Connected to ChromaDB: **`paper_collection`** | Total Chunks: **{paper_collection.count()}**")
    
    test_q = st.text_input("Vector Search Query:", value="LoRA and parameter efficient fine-tuning techniques")
    k_num = st.slider("Retrieve Top-N Chunks", min_value=1, max_value=10, value=3)
    
    if st.button("Run Vector Similarity Search", use_container_width=True):
        res = paper_collection.query(query_texts=[test_q], n_results=k_num)
        docs = res.get("documents", [[]])[0]
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0] if "distances" in res and res["distances"] else [None] * len(docs)
        
        for idx, (doc_id, doc, dist) in enumerate(zip(ids, docs, distances), 1):
            dist_badge = f'<span class="pill pill-cyan">Distance: {round(dist, 4)}</span>' if dist is not None else ""
            st.markdown(f"#### 📄 Result #{idx}: Chunk `{doc_id}` {dist_badge}", unsafe_allow_html=True)
            st.markdown(f'<div class="context-terminal">{html.escape(doc)}</div>', unsafe_allow_html=True)
