import os
import html
from pathlib import Path
from typing import Dict, Any, Tuple, List

import streamlit as st
import pandas as pd
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
import chromadb
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GraphRAG & Vector RAG Explorer",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")

# Custom CSS for polished, modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4D96FF, #9D4EDD, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background-color: #1e1e2f;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #2d2d48;
        text-align: center;
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4D96FF;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-graph {
        background-color: rgba(77, 150, 255, 0.2);
        color: #4D96FF;
        border: 1px solid #4D96FF;
    }
    .badge-vector {
        background-color: rgba(157, 78, 221, 0.2);
        color: #9D4EDD;
        border: 1px solid #9D4EDD;
    }
    .chat-context-box {
        background-color: #161626;
        border-radius: 8px;
        padding: 12px;
        font-family: monospace;
        font-size: 0.82rem;
        max-height: 250px;
        overflow-y: auto;
        border: 1px solid #2d2d48;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Cached Data & Client Loaders
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
# 3. Retrieval & QA Engines
# -----------------------------------------------------------------------------
def query_graphrag_local(query: str, entities: pd.DataFrame, rels: pd.DataFrame, llm: ChatOpenAI, top_k: int = 15) -> Tuple[str, str]:
    """Execute GraphRAG Local Search focused on entity neighborhoods and edges."""
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
    """Execute GraphRAG Global Search across community summaries."""
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
    """Execute GraphRAG DRIFT Search (Dynamic Reasoning and Inference with Flexible Traversal)."""
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
    """Execute dense vector RAG over ChromaDB."""
    results = collection.query(query_texts=[query], n_results=num_results)
    docs = results.get("documents", [[]])[0]
    
    context_lines = []
    for idx, doc in enumerate(docs, 1):
        context_lines.append(f"--- Chunk #{idx} ---\n{doc.strip()}\n")
    
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
# 4. Sidebar Controls & Info
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    
    # Mode selector
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
        help="Select the retrieval paradigm to use."
    )
    
    st.divider()
    st.markdown("### 🎛️ Parameters")
    temp = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    top_k_chunks = st.slider("Vector Chunks (Top-K)", min_value=1, max_value=10, value=4)
    
    st.divider()
    st.markdown("### 📊 Shared Database Stats")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Entities", len(entities_df))
        st.metric("Communities", len(community_df))
    with col_b:
        st.metric("Relations", len(relationships_df))
        st.metric("Vector Chunks", paper_collection.count())
    
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # API key check
    if not api_key:
        st.warning("⚠️ No `OPENAI_API_KEY` found in `.env`!")
    else:
        st.success("✅ OpenAI API Connected (`gpt-4o-mini`)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I am your **GraphRAG & Vector RAG AI Assistant**.\n\nI am connected to your shared knowledge base built from *The Ultimate Guide to Fine-Tuning LLMs*. You can ask me questions using **GraphRAG Global/Local/DRIFT Search**, compare with **Vector RAG**, or inspect the **Knowledge Graph**!",
            "mode": "System",
            "context": None
        }
    ]

# -----------------------------------------------------------------------------
# 5. Header and Tabs
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🕸️ GraphRAG & Vector RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-modal knowledge extraction, community graph reasoning, and dense vector search explorer</div>', unsafe_allow_html=True)

tab_chat, tab_graph, tab_data, tab_vector = st.tabs([
    "💬 AI Chatbot",
    "🕸️ Interactive Graph",
    "📊 GraphRAG Parquet Explorer",
    "🗄️ ChromaDB Store"
])

# =============================================================================
# TAB 1: Chatbot Interface
# =============================================================================
with tab_chat:
    # Example prompt chips
    st.markdown("**💡 Example Queries:**")
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
            if st.button(sample_queries[i], key=f"sample_{i}", use_container_width=True):
                clicked_query = sample_queries[i]
    
    st.divider()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "mode" in msg and msg["mode"] != "System":
                badge_class = "badge-vector" if "Vector" in msg["mode"] else "badge-graph"
                st.markdown(f'<span class="badge {badge_class}">{msg["mode"]}</span>', unsafe_allow_html=True)
            
            st.markdown(msg["content"])
            
            if msg.get("context"):
                with st.expander("🔍 Inspect Retrieved Grounding Context"):
                    st.markdown(f'<div class="chat-context-box">{html.escape(msg["context"])}</div>', unsafe_allow_html=True)
            
            if msg.get("comparison"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🌐 GraphRAG DRIFT Answer")
                    st.markdown(msg["comparison"]["graph_answer"])
                    with st.expander("Graph Context"):
                        st.markdown(f'<div class="chat-context-box">{html.escape(msg["comparison"]["graph_context"])}</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown("#### 📚 ChromaDB Vector RAG Answer")
                    st.markdown(msg["comparison"]["vector_answer"])
                    with st.expander("Vector Context"):
                        st.markdown(f'<div class="chat-context-box">{html.escape(msg["comparison"]["vector_context"])}</div>', unsafe_allow_html=True)

    # User Input
    user_input = st.chat_input("Ask a question about Fine-Tuning, PEFT, RAG, or Graph Communities...") or clicked_query

    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Assistant generation
        llm = get_llm(temperature=temp)
        with st.chat_message("assistant"):
            with st.spinner(f"Generating answer using {search_mode}..."):
                try:
                    if "Global" in search_mode:
                        ans, ctx = query_graphrag_global(user_input, community_df, llm)
                        st.markdown('<span class="badge badge-graph">GraphRAG: Global</span>', unsafe_allow_html=True)
                        st.markdown(ans)
                        with st.expander("🔍 Inspect Retrieved Grounding Context"):
                            st.markdown(f'<div class="chat-context-box">{html.escape(ctx)}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ans,
                            "mode": "GraphRAG: Global",
                            "context": ctx
                        })
                    
                    elif "Local" in search_mode:
                        ans, ctx = query_graphrag_local(user_input, entities_df, relationships_df, llm)
                        st.markdown('<span class="badge badge-graph">GraphRAG: Local</span>', unsafe_allow_html=True)
                        st.markdown(ans)
                        with st.expander("🔍 Inspect Retrieved Grounding Context"):
                            st.markdown(f'<div class="chat-context-box">{html.escape(ctx)}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ans,
                            "mode": "GraphRAG: Local",
                            "context": ctx
                        })
                    
                    elif "DRIFT" in search_mode:
                        ans, ctx = query_graphrag_drift(user_input, community_df, relationships_df, llm)
                        st.markdown('<span class="badge badge-graph">GraphRAG: DRIFT</span>', unsafe_allow_html=True)
                        st.markdown(ans)
                        with st.expander("🔍 Inspect Retrieved Grounding Context"):
                            st.markdown(f'<div class="chat-context-box">{html.escape(ctx)}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ans,
                            "mode": "GraphRAG: DRIFT",
                            "context": ctx
                        })
                    
                    elif "Vector RAG" in search_mode:
                        ans, ctx = query_chroma_rag(user_input, paper_collection, llm, num_results=top_k_chunks)
                        st.markdown('<span class="badge badge-vector">ChromaDB: Vector RAG</span>', unsafe_allow_html=True)
                        st.markdown(ans)
                        with st.expander("🔍 Inspect Retrieved Grounding Context"):
                            st.markdown(f'<div class="chat-context-box">{html.escape(ctx)}</div>', unsafe_allow_html=True)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ans,
                            "mode": "ChromaDB: Vector RAG",
                            "context": ctx
                        })
                    
                    else: # Side-by-Side Comparison
                        g_ans, g_ctx = query_graphrag_drift(user_input, community_df, relationships_df, llm)
                        v_ans, v_ctx = query_chroma_rag(user_input, paper_collection, llm, num_results=top_k_chunks)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### 🌐 GraphRAG DRIFT Answer")
                            st.markdown(g_ans)
                            with st.expander("Graph Context"):
                                st.markdown(f'<div class="chat-context-box">{html.escape(g_ctx)}</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown("#### 📚 ChromaDB Vector RAG Answer")
                            st.markdown(v_ans)
                            with st.expander("Vector Context"):
                                st.markdown(f'<div class="chat-context-box">{html.escape(v_ctx)}</div>', unsafe_allow_html=True)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "⚔️ **Side-by-Side Comparison Generated**",
                            "mode": "Comparison",
                            "comparison": {
                                "graph_answer": g_ans,
                                "graph_context": g_ctx,
                                "vector_answer": v_ans,
                                "vector_context": v_ctx
                            }
                        })
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")

# =============================================================================
# TAB 2: Interactive Knowledge Graph
# =============================================================================
with tab_graph:
    st.markdown("### 🕸️ Interactive GraphRAG Knowledge Graph")
    st.markdown("Explore nodes, drag entities, zoom in/out, and inspect relationships across community clusters.")
    
    html_file = Path("./notebook/interactive_graph.html")
    if html_file.exists():
        raw_html = html_file.read_text(encoding="utf-8")
        st.components.v1.html(raw_html, height=650, scrolling=True)
    else:
        st.info("Interactive HTML not found at `./notebook/interactive_graph.html`. Run `graph_examples.ipynb` Cell 25 to generate it.")
    
    # Graph summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="stat-card"><div class="stat-val">' + str(len(nodes_df)) + '</div><div class="stat-label">Total Graph Nodes</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="stat-card"><div class="stat-val">' + str(len(relationships_df)) + '</div><div class="stat-label">Total Relationships</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="stat-card"><div class="stat-val">' + str(len(community_df)) + '</div><div class="stat-label">Leiden Communities</div></div>', unsafe_allow_html=True)
    with m4:
        avg_deg = round(nodes_df['degree'].mean(), 2) if not nodes_df.empty and 'degree' in nodes_df.columns else 0
        st.markdown('<div class="stat-card"><div class="stat-val">' + str(avg_deg) + '</div><div class="stat-label">Avg Node Degree</div></div>', unsafe_allow_html=True)

# =============================================================================
# TAB 3: GraphRAG Parquet Explorer
# =============================================================================
with tab_data:
    st.markdown("### 📊 GraphRAG Parquet Table Viewer")
    st.markdown("Inspect and search the 4 core Parquet tables generated by the GraphRAG pipeline.")
    
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "🏷️ Entities (create_final_entities)",
        "🔗 Relationships (create_final_relationships)",
        "🌐 Nodes (create_final_nodes)",
        "📑 Community Reports (create_final_community_reports)"
    ])
    
    with sub_tab1:
        st.markdown(f"**Total Entities:** {len(entities_df)}")
        e_filter = st.text_input("Filter entities by title or description:", key="filter_entities")
        filtered_e = entities_df[entities_df['title'].str.contains(e_filter, case=False, na=False) | entities_df['description'].str.contains(e_filter, case=False, na=False)] if e_filter else entities_df
        st.dataframe(filtered_e, use_container_width=True)
        
    with sub_tab2:
        st.markdown(f"**Total Relationships:** {len(relationships_df)}")
        r_filter = st.text_input("Filter relationships by source, target, or description:", key="filter_rels")
        filtered_r = relationships_df[relationships_df['source'].str.contains(r_filter, case=False, na=False) | relationships_df['target'].str.contains(r_filter, case=False, na=False) | relationships_df['description'].str.contains(r_filter, case=False, na=False)] if r_filter else relationships_df
        st.dataframe(filtered_r, use_container_width=True)
        
    with sub_tab3:
        st.markdown(f"**Total Nodes with Community IDs:** {len(nodes_df)}")
        st.dataframe(nodes_df, use_container_width=True)
        
    with sub_tab4:
        st.markdown(f"**Total Community Reports:** {len(community_df)}")
        st.dataframe(community_df[['human_readable_id', 'community', 'title', 'summary', 'rank']], use_container_width=True)
        for _, row in community_df.iterrows():
            with st.expander(f"📑 {row['title']} (Rank: {row['rank']})"):
                st.markdown(row['full_content'])

# =============================================================================
# TAB 4: ChromaDB Store
# =============================================================================
with tab_vector:
    st.markdown("### 🗄️ ChromaDB Vector Collection Inspector")
    st.markdown("Inspect raw document chunks and run standalone vector similarity queries.")
    
    st.info(f"Connected to ChromaDB at `./notebook/chromadb` | Collection: **`paper_collection`** | Total Chunks: **{paper_collection.count()}**")
    
    test_query = st.text_input("Test Vector Semantic Retrieval:", value="LoRA and parameter efficient fine-tuning techniques")
    k_val = st.slider("Retrieve Top-N Chunks:", min_value=1, max_value=10, value=3)
    
    if st.button("Run Vector Search", use_container_width=True):
        res = paper_collection.query(query_texts=[test_query], n_results=k_val)
        docs = res.get("documents", [[]])[0]
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0] if "distances" in res and res["distances"] else [None] * len(docs)
        
        for idx, (doc_id, doc, dist) in enumerate(zip(ids, docs, distances), 1):
            dist_str = f" | Distance: `{round(dist, 4)}`" if dist is not None else ""
            with st.expander(f"📄 Result #{idx}: Chunk ID `{doc_id}`{dist_str}", expanded=True):
                st.write(doc)
