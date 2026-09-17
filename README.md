# 🏛️ Enterprise Knowledge Intelligence Platform (GraphRAG)

[![Version](https://img.shields.io/badge/Version-2.4.1-blue.svg?style=flat-square)](#)
[![Compliance](https://img.shields.io/badge/Compliance-SOC2_Audit_Ready-emerald.svg?style=flat-square)](#)
[![Design](https://img.shields.io/badge/Design-Enterprise_Minimalist-indigo.svg?style=flat-square)](#)
[![Python](https://img.shields.io/badge/Python-3.13+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](#)

An enterprise-grade autonomous knowledge extraction, community graph reasoning, and hybrid vector intelligence platform built on **Microsoft GraphRAG**, **ChromaDB**, and **Enterprise Document Intelligence**.

---

# 🧠 Conceptual Foundation: Knowledge Graph RAG

<p align="center">
  <img src="./media/graph_start.png" width="600" alt="Knowledge Graph Architecture" />
</p>
<p align="center">
  <em><a href="https://deeppavlov.ai/research/tpost/bn15u1y4v1-improving-knowledge-graph-completion-wit">Improving Knowledge Graph Completion with Generative LM and neighbors</a></em>
</p>

In the evolving landscape of AI and information retrieval, **Knowledge Graphs** have emerged as a powerful way to represent complex, interconnected information. A knowledge graph is a structured data model or topology used to represent and operate on data. They store interlinked descriptions of entities — objects, events, situations, or abstract concepts — while encoding the semantics and relationships connecting them. *(Source: Wikipedia)*

### Mirroring Human Cognition in Data

What makes knowledge graphs particularly powerful is their ability to **mirror human cognition in data**. Rather than storing isolated text chunks, they explicitly map relationships between concepts through both semantic and relational connections. This approach closely parallels how human memory and reasoning operate — not as flat lists of facts, but as an interconnected web of associations.

<p align="center">
  <img src="./media/coffee_graph_ex.png" width="400" alt="Coffee Concept Graph Example" />
</p>

Looking at a concept like **"coffee,"** we don't just perceive it as a beverage; we automatically associate it with coffee beans, brewing methods, caffeine, morning routines, roasting profiles, and social spaces. Knowledge graphs capture these natural associations in a structured, traversable format.

---

### The Limitation of Traditional Vector RAG

Traditional RAG systems, while effective at semantic similarity-based retrieval across isolated text chunks, often struggle with questions requiring:
1. **Broader Conceptual Synthesis**: "What are the overarching themes and trade-offs across this document collection?"
2. **Multi-Hop Traversal**: Connecting Concept A to Concept C through an intermediate Concept B across separate documents.
3. **Global Dataset Understanding**: Identifying holistic patterns that no single text chunk explicitly states.

<p align="center">
  <img src="./media/table_comp.png" width="600" alt="Comparison: Knowledge Graphs vs Flat Vectors" />
</p>
<p align="center">
  <em><a href="https://zilliz.com/learn/what-is-knowledge-graph">What is a Knowledge Graph (KG)?</a></em>
</p>

Knowledge Graph RAG addresses this limitation by introducing a **structured, hierarchical approach to information organization and retrieval**:
- **Graph Traversal**: Explores explicit relationship pathways between entities.
- **Multi-Hop Reasoning**: Uncovers non-obvious connections across diverse sources.
- **Explainable Provenance**: Every generated response links directly to verified entity nodes and source documents.

---

## ⚙️ How GraphRAG Works Behind the Scenes

While early knowledge graphs required labor-intensive manual curation by domain experts, modern LLMs automate the end-to-end extraction and graph indexing pipeline:

<p align="center">
  <img src="./media/graph_building.png" width="800" alt="Graph Building Pipeline" />
</p>

### 1. Document Chunking & Information Extraction
Raw documents (PDF, TXT, MD) are ingested, cleaned, and split into overlapping text chunks. An LLM performs zero-shot or few-shot extraction to identify:
- **Entities**: Named concepts, technologies, organizations, and methods (`./media/entities.png`).
- **Relationships**: Typed, directed connections between entities with descriptive summaries and weight metrics (`./media/relationship.png`).
- **Claims / Covariates**: Factual statements grounding each relationship back to source text units.

<p align="center">
  <img src="./media/entities.png" width="450" alt="Entities" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./media/relationship.png" width="380" alt="Relationships" />
</p>

### 2. Hierarchical Community Detection (Leiden / Louvain Algorithm)
Once the knowledge network is formed, graph community detection algorithms partition the graph into hierarchical clusters (communities of related concepts). 

<p align="center">
  <img src="./media/leidan.png" width="550" alt="Leiden Community Detection" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./media/communities.png" width="380" alt="Hierarchical Communities" />
</p>

- High-level communities represent macro themes (e.g., "LLM Fine-Tuning Strategies").
- Lower-level sub-communities capture granular technical specializations (e.g., "LoRA Hyperparameter Tuning", "DPO vs RLHF").
- An LLM generates pre-computed **Community Reports** summarizing the key insights, findings, and risks for each cluster.

---

## 🔍 Retrieval Paradigms Compared

This platform implements Microsoft GraphRAG's primary search paradigms alongside standard dense vector search:

<p align="center">
  <img src="./media/kg_retrieval.png" width="600" alt="Knowledge Graph Retrieval Paradigms" />
</p>

| Paradigm | Visual Architecture | Best Suited For | How It Works |
| :--- | :--- | :--- | :--- |
| **Local Search** | <img src="./media/local_search.png" width="300" /> | Specific entities, direct relationships, targeted facts | Identifies seed entities via vector search, traverses 1-hop & 2-hop neighbors, subgraphs, and associated text chunks. |
| **Global Search** | <img src="./media/global_search.png" width="300" /> | Broad, thematic, holistic queries across entire corpus | Synthesizes hierarchical **Community Reports** using map-reduce summarization without querying raw text chunks. |
| **DRIFT Search** | <img src="./media/drift_search.png" width="300" /> | Complex multi-faceted inquiries | Dual-stage: starts with broad community reports, branches into entity graphs, then consolidates insights. |
| **Standard Vector RAG** | <img src="./media/basic_retrieval.png" width="300" /> | Quick passage lookup, raw text quotes | Embeds query, retrieves top-$k$ nearest chunk embeddings from ChromaDB. |

---

## 🏗️ Enterprise Sub-Tree Architecture

The platform is engineered with a clean, decoupled **sub-tree architecture** ensuring separation of concerns between core AI engines, persistent storage vaults, automation agents, and the executive web portal:

```
GraphRAG-Breakdown/
├── app.py                          # Enterprise Web Application (Executive UI)
├── CHANGELOG.md                    # Auto-generated audit log by Change Management Agent
├── VERSION                         # Semantic Versioning (2.4.1)
├── requirements.txt                # Production dependencies
│
├── core/                           # Sub-Tree: Core Enterprise Services
│   ├── __init__.py                 # Clean package exports
│   ├── pipeline.py                 # GraphRAG engine, Louvain clustering & ChromaDB sync
│   ├── vault.py                    # Document Retention Vault & Citation Attribution Builder
│   └── change_manager.py           # Change Management Agent (QA, Versioning, Git sync)
│
├── vault/                          # Sub-Tree: Managed Document Store
│   ├── documents/                  # Persisted uploaded source files (PDF, TXT, MD)
│   └── catalog.json                # Document registry with SHA-256 hashes & chunk mappings
│
├── scripts/                        # Sub-Tree: Automation & Governance
│   ├── qa_check.py                 # Automated test suite (syntax, parquet tables, ChromaDB)
│   └── sync_and_push.py            # Pre-push automation CLI with Change Management integration
│
├── ragtest/output/                 # Parquet datasets (entities, relationships, nodes, reports)
├── notebook/                       # Interactive PyVis physics network & ChromaDB store
└── media/                          # Conceptual diagrams, architectural graphics & visual assets
```

---

## 🌟 Production Capabilities

1. **📑 Enterprise Document Retention Vault (`core/vault.py`)**:
   - Upload and permanently retain enterprise documents (PDF, TXT, MD).
   - SHA-256 cryptographic verification prevents duplicate ingestion.
   - Generates inline verified citations `[1]`, `[2]` linking answers directly to specific documents and page numbers.

2. **🤖 Multi-Strategy Search & Comparative Audit**:
   - **🌐 Strategic Synthesis (Global)**: Hierarchical summaries across all strategic knowledge domains.
   - **🔍 Targeted Lookup (Local)**: Entity-centric traversal with verified source proof.
   - **🌀 Deep-Dive Cross-Functional (DRIFT)**: Multi-hop reasoning connecting macro strategy with operational details.
   - **📚 Standard Document Search**: Dense vector similarity search across source passages.
   - **⚖️ Side-by-Side Comparative Audit**: Real-time evaluation comparing Graph Intelligence against flat vector retrieval.

3. **🕸️ Knowledge Concept Map**:
   - Interactive physics-based PyVis network with Louvain community coloring, node degree sizing, and real-time concept exploration.

4. **📊 Enterprise Data Catalog**:
   - Live inspection and filtering of extracted entities, relationships, graph nodes, and community reports.

5. **🛡️ Change Management Agent (`core/change_manager.py`)**:
   - Automated semantic versioning (`VERSION`), release audit log (`CHANGELOG.md`), automated test validation (`scripts/qa_check.py`), and pre-push git synchronization.

---

## 🚀 Quickstart

### 1. Prerequisites & Installation
```bash
# Clone the repository
git clone https://github.com/himanshusin/graph_rag.git
cd graph_rag

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Web Application
```bash
streamlit run app.py
```
Access the application at **[http://localhost:8501](http://localhost:8501)**.

---

## 🛡️ Change Management & Pre-Push Workflow

Before pushing changes to GitHub, execute the pre-push automation hook. This runs the automated QA test suite, increments the semantic version, compiles the changelog, and pushes cleanly:

```bash
# Automated Patch Bump (e.g., v2.3.1 -> v2.3.2)
python scripts/sync_and_push.py --bump patch --msg "feat: your change description"

# Or run QA checks standalone
python scripts/qa_check.py
```
