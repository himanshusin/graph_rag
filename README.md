# 🏛️ GraphRAG Knowledge Workspace

[![Version](https://img.shields.io/badge/Version-2.4.1-blue.svg?style=flat-square)](#)
[![Design](https://img.shields.io/badge/Design-Analyst_Workspace-indigo.svg?style=flat-square)](#)
[![Python](https://img.shields.io/badge/Python-3.13+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](#)

Knowledge extraction, community graph reasoning and hybrid vector retrieval over a document vault, built on **GraphRAG** concepts, **ChromaDB** and **Streamlit**. Analysts ask a question, read the answer, and check where every claim came from.

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

## 🏗️ Architecture

Engines, presentation and screens are separated so the reasoning logic can be tested without Streamlit and the design tokens live in one place.

```
GraphRAG-Breakdown/
├── app.py                          # Entry point: sidebar, query-param routing
├── VERSION                         # Semantic versioning (2.4.1)
├── CHANGELOG.md                    # Generated from git history by the change agent
│
├── core/                           # Engines (no Streamlit imports)
│   ├── pipeline.py                 # Chunking, extraction, Louvain, LLM community reports, exports
│   ├── rag.py                      # Retrieval + generation for the five modes
│   ├── vault.py                    # Document retention, table extraction, index state
│   └── change_manager.py           # Versioning, changelog, health checks
│
├── ui/                             # Presentation
│   ├── theme.css                   # Design tokens and every component rule
│   ├── tokens.py                   # Token values, escaping and query-string helpers
│   ├── components.py               # HTML fragments (nav, evidence cards, tables, timeline)
│   └── data.py                     # Cached loaders for parquet, ChromaDB, vault, QA
│
├── screens/                        # One module per screen
│   ├── search.py                   # Thread, streaming answers, evidence rail, compare
│   ├── vault.py                    # Registry, ingest progress row, inspector
│   ├── concept_map.py              # PyVis network, community filter, node inspector
│   ├── catalog.py                  # Concepts, relationships, nodes, domain briefs
│   └── governance.py               # QA check list and changelog timeline
│
├── scripts/
│   ├── ui_qa_agent.py              # Design + functional QA suite (no LLM calls)
│   ├── app_smoke.py                # Drives the real app via Streamlit AppTest
│   ├── pipeline_e2e.py             # Live indexing + answering test (calls the API)
│   ├── qa_check.py                 # Health checks + QA suite
│   └── sync_and_push.py            # Pre-push orchestration
│
├── vault/                          # Retained documents, extracted tables, catalog.json
├── ragtest/output/                 # Parquet artifacts (entities, relationships, nodes, reports)
├── notebook/                       # Concept map HTML and the ChromaDB store
└── media/                          # Diagrams used in this README
```

---

## 🌟 Capabilities

1. **Document vault (`core/vault.py`)**
   - Retains PDF, TXT and MD uploads, deduplicated by SHA-256.
   - Extracts structured tables with per-row facts, so numeric questions can be answered from the table rather than from prose.
   - Tracks index state per document (`chunks_indexed` of `chunks_total`) and purges a document's vectors when it is deleted.

2. **Indexing pipeline (`core/pipeline.py`)**
   - Rebuilds the graph **cumulatively over the whole vault**, so several documents share one graph.
   - Extracts typed relationships (`extends`, `reduces`, `evaluated on`, …) alongside entities and weights.
   - Writes **LLM-authored community reports** — a real title, summary, findings and importance rating per community, not a template.
   - Namespaces chunk ids per document (`doc_abc::chunk_3`) and tags every vector with `document_id`, so each citation traces back to its source file.

3. **Five reasoning modes (`core/rag.py`)**
   - **Global synthesis** — ranks community reports by relevance, with importance as the tie-breaker.
   - **Local lookup** — scores concepts against the question, then pulls the relationships that touch them.
   - **DRIFT deep-dive** — a global pass picks the domains, a local pass drills into their concepts.
   - **Vector passages** — nearest passages from ChromaDB, dropping hits past the relevance threshold and saying so.
   - **Compare** — the graph and vector engines run concurrently, side by side, each with its own grounding score.

4. **Evidence that stays adjacent**
   - Every `[n]` in an answer links to a visible evidence card; numbers with no matching card stay plain text.
   - Cards open the screen that owns the source: passages open the Vault inspector, concepts the Concept map, domains the Catalog.
   - `grounded n%` is computed per answer as cited sentences ÷ total sentences.

5. **Governance (`core/change_manager.py`)**
   - Eleven measured health checks — syntax, each parquet, ChromaDB reachability, vector provenance, vault SHA-256 integrity, concept map, API key — each individually timed, with failures listed first.
   - Semantic versioning, changelog generated from git history, README badge sync.

---

## 🚀 Quickstart

### 1. Install
```bash
git clone https://github.com/himanshusin/graph_rag.git
cd graph_rag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)**.

### 4. Build the knowledge base
The graph starts empty. Open **Vault → Add document**, upload a PDF or text file, leave *Re-index graph now* on and set the chunk limit, then **Upload & retain**. Progress appears as a row in the registry. You can also attach a file directly in the Search composer.

Indexing rebuilds from every retained document, so adding a second file produces one graph spanning both.

---

## 🧪 Testing

```bash
# Design tokens, components, retrieval accuracy, pipeline, vault, governance.
# No LLM calls, so it is free and deterministic.
python scripts/ui_qa_agent.py

# Runs the real app for every screen via Streamlit AppTest and asserts
# what rendered, in both the empty and populated states.
python scripts/app_smoke.py

# Live: indexes a sample document end to end and answers in every mode,
# checking that citations resolve. Calls the OpenAI API.
python scripts/pipeline_e2e.py

# Health checks followed by the QA suite.
python scripts/qa_check.py
```

---

## 🛡️ Change management

```bash
# Health checks, QA suite, version bump, changelog, commit and push.
python scripts/sync_and_push.py --bump patch --msg "feat: your change description"

# Sync without pushing.
python scripts/sync_and_push.py --no-push
```

---

## 🎨 Design

The UI implements the handoff in `Graph RAG UI mockups/` — direction **1a** (sidebar, thread, evidence rail) as the shell, **1b**'s two-column block for Compare, and **1c**'s empty state. Tokens live in `ui/theme.css` as `:root` custom properties; edit them there rather than in Python. `scripts/ui_qa_agent.py` asserts the token values, typography, palette and the guide's acceptance checklist, so a drift from the handoff fails the suite.
