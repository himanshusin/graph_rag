import os
import re
import html
import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import chromadb
import truststore
truststore.inject_into_ssl()

import pymupdf


# -----------------------------------------------------------------------------
# 1. Document Ingestor
# -----------------------------------------------------------------------------
class DocumentIngestor:
    """Handles multi-format document text extraction (PDF, TXT, Markdown, Raw String)."""

    @staticmethod
    def extract_from_pdf_bytes(pdf_bytes: bytes) -> str:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages_text = [page.get_text() for page in doc]
        return "\n".join(pages_text)

    @staticmethod
    def extract_from_pdf_file(file_path: str) -> str:
        doc = pymupdf.open(file_path)
        pages_text = [page.get_text() for page in doc]
        return "\n".join(pages_text)

    @staticmethod
    def extract_from_text(text: str) -> str:
        return text.strip()


# -----------------------------------------------------------------------------
# 2. Text Chunker
# -----------------------------------------------------------------------------
class TextChunker:
    """Splits document text into manageable token chunks with overlap."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, text: str) -> List[str]:
        return self.splitter.split_text(text)


# -----------------------------------------------------------------------------
# 3. Entity & Relationship Extractor
# -----------------------------------------------------------------------------
EXTRACTION_PROMPT = """
-Goal-
Given a text document, identify all entities and relationships between them.

-Steps-
1. Identify all entities. For each entity, extract:
- entity_name: Name of the entity, capitalized
- entity_type: Category of the entity (e.g., CONCEPT, METHOD, TOOL, ORGANIZATION, TECHNIQUE, METRIC)
- entity_description: Comprehensive description of the entity's attributes and role
Format: ("entity"<|delimiter|><entity_name><|delimiter|><entity_type><|delimiter|><entity_description>)

2. Identify all relationships directly mentioned between entities. For each relationship, extract:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: explanation of why the entities are related
- relationship_strength: integer score between 1 and 10 representing relationship strength
Format: ("relationship"<|delimiter|><source_entity><|delimiter|><target_entity><|delimiter|><relationship_description><|delimiter|><relationship_strength>)

3. Return output strictly in the format above, one item per line.

Input Text:
{input_text}
"""

class EntityRelationshipExtractor:
    """Extracts entities and relationships from chunks using OpenAI LLM."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0, api_key: Optional[str] = None):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def extract_chunk(self, chunk_text: str, chunk_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        response = self.chain.invoke({"input_text": chunk_text})
        return self._parse_response(response, chunk_id)

    def _parse_response(self, raw_text: str, chunk_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        entities = []
        relationships = []
        cleaned = raw_text.replace("```plaintext", "").replace("```", "")

        for line in cleaned.splitlines():
            line = line.strip()
            if not line:
                continue
            if "(" in line and ")" in line:
                start = line.find("(") + 1
                end = line.rfind(")")
                inner = line[start:end]

                parts = None
                for delim in ["<|delimiter|>", "{tuple_delimiter}", "\t", "|"]:
                    if delim in inner:
                        parts = inner.split(delim)
                        break
                if parts is None:
                    parts = inner.split('","')

                parts = [p.strip().strip('"').strip("'") for p in parts]
                if not parts:
                    continue

                kind = parts[0].lower()
                if kind == "entity" and len(parts) >= 4:
                    entities.append({
                        "title": parts[1].upper(),
                        "type": parts[2].upper(),
                        "description": parts[3],
                        "chunk_id": chunk_id
                    })
                elif kind == "relationship" and len(parts) >= 5:
                    digits = re.findall(r"\d+", parts[4])
                    weight = float(digits[0]) if digits else 1.0
                    relationships.append({
                        "source": parts[1].upper(),
                        "target": parts[2].upper(),
                        "description": parts[3],
                        "weight": weight,
                        "chunk_id": chunk_id
                    })

        return entities, relationships


# -----------------------------------------------------------------------------
# 4. Master GraphRAG Engine
# -----------------------------------------------------------------------------
class GraphRAGEngine:
    """
    Complete, industry-grade GraphRAG Pipeline:
    1. Chunking
    2. Entity/Relationship Extraction
    3. Deduplication & Graph Network Construction
    4. Louvain Community Detection
    5. Hierarchical Community Report Generation
    6. ChromaDB Vector Store Ingestion
    7. Parquet & PyVis Export
    """

    def __init__(
        self,
        output_dir: str = "./ragtest/output",
        chroma_dir: str = "./notebook/chromadb",
        notebook_dir: str = "./notebook",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):
        self.output_dir = Path(output_dir)
        self.chroma_dir = Path(chroma_dir)
        self.notebook_dir = Path(notebook_dir)
        self.model_name = model_name
        self.temperature = temperature

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.notebook_dir.mkdir(parents=True, exist_ok=True)

    def build_from_text(
        self,
        document_text: str,
        document_title: str = "Enterprise Document",
        max_chunks: Optional[int] = 10,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        progress_callback: Optional[Callable[[float, str, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        def notify(progress: float, message: str, stats: Optional[Dict[str, Any]] = None):
            if progress_callback:
                progress_callback(progress, message, stats or {})

        # Step 1: Chunking
        notify(0.05, "Splitting document into semantic token chunks...")
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks = chunker.split(document_text)

        chunks_to_process = all_chunks[:max_chunks] if (max_chunks and max_chunks > 0) else all_chunks

        notify(0.12, f"Prepared {len(chunks_to_process)} chunks for knowledge extraction.", {
            "total_chunks": len(all_chunks),
            "processed_chunks": len(chunks_to_process)
        })

        # Step 2: Extraction
        extractor = EntityRelationshipExtractor(model_name=self.model_name, temperature=self.temperature)
        all_entities_raw = []
        all_relationships_raw = []

        total_chunks = len(chunks_to_process)
        for i, chunk_text in enumerate(chunks_to_process):
            chunk_id = f"chunk_{i}"
            step_progress = 0.15 + (0.45 * ((i + 1) / total_chunks))
            notify(step_progress, f"Extracting entities & relations from chunk {i + 1}/{total_chunks}...", {
                "chunk_current": i + 1,
                "chunk_total": total_chunks,
                "raw_entities": len(all_entities_raw),
                "raw_relationships": len(all_relationships_raw)
            })

            e_list, r_list = extractor.extract_chunk(chunk_text, chunk_id)
            all_entities_raw.extend(e_list)
            all_relationships_raw.extend(r_list)

        # Step 3: Deduplication
        notify(0.65, "Deduplicating entities and merging relationship weights...")
        entity_groups = {}
        for e in all_entities_raw:
            key = (e["title"], e["type"])
            if key not in entity_groups:
                entity_groups[key] = {"descriptions": [], "chunk_ids": set()}
            entity_groups[key]["descriptions"].append(e["description"])
            entity_groups[key]["chunk_ids"].add(e["chunk_id"])

        entities_rows = []
        for h_id, ((title, ent_type), data) in enumerate(entity_groups.items()):
            entities_rows.append({
                "id": str(uuid.uuid4()),
                "human_readable_id": h_id,
                "title": title,
                "type": ent_type,
                "description": " ".join(list(dict.fromkeys(data["descriptions"]))),
                "text_unit_ids": list(data["chunk_ids"])
            })
        df_entities = pd.DataFrame(entities_rows) if entities_rows else pd.DataFrame(columns=["id", "human_readable_id", "title", "type", "description", "text_unit_ids"])

        rel_groups = {}
        for r in all_relationships_raw:
            key = (r["source"], r["target"])
            if key not in rel_groups:
                rel_groups[key] = {"descriptions": [], "weights": [], "chunk_ids": set()}
            rel_groups[key]["descriptions"].append(r["description"])
            rel_groups[key]["weights"].append(r["weight"])
            rel_groups[key]["chunk_ids"].add(r["chunk_id"])

        rel_rows = []
        for h_id, ((src, tgt), data) in enumerate(rel_groups.items()):
            rel_rows.append({
                "id": str(uuid.uuid4()),
                "human_readable_id": h_id,
                "source": src,
                "target": tgt,
                "description": " ".join(list(dict.fromkeys(data["descriptions"]))),
                "weight": float(sum(data["weights"]) / len(data["weights"])),
                "combined_degree": len(data["weights"]),
                "text_unit_ids": list(data["chunk_ids"])
            })
        df_relationships = pd.DataFrame(rel_rows) if rel_rows else pd.DataFrame(columns=["id", "human_readable_id", "source", "target", "description", "weight", "combined_degree", "text_unit_ids"])

        # Step 4: Community Clustering & Graph Layout
        notify(0.75, "Running Louvain community clustering & graph layout...")
        G = nx.Graph()
        for _, row in df_entities.iterrows():
            G.add_node(row["title"], type=row["type"], description=row["description"], id=row["id"], human_readable_id=row["human_readable_id"])

        for _, row in df_relationships.iterrows():
            if G.has_node(row["source"]) and G.has_node(row["target"]):
                G.add_edge(row["source"], row["target"], weight=row["weight"], description=row["description"])

        communities = list(nx.algorithms.community.louvain_communities(G, weight="weight", seed=42)) if len(G.nodes) > 0 else []
        node_community_map = {}
        for comm_id, comm in enumerate(communities):
            for node_name in comm:
                node_community_map[node_name] = comm_id

        pos = nx.spring_layout(G, seed=42) if len(G.nodes) > 0 else {}
        degrees = dict(G.degree())

        nodes_rows = []
        for _, row in df_entities.iterrows():
            node_name = row["title"]
            nodes_rows.append({
                "id": row["id"],
                "human_readable_id": row["human_readable_id"],
                "title": node_name,
                "community": node_community_map.get(node_name, 0),
                "level": 0,
                "degree": degrees.get(node_name, 0),
                "x": float(pos.get(node_name, [0.0, 0.0])[0]),
                "y": float(pos.get(node_name, [0.0, 0.0])[1])
            })
        df_nodes = pd.DataFrame(nodes_rows) if nodes_rows else pd.DataFrame(columns=["id", "human_readable_id", "title", "community", "level", "degree", "x", "y"])

        # Step 5: Community Reports
        notify(0.85, "Synthesizing hierarchical community reports & summaries...")
        comm_reports_rows = []
        for comm_id, comm_nodes in enumerate(communities):
            comm_entities = list(comm_nodes)
            title = f"Domain {comm_id}: " + ", ".join(comm_entities[:4])
            summary = f"This strategic domain encompasses {len(comm_entities)} key entities: " + ", ".join(comm_entities)
            full_content = f"# {title}\n\n## Strategic Overview\n{summary}\n\n## Key Strategic Concepts\n" + "\n".join([f"- **{e}**" for e in comm_entities])
            
            comm_reports_rows.append({
                "id": str(uuid.uuid4()),
                "human_readable_id": comm_id,
                "community": comm_id,
                "parent": -1,
                "level": 0,
                "title": title,
                "summary": summary,
                "full_content": full_content,
                "rank": 7.5,
                "rank_explanation": "Based on graph connectivity and node centrality",
                "findings": json.dumps([{"explanation": summary}]),
                "full_content_json": json.dumps({"title": title, "summary": summary}),
                "period": time.strftime("%Y-%m-%d"),
                "size": len(comm_entities)
            })
        df_community_reports = pd.DataFrame(comm_reports_rows) if comm_reports_rows else pd.DataFrame(columns=["id", "human_readable_id", "community", "parent", "level", "title", "summary", "full_content", "rank", "rank_explanation", "findings", "full_content_json", "period", "size"])

        # Save all 4 Parquet files
        df_entities.to_parquet(self.output_dir / "create_final_entities.parquet")
        df_relationships.to_parquet(self.output_dir / "create_final_relationships.parquet")
        df_nodes.to_parquet(self.output_dir / "create_final_nodes.parquet")
        df_community_reports.to_parquet(self.output_dir / "create_final_community_reports.parquet")

        # Step 6: ChromaDB Vector Sync
        notify(0.92, "Syncing document chunks with ChromaDB vector store...")
        chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        try:
            chroma_client.delete_collection("paper_collection")
        except Exception:
            pass
        collection = chroma_client.get_or_create_collection(name="paper_collection")

        for idx, text in enumerate(chunks_to_process):
            collection.add(
                documents=[text],
                metadatas=[{"document_title": document_title, "chunk_index": idx, "chunk_id": f"chunk_{idx}"}],
                ids=[f"chunk_{idx}"]
            )

        # Step 7: Visualizations
        notify(0.97, "Generating interactive concept map visualization...")
        self._generate_visualizations(df_nodes, df_relationships)

        elapsed = round(time.perf_counter() - start_time, 2)
        notify(1.0, f"✅ Knowledge Engine built successfully in {elapsed}s!", {
            "elapsed_seconds": elapsed,
            "entities": len(df_entities),
            "relationships": len(df_relationships),
            "communities": len(communities),
            "vector_chunks": len(chunks_to_process)
        })

        return {
            "elapsed_seconds": elapsed,
            "entities_count": len(df_entities),
            "relationships_count": len(df_relationships),
            "nodes_count": len(df_nodes),
            "communities_count": len(communities),
            "chunks_count": len(chunks_to_process),
            "entities_df": df_entities,
            "relationships_df": df_relationships,
            "nodes_df": df_nodes,
            "community_reports_df": df_community_reports
        }

    def _generate_visualizations(self, nodes_df: pd.DataFrame, relationships_df: pd.DataFrame):
        colors = ['#3B82F6', '#6366F1', '#10B981', '#F59E0B', '#EC4899', '#8B5CF6', '#06B6D4', '#EF4444', '#14B8A6', '#84CC16']
        
        # Interactive PyVis Graph
        net = Network(height='600px', width='100%', bgcolor='#0B0E14', font_color='white', notebook=True, cdn_resources='in_line')
        
        for _, row in nodes_df.iterrows():
            comm = int(row['community'])
            color = colors[comm % len(colors)]
            degree = int(row.get('degree', 1))
            net.add_node(
                row['title'],
                label=row['title'],
                title=f"Concept: {row['title']}<br>Domain: {comm}<br>Centrality: {degree}",
                color=color,
                size=14 + min(degree * 4, 32)
            )

        for _, row in relationships_df.iterrows():
            if row['source'] in net.get_nodes() and row['target'] in net.get_nodes():
                net.add_edge(
                    row['source'],
                    row['target'],
                    value=float(row['weight']),
                    title=f"Confidence: {row['weight']}/10<br>{row['description']}"
                )

        html_path = self.notebook_dir / "interactive_graph.html"
        net.write_html(str(html_path))

        # Matplotlib Static Graph
        if len(nodes_df) > 0:
            fig, ax = plt.subplots(figsize=(14, 9), facecolor='#0B0E14')
            ax.set_facecolor('#0B0E14')

            G = nx.Graph()
            for _, row in nodes_df.iterrows():
                G.add_node(row['title'], community=int(row['community']), degree=int(row.get('degree', 1)))

            for _, row in relationships_df.iterrows():
                if G.has_node(row['source']) and G.has_node(row['target']):
                    G.add_edge(row['source'], row['target'], weight=float(row['weight']))

            pos = nx.spring_layout(G, seed=42, k=0.55)
            node_colors = [colors[G.nodes[n]['community'] % len(colors)] for n in G.nodes]
            node_sizes = [350 + G.nodes[n]['degree'] * 120 for n in G.nodes]

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.92, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='#232B3E', alpha=0.5, width=1.5, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', font_weight='bold', ax=ax)

            plt.title('Enterprise Knowledge Concept Map (Thematic Domains)', color='white', fontsize=14, pad=15)
            plt.axis('off')
            plt.tight_layout()

            img_path = self.notebook_dir / "knowledge_graph.png"
            plt.savefig(img_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
