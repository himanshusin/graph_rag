"""GraphRAG indexing pipeline: chunking, extraction, clustering, community reports.

The engine builds the whole knowledge base from a set of documents so that the
graph, the community reports, the vector store and the concept map always
describe the same corpus. Rebuilding is cumulative over the vault rather than
per upload, which is what lets several documents share one graph.
"""

import os
import re
import json
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
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

CHROMA_COLLECTION = "paper_collection"

# Community palette from the developer guide §2, in order.
COMMUNITY_PALETTE = [
    "#3056D3", "#1F8A5B", "#B7791F", "#8A8A83",
    "#B03A2E", "#6B4FBB", "#2A8FA8", "#C25E9A",
]

ENTITY_COLUMNS = ["id", "human_readable_id", "title", "type", "description", "text_unit_ids"]
RELATIONSHIP_COLUMNS = [
    "id", "human_readable_id", "source", "target", "type", "description",
    "weight", "combined_degree", "text_unit_ids",
]
NODE_COLUMNS = ["id", "human_readable_id", "title", "community", "level", "degree", "x", "y"]
REPORT_COLUMNS = [
    "id", "human_readable_id", "community", "parent", "level", "title", "summary",
    "full_content", "rank", "rank_explanation", "findings", "full_content_json",
    "period", "size",
]


# -----------------------------------------------------------------------------
# 1. Document ingestor
# -----------------------------------------------------------------------------
class DocumentIngestor:
    """Multi-format text and structured table extraction (PDF, TXT, Markdown)."""

    @staticmethod
    def extract_from_pdf_bytes(pdf_bytes: bytes) -> str:
        """Extract pages and structured tables from PDF bytes using the PyMuPDF table finder."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages_content = []
        for p_idx, page in enumerate(doc):
            page_num = p_idx + 1
            blocks = []
            try:
                tabs = page.find_tables()
                if tabs and tabs.tables:
                    for tab in tabs.tables:
                        df = tab.to_pandas()
                        if not df.empty and len(df.columns) >= 2:
                            df = df.map(lambda x: str(x).replace("\n", " ").strip() if pd.notna(x) else "")
                            df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
                            md_tab = tab.to_markdown()
                            row_facts = []
                            for r_idx, row in df.iterrows():
                                items = [
                                    f"[{col}] = {val}" for col, val in row.items()
                                    if val and str(val).lower() not in ["none", "nan", ""]
                                ]
                                if items:
                                    row_facts.append(f"Row {r_idx + 1}: " + ", ".join(items))
                            blocks.append(
                                f"\n[STRUCTURED_TABLE Page {page_num}]\n{md_tab}\n"
                                f"**Structured Metric Facts:**\n" + "\n".join(row_facts) + "\n[/STRUCTURED_TABLE]\n"
                            )
            except Exception:
                pass
            text = page.get_text().strip()
            if text:
                blocks.append(text)
            pages_content.append("\n\n".join(blocks))
        return "\n\n--- PAGE BREAK ---\n\n".join(pages_content)

    @staticmethod
    def extract_from_pdf_file(file_path: str) -> str:
        with open(file_path, "rb") as f:
            return DocumentIngestor.extract_from_pdf_bytes(f.read())

    @staticmethod
    def extract_from_text(text: str) -> str:
        return text.strip()


# -----------------------------------------------------------------------------
# 2. Text chunker (table preserving)
# -----------------------------------------------------------------------------
class TextChunker:
    """Splits document text into token chunks with overlap while keeping tables intact."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, text: str) -> List[str]:
        if "[STRUCTURED_TABLE" in text:
            chunks = []
            parts = re.split(r'(\[STRUCTURED_TABLE.*?\[/STRUCTURED_TABLE\])', text, flags=re.DOTALL)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part.startswith("[STRUCTURED_TABLE"):
                    if len(part) < self.chunk_size * 4:
                        chunks.append(part)
                    else:
                        chunks.extend(self.splitter.split_text(part))
                else:
                    chunks.extend(self.splitter.split_text(part))
            return [c for c in chunks if c.strip()]
        return self.splitter.split_text(text)


# -----------------------------------------------------------------------------
# 3. Entity & relationship extractor
# -----------------------------------------------------------------------------
EXTRACTION_PROMPT = """
-Goal-
Given a text document, identify all entities, quantitative metrics, numerical values, and relationships between them.

-Steps-
1. Identify all entities. Categories include:
- CONCEPT, METHOD, TOOL, ORGANIZATION, TECHNIQUE
- METRIC: Key metrics or properties being evaluated (e.g., MMLU ACCURACY, LATENCY, REVENUE, PERPLEXITY, LOSS, THROUGHPUT, PARAMETERS)
- BENCHMARK: Datasets, testbeds, or evaluation settings (e.g., GSM8K, HUMANEVAL, GLUE, SQUAD, Q3 FY2024, 5-SHOT)
- QUANTITY: Exact quantitative figures, measurements, percentages, or values when significant (e.g., 89.2%, 14.5 MS, $4.2 BILLION, 7B PARAMETERS, 128K TOKENS)
- TIMEFRAME: Fiscal periods, dates, or quarters (e.g., Q3 2024, 2023, ANNUAL)

For each entity, extract:
- entity_name: Name of the entity, capitalized
- entity_type: Category (from the list above)
- entity_description: Comprehensive description including any exact numbers, units, scope, and technical context
Format: ("entity"<|delimiter|><entity_name><|delimiter|><entity_type><|delimiter|><entity_description>)

2. Identify all relationships directly mentioned between entities, paying special attention to quantitative, comparative, and evaluative relationships:
- source_entity: name of the source entity (e.g., "LLAMA-3-8B", "SALES DIVISION")
- target_entity: name of the target entity (e.g., "MMLU ACCURACY", "GSM8K", "89.2%", "REVENUE")
- relationship_type: one lower-case verb or short verb phrase naming the link, with no spaces around it (e.g., extends, reduces, measured by, alternative to, requires, improves, part of, evaluated on)
- relationship_description: explanation of why the entities are related. CRITICAL: ALWAYS preserve exact numbers, percentages, dollar amounts, performance scores, and comparative deltas in this description.
- relationship_strength: integer score between 1 and 10 representing relationship strength/confidence
Format: ("relationship"<|delimiter|><source_entity><|delimiter|><target_entity><|delimiter|><relationship_type><|delimiter|><relationship_description><|delimiter|><relationship_strength>)

3. Return output strictly in the format above, one item per line. Do not output anything else.

Input Text:
{input_text}
"""


class EntityRelationshipExtractor:
    """Extracts entities and relationships from chunks using an OpenAI chat model."""

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
                    # The typed form has 6 fields; tolerate the untyped 5-field form.
                    if len(parts) >= 6:
                        rel_type, description, strength = parts[3], parts[4], parts[5]
                    else:
                        rel_type, description, strength = "", parts[3], parts[4]
                    digits = re.findall(r"\d+", strength)
                    weight = float(digits[0]) if digits else 1.0
                    relationships.append({
                        "source": parts[1].upper(),
                        "target": parts[2].upper(),
                        "type": (rel_type or "related").strip().lower()[:40],
                        "description": description,
                        "weight": max(1.0, min(10.0, weight)),
                        "chunk_id": chunk_id
                    })

        return entities, relationships


# -----------------------------------------------------------------------------
# 4. Community report synthesizer
# -----------------------------------------------------------------------------
REPORT_PROMPT = """You are an analyst writing a report about one community of related
concepts extracted from a document corpus.

Write the report as strict JSON with exactly these keys:
- "title": a short specific noun phrase naming the theme of this community (no
  "Community N" prefixes, no numbering, 3-8 words)
- "summary": 2-4 sentences describing what this community covers, the concrete
  relationships inside it, and any exact numbers, percentages or units that
  appear in the source descriptions. Preserve figures verbatim.
- "findings": a list of 2-4 objects, each {{"summary": "<short claim>", "explanation": "<2-3 sentences with supporting detail and exact figures>"}}
- "rating": a number from 1 to 10 for how important this community is for
  understanding the corpus, where 10 means central and 1 means peripheral
- "rating_explanation": one sentence justifying the rating

Base every statement only on the entities and relationships given below. Do not
invent facts and do not add numbers that are not present.

Entities in this community:
{entities}

Relationships inside this community:
{relationships}

Return only the JSON object.
"""


class CommunityReportSynthesizer:
    """Generates community reports with an LLM, mirroring GraphRAG's report step."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2, api_key: Optional[str] = None):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.chain = ChatPromptTemplate.from_template(REPORT_PROMPT) | self.llm | StrOutputParser()

    def synthesize(
        self,
        entity_rows: List[Dict[str, Any]],
        relationship_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return title, summary, findings, rating and rating_explanation for a community."""
        entities_text = "\n".join(
            f"- {r['title']} ({r['type']}): {_clip(r.get('description', ''), 400)}"
            for r in entity_rows
        ) or "- (none)"
        relationships_text = "\n".join(
            f"- {r['source']} {r.get('type') or 'relates to'} {r['target']} "
            f"(strength {r['weight']:.0f}/10): {_clip(r.get('description', ''), 300)}"
            for r in relationship_rows
        ) or "- (no relationships inside this community)"

        raw = self.chain.invoke({"entities": entities_text, "relationships": relationships_text})
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> Dict[str, Any]:
        """Parse the model's JSON, tolerating code fences and surrounding prose."""
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
        data = json.loads(text)

        findings = []
        for item in data.get("findings") or []:
            if isinstance(item, dict):
                findings.append({
                    "summary": str(item.get("summary", "")).strip(),
                    "explanation": str(item.get("explanation", "")).strip(),
                })
            elif isinstance(item, str):
                findings.append({"summary": item.strip(), "explanation": ""})

        rating = data.get("rating")
        try:
            rating = round(float(rating), 1)
        except (TypeError, ValueError):
            rating = None
        if rating is not None:
            rating = max(1.0, min(10.0, rating))

        return {
            "title": str(data.get("title", "")).strip(),
            "summary": str(data.get("summary", "")).strip(),
            "findings": findings,
            "rating": rating,
            "rating_explanation": str(data.get("rating_explanation", "")).strip(),
        }


def _clip(text: Any, limit: int) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "…"


def _fallback_rank(entity_count: int, internal_edges: int, max_entities: int) -> Tuple[float, str]:
    """Derive an importance rank from graph structure when the LLM rating is unusable."""
    if max_entities <= 0:
        return 1.0, "Empty community."
    size_share = entity_count / max_entities
    density = internal_edges / max(1, entity_count)
    score = 1.0 + 9.0 * min(1.0, 0.6 * size_share + 0.4 * min(1.0, density / 2.0))
    return (
        round(score, 1),
        f"Derived from {entity_count} concepts and {internal_edges} internal relationships.",
    )


# -----------------------------------------------------------------------------
# 5. GraphRAG engine
# -----------------------------------------------------------------------------
class GraphRAGEngine:
    """
    Builds the knowledge base from a set of documents:
    1. Chunking
    2. Entity / relationship extraction
    3. Deduplication and graph construction
    4. Louvain community detection
    5. LLM community report synthesis
    6. ChromaDB vector ingestion
    7. Parquet and concept-map export
    """

    STEPS = ["Chunk documents", "Extract concepts", "Cluster communities",
             "Write reports", "Sync vectors", "Draw concept map"]

    def __init__(
        self,
        output_dir: str = "./ragtest/output",
        chroma_dir: str = "./notebook/chromadb",
        notebook_dir: str = "./notebook",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.chroma_dir = Path(chroma_dir)
        self.notebook_dir = Path(notebook_dir)
        self.model_name = model_name
        self.temperature = temperature
        self.max_workers = max_workers

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.notebook_dir.mkdir(parents=True, exist_ok=True)

    # -- public API ----------------------------------------------------------
    def build_from_text(
        self,
        document_text: str,
        document_title: str = "Document",
        document_id: Optional[str] = None,
        max_chunks: Optional[int] = 10,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        progress_callback: Optional[Callable[[float, str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Build the knowledge base from a single document."""
        return self.build_from_documents(
            [{
                "id": document_id or "doc_adhoc",
                "title": document_title,
                "text": document_text,
            }],
            max_chunks_per_doc=max_chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_callback=progress_callback,
        )

    def build_from_documents(
        self,
        documents: List[Dict[str, Any]],
        max_chunks_per_doc: Optional[int] = 10,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        progress_callback: Optional[Callable[[float, str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Rebuild the whole knowledge base from the given documents.

        Each document is a mapping with ``id``, ``title`` and ``text``. Chunk ids
        are namespaced by document id so the vector store stays deletable per
        document and every citation can be traced back to its source file.
        """
        start_time = time.perf_counter()

        def notify(progress: float, message: str, stats: Optional[Dict[str, Any]] = None):
            if progress_callback:
                progress_callback(progress, message, stats or {})

        # Step 1: chunking, per document
        notify(0.04, "Splitting documents into token chunks", {"step": "Chunk documents"})
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        units: List[Dict[str, Any]] = []
        per_document: Dict[str, Dict[str, Any]] = {}

        for doc in documents:
            doc_id = str(doc.get("id") or "doc_unknown")
            title = str(doc.get("title") or doc_id)
            text = str(doc.get("text") or "")
            all_chunks = chunker.split(text) if text.strip() else []
            selected = all_chunks[:max_chunks_per_doc] if (max_chunks_per_doc and max_chunks_per_doc > 0) else all_chunks
            per_document[doc_id] = {
                "document_id": doc_id,
                "document_title": title,
                "chunks_total": len(all_chunks),
                "chunks_indexed": len(selected),
            }
            for local_idx, chunk_text in enumerate(selected):
                units.append({
                    "chunk_id": f"{doc_id}::chunk_{local_idx}",
                    "document_id": doc_id,
                    "document_title": title,
                    "chunk_index": local_idx,
                    "text": chunk_text,
                })

        total_units = len(units)
        notify(0.10, f"Prepared {total_units} chunks across {len(documents)} documents", {
            "step": "Chunk documents",
            "chunk_total": total_units,
            "documents": len(documents),
        })

        if total_units == 0:
            raise ValueError("No text could be chunked from the supplied documents.")

        # Step 2: extraction, parallel across chunks
        extractor = EntityRelationshipExtractor(model_name=self.model_name, temperature=self.temperature)
        all_entities_raw: List[Dict[str, Any]] = []
        all_relationships_raw: List[Dict[str, Any]] = []
        completed = 0

        def extract(unit: Dict[str, Any]):
            return extractor.extract_chunk(unit["text"], unit["chunk_id"])

        with ThreadPoolExecutor(max_workers=min(self.max_workers, total_units)) as pool:
            for entities, relationships in pool.map(extract, units):
                all_entities_raw.extend(entities)
                all_relationships_raw.extend(relationships)
                completed += 1
                notify(
                    0.12 + 0.43 * (completed / total_units),
                    f"Extracting concepts from chunk {completed} of {total_units}",
                    {
                        "step": "Extract concepts",
                        "chunk_current": completed,
                        "chunk_total": total_units,
                        "raw_entities": len(all_entities_raw),
                        "raw_relationships": len(all_relationships_raw),
                    },
                )

        # Step 3: deduplication
        notify(0.58, "Deduplicating concepts and merging relationship weights", {"step": "Cluster communities"})
        df_entities = self._build_entities(all_entities_raw)
        df_relationships = self._build_relationships(all_relationships_raw, set(df_entities["title"]))

        # Step 4: community detection and layout
        notify(0.64, "Detecting communities with Louvain clustering", {"step": "Cluster communities"})
        graph = nx.Graph()
        for _, row in df_entities.iterrows():
            graph.add_node(row["title"], type=row["type"], description=row["description"])
        for _, row in df_relationships.iterrows():
            if graph.has_node(row["source"]) and graph.has_node(row["target"]):
                graph.add_edge(row["source"], row["target"], weight=row["weight"], description=row["description"])

        communities = (
            list(nx.algorithms.community.louvain_communities(graph, weight="weight", seed=42))
            if graph.number_of_nodes() else []
        )
        communities.sort(key=len, reverse=True)
        community_of = {
            node: comm_id
            for comm_id, members in enumerate(communities)
            for node in members
        }

        positions = nx.spring_layout(graph, seed=42) if graph.number_of_nodes() else {}
        degrees = dict(graph.degree())
        df_nodes = self._build_nodes(df_entities, community_of, degrees, positions)

        # Step 5: community reports
        notify(0.70, f"Writing reports for {len(communities)} communities", {"step": "Write reports"})
        df_reports = self._build_reports(
            communities, df_entities, df_relationships, graph, notify
        )

        # attach the report rank back onto the nodes so the map can size by rank
        rank_by_community = (
            dict(zip(df_reports["community"], df_reports["rank"])) if not df_reports.empty else {}
        )
        df_nodes["community_rank"] = df_nodes["community"].map(rank_by_community).fillna(0.0).astype(float)

        # Step 6: persist parquet
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df_entities.to_parquet(self.output_dir / "create_final_entities.parquet")
        df_relationships.to_parquet(self.output_dir / "create_final_relationships.parquet")
        df_nodes.to_parquet(self.output_dir / "create_final_nodes.parquet")
        df_reports.to_parquet(self.output_dir / "create_final_community_reports.parquet")

        # Step 7: vector sync
        notify(0.90, "Syncing chunks with the vector store", {"step": "Sync vectors"})
        self._sync_chroma(units)

        # Step 8: concept map
        notify(0.96, "Drawing the concept map", {"step": "Draw concept map"})
        self._generate_visualizations(df_nodes, df_relationships)

        elapsed = round(time.perf_counter() - start_time, 2)
        result = {
            "elapsed_seconds": elapsed,
            "entities_count": len(df_entities),
            "relationships_count": len(df_relationships),
            "nodes_count": len(df_nodes),
            "communities_count": len(communities),
            "chunks_count": total_units,
            "documents": list(per_document.values()),
            "entities_df": df_entities,
            "relationships_df": df_relationships,
            "nodes_df": df_nodes,
            "community_reports_df": df_reports,
        }
        notify(1.0, f"Indexed {len(df_entities)} concepts in {elapsed}s", {
            "step": "Draw concept map",
            "elapsed_seconds": elapsed,
            "entities": len(df_entities),
            "relationships": len(df_relationships),
            "communities": len(communities),
            "vector_chunks": total_units,
        })
        return result

    # -- frame builders ------------------------------------------------------
    @staticmethod
    def _build_entities(raw: List[Dict[str, Any]]) -> pd.DataFrame:
        groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in raw:
            key = (item["title"], item["type"])
            bucket = groups.setdefault(key, {"descriptions": [], "chunk_ids": set()})
            bucket["descriptions"].append(item["description"])
            bucket["chunk_ids"].add(item["chunk_id"])

        rows = []
        for h_id, ((title, ent_type), data) in enumerate(groups.items()):
            unique = list(dict.fromkeys(d.strip() for d in data["descriptions"] if d and d.strip()))
            rows.append({
                "id": str(uuid.uuid4()),
                "human_readable_id": h_id,
                "title": title,
                "type": ent_type,
                "description": _clip(" ".join(unique), 1200),
                "text_unit_ids": sorted(data["chunk_ids"]),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=ENTITY_COLUMNS)

    @staticmethod
    def _build_relationships(raw: List[Dict[str, Any]], known_titles: set) -> pd.DataFrame:
        groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in raw:
            # Keep only relationships whose endpoints were also extracted as entities,
            # so the graph, the catalog and the concept map agree on the node set.
            if item["source"] not in known_titles or item["target"] not in known_titles:
                continue
            if item["source"] == item["target"]:
                continue
            key = tuple(sorted((item["source"], item["target"])))
            bucket = groups.setdefault(
                key, {"descriptions": [], "weights": [], "types": [], "chunk_ids": set()}
            )
            bucket["descriptions"].append(item["description"])
            bucket["weights"].append(item["weight"])
            bucket["types"].append(item.get("type") or "related")
            bucket["chunk_ids"].add(item["chunk_id"])

        rows = []
        for h_id, ((src, tgt), data) in enumerate(groups.items()):
            unique = list(dict.fromkeys(d.strip() for d in data["descriptions"] if d and d.strip()))
            # Several chunks can describe the same pair; keep the most frequent label.
            rel_type = Counter(data["types"]).most_common(1)[0][0]
            rows.append({
                "id": str(uuid.uuid4()),
                "human_readable_id": h_id,
                "source": src,
                "target": tgt,
                "type": rel_type,
                "description": _clip(" ".join(unique), 800),
                "weight": round(float(sum(data["weights"]) / len(data["weights"])), 1),
                "combined_degree": len(data["chunk_ids"]),
                "text_unit_ids": sorted(data["chunk_ids"]),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=RELATIONSHIP_COLUMNS)

    @staticmethod
    def _build_nodes(
        df_entities: pd.DataFrame,
        community_of: Dict[str, int],
        degrees: Dict[str, int],
        positions: Dict[str, Any],
    ) -> pd.DataFrame:
        rows = []
        for _, row in df_entities.iterrows():
            title = row["title"]
            point = positions.get(title, [0.0, 0.0])
            rows.append({
                "id": row["id"],
                "human_readable_id": row["human_readable_id"],
                "title": title,
                "community": int(community_of.get(title, 0)),
                "level": 0,
                "degree": int(degrees.get(title, 0)),
                "x": float(point[0]),
                "y": float(point[1]),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=NODE_COLUMNS)

    def _build_reports(
        self,
        communities: List[set],
        df_entities: pd.DataFrame,
        df_relationships: pd.DataFrame,
        graph: nx.Graph,
        notify: Callable[..., None],
    ) -> pd.DataFrame:
        if not communities:
            return pd.DataFrame(columns=REPORT_COLUMNS)

        entities_by_title = {row["title"]: row for row in df_entities.to_dict("records")}
        relationship_records = df_relationships.to_dict("records")
        max_size = max(len(c) for c in communities)
        synthesizer = CommunityReportSynthesizer(model_name=self.model_name, temperature=0.2)
        period = time.strftime("%Y-%m-%d")

        def build_one(indexed: Tuple[int, set]) -> Dict[str, Any]:
            comm_id, members = indexed
            member_rows = [entities_by_title[t] for t in members if t in entities_by_title]
            member_rows.sort(key=lambda r: graph.degree(r["title"]) if graph.has_node(r["title"]) else 0, reverse=True)
            internal = [
                r for r in relationship_records
                if r["source"] in members and r["target"] in members
            ]
            internal.sort(key=lambda r: r["weight"], reverse=True)

            fallback_rank, fallback_reason = _fallback_rank(len(member_rows), len(internal), max_size)
            top_names = [r["title"] for r in member_rows[:4]]

            report = None
            try:
                report = synthesizer.synthesize(member_rows[:30], internal[:40])
            except Exception as exc:  # keep indexing resilient to a single bad response
                notify(0.0, f"Report synthesis fell back for community {comm_id}: {exc}",
                       {"step": "Write reports", "community": comm_id})

            if report and report.get("title") and report.get("summary"):
                title = report["title"]
                summary = report["summary"]
                findings = report["findings"] or [{"summary": title, "explanation": summary}]
                rank = report["rating"] if report["rating"] is not None else fallback_rank
                rank_explanation = report["rating_explanation"] or fallback_reason
            else:
                title = ", ".join(top_names) if top_names else f"Community {comm_id}"
                summary = (
                    f"{len(member_rows)} related concepts including {', '.join(top_names)}, "
                    f"connected by {len(internal)} relationships."
                )
                findings = [{"summary": title, "explanation": summary}]
                rank = fallback_rank
                rank_explanation = fallback_reason

            body_lines = [f"# {title}", "", "## Summary", summary, ""]
            for finding in findings:
                body_lines.append(f"## {finding['summary']}")
                if finding["explanation"]:
                    body_lines.append(finding["explanation"])
                body_lines.append("")
            body_lines.append("## Concepts in this community")
            body_lines.extend(
                f"- **{r['title']}** ({r['type']}) · degree "
                f"{graph.degree(r['title']) if graph.has_node(r['title']) else 0}"
                for r in member_rows
            )
            if internal:
                body_lines.extend(["", "## Relationships"])
                body_lines.extend(
                    f"- {r['source']} → {r['target']} · {r.get('type') or 'related'} "
                    f"({r['weight']:.0f}/10): {r['description']}"
                    for r in internal[:20]
                )
            full_content = "\n".join(body_lines)

            return {
                "id": str(uuid.uuid4()),
                "human_readable_id": comm_id,
                "community": comm_id,
                "parent": -1,
                "level": 0,
                "title": title,
                "summary": summary,
                "full_content": full_content,
                "rank": float(rank),
                "rank_explanation": rank_explanation,
                "findings": json.dumps(findings),
                "full_content_json": json.dumps({
                    "title": title,
                    "summary": summary,
                    "findings": findings,
                    "rating": float(rank),
                    "rating_explanation": rank_explanation,
                }),
                "period": period,
                "size": len(member_rows),
            }

        rows: List[Dict[str, Any]] = []
        indexed = list(enumerate(communities))
        done = 0
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(indexed))) as pool:
            for row in pool.map(build_one, indexed):
                rows.append(row)
                done += 1
                notify(
                    0.70 + 0.18 * (done / len(indexed)),
                    f"Wrote report {done} of {len(indexed)}",
                    {"step": "Write reports", "report_current": done, "report_total": len(indexed)},
                )

        rows.sort(key=lambda r: r["community"])
        return pd.DataFrame(rows)

    # -- sinks ---------------------------------------------------------------
    def _sync_chroma(self, units: List[Dict[str, Any]]):
        """Replace the collection with the current chunk set, carrying provenance metadata."""
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION)

        batch = 64
        for start in range(0, len(units), batch):
            window = units[start:start + batch]
            collection.add(
                documents=[u["text"] for u in window],
                metadatas=[{
                    "document_id": u["document_id"],
                    "document_title": u["document_title"],
                    "chunk_index": u["chunk_index"],
                    "chunk_id": u["chunk_id"],
                } for u in window],
                ids=[u["chunk_id"] for u in window],
            )

    def _generate_visualizations(self, nodes_df: pd.DataFrame, relationships_df: pd.DataFrame):
        write_concept_map(
            nodes_df,
            relationships_df,
            self.notebook_dir / "interactive_graph.html",
        )

        if nodes_df.empty:
            return

        fig, ax = plt.subplots(figsize=(14, 9), facecolor="#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        graph = nx.Graph()
        for _, row in nodes_df.iterrows():
            graph.add_node(row["title"], community=int(row["community"]), degree=int(row.get("degree", 1)))
        for _, row in relationships_df.iterrows():
            if graph.has_node(row["source"]) and graph.has_node(row["target"]):
                graph.add_edge(row["source"], row["target"], weight=float(row["weight"]))

        positions = nx.spring_layout(graph, seed=42, k=0.55)
        node_colors = [COMMUNITY_PALETTE[graph.nodes[n]["community"] % len(COMMUNITY_PALETTE)] for n in graph.nodes]
        node_sizes = [350 + graph.nodes[n]["degree"] * 120 for n in graph.nodes]

        nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=node_sizes, alpha=0.92, ax=ax)
        nx.draw_networkx_edges(graph, positions, edge_color="#D6D6D0", alpha=0.9, width=1.2, ax=ax)
        nx.draw_networkx_labels(graph, positions, font_size=8, font_color="#111110", ax=ax)

        ax.set_title("Concept map · thematic domains", color="#111110", fontsize=14, pad=15)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(self.notebook_dir / "knowledge_graph.png", dpi=150,
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)


# -----------------------------------------------------------------------------
# 6. Concept map rendering
# -----------------------------------------------------------------------------
def write_concept_map(
    nodes_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    target_path: Path,
    size_by: str = "degree",
    communities: Optional[List[int]] = None,
    selected: Optional[str] = None,
    height: int = 620,
) -> Path:
    """Write the PyVis concept map with the guide's light palette.

    ``size_by`` is ``degree`` or ``rank``; ``communities`` restricts the drawing
    to those community ids; ``selected`` highlights one node.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    frame = nodes_df.copy()
    if communities:
        frame = frame[frame["community"].astype(int).isin([int(c) for c in communities])]

    net = Network(
        height=f"{height}px", width="100%", bgcolor="#FFFFFF", font_color="#111110",
        cdn_resources="in_line", directed=False,
    )
    net.set_options(json.dumps({
        "physics": {
            "enabled": True,
            "barnesHut": {"gravitationalConstant": -12000, "springLength": 130, "springConstant": 0.02},
            "stabilization": {"iterations": 180},
        },
        "interaction": {"hover": True, "tooltipDelay": 120, "navigationButtons": True, "keyboard": True},
        "nodes": {"borderWidth": 1, "borderWidthSelected": 2,
                  "font": {"face": "Geist, system-ui, sans-serif", "size": 12, "color": "#111110"}},
        "edges": {"color": {"color": "#D6D6D0", "highlight": "#3056D3", "hover": "#3056D3"},
                  "smooth": {"enabled": True, "type": "continuous"}, "width": 1},
    }))

    if "community_rank" not in frame.columns:
        frame["community_rank"] = 0.0

    for _, row in frame.iterrows():
        community = int(row["community"])
        color = COMMUNITY_PALETTE[community % len(COMMUNITY_PALETTE)]
        degree = int(row.get("degree", 0) or 0)
        rank = float(row.get("community_rank", 0) or 0)
        title_text = str(row["title"])
        magnitude = rank * 2.2 if size_by == "rank" else degree * 3.0
        is_selected = selected is not None and title_text == selected
        net.add_node(
            title_text,
            label=title_text,
            title=f"{title_text}\nCommunity {community} · degree {degree}",
            color={
                "background": color,
                "border": "#111110" if is_selected else color,
                "highlight": {"background": color, "border": "#111110"},
                "hover": {"background": color, "border": "#3056D3"},
            },
            borderWidth=3 if is_selected else 1,
            size=12 + min(magnitude, 30),
        )

    present = set(net.get_nodes())
    for _, row in relationships_df.iterrows():
        source, target = str(row["source"]), str(row["target"])
        if source in present and target in present:
            weight = float(row.get("weight", 1.0) or 1.0)
            net.add_edge(
                source, target,
                value=weight,
                title=f"{source} → {target} · strength {weight:.0f}/10",
            )

    net.write_html(str(target_path), notebook=False, open_browser=False)

    # Node clicks ask the host page to open the concept in the inspector. Streamlit
    # sandboxes component iframes, so the parent-navigation attempt can be refused;
    # the "Find node" select box in the top bar is the guaranteed selection path.
    handler = """
<script type="text/javascript">
(function () {
  if (typeof network === "undefined") { return; }
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    network.setOptions({ physics: { enabled: false } });
  }
  network.on("selectNode", function (params) {
    if (!params.nodes || !params.nodes.length) { return; }
    var node = params.nodes[0];
    var search = "?screen=concepts&node=" + encodeURIComponent(node);
    try { window.parent.postMessage({ type: "graphrag:selectNode", node: node }, "*"); } catch (e) {}
    try { window.top.location.search = search; return; } catch (e) {}
    var link = document.createElement("a");
    link.href = search; link.target = "_top";
    document.body.appendChild(link); link.click();
  });
})();
</script>
"""
    html = target_path.read_text(encoding="utf-8")
    html = html.replace("</body>", f"{handler}</body>") if "</body>" in html else html + handler
    target_path.write_text(html, encoding="utf-8")
    return target_path
