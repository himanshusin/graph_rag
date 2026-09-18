"""GraphRAG indexing pipeline: chunking, extraction, clustering, community reports.

The engine builds the whole knowledge base from a set of documents so that the
graph, the community reports, the vector store and the concept map always
describe the same corpus. Rebuilding is cumulative over the vault rather than
per upload, which is what lets several documents share one graph.
"""

import hashlib
import os
import queue
import re
import json
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable, TYPE_CHECKING

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

if TYPE_CHECKING:
    from core.cache import ExtractionCache, ReportCache
    from core.ledger import UsageLedger

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
PROMPT_VERSION = "json-v1"   # part of the extraction cache key; bump when prompts change

# JSON replaces GraphRAG's legacy ("entity"<|delimiter|>...) format. Measured
# 2026-09-18: an 8B-class open-weight model reproduces that custom format
# unreliably — it emitted the literal placeholder <NAME> and mangled the
# delimiter — while naming the right entities. A schema the provider enforces
# removes the largest source of non-frontier-model failure, and makes extraction
# quality far less dependent on which route is selected.
EXTRACTION_PROMPT_JSON = """You extract a knowledge graph from one chunk of a document.

Return ONLY a JSON object with exactly these two keys:

{{
  "entities": [
    {{"name": "ENTITY NAME IN CAPITALS",
      "type": "CONCEPT|METHOD|TOOL|ORGANIZATION|TECHNIQUE|METRIC|BENCHMARK|QUANTITY|TIMEFRAME",
      "description": "what it is, including exact numbers, units and scope"}}
  ],
  "relationships": [
    {{"source": "ENTITY NAME", "target": "ENTITY NAME",
      "type": "short lower-case verb phrase, e.g. extends, reduces, measured by, evaluated on",
      "description": "why they are related. ALWAYS keep exact numbers, percentages and deltas.",
      "strength": 7}}
  ]
}}

Rules:
- Extract EVERY entity you can find, then find the relationships BETWEEN them.
- Relationships matter as much as entities. A chunk with 10 entities usually has
  at least 5 relationships. Look for: what measures what, what improves what,
  what is part of what, what is evaluated on what, what is an alternative to what.
- "source" and "target" MUST be names that appear in your own "entities" list.
- METRIC is a property being measured; BENCHMARK is a dataset or evaluation setting;
  QUANTITY is an exact figure; TIMEFRAME is a period.
- "strength" is an integer 1-10.
- Preserve figures verbatim. Invent nothing.

Input text:
{input_text}
"""

# Retained so extractions cached under the old format still parse.
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
    """Extracts entities and relationships from chunks via any configured provider.

    Results are cached by content hash, so a rebuild only pays for chunks whose
    text, prompt or model actually changed. The cache key includes the provider
    and model: without that, switching routes would read back another model's
    extractions and report them as this one's, and a re-index after a switch
    would yield a graph with two different naming conventions in it.
    """

    def __init__(
        self,
        provider=None,
        temperature: float = 0.0,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: Optional["ExtractionCache"] = None,
        usage: Optional["UsageLedger"] = None,
    ):
        from core import providers as provider_registry

        if provider is None:
            provider = provider_registry.get("openai")
            if model_name:
                provider = replace(provider, model=model_name)
        self.provider = provider
        self.cache = cache
        self.usage = usage
        self.llm = provider_registry.build_llm(
            provider, temperature=temperature, json_mode=True
        )
        self.prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT_JSON)
        self.chain = self.prompt | self.llm

    def cache_key(self, chunk_text: str) -> str:
        digest = hashlib.sha256(
            "|".join([chunk_text, PROMPT_VERSION, self.provider.key, self.provider.model])
            .encode("utf-8")
        ).hexdigest()
        return digest

    def extract_chunk(self, chunk_text: str, chunk_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        key = self.cache_key(chunk_text)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                if self.usage is not None:
                    self.usage.record_cache_hit("extraction")
                return self._retag(hit, chunk_id)

        response = self.chain.invoke({"input_text": chunk_text})
        raw = getattr(response, "content", response)
        if self.usage is not None:
            self.usage.record_call("extraction", self.provider, response)

        entities, relationships = self._parse_json(str(raw), chunk_id)
        if not entities and not relationships:
            # Tolerate a model that ignored json_mode and answered in the old
            # delimiter format rather than losing the chunk entirely.
            entities, relationships = self._parse_response(str(raw), chunk_id)

        if self.cache is not None:
            self.cache.put(key, {"entities": entities, "relationships": relationships})
        return entities, relationships

    @staticmethod
    def _retag(payload: Dict[str, Any], chunk_id: str):
        """A cached result was produced for some other chunk id; re-stamp it."""
        entities = [dict(e, chunk_id=chunk_id) for e in payload.get("entities", [])]
        relationships = [dict(r, chunk_id=chunk_id) for r in payload.get("relationships", [])]
        return entities, relationships

    @staticmethod
    def _parse_json(raw_text: str, chunk_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse the schema'd response, tolerating fences and surrounding prose."""
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return [], []
        try:
            data = json.loads(text[start:end + 1])
        except ValueError:
            return [], []

        entities: List[Dict[str, Any]] = []
        seen_titles = set()
        for item in data.get("entities") or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("name") or "").strip().upper()
            if not title:
                continue
            entities.append({
                "title": title,
                "type": str(item.get("type") or "CONCEPT").strip().upper()[:40],
                "description": str(item.get("description") or "").strip(),
                "chunk_id": chunk_id,
            })
            seen_titles.add(title)

        relationships: List[Dict[str, Any]] = []
        for item in data.get("relationships") or []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip().upper()
            target = str(item.get("target") or "").strip().upper()
            if not source or not target or source == target:
                continue
            try:
                weight = float(item.get("strength", 5))
            except (TypeError, ValueError):
                weight = 5.0
            relationships.append({
                "source": source,
                "target": target,
                "type": str(item.get("type") or "related").strip().lower()[:40],
                "description": str(item.get("description") or "").strip(),
                "weight": max(1.0, min(10.0, weight)),
                "chunk_id": chunk_id,
            })

        return entities, relationships

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

    def __init__(
        self,
        provider=None,
        temperature: float = 0.2,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        usage: Optional["UsageLedger"] = None,
    ):
        from core import providers as provider_registry

        if provider is None:
            provider = provider_registry.get("openai")
            if model_name:
                provider = replace(provider, model=model_name)
        self.provider = provider
        self.usage = usage
        self.llm = provider_registry.build_llm(
            provider, temperature=temperature, json_mode=True
        )
        self.chain = ChatPromptTemplate.from_template(REPORT_PROMPT) | self.llm

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

        response = self.chain.invoke(
            {"entities": entities_text, "relationships": relationships_text}
        )
        if self.usage is not None:
            self.usage.record_call("reports", self.provider, response)
        return self._parse(str(getattr(response, "content", response)))

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


def _is_trivial_community(member_rows: List[Dict[str, Any]], internal: List[Dict[str, Any]]) -> bool:
    """Is there anything here worth spending a model call on?

    A community is trivial when it has at most two concepts and no relationship
    between them: there is no structure to summarize, so a report would be the
    model restating one entity's description as analysis.
    """
    return len(member_rows) <= 2 and not internal


def _describe_trivial(member_rows: List[Dict[str, Any]], comm_id: int) -> Tuple[str, str]:
    """Deterministic title and summary for a community with no internal structure."""
    if not member_rows:
        return f"Empty community {comm_id}", "This community has no concepts."
    names = [r["title"] for r in member_rows]
    title = " and ".join(names) if len(names) > 1 else names[0]
    kinds = ", ".join(sorted({str(r.get("type", "CONCEPT")) for r in member_rows}))
    descriptions = " ".join(
        _clip(r.get("description", ""), 300) for r in member_rows if r.get("description")
    )
    lead = (
        f"An isolated concept ({kinds}) with no extracted relationships to the rest of "
        f"the corpus."
        if len(names) == 1 else
        f"Two concepts ({kinds}) grouped together with no extracted relationship between them."
    )
    return title, f"{lead} {descriptions}".strip()


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


class ProgressRelay:
    """Funnels progress from worker threads onto the calling thread.

    Streamlit widgets can only be written from the thread that owns the script
    run; a call from a pool worker is silently dropped with a "missing
    ScriptRunContext" warning. Worker messages are queued here and flushed the
    next time the owning thread reports, so report-stage progress is not lost.
    """

    def __init__(self, callback: Optional[Callable[[float, str, Dict[str, Any]], None]]):
        self._callback = callback
        self._queue: "queue.Queue" = queue.Queue()
        self._owner = threading.get_ident()

    def __call__(self, progress: float, message: str, stats: Optional[Dict[str, Any]] = None):
        item = (float(progress), str(message), dict(stats or {}))
        if threading.get_ident() == self._owner:
            self.flush()
            self._emit(item)
        else:
            self._queue.put(item)

    def flush(self) -> None:
        """Drain queued worker messages. Safe to call only from the owner."""
        while True:
            try:
                self._emit(self._queue.get_nowait())
            except queue.Empty:
                return

    def _emit(self, item) -> None:
        if not self._callback:
            return
        try:
            self._callback(*item)
        except Exception:
            # Progress reporting must never take the build down.
            pass


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

    # Reports are the slowest stage and are written last, so an interrupted run
    # still leaves a queryable graph, vectors and concept map behind.
    STEPS = ["Chunk documents", "Extract concepts", "Cluster communities",
             "Write graph", "Sync vectors", "Draw concept map", "Write reports"]

    def __init__(
        self,
        output_dir: str = "./ragtest/output",
        chroma_dir: str = "./notebook/chromadb",
        notebook_dir: str = "./notebook",
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_workers: int = 4,
        extraction_provider=None,
        report_provider=None,
        cache: Optional["ExtractionCache"] = None,
        usage: Optional["UsageLedger"] = None,
    ):
        from core import providers as provider_registry

        self.output_dir = Path(output_dir)
        self.chroma_dir = Path(chroma_dir)
        self.notebook_dir = Path(notebook_dir)
        self.temperature = temperature

        # Extraction and reports can sit on different routes: extraction is bulk
        # structured work that suits a cheap open-weight model, while reports are
        # closer to the analyst-facing output.
        default = provider_registry.get("openai")
        if model_name:
            default = replace(default, model=model_name)
        self.extraction_provider = extraction_provider or default
        self.report_provider = report_provider or self.extraction_provider
        self.model_name = self.extraction_provider.model

        # Concurrency is a property of the backend, not a constant. Ollama
        # serialises, and raising its parallelism multiplies KV cache instead of
        # throughput; hosted routes rate-limit instead.
        self.max_workers = max(1, min(max_workers, self.extraction_provider.max_workers))

        self.cache = cache
        self.usage = usage

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

        notify = ProgressRelay(progress_callback)

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
        extractor = EntityRelationshipExtractor(
            provider=self.extraction_provider,
            temperature=self.temperature,
            cache=self.cache,
            usage=self.usage,
        )
        all_entities_raw: List[Dict[str, Any]] = []
        all_relationships_raw: List[Dict[str, Any]] = []
        completed = 0

        def extract(unit: Dict[str, Any]):
            return extractor.extract_chunk(unit["text"], unit["chunk_id"])

        with ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, total_units))) as pool:
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

        # Step 5: persist the graph now, before the slow report stage. Everything
        # written here survives an interruption during report synthesis.
        df_nodes["community_rank"] = 0.0
        notify(0.68, f"Writing {len(df_entities)} concepts to the graph", {"step": "Write graph"})
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df_entities.to_parquet(self.output_dir / "create_final_entities.parquet")
        df_relationships.to_parquet(self.output_dir / "create_final_relationships.parquet")
        df_nodes.to_parquet(self.output_dir / "create_final_nodes.parquet")

        # Step 6: vector sync
        notify(0.72, "Syncing chunks with the vector store", {"step": "Sync vectors"})
        self._sync_chroma(units)

        # Step 7: concept map
        notify(0.76, "Drawing the concept map", {"step": "Draw concept map"})
        self._generate_visualizations(df_nodes, df_relationships)

        # Step 8: community reports, the slowest stage
        notify(0.80, f"Writing reports for {len(communities)} communities", {"step": "Write reports"})
        df_reports = self._build_reports(
            communities, df_entities, df_relationships, graph, notify
        )
        df_reports.to_parquet(self.output_dir / "create_final_community_reports.parquet")

        # Report importance feeds the concept map's size-by-rank, so the nodes
        # are rewritten once the ranks exist.
        rank_by_community = (
            dict(zip(df_reports["community"], df_reports["rank"])) if not df_reports.empty else {}
        )
        df_nodes["community_rank"] = (
            df_nodes["community"].map(rank_by_community).fillna(0.0).astype(float)
        )
        df_nodes.to_parquet(self.output_dir / "create_final_nodes.parquet")
        notify.flush()

        elapsed = round(time.perf_counter() - start_time, 2)
        usage_summary = self.usage.summary() if self.usage is not None else {}
        cache_stats = self.cache.stats() if self.cache is not None else {}
        result = {
            "elapsed_seconds": elapsed,
            "usage": usage_summary,
            "cache": cache_stats,
            "extraction_provider": self.extraction_provider.key,
            "extraction_model": self.extraction_provider.model,
            "report_provider": self.report_provider.key,
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
            "step": "Write reports",
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

        usage = self.usage
        entities_by_title = {row["title"]: row for row in df_entities.to_dict("records")}
        relationship_records = df_relationships.to_dict("records")
        max_size = max(len(c) for c in communities)
        synthesizer = CommunityReportSynthesizer(
            provider=self.report_provider, temperature=0.2, usage=usage
        )
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
            if _is_trivial_community(member_rows, internal):
                # A community of one or two concepts with no relationship between
                # them has nothing for an analyst to synthesize. Measured on this
                # corpus, 206 of 228 communities were of this shape — 90% of the
                # report stage spent inventing narrative about a single node.
                # The deterministic description below is both cheaper and more
                # honest than a model's four findings about one entity.
                if usage is not None:
                    usage.record_templated("reports")
            else:
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
            elif _is_trivial_community(member_rows, internal):
                title, summary = _describe_trivial(member_rows, comm_id)
                findings = [{"summary": title, "explanation": summary}]
                rank = fallback_rank
                rank_explanation = fallback_reason
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
        report_workers = min(self.report_provider.max_workers, self.max_workers, len(indexed))
        with ThreadPoolExecutor(max_workers=max(1, report_workers)) as pool:
            for row in pool.map(build_one, indexed):
                rows.append(row)
                done += 1
                notify(
                    0.80 + 0.18 * (done / len(indexed)),
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
