"""Reasoning engines for the five search modes.

Retrieval is separated from generation so the UI can fill the evidence rail
before the answer starts streaming, and so each mode can be tested without
Streamlit.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import pandas as pd

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

MODE_KEYS = ["global", "local", "drift", "vector", "compare"]

MODE_LABELS = {
    "global": "Global synthesis",
    "local": "Local lookup",
    "drift": "DRIFT deep-dive",
    "vector": "Vector passages",
    "compare": "Compare graph vs vector",
}

MODE_DESCRIPTIONS = {
    "global": "Synthesizes community reports across the whole corpus",
    "local": "Looks up the concepts and relationships closest to the question",
    "drift": "Starts from the top domains, then drills into their concepts",
    "vector": "Retrieves the closest document passages from the vector store",
    "compare": "Runs the graph and the vector engine side by side",
}

# Citation kinds each source filter admits.
SOURCE_KINDS = {
    "all": {"Domain", "Concept", "Relationship", "Passage", "Table"},
    "graph": {"Domain", "Concept", "Relationship"},
    "documents": {"Passage"},
    "tables": {"Table"},
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at", "by",
    "from", "is", "are", "was", "were", "be", "been", "being", "that", "this", "these",
    "those", "it", "its", "as", "what", "which", "who", "whom", "how", "why", "when",
    "where", "do", "does", "did", "can", "could", "should", "would", "will", "about",
    "between", "across", "into", "than", "then", "there", "their", "them", "i", "we",
    "you", "he", "she", "they", "my", "our", "your", "if", "not", "no", "but", "so",
}

# Vector hits further than this are treated as unrelated. ChromaDB's default
# embedding uses squared L2 distance, where unrelated sentence pairs land well
# above 1.8 and related ones typically below 1.4.
MAX_VECTOR_DISTANCE = 1.8


def tokenize(text: Any) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(text or "").lower()) if t not in STOPWORDS and len(t) > 1]


def lexical_score(query_tokens: Sequence[str], *fields: Any) -> float:
    """Score a record against the query by weighted token overlap.

    Earlier fields weigh more, so a title match outranks a description match.
    """
    if not query_tokens:
        return 0.0
    query_set = set(query_tokens)
    score = 0.0
    for position, value in enumerate(fields):
        weight = 3.0 / (position + 1)
        tokens = set(tokenize(value))
        if not tokens:
            continue
        overlap = query_set & tokens
        if overlap:
            score += weight * len(overlap) / math.sqrt(len(tokens))
    return score


@dataclass
class Retrieval:
    """Everything gathered for one run, before the model is called."""

    mode: str
    query: str
    context: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    exhausted: bool = False  # nothing relevant was found

    def summary(self) -> str:
        parts = [f"{value} {name}" for name, value in self.counts.items() if value]
        return " · ".join(parts)


# -----------------------------------------------------------------------------
# Graph data access
# -----------------------------------------------------------------------------
@dataclass
class GraphData:
    entities: pd.DataFrame
    relationships: pd.DataFrame
    nodes: pd.DataFrame
    reports: pd.DataFrame

    @property
    def is_empty(self) -> bool:
        return self.entities.empty and self.reports.empty

    def degree_of(self, title: str) -> int:
        if self.nodes.empty or "title" not in self.nodes.columns:
            return 0
        match = self.nodes[self.nodes["title"].astype(str) == str(title)]
        if match.empty:
            return 0
        return int(match.iloc[0].get("degree", 0) or 0)

    def community_of(self, title: str) -> Optional[int]:
        if self.nodes.empty or "title" not in self.nodes.columns:
            return None
        match = self.nodes[self.nodes["title"].astype(str) == str(title)]
        if match.empty:
            return None
        try:
            return int(match.iloc[0]["community"])
        except (KeyError, TypeError, ValueError):
            return None

    def community_title(self, community: Any) -> str:
        if self.reports.empty:
            return f"community {community}"
        match = self.reports[self.reports["community"].astype(str) == str(community)]
        if match.empty:
            return f"community {community}"
        return str(match.iloc[0]["title"])

    def titles_in_community(self, community: Any) -> List[str]:
        if self.nodes.empty:
            return []
        match = self.nodes[self.nodes["community"].astype(str) == str(community)]
        return [str(t) for t in match["title"].tolist()]


# -----------------------------------------------------------------------------
# Retrievers
# -----------------------------------------------------------------------------
def _table_citations(
    query: str,
    vault,
    citations: List[Dict[str, Any]],
    context_lines: List[str],
    allowed: set,
    top_k: int = 2,
) -> int:
    """Append matched structured tables to the context and the citation list."""
    if "Table" not in allowed or vault is None:
        return 0
    matched = vault.find_relevant_tables(query, top_k=top_k)
    if not matched:
        return 0

    context_lines.append("\n### Structured tables")
    for table in matched:
        index = len(citations) + 1
        context_lines.append(
            f"#### [{index}] Table: {table.get('title', 'Extracted table')} "
            f"(page {table.get('page', 1)})"
        )
        context_lines.append(table.get("markdown", ""))
        if table.get("row_facts"):
            context_lines.append("Row facts:\n" + "\n".join(table["row_facts"][:8]))
        citations.append({
            "index": index,
            "kind": "Table",
            "name": table.get("title", "Extracted table"),
            "meta": f"p. {table.get('page', 1)}",
            "source": "tables",
            "document_id": table.get("doc_id"),
            "table_id": table.get("table_id"),
            "columns": table.get("columns", []),
            "rows_count": table.get("rows_count", 0),
            "records": table.get("records", [])[:4],
            "table_markdown": table.get("markdown", ""),
            "excerpt": (
                f"{len(table.get('columns', []))} columns × {table.get('rows_count', 0)} rows: "
                f"{', '.join(table.get('columns', [])[:4])}"
            ),
        })
    return len(matched)


def retrieve_global(
    query: str, data: GraphData, vault=None, *, source: str = "all", max_reports: int = 8
) -> Retrieval:
    """Rank community reports by relevance and importance, then summarize across them."""
    allowed = SOURCE_KINDS.get(source, SOURCE_KINDS["all"])
    tokens = tokenize(query)
    citations: List[Dict[str, Any]] = []
    context_lines: List[str] = []
    counts = {"reports": 0, "tables": 0}

    if "Domain" in allowed and not data.reports.empty:
        scored = []
        for row in data.reports.to_dict("records"):
            rank = float(row.get("rank", 0) or 0)
            relevance = lexical_score(tokens, row.get("title"), row.get("summary"))
            # Importance breaks ties when the question is broad and matches little.
            scored.append((relevance * 2.0 + rank / 10.0, relevance, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = [row for _score, _rel, row in scored[:max_reports]]

        context_lines.append("### Domain reports")
        for row in chosen:
            index = len(citations) + 1
            context_lines.append(
                f"#### [{index}] {row['title']} (importance {float(row.get('rank', 0)):.1f}/10)\n"
                f"{row.get('summary', '')}\n"
            )
            citations.append({
                "index": index,
                "kind": "Domain",
                "name": row["title"],
                "meta": f"rank {float(row.get('rank', 0)):.1f}",
                "source": "graph",
                "community": row.get("community"),
                "excerpt": row.get("summary", ""),
            })
        counts["reports"] = len(chosen)

    counts["tables"] = _table_citations(query, vault, citations, context_lines, allowed)
    exhausted = not citations
    if exhausted:
        context_lines.append("No domain reports are available for this corpus.")

    return Retrieval(
        mode="global",
        query=query,
        context="\n".join(context_lines),
        citations=citations,
        counts=counts,
        exhausted=exhausted,
    )


def retrieve_local(
    query: str,
    data: GraphData,
    vault=None,
    *,
    source: str = "all",
    top_entities: int = 12,
    top_relationships: int = 14,
) -> Retrieval:
    """Select the concepts closest to the question, then their relationships."""
    allowed = SOURCE_KINDS.get(source, SOURCE_KINDS["all"])
    tokens = tokenize(query)
    citations: List[Dict[str, Any]] = []
    context_lines: List[str] = []
    counts = {"concepts": 0, "relationships": 0, "tables": 0}
    selected_titles: List[str] = []

    if "Concept" in allowed and not data.entities.empty:
        scored = []
        for row in data.entities.to_dict("records"):
            relevance = lexical_score(tokens, row.get("title"), row.get("type"), row.get("description"))
            if relevance <= 0:
                continue
            degree = data.degree_of(row["title"])
            scored.append((relevance + min(degree, 20) * 0.02, row))
        scored.sort(key=lambda item: item[0], reverse=True)

        if not scored:
            # Nothing matched lexically: fall back to the most connected concepts
            # so a broad question still gets the backbone of the graph.
            fallback = data.entities.to_dict("records")
            fallback.sort(key=lambda r: data.degree_of(r["title"]), reverse=True)
            scored = [(0.0, row) for row in fallback[:top_entities]]

        chosen = [row for _score, row in scored[:top_entities]]
        selected_titles = [str(row["title"]) for row in chosen]

        context_lines.append("### Concepts")
        for row in chosen:
            index = len(citations) + 1
            community = data.community_of(row["title"])
            context_lines.append(
                f"- [{index}] {row['title']} ({row['type']}): {row.get('description', '')}"
            )
            citations.append({
                "index": index,
                "kind": "Concept",
                "name": row["title"],
                "meta": str(row.get("type", "")),
                "source": "graph",
                "entity_type": row.get("type"),
                "community": community,
                "degree": data.degree_of(row["title"]),
                "excerpt": row.get("description", ""),
            })
        counts["concepts"] = len(chosen)

    if "Relationship" in allowed and not data.relationships.empty:
        selected = set(selected_titles)
        records = data.relationships.to_dict("records")
        scored_rels = []
        for row in records:
            touching = (str(row["source"]) in selected) + (str(row["target"]) in selected)
            if not touching:
                continue
            relevance = lexical_score(tokens, row.get("description"))
            scored_rels.append((touching * 1.0 + relevance + float(row.get("weight", 0) or 0) / 20.0, row))
        scored_rels.sort(key=lambda item: item[0], reverse=True)
        chosen_rels = [row for _s, row in scored_rels[:top_relationships]]

        if chosen_rels:
            context_lines.append("\n### Relationships")
            for row in chosen_rels:
                context_lines.append(
                    f"- {row['source']} {row.get('type') or 'relates to'} {row['target']} "
                    f"(strength {float(row.get('weight', 0)):.0f}/10): {row.get('description', '')}"
                )
            counts["relationships"] = len(chosen_rels)

    counts["tables"] = _table_citations(query, vault, citations, context_lines, allowed)
    exhausted = not citations
    if exhausted:
        context_lines.append("No concepts are indexed for this corpus.")

    return Retrieval(
        mode="local",
        query=query,
        context="\n".join(context_lines),
        citations=citations,
        counts=counts,
        exhausted=exhausted,
    )


def retrieve_drift(
    query: str,
    data: GraphData,
    vault=None,
    *,
    source: str = "all",
    max_communities: int = 4,
) -> Retrieval:
    """Start from the most relevant domains, then drill into their concepts and links.

    This mirrors DRIFT's two stages: a global pass to choose where to look,
    followed by a local pass inside the chosen communities.
    """
    allowed = SOURCE_KINDS.get(source, SOURCE_KINDS["all"])
    tokens = tokenize(query)
    citations: List[Dict[str, Any]] = []
    context_lines: List[str] = []
    counts = {"reports": 0, "concepts": 0, "relationships": 0, "tables": 0}

    chosen_communities: List[Any] = []
    if not data.reports.empty and "Domain" in allowed:
        scored = []
        for row in data.reports.to_dict("records"):
            rank = float(row.get("rank", 0) or 0)
            relevance = lexical_score(tokens, row.get("title"), row.get("summary"), row.get("full_content"))
            scored.append((relevance * 2.0 + rank / 10.0, relevance, row))
        scored.sort(key=lambda item: item[0], reverse=True)

        # Drill only into domains the question actually touches. When nothing
        # matches, fall back to the most important domains so a broad question
        # still gets an answer.
        relevant = [item for item in scored if item[1] > 0]
        pool = relevant if relevant else scored
        top = [row for _s, _rel, row in pool[:max_communities]]

        context_lines.append("### Stage 1 · domain reports")
        for row in top:
            index = len(citations) + 1
            chosen_communities.append(row.get("community"))
            context_lines.append(
                f"#### [{index}] {row['title']} (importance {float(row.get('rank', 0)):.1f}/10)\n"
                f"{row.get('summary', '')}\n"
            )
            citations.append({
                "index": index,
                "kind": "Domain",
                "name": row["title"],
                "meta": f"rank {float(row.get('rank', 0)):.1f}",
                "source": "graph",
                "community": row.get("community"),
                "excerpt": row.get("summary", ""),
            })
        counts["reports"] = len(top)

    # Stage 2: the concepts and relationships inside those communities.
    community_titles = set()
    for community in chosen_communities:
        community_titles.update(data.titles_in_community(community))

    if community_titles and "Concept" in allowed and not data.entities.empty:
        inside = [
            row for row in data.entities.to_dict("records")
            if str(row["title"]) in community_titles
        ]
        inside.sort(
            key=lambda r: (
                lexical_score(tokens, r.get("title"), r.get("description")),
                data.degree_of(r["title"]),
            ),
            reverse=True,
        )
        chosen = inside[:8]
        if chosen:
            context_lines.append("\n### Stage 2 · concepts inside those domains")
            for row in chosen:
                index = len(citations) + 1
                context_lines.append(
                    f"- [{index}] {row['title']} ({row['type']}): {row.get('description', '')}"
                )
                citations.append({
                    "index": index,
                    "kind": "Concept",
                    "name": row["title"],
                    "meta": str(row.get("type", "")),
                    "source": "graph",
                    "entity_type": row.get("type"),
                    "community": data.community_of(row["title"]),
                    "degree": data.degree_of(row["title"]),
                    "excerpt": row.get("description", ""),
                })
            counts["concepts"] = len(chosen)

    if not data.relationships.empty and "Relationship" in allowed:
        records = data.relationships.to_dict("records")
        relevant = [
            row for row in records
            if (not community_titles)
            or str(row["source"]) in community_titles
            or str(row["target"]) in community_titles
        ]
        relevant.sort(
            key=lambda r: lexical_score(tokens, r.get("description")) + float(r.get("weight", 0) or 0) / 20.0,
            reverse=True,
        )
        chosen_rels = relevant[:20]
        if chosen_rels:
            # The strongest link is citable, so it carries a marker in the context.
            top_link = chosen_rels[0]
            link_index = len(citations) + 1
            context_lines.append("\n### Cross-domain relationships")
            for position, row in enumerate(chosen_rels):
                marker = f"[{link_index}] " if position == 0 else ""
                context_lines.append(
                    f"- {marker}{row['source']} {row.get('type') or 'relates to'} {row['target']} "
                    f"(strength {float(row.get('weight', 0)):.0f}/10): {row.get('description', '')}"
                )
            counts["relationships"] = len(chosen_rels)
            citations.append({
                "index": link_index,
                "kind": "Relationship",
                "name": f"{top_link['source']} → {top_link['target']}",
                "meta": f"{top_link.get('type') or 'related'} · {float(top_link.get('weight', 0)):.0f}/10",
                "source": "graph",
                "excerpt": top_link.get("description", ""),
            })

    counts["tables"] = _table_citations(query, vault, citations, context_lines, allowed)
    exhausted = not citations
    if exhausted:
        context_lines.append("No graph artifacts are available for this corpus.")

    return Retrieval(
        mode="drift",
        query=query,
        context="\n".join(context_lines),
        citations=citations,
        counts=counts,
        exhausted=exhausted,
    )


def retrieve_vector(
    query: str,
    collection,
    vault=None,
    *,
    source: str = "all",
    top_k: int = 4,
    max_distance: float = MAX_VECTOR_DISTANCE,
) -> Retrieval:
    """Retrieve the closest passages, dropping hits beyond the distance threshold."""
    allowed = SOURCE_KINDS.get(source, SOURCE_KINDS["all"])
    citations: List[Dict[str, Any]] = []
    context_lines: List[str] = []
    notes: List[str] = []
    counts = {"passages": 0, "tables": 0}

    if "Passage" in allowed and collection is not None:
        available = collection.count()
        if available:
            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, available),
                include=["documents", "metadatas", "distances"],
            )
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            identifiers = (results.get("ids") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]

            kept = 0
            for document, meta, chunk_id, distance in zip(documents, metadatas, identifiers, distances):
                meta = meta or {}
                if distance is not None and distance > max_distance:
                    continue
                kept += 1
                index = len(citations) + 1
                title = meta.get("document_title") or "Untitled passage"
                context_lines.append(
                    f"--- [{index}] {title} · {chunk_id} ---\n{str(document).strip()}\n"
                )
                citations.append({
                    "index": index,
                    "kind": "Passage",
                    "name": title,
                    "meta": f"chunk {meta.get('chunk_index', chunk_id)}",
                    "source": "documents",
                    "document_id": meta.get("document_id"),
                    "chunk_id": chunk_id,
                    "distance": round(float(distance), 3) if distance is not None else None,
                    "quote": str(document).strip(),
                })
            counts["passages"] = kept
            if not kept:
                notes.append(
                    f"No passage came within the relevance threshold "
                    f"(closest distance {min(distances):.2f} of {max_distance:.1f})."
                    if distances else "No passages were returned."
                )
        else:
            notes.append("The vector store holds no passages yet.")

    counts["tables"] = _table_citations(query, vault, citations, context_lines, allowed)
    exhausted = not citations
    if exhausted and not notes:
        notes.append("No document passages are available.")

    return Retrieval(
        mode="vector",
        query=query,
        context="\n".join(context_lines) or "(no retrieved context)",
        citations=citations,
        notes=notes,
        counts=counts,
        exhausted=exhausted,
    )


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------
CITATION_RULE = (
    "Cite every claim with bracketed footnotes that match the numbers in the "
    "context, like [1] or [2]. Only use numbers that appear in the context. "
    "Quote figures, percentages and units exactly as they are written."
)

PROMPTS = {
    "global": (
        "You are an analyst summarizing what a document corpus says, working only from the "
        "domain reports and tables below.\n" + CITATION_RULE + "\n\n"
        "Structure the answer as short labelled paragraphs: Summary, Evidence, Trade-offs, "
        "Recommendation. Keep it under 250 words.\n\n"
        "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    ),
    "local": (
        "You are an analyst answering a specific question from a knowledge graph of concepts "
        "and relationships.\n" + CITATION_RULE + "\n\n"
        "Lead with the direct answer in one sentence, then give the supporting concepts and "
        "relationships as bullets. Keep it under 200 words.\n\n"
        "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    ),
    "drift": (
        "You are an analyst connecting themes across a corpus to the detailed concepts "
        "underneath them. The context has a stage 1 domain pass and a stage 2 concept pass.\n"
        + CITATION_RULE + "\n\n"
        "Use bold lead-ins for each point, name the dependency chains explicitly, and state "
        "the risks. Keep it under 260 words.\n\n"
        "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    ),
    "vector": (
        "You are an analyst answering strictly from the retrieved document passages below.\n"
        + CITATION_RULE + "\n\n"
        "If part of the question is not covered by the passages, say so in a short sentence "
        "instead of leaving it out. Never fill gaps from prior knowledge. Keep it under 200 words.\n\n"
        "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    ),
}


def answer_stream(retrieval: Retrieval, llm) -> Iterator[str]:
    """Stream the answer for a retrieval. Yields text fragments."""
    if retrieval.exhausted:
        yield _exhausted_message(retrieval)
        return
    template = PROMPTS[retrieval.mode]
    chain = ChatPromptTemplate.from_template(template) | llm | StrOutputParser()
    prefix = ""
    if retrieval.notes:
        prefix = " ".join(retrieval.notes) + "\n\n"
    if prefix:
        yield prefix
    yield from chain.stream({"context": retrieval.context, "query": retrieval.query})


def answer(retrieval: Retrieval, llm) -> str:
    """Non-streaming variant, used by the test suite."""
    return "".join(answer_stream(retrieval, llm))


def _exhausted_message(retrieval: Retrieval) -> str:
    if retrieval.mode == "vector":
        note = retrieval.notes[0] if retrieval.notes else "No passages matched."
        return f"{note} Try Global synthesis for a corpus-level answer."
    return (
        "The knowledge graph has not been built yet, so there is nothing to reason over. "
        "Add a document in the Vault and index it to start."
    )


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def retrieve(
    mode: str,
    query: str,
    *,
    data: GraphData,
    collection=None,
    vault=None,
    source: str = "all",
    top_k: int = 4,
) -> Retrieval:
    """Run the retriever for one mode."""
    if mode == "global":
        return retrieve_global(query, data, vault, source=source)
    if mode == "local":
        return retrieve_local(query, data, vault, source=source)
    if mode == "drift":
        return retrieve_drift(query, data, vault, source=source)
    if mode == "vector":
        return retrieve_vector(query, collection, vault, source=source, top_k=top_k)
    raise ValueError(f"{mode!r} has no single retriever; use retrieve_compare")


def retrieve_compare(
    query: str,
    *,
    data: GraphData,
    collection=None,
    vault=None,
    source: str = "all",
    top_k: int = 4,
    graph_mode: str = "drift",
) -> Dict[str, Retrieval]:
    """Retrieve for both sides of a comparison concurrently."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        graph_future = pool.submit(
            retrieve, graph_mode, query, data=data, collection=collection, vault=vault, source=source
        )
        vector_future = pool.submit(
            retrieve_vector, query, collection, vault, source=source, top_k=top_k
        )
        return {"graph": graph_future.result(), "vector": vector_future.result()}


def renumber(*retrievals: Retrieval) -> List[Dict[str, Any]]:
    """Give the citations of several retrievals one continuous numbering.

    Each retrieval's own citation numbers stay valid inside its own answer, so
    this returns the merged list for the rail without mutating the originals.
    """
    merged: List[Dict[str, Any]] = []
    for retrieval in retrievals:
        for citation in retrieval.citations:
            copy = dict(citation)
            copy["index"] = len(merged) + 1
            copy["local_index"] = citation.get("index")
            copy["side"] = retrieval.mode
            merged.append(copy)
    return merged


def compute_grounding(text: str) -> float:
    """Share of answer sentences that carry at least one [n] citation."""
    if not text:
        return 0.0
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 2]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[\d+\]", s))
    return round(cited / len(sentences), 2)


def suggested_prompts(data: GraphData, limit: int = 4) -> List[Dict[str, str]]:
    """Build starter questions from the corpus that is actually indexed.

    Falls back to nothing when the graph is empty, so the empty state never
    advertises questions the app cannot answer.
    """
    if data.is_empty:
        return []

    prompts: List[Dict[str, str]] = []

    if not data.reports.empty:
        reports = data.reports.sort_values("rank", ascending=False).to_dict("records")
        if reports:
            prompts.append({
                "text": f"What are the main findings about {_phrase(reports[0]['title'])}?",
                "mode": "global",
            })
        if len(reports) > 1:
            prompts.append({
                "text": (
                    f"How do {_phrase(reports[0]['title'])} and "
                    f"{_phrase(reports[1]['title'])} relate?"
                ),
                "mode": "drift",
            })

    if not data.entities.empty and not data.nodes.empty:
        ranked = data.nodes.sort_values("degree", ascending=False).to_dict("records")
        if ranked:
            prompts.append({
                "text": f"What is {_phrase(ranked[0]['title'])} and what does it connect to?",
                "mode": "local",
            })

    if not data.reports.empty:
        prompts.append({
            "text": "What are the core themes across this corpus?",
            "mode": "global",
        })

    deduped: List[Dict[str, str]] = []
    seen = set()
    for prompt in prompts:
        if prompt["text"].lower() in seen:
            continue
        seen.add(prompt["text"].lower())
        deduped.append(prompt)
    return deduped[:limit]


def _phrase(title: Any) -> str:
    """Turn an upper-case graph title into something readable inside a sentence."""
    text = str(title or "").strip()
    if not text:
        return "this corpus"
    if text.isupper():
        return text.lower()
    return text
