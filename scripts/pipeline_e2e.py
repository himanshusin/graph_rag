#!/usr/bin/env python3
"""Live end-to-end test of indexing and answering.

Builds a real knowledge base from a sample document in a temporary directory —
real LLM extraction, real community reports, real ChromaDB, real parquet and a
real concept map — then answers a question in every mode and checks that the
citations resolve.

This calls the OpenAI API, so it costs a small amount and is not part of the
free QA suite. Run it after changing the pipeline or the prompts:

    python scripts/pipeline_e2e.py
"""

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from dotenv import load_dotenv

from core import rag
from core.pipeline import GraphRAGEngine
from core.vault import DocumentVault
from ui import components as c

load_dotenv(ROOT_DIR / ".env")

MODEL = "gpt-4o-mini"

SAMPLE_DOCUMENT = """# Fine-tuning and retrieval trade-offs

## Parameter-efficient fine-tuning

LoRA (Low-Rank Adaptation) freezes the pretrained weights of a transformer and
injects trainable rank-decomposition matrices into each layer. It reduces the
number of trainable parameters to between 0.1% and 1% of the base model. QLoRA
extends LoRA by quantizing the frozen base model to 4-bit NormalFloat, which
allows a 65B parameter model to be fine-tuned on a single 48 GB GPU.

Full fine-tuning updates every weight and therefore needs roughly 3 times the
GPU memory of LoRA for the same model size. Adapter modules are the predecessor
of LoRA: small bottleneck layers inserted between transformer blocks.

## Retrieval-augmented generation

Retrieval-Augmented Generation (RAG) retrieves passages from an external corpus
at query time instead of encoding knowledge in the weights. RAG adds roughly
180 to 340 milliseconds of latency per query, but keeps answers current without
retraining. Fine-tuned weights encode a snapshot of the corpus and drift as the
source documents change.

## Benchmark comparison

| Method | Compute | MMLU | Memory |
| --- | --- | --- | --- |
| Full fine-tuning | 1.0x | 64.1 | 80 GB |
| LoRA | 0.8x | 62.9 | 26 GB |
| QLoRA | 0.5x | 62.1 | 14 GB |
| RAG | n/a | 61.4 | 8 GB |

## Evaluation and deployment risk

Quantization below 4 bits shows a documented accuracy cliff on MMLU. The
recommended mitigation is a staged rollout with a regression suite. Knowledge
distillation produces a smaller serving footprint, which lowers inference cost.
"""

TEMPLATE_PHRASES = [
    "This community contains",
    "This strategic domain encompasses",
    "key entities:",
]


class PipelineE2E:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.build: Dict[str, Any] = {}
        self.tmp: Path = Path()

    def log(self, name: str, passed: bool, detail: str, error: str = ""):
        self.results.append({"test": name, "passed": passed, "detail": detail, "error": error})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        if error:
            print(f"       {error}")

    def check(self, name: str, fn: Callable[[], Any]):
        try:
            self.log(name, True, fn() or "ok")
        except AssertionError as exc:
            self.log(name, False, "assertion failed", str(exc))
        except Exception as exc:
            self.log(name, False, "raised",
                     f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")

    # -------------------------------------------------------------------------
    def run_index(self, tmp: Path) -> Dict[str, Any]:
        """Store the sample in a fresh vault, then index the whole vault."""
        vault = DocumentVault(vault_dir=str(tmp / "vault"))
        stored = vault.store_document(
            filename="trade_offs.md",
            content_bytes=SAMPLE_DOCUMENT.encode("utf-8"),
            source_type="txt",
            metadata={"origin": "e2e test"},
        )
        self.vault = vault
        self.stored = stored

        engine = GraphRAGEngine(
            output_dir=str(tmp / "output"),
            chroma_dir=str(tmp / "chromadb"),
            notebook_dir=str(tmp / "notebook"),
            model_name=MODEL,
            temperature=0.0,
        )

        steps: List[str] = []

        def on_progress(fraction: float, message: str, stats: Dict[str, Any]):
            step = stats.get("step")
            if step and step not in steps:
                steps.append(step)
            print(f"       {fraction * 100:5.1f}%  {message}")

        documents = [{
            "id": stored["id"],
            "title": stored["title"],
            "text": vault.get_document_text(stored["id"]),
        }]
        result = engine.build_from_documents(
            documents, max_chunks_per_doc=6, progress_callback=on_progress
        )
        for entry in result["documents"]:
            vault.mark_indexed(entry["document_id"], entry["chunks_indexed"], entry["chunks_total"])
        result["steps"] = steps
        return result

    # -------------------------------------------------------------------------
    def test_artifacts(self):
        tmp, result = self.tmp, self.build
        out = tmp / "output"

        def parquet_files():
            expected = {
                "create_final_entities.parquet": ["title", "type", "description", "text_unit_ids"],
                "create_final_relationships.parquet": ["source", "target", "type", "weight"],
                "create_final_nodes.parquet": ["title", "community", "degree"],
                "create_final_community_reports.parquet": ["community", "title", "summary",
                                                           "rank", "rank_explanation", "findings",
                                                           "full_content"],
            }
            rows = {}
            for name, columns in expected.items():
                path = out / name
                assert path.exists(), f"{name} was not written"
                frame = pd.read_parquet(path)
                assert not frame.empty, f"{name} is empty"
                missing = [x for x in columns if x not in frame.columns]
                assert not missing, f"{name} missing columns {missing}"
                rows[name.replace("create_final_", "").replace(".parquet", "")] = len(frame)
            return ", ".join(f"{k} {v}" for k, v in rows.items())

        def progress_steps():
            steps = result["steps"]
            assert steps == GraphRAGEngine.STEPS, f"steps reported: {steps}"
            return " → ".join(steps)

        def entities_extracted():
            frame = pd.read_parquet(out / "create_final_entities.parquet")
            titles = set(frame["title"])
            assert len(frame) >= 8, f"only {len(frame)} concepts extracted"
            assert all(t == t.upper() for t in titles), "titles are not normalised to upper case"
            assert len(titles) == len(frame.drop_duplicates(subset=["title", "type"])), \
                "duplicate (title, type) rows survived"
            found = [t for t in ("LORA", "QLORA", "RAG", "RETRIEVAL-AUGMENTED GENERATION")
                     if t in titles]
            assert found, f"no expected concept extracted from the sample: {sorted(titles)[:12]}"
            assert all(isinstance(x, (list, tuple)) or hasattr(x, "tolist")
                       for x in frame["text_unit_ids"]), "text unit ids are not lists"
            return f"{len(frame)} concepts including {found}"

        def relationships_consistent():
            entities = pd.read_parquet(out / "create_final_entities.parquet")
            frame = pd.read_parquet(out / "create_final_relationships.parquet")
            titles = set(entities["title"])
            dangling = {s for s in frame["source"] if s not in titles} | \
                       {t for t in frame["target"] if t not in titles}
            assert not dangling, f"relationships point at unknown nodes: {dangling}"
            assert (frame["weight"] >= 1).all() and (frame["weight"] <= 10).all(), "weights out of range"
            assert (frame["source"] != frame["target"]).all(), "self loops present"
            typed = frame[frame["type"].astype(str).str.len() > 0]
            assert len(typed) == len(frame), "some relationships have no type"
            non_default = frame[frame["type"] != "related"]
            assert not non_default.empty, \
                "every relationship fell back to 'related'; the type field is not being extracted"
            return (f"{len(frame)} relationships, {len(non_default)} with a specific type "
                    f"e.g. {sorted(set(non_default['type']))[:4]}")

        def nodes_clustered():
            frame = pd.read_parquet(out / "create_final_nodes.parquet")
            entities = pd.read_parquet(out / "create_final_entities.parquet")
            assert len(frame) == len(entities), "node count does not match the concept count"
            assert frame["degree"].sum() > 0, "every node has degree 0"
            assert frame["community"].nunique() >= 1
            assert "community_rank" in frame.columns, "nodes carry no report rank for size-by-rank"
            assert frame["community_rank"].max() > 0
            return (f"{len(frame)} nodes in {frame['community'].nunique()} communities, "
                    f"max degree {int(frame['degree'].max())}")

        def reports_are_llm_written():
            frame = pd.read_parquet(out / "create_final_community_reports.parquet")
            assert not frame.empty, "no community reports written"
            for phrase in TEMPLATE_PHRASES:
                offenders = frame[frame["summary"].str.contains(phrase, regex=False, na=False)]
                assert offenders.empty, \
                    f"placeholder summary still generated: {offenders.iloc[0]['summary'][:90]}"
            assert not frame["title"].str.match(r"^(Domain|Community)\s+\d+:").any(), \
                "titles still use the 'Community N:' template"
            ranks = set(frame["rank"])
            assert ranks != {7.5}, "every report still carries the hard-coded rank 7.5"
            assert all(1.0 <= r <= 10.0 for r in ranks), f"ranks out of range: {ranks}"
            assert frame["rank_explanation"].str.len().min() > 0, "missing rank explanations"
            for raw in frame["findings"]:
                findings = json.loads(raw)
                assert isinstance(findings, list) and findings, "findings is not a populated list"
                assert "summary" in findings[0], findings[0]
            longest = frame.loc[frame["summary"].str.len().idxmax()]
            assert len(longest["summary"]) > 80, "summaries are too short to be real reports"
            assert "## Summary" in longest["full_content"], "full content is not a structured report"
            return (f"{len(frame)} reports, ranks {sorted(ranks)}, "
                    f"e.g. {longest['title']!r}")

        def vectors_carry_provenance():
            import chromadb
            client = chromadb.PersistentClient(path=str(tmp / "chromadb"))
            collection = client.get_or_create_collection("paper_collection")
            count = collection.count()
            assert count == result["chunks_count"], \
                f"chroma has {count} chunks, expected {result['chunks_count']}"
            got = collection.get(include=["metadatas"])
            metadatas = [m or {} for m in got["metadatas"]]
            assert all(m.get("document_id") == self.stored["id"] for m in metadatas), \
                "chunks are missing document_id"
            assert all(m.get("document_title") for m in metadatas), "chunks are missing the title"
            assert all("::chunk_" in i for i in got["ids"]), \
                f"chunk ids are not namespaced per document: {got['ids'][:3]}"
            return f"{count} chunks, all tagged with document_id and a namespaced id"

        def concept_map_written():
            path = tmp / "notebook" / "interactive_graph.html"
            assert path.exists(), "concept map was not written"
            html = path.read_text(encoding="utf-8")
            assert "#FFFFFF" in html and "#D6D6D0" in html
            assert "selectNode" in html and "Geist" in html
            assert (tmp / "notebook" / "knowledge_graph.png").exists(), "static graph missing"
            return f"{path.stat().st_size // 1024} KB light-theme map with a selection handler"

        def vault_index_state():
            document = self.vault.get_document(self.stored["id"])
            assert document["status"] == "Indexed", document["status"]
            assert document["chunks_indexed"] == result["chunks_count"], document
            assert document["chunks_total"] >= document["chunks_indexed"]
            assert document["indexed_at"]
            tables = self.vault.get_tables(self.stored["id"])
            assert tables, "the benchmark table was not extracted"
            assert tables[0]["columns"] == ["Method", "Compute", "MMLU", "Memory"], tables[0]["columns"]
            assert tables[0]["rows_count"] == 4, tables[0]["rows_count"]
            return (f"chunks {document['chunks_indexed']}/{document['chunks_total']} indexed, "
                    f"{len(tables)} table with {tables[0]['rows_count']} rows")

        for name, fn in [
            ("Parquet artifacts", parquet_files),
            ("Progress steps reported", progress_steps),
            ("Concepts extracted", entities_extracted),
            ("Relationships consistent and typed", relationships_consistent),
            ("Communities and degrees", nodes_clustered),
            ("Community reports are written, not templated", reports_are_llm_written),
            ("Vector store provenance", vectors_carry_provenance),
            ("Concept map artifact", concept_map_written),
            ("Vault index state", vault_index_state),
        ]:
            self.check(name, fn)

    # -------------------------------------------------------------------------
    def test_answers(self):
        """Ask a real question in every mode and verify the citations resolve."""
        tmp = self.tmp
        out = tmp / "output"
        graph = rag.GraphData(
            entities=pd.read_parquet(out / "create_final_entities.parquet"),
            relationships=pd.read_parquet(out / "create_final_relationships.parquet"),
            nodes=pd.read_parquet(out / "create_final_nodes.parquet"),
            reports=pd.read_parquet(out / "create_final_community_reports.parquet"),
        )
        import chromadb
        from langchain_openai import ChatOpenAI

        collection = chromadb.PersistentClient(
            path=str(tmp / "chromadb")
        ).get_or_create_collection("paper_collection")
        llm = ChatOpenAI(model=MODEL, temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))
        question = "How much GPU memory does LoRA save compared with full fine-tuning?"

        def answer_mode(mode: str) -> str:
            retrieval = rag.retrieve(
                mode, question, data=graph, collection=collection,
                vault=self.vault, source="all", top_k=4,
            )
            assert retrieval.citations, f"{mode}: no citations retrieved"
            indices = [x["index"] for x in retrieval.citations]
            assert indices == list(range(1, len(indices) + 1)), f"{mode}: {indices}"

            text = rag.answer(retrieval, llm)
            assert len(text) > 80, f"{mode}: answer too short: {text!r}"

            marked = c.mark_citations(text, retrieval.citations)
            cited = {int(n) for n in rag.re.findall(r"\[(\d+)\]", text)}
            resolvable = {n for n in cited if n <= len(retrieval.citations)}
            assert resolvable, f"{mode}: the answer cited nothing resolvable ({cited})"
            for index in resolvable:
                assert f'href="#cite-{index}"' in marked, f"{mode}: [{index}] did not become a link"
            unresolved = cited - resolvable
            for index in unresolved:
                assert f'href="#cite-{index}"' not in marked, \
                    f"{mode}: [{index}] linked to a card that does not exist"

            grounding = rag.compute_grounding(text)
            assert grounding > 0, f"{mode}: nothing in the answer is cited"
            return (f"{len(retrieval.citations)} citations, {len(resolvable)} used, "
                    f"grounding {int(grounding * 100)}%, {len(text)} chars")

        for mode in ("global", "local", "drift", "vector"):
            self.check(f"Answer · {mode}", lambda m=mode: answer_mode(m))

        def compare_mode():
            sides = rag.retrieve_compare(
                question, data=graph, collection=collection, vault=self.vault, top_k=4
            )
            graph_text = rag.answer(sides["graph"], llm)
            vector_text = rag.answer(sides["vector"], llm)
            assert len(graph_text) > 80 and len(vector_text) > 80
            merged = rag.renumber(sides["graph"], sides["vector"])
            assert [x["index"] for x in merged] == list(range(1, len(merged) + 1))
            # each column's own answer must still resolve against its own citations
            for side, text in (("graph", graph_text), ("vector", vector_text)):
                marked = c.mark_citations(text, sides[side].citations)
                cited = {int(n) for n in rag.re.findall(r"\[(\d+)\]", text)}
                for index in cited:
                    if index <= len(sides[side].citations):
                        assert f'href="#cite-{index}"' in marked, f"{side}: [{index}] unresolved"
            g_ground = rag.compute_grounding(graph_text)
            v_ground = rag.compute_grounding(vector_text)
            return (f"graph {len(sides['graph'].citations)} citations grounded "
                    f"{int(g_ground * 100)}%, vector {len(sides['vector'].citations)} citations "
                    f"grounded {int(v_ground * 100)}%")

        self.check("Answer · compare", compare_mode)

        def numbers_survive():
            """A quantitative question must come back with figures from the source."""
            retrieval = rag.retrieve("vector", "What is the memory use of LoRA and QLoRA?",
                                     data=graph, collection=collection, vault=self.vault, top_k=4)
            text = rag.answer(retrieval, llm)
            figures = [f for f in ("26 GB", "14 GB", "80 GB", "3 times", "48 GB", "0.8x", "0.5x")
                       if f.lower() in text.lower()]
            assert figures, f"no source figure survived into the answer: {text[:200]}"
            return f"answer quotes {figures}"

        self.check("Quantitative grounding", numbers_survive)

        def prompts_from_corpus():
            prompts = rag.suggested_prompts(graph)
            assert prompts, "no starter prompts derived"
            joined = " ".join(p["text"] for p in prompts)
            assert "RAG and fine-tuning" not in joined, "prompts are the old hard-coded set"
            return f"{len(prompts)} prompts, e.g. {prompts[0]['text']!r}"

        self.check("Suggested prompts from the built graph", prompts_from_corpus)

    # -------------------------------------------------------------------------
    def run_all(self) -> bool:
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set; the live pipeline test cannot run.")
            return False

        print("=" * 72)
        print("GraphRAG knowledge workspace · live indexing and answering test")
        print("=" * 72)
        with tempfile.TemporaryDirectory() as tmp:
            self.tmp = Path(tmp)
            print("\n--- Indexing the sample document ---")
            try:
                self.build = self.run_index(self.tmp)
            except Exception as exc:
                self.log("Indexing run", False, "raised",
                         f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}")
                return False
            self.log("Indexing run", True,
                     f"{self.build['entities_count']} concepts, "
                     f"{self.build['relationships_count']} relationships, "
                     f"{self.build['communities_count']} communities, "
                     f"{self.build['chunks_count']} chunks in {self.build['elapsed_seconds']}s")

            print("\n--- Artifacts ---")
            self.test_artifacts()
            print("\n--- Answers ---")
            self.test_answers()

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        print("\n" + "=" * 72)
        print(f"SUMMARY: {passed} / {total} checks passed ({total - passed} failures)")
        print("=" * 72)
        if passed != total:
            for result in self.results:
                if not result["passed"]:
                    print(f"  - {result['test']}: {result['error']}")
            return False
        return True


if __name__ == "__main__":
    sys.exit(0 if PipelineE2E().run_all() else 1)
