#!/usr/bin/env python3
"""End-to-end QA suite for the GraphRAG knowledge workspace.

Checks the design tokens and components against the mockups and the developer
guide, then exercises retrieval, the indexing frame builders, the vault
lifecycle and the governance checks against real files in a temporary root.

No test calls an LLM, so the suite is free to run and deterministic.
"""

import json
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from langchain_core.runnables import RunnableLambda

from core import rag
from core.change_manager import ChangeManagementAgent
from core.pipeline import (
    COMMUNITY_PALETTE, CommunityReportSynthesizer, EntityRelationshipExtractor,
    GraphRAGEngine, _fallback_rank, write_concept_map,
)
from core.vault import DocumentVault
from ui import components as c
from ui import tokens

THEME_CSS = (ROOT_DIR / "ui" / "theme.css").read_text(encoding="utf-8")

# Tokens from the developer guide §2 that must appear verbatim in the sheet.
GUIDE_TOKENS = {
    "--k-canvas": "#F7F7F5",
    "--k-panel": "#FFFFFF",
    "--k-hover": "#F1F1EE",
    "--k-border": "#E8E8E4",
    "--k-border-strong": "#E0E0DB",
    "--k-border-input": "#D6D6D0",
    "--k-text": "#111110",
    "--k-muted": "#55554F",
    "--k-faint": "#8A8A83",
    "--k-accent": "#3056D3",
    "--k-accent-soft": "#EEF2FD",
    "--k-accent-border": "#C9D3F5",
    "--k-accent-bg": "#FBFCFF",
    "--k-success": "#1F8A5B",
    "--k-success-bg": "#E8F5EE",
    "--k-warn": "#B7791F",
    "--k-warn-bg": "#FBF3E4",
    "--k-danger": "#B03A2E",
    "--k-highlight": "#FFF0B3",
}

# Pictographs, dingbats and variation selectors. The check marks the mockup
# itself uses for QA rows are allowed; nothing else is.
EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️⬀-⯿]")
ALLOWED_GLYPHS = {"✓", "✕"}  # the mockup's check and cross marks


def find_emoji(text: str) -> List[str]:
    return [g for g in EMOJI_RE.findall(text) if g not in ALLOWED_GLYPHS]


SOURCE_FILES = ["app.py", "core/pipeline.py", "core/rag.py", "core/vault.py",
                "core/change_manager.py", "ui/components.py", "ui/tokens.py",
                "ui/data.py", "screens/search.py", "screens/vault.py",
                "screens/concept_map.py", "screens/catalog.py", "screens/governance.py"]


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
def sample_graph() -> rag.GraphData:
    """A small graph where the relevant concept is deliberately not row zero.

    Local lookup used to feed the model ``entities.head(15)``, so a fixture whose
    first rows are irrelevant catches that regression.
    """
    entities = pd.DataFrame([
        {"id": "e0", "human_readable_id": 0, "title": "DATA CLEANING", "type": "TECHNIQUE",
         "description": "Removing duplicates and noise from training corpora before training.",
         "text_unit_ids": ["doc_a::chunk_0"]},
        {"id": "e1", "human_readable_id": 1, "title": "REVENUE", "type": "METRIC",
         "description": "Quarterly revenue reported at 4.2 billion dollars.",
         "text_unit_ids": ["doc_a::chunk_1"]},
        {"id": "e2", "human_readable_id": 2, "title": "LORA", "type": "METHOD",
         "description": "Low-Rank Adaptation freezes pretrained weights and injects "
                        "trainable rank-decomposition matrices, cutting trainable parameters to 0.1%.",
         "text_unit_ids": ["doc_a::chunk_2", "doc_a::chunk_3"]},
        {"id": "e3", "human_readable_id": 3, "title": "GPU MEMORY", "type": "RESOURCE",
         "description": "Accelerator memory budget, reduced up to 3x by adapter methods.",
         "text_unit_ids": ["doc_a::chunk_2"]},
        {"id": "e4", "human_readable_id": 4, "title": "QLORA", "type": "METHOD",
         "description": "Quantized LoRA using 4-bit NormalFloat for the frozen base model.",
         "text_unit_ids": ["doc_a::chunk_3"]},
    ])
    relationships = pd.DataFrame([
        {"id": "r0", "human_readable_id": 0, "source": "DATA CLEANING", "target": "REVENUE",
         "type": "improves", "description": "Cleaner data improved reported revenue quality.",
         "weight": 4.0, "combined_degree": 1, "text_unit_ids": ["doc_a::chunk_1"]},
        {"id": "r1", "human_readable_id": 1, "source": "LORA", "target": "GPU MEMORY",
         "type": "reduces", "description": "LoRA reduces GPU memory use by up to 3x versus full fine-tuning.",
         "weight": 8.0, "combined_degree": 2, "text_unit_ids": ["doc_a::chunk_2"]},
        {"id": "r2", "human_readable_id": 2, "source": "LORA", "target": "QLORA",
         "type": "extends", "description": "QLoRA extends LoRA with 4-bit quantization of the base.",
         "weight": 9.0, "combined_degree": 2, "text_unit_ids": ["doc_a::chunk_3"]},
    ])
    nodes = pd.DataFrame([
        {"id": "e0", "human_readable_id": 0, "title": "DATA CLEANING", "community": 1,
         "level": 0, "degree": 1, "x": 0.0, "y": 0.0, "community_rank": 4.0},
        {"id": "e1", "human_readable_id": 1, "title": "REVENUE", "community": 1,
         "level": 0, "degree": 1, "x": 0.1, "y": 0.1, "community_rank": 4.0},
        {"id": "e2", "human_readable_id": 2, "title": "LORA", "community": 0,
         "level": 0, "degree": 2, "x": 0.2, "y": 0.2, "community_rank": 8.5},
        {"id": "e3", "human_readable_id": 3, "title": "GPU MEMORY", "community": 0,
         "level": 0, "degree": 1, "x": 0.3, "y": 0.3, "community_rank": 8.5},
        {"id": "e4", "human_readable_id": 4, "title": "QLORA", "community": 0,
         "level": 0, "degree": 1, "x": 0.4, "y": 0.4, "community_rank": 8.5},
    ])
    reports = pd.DataFrame([
        {"id": "c0", "human_readable_id": 0, "community": 0, "parent": -1, "level": 0,
         "title": "Parameter-efficient fine-tuning and memory",
         "summary": "LoRA and QLoRA cut trainable parameters and reduce GPU memory by up to 3x.",
         "full_content": "# Parameter-efficient fine-tuning\nLoRA reduces GPU memory.",
         "rank": 8.5, "rank_explanation": "Central to the corpus.",
         "findings": json.dumps([{"summary": "LoRA cuts memory", "explanation": "up to 3x"}]),
         "full_content_json": "{}", "period": "2026-09-17", "size": 3},
        {"id": "c1", "human_readable_id": 1, "community": 1, "parent": -1, "level": 0,
         "title": "Data preparation and revenue reporting",
         "summary": "Data cleaning practices and the 4.2 billion dollar revenue figure.",
         "full_content": "# Data preparation\nCleaning improves revenue quality.",
         "rank": 4.0, "rank_explanation": "Peripheral.",
         "findings": json.dumps([{"summary": "Cleaning matters", "explanation": "quality"}]),
         "full_content_json": "{}", "period": "2026-09-17", "size": 2},
    ])
    return rag.GraphData(entities=entities, relationships=relationships,
                         nodes=nodes, reports=reports)


class FakeCollection:
    """Stand-in for a Chroma collection with controllable distances."""

    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents

    def count(self) -> int:
        return len(self.documents)

    def query(self, query_texts, n_results, include=None):
        window = self.documents[:n_results]
        return {
            "documents": [[d["text"] for d in window]],
            "metadatas": [[d["meta"] for d in window]],
            "ids": [[d["id"] for d in window]],
            "distances": [[d["distance"] for d in window]],
        }


def fake_llm(text: str) -> RunnableLambda:
    """A runnable that satisfies ``prompt | llm | StrOutputParser()``."""
    return RunnableLambda(lambda _prompt: text)


# -----------------------------------------------------------------------------
# Harness
# -----------------------------------------------------------------------------
class UIQAAgent:
    def __init__(self, root_dir: Path = ROOT_DIR):
        self.root_dir = root_dir
        self.vault = DocumentVault(vault_dir=str(root_dir / "vault"))
        self.change_agent = ChangeManagementAgent(root_dir=str(root_dir))
        self.results: List[Dict[str, Any]] = []

    def log_check(self, category: str, test_name: str, passed: bool, detail: str, error: str = ""):
        self.results.append({
            "category": category, "test": test_name,
            "passed": bool(passed), "detail": detail, "error": error,
        })
        print(f"[{'PASS' if passed else 'FAIL'}] {category} | {test_name}: {detail}")
        if error:
            print(f"       {error}")

    def check(self, category: str, name: str, fn: Callable[[], Any]):
        """Run one assertion block. ``fn`` returns a detail string or raises."""
        try:
            detail = fn()
            self.log_check(category, name, True, detail or "ok")
        except AssertionError as exc:
            self.log_check(category, name, False, "assertion failed", str(exc))
        except Exception as exc:
            self.log_check(category, name, False, "raised",
                           f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")

    # -------------------------------------------------------------------------
    # 1. Design tokens and acceptance checklist
    # -------------------------------------------------------------------------
    def audit_design_tokens(self):
        print("\n--- 1. Design tokens and the acceptance checklist ---")

        def theme_config():
            path = self.root_dir / ".streamlit/config.toml"
            assert path.exists(), "missing .streamlit/config.toml"
            text = path.read_text(encoding="utf-8")
            assert 'base = "light"' in text or 'base="light"' in text, "theme is not light"
            assert "#3056D3" in text, "primaryColor is not the accent"
            assert "#F7F7F5" in text, "backgroundColor is not the canvas"
            return "light theme, accent #3056D3, canvas #F7F7F5"

        def guide_tokens():
            missing = [
                f"{name}:{value}" for name, value in GUIDE_TOKENS.items()
                if f"{name}:{value}" not in THEME_CSS.replace(" ", "")
            ]
            assert not missing, f"tokens absent or wrong: {missing}"
            return f"all {len(GUIDE_TOKENS)} guide §2 color tokens declared verbatim"

        def typography():
            assert "family=Geist:wght@400;500;600" in THEME_CSS, "Geist weights not imported"
            assert "Geist+Mono:wght@400;500" in THEME_CSS, "Geist Mono not imported"
            assert "Plus Jakarta" not in THEME_CSS, "old display face still referenced"
            assert "JetBrains" not in THEME_CSS, "old mono face still referenced"
            weights = set(re.findall(r"font-weight:\s*(\d{3})", THEME_CSS))
            weights |= set(re.findall(r"font:\s*(\d{3})\s", THEME_CSS))
            illegal = weights - {"400", "450", "500", "600"}
            assert not illegal, f"weights outside 400/500/600: {sorted(illegal)}"
            return f"Geist + Geist Mono, weights {sorted(weights)}"

        def palette():
            expected = ["#3056D3", "#1F8A5B", "#B7791F", "#8A8A83",
                        "#B03A2E", "#6B4FBB", "#2A8FA8", "#C25E9A"]
            assert tokens.COMMUNITY_COLORS == expected, tokens.COMMUNITY_COLORS
            assert COMMUNITY_PALETTE == expected, COMMUNITY_PALETTE
            assert len(expected) == 8, "palette must cap at 8 hues"
            return "8 community hues, identical in ui.tokens and core.pipeline"

        def no_emoji():
            offenders = {}
            for name in SOURCE_FILES:
                found = find_emoji((self.root_dir / name).read_text(encoding="utf-8"))
                if found:
                    offenders[name] = found
            assert not offenders, f"emoji present: {offenders}"
            css_emoji = find_emoji(THEME_CSS)
            assert not css_emoji, f"emoji in theme.css: {css_emoji}"
            return f"0 emoji across {len(SOURCE_FILES)} source files and theme.css"

        def no_gradients_or_glows():
            lowered = THEME_CSS.lower()
            assert "gradient" not in lowered, "gradient present in the stylesheet"
            shadows = re.findall(r"box-shadow:([^;]+);", lowered)
            allowed = ("0 1px 2px", "0 8px 24px", "0 1px 3px", "none")
            bad = [s.strip() for s in shadows if not any(a in s for a in allowed)]
            assert not bad, f"shadows outside the guide: {bad}"
            assert "text-transform:uppercase" in lowered.replace(" ", ""), \
                "eyebrow uppercase rule missing"
            return f"no gradients, {len(shadows)} shadows all within the guide"

        def responsive_and_motion():
            assert "max-width:1200px" in THEME_CSS, "1200px breakpoint missing"
            assert "max-width:900px" in THEME_CSS, "900px breakpoint missing"
            assert "prefers-reduced-motion" in THEME_CSS, "reduced-motion block missing"
            assert "focus-visible" in THEME_CSS, "focus ring missing"
            assert "outline:2px solid var(--k-accent); outline-offset:2px" in THEME_CSS, \
                "focus ring is not 2px accent at 2px offset"
            assert "min-height:32px" in THEME_CSS, "32px hit targets missing"
            return "1200 / 900 breakpoints, reduced motion, 2px accent focus ring, 32px targets"

        def danger_tag_exists():
            # grounding_tag emits k-tag--danger below 70%, so the class must exist.
            assert ".k-tag--danger" in THEME_CSS, "k-tag--danger is emitted but never styled"
            assert ".k-tag--warn" in THEME_CSS and ".k-tag--accent" in THEME_CSS
            return "ok / warn / danger / accent tag variants all styled"

        for name, fn in [
            ("Streamlit light theme", theme_config),
            ("Guide §2 color tokens", guide_tokens),
            ("Typography", typography),
            ("Community palette", palette),
            ("No emoji in UI or copy", no_emoji),
            ("No gradients, glows or stray shadows", no_gradients_or_glows),
            ("Responsive and motion rules", responsive_and_motion),
            ("Grounding tag variants styled", danger_tag_exists),
        ]:
            self.check("Design", name, fn)

    # -------------------------------------------------------------------------
    # 2. Citations, grounding and components
    # -------------------------------------------------------------------------
    def audit_components(self):
        print("\n--- 2. Citations, grounding and components ---")

        def citation_resolution():
            citations = [
                {"index": 1, "kind": "Domain", "name": "PEFT"},
                {"index": 2, "kind": "Table", "name": "Benchmarks"},
            ]
            text = "RAG is cheaper [1] but fine-tuning wins on latency [2]. Unsupported [7]."
            out = c.mark_citations(text, citations)
            assert 'href="#cite-1"' in out and 'href="#cite-2"' in out, out
            assert "[7]" in out and 'href="#cite-7"' not in out, \
                "an unresolvable number must stay plain text"
            assert 'aria-label="Source 1: Domain PEFT"' in out, "aria-label missing"
            assert out.count("k-cite") == 2
            return "resolvable [n] link to #cite-n with aria-labels; [7] left plain"

        def citation_ids_match_cards():
            citations = [{"index": 1, "kind": "Concept", "name": "LORA", "excerpt": "x"}]
            rail = c.evidence_rail(citations)
            marked = c.mark_citations("Claim [1].", citations)
            assert 'id="cite-1"' in rail, rail
            assert 'href="#cite-1"' in marked
            return "every [n] has a matching card id in the rail"

        def grounding_thresholds():
            assert "grounded 100%" in c.grounding_tag(1.0)
            assert "k-tag--warn" not in c.grounding_tag(0.95)
            assert "k-tag--warn" in c.grounding_tag(0.75), c.grounding_tag(0.75)
            assert "k-tag--danger" in c.grounding_tag(0.40), c.grounding_tag(0.40)
            # boundaries
            assert "k-tag--warn" in c.grounding_tag(0.89)
            assert "k-tag--warn" not in c.grounding_tag(0.90)
            assert "k-tag--danger" in c.grounding_tag(0.69)
            return ">=90 ok, 70-89 warn, <70 danger, boundaries included"

        def grounding_formula():
            text = "First claim [1]. Second claim without a source. Third one [2]."
            assert rag.compute_grounding(text) == 0.67, rag.compute_grounding(text)
            assert rag.compute_grounding("No citations at all.") == 0.0
            assert rag.compute_grounding("") == 0.0
            assert rag.compute_grounding("Only cited [1].") == 1.0
            return "cited sentences / total sentences, 2 of 3 = 0.67"

        def evidence_card_bodies():
            passage = c.evidence_card({
                "index": 1, "kind": "Passage", "name": "Guide.pdf", "meta": "chunk 7",
                "quote": "fine-tuned weights encode a snapshot", "distance": 0.82,
                "document_id": "doc_abc",
            })
            assert "k-ev__quote" in passage, "passage body is not the mono quote block"
            assert "screen=vault&doc=doc_abc" in passage, "passage does not open the vault"

            table = c.evidence_card({
                "index": 2, "kind": "Table", "name": "Benchmarks", "meta": "p. 42",
                "columns": ["Method", "Compute", "MMLU"],
                "records": [{"Method": "LoRA", "Compute": "0.8x", "MMLU": "62.9"}],
                "rows_count": 6, "document_id": "doc_abc", "table_id": "t1",
            })
            assert "k-ev__minitable" in table and "<th>Method</th>" in table, table
            assert "5 more rows" in table, "mini table must report the remaining rows"

            concept = c.evidence_card({
                "index": 3, "kind": "Concept", "name": "LORA", "excerpt": "Low-Rank Adaptation",
                "entity_type": "METHOD", "degree": 14,
            })
            assert "k-ev__text" in concept and "14 relationships" in concept
            assert "screen=concepts&node=LORA" in concept, "concept does not open the map"

            domain = c.evidence_card({"index": 4, "kind": "Domain", "name": "PEFT",
                                      "community": 3, "excerpt": "summary"})
            assert "tab=briefs" in domain and "community=3" in domain
            return "passage, table, concept and domain cards each render their own body and link"

        def rail_strikes_disallowed_kinds():
            citations = [
                {"index": 1, "kind": "Concept", "name": "LORA", "excerpt": "x"},
                {"index": 2, "kind": "Passage", "name": "Guide", "quote": "y"},
            ]
            rail = c.evidence_rail(citations, allowed_kinds=rag.SOURCE_KINDS["documents"])
            assert rail.count("k-ev--muted") == 1, rail
            assert ".k-ev--muted .k-ev__title { text-decoration:line-through; }" in THEME_CSS \
                or "text-decoration:line-through" in THEME_CSS
            return "kinds excluded by the source filter render struck through"

        def escaping():
            hostile = '<img src=x onerror="alert(1)">'
            card = c.evidence_card({"index": 1, "kind": "Concept", "name": hostile,
                                    "excerpt": hostile})
            assert "<img" not in card, card
            assert "&lt;img" in card
            marked = c.mark_citations("text", [{"index": 1, "kind": "X", "name": hostile}])
            assert "<img" not in marked
            table = c.mini_table(["<b>col</b>"], [{"<b>col</b>": "<i>v</i>"}], 1)
            assert "<b>" not in table.replace("<b>col</b>", "")
            return "hostile markup in graph titles is escaped everywhere"

        def modes_and_copy():
            assert [k for k, _ in c.MODES] == ["global", "local", "drift", "vector", "compare"]
            labels = [label for _k, label in c.MODES]
            assert labels == ["Global synthesis", "Local lookup", "DRIFT deep-dive",
                             "Vector passages", "Compare graph vs vector"], labels
            for banned in ("Enterprise", "Executive", "Strategic"):
                for name in ["ui/components.py", "screens/search.py", "ui/tokens.py"]:
                    text = (self.root_dir / name).read_text(encoding="utf-8")
                    assert banned not in text, f"{banned!r} still in {name}"
            return "five plain mode names, no Enterprise / Executive / Strategic prefixes"

        def mode_sync():
            sidebar = c.reasoning_modes("drift")
            composer = c.composer_modes("drift")
            assert 'mode=drift' in sidebar and "k-mode-dot--active" in sidebar
            assert 'mode=drift' in composer and "k-pill--active" in composer
            # both write the same query key, so they cannot diverge
            assert sidebar.count("mode=") == 5 and composer.count("mode=") == 5
            return "sidebar and composer both drive ?mode=, so they stay in sync"

        def states():
            assert "Ask the knowledge base." in c.suggested_prompts([])
            assert "Add a document to start." in c.suggested_prompts([])
            prompts = c.suggested_prompts([{"text": "What is LoRA?", "mode": "local"}])
            assert "prompt=1" in prompts and "Local" in prompts
            skeleton = c.answer_skeleton()
            assert skeleton.count("k-skel__bar") == 3, "loading state needs three bars"
            assert 'role="status"' in skeleton
            error = c.error_block("Couldn't complete this search.", "RuntimeError: boom")
            assert tokens.esc("Couldn't complete this search.") in error, error
            assert "k-error__detail" in error and "RuntimeError: boom" in error
            assert c.evidence_skeleton(2).count("k-skel--card") == 2
            return "empty, loading and error states match §6"

        def check_list_order():
            checks = [
                {"label": "A passes", "ok": True, "detail": "1 row", "duration": 0.1},
                {"label": "B fails", "ok": False, "detail": "missing", "error": "boom", "duration": 0.2},
            ]
            html = c.check_list(checks)
            assert html.index("B fails") < html.index("A passes"), "failures must sort first"
            assert "k-check__error" in html and "boom" in html
            return "failed checks first, with the error text beneath"

        def status_never_colour_only():
            for kind, label in [("ok", "Indexed"), ("progress", "Extracting 3 / 5"),
                                ("warn", "Needs re-index"), ("danger", "Failed")]:
                html = c.status_dot(kind, label)
                assert label in html, f"{kind} dot has no text label"
            return "every status dot carries its text label"

        for name, fn in [
            ("Citation resolution", citation_resolution),
            ("Citation ids match rail cards", citation_ids_match_cards),
            ("Grounding tag thresholds", grounding_thresholds),
            ("Grounding formula", grounding_formula),
            ("Evidence card bodies and links", evidence_card_bodies),
            ("Source filter strikes disallowed kinds", rail_strikes_disallowed_kinds),
            ("HTML escaping", escaping),
            ("Mode naming and copy rules", modes_and_copy),
            ("Sidebar and composer mode sync", mode_sync),
            ("Empty, loading and error states", states),
            ("QA check list ordering", check_list_order),
            ("Status is never colour-only", status_never_colour_only),
        ]:
            self.check("Components", name, fn)

    # -------------------------------------------------------------------------
    # 3. Retrieval accuracy
    # -------------------------------------------------------------------------
    def audit_retrieval(self):
        print("\n--- 3. Retrieval accuracy ---")
        graph = sample_graph()

        def local_picks_relevant_not_first():
            result = rag.retrieve_local("How does LoRA reduce GPU memory?", graph)
            names = [x["name"] for x in result.citations]
            assert "LORA" in names[:2], f"LoRA not ranked into the top 2: {names}"
            assert "GPU MEMORY" in names[:3], f"GPU MEMORY missing: {names}"
            assert names[0] != "DATA CLEANING", \
                "first row of the frame is being returned instead of the relevant concept"
            assert "REVENUE" not in names, f"irrelevant concept retrieved: {names}"
            return f"ranked {names[:3]} for a LoRA question, irrelevant rows dropped"

        def local_includes_touching_relationships():
            result = rag.retrieve_local("How does LoRA reduce GPU memory?", graph)
            assert result.counts["relationships"] >= 1, result.counts
            assert "reduces" in result.context, "relationship type missing from the context"
            assert "DATA CLEANING improves REVENUE" not in result.context, \
                "relationships unrelated to the selected concepts leaked in"
            return f"{result.counts['relationships']} relationships, typed, all touching the selection"

        def local_fallback_when_nothing_matches():
            result = rag.retrieve_local("zzzz unrelated question", graph)
            assert result.citations, "a broad question should fall back to the graph backbone"
            assert result.citations[0]["name"] == "LORA", \
                f"fallback should lead with the most connected node, got {result.citations[0]['name']}"
            return "no lexical match falls back to the highest-degree concepts"

        def global_ranks_by_relevance_then_rank():
            result = rag.retrieve_global("What reduces GPU memory?", graph)
            assert result.citations[0]["name"] == "Parameter-efficient fine-tuning and memory", \
                result.citations[0]["name"]
            assert "rank 8.5" in result.citations[0]["meta"]
            broad = rag.retrieve_global("summarise everything", graph)
            assert broad.citations[0]["meta"] == "rank 8.5", \
                "importance should break ties for a broad question"
            return "relevance first, report importance as the tie-breaker"

        def drift_drills_into_chosen_domains():
            result = rag.retrieve_drift("How does LoRA reduce GPU memory?", graph)
            assert "Stage 1" in result.context and "Stage 2" in result.context
            concepts = [x["name"] for x in result.citations if x["kind"] == "Concept"]
            assert concepts, "stage 2 produced no concepts"
            assert "REVENUE" not in concepts, \
                f"stage 2 leaked concepts from an unselected community: {concepts}"
            kinds = {x["kind"] for x in result.citations}
            assert "Relationship" in kinds, "DRIFT should cite its strongest relationship"
            return f"two stages, concepts {concepts} confined to the selected communities"

        def source_filter_restricts_kinds():
            graph_only = rag.retrieve_local("LoRA", graph, source="graph")
            assert {x["kind"] for x in graph_only.citations} <= {"Concept", "Relationship"}, \
                {x["kind"] for x in graph_only.citations}
            docs_only = rag.retrieve_local("LoRA", graph, source="documents")
            assert not docs_only.citations, \
                "documents-only must not return graph citations"
            assert docs_only.exhausted
            return "graph filter yields graph kinds; documents filter yields none from the graph"

        def citation_numbering_is_contiguous():
            for mode in ("global", "local", "drift"):
                result = rag.retrieve(mode, "LoRA and GPU memory", data=graph)
                indices = [x["index"] for x in result.citations]
                assert indices == list(range(1, len(indices) + 1)), f"{mode}: {indices}"
                for index in indices:
                    assert f"[{index}]" in result.context, \
                        f"{mode}: context has no marker for [{index}]"
            return "every mode numbers citations 1..n and marks each one in the context"

        def vector_drops_far_hits():
            collection = FakeCollection([
                {"id": "doc_a::chunk_0", "text": "LoRA freezes weights.", "distance": 0.6,
                 "meta": {"document_id": "doc_a", "document_title": "Guide.pdf", "chunk_index": 0}},
                {"id": "doc_a::chunk_1", "text": "Unrelated boilerplate.", "distance": 2.4,
                 "meta": {"document_id": "doc_a", "document_title": "Guide.pdf", "chunk_index": 1}},
            ])
            result = rag.retrieve_vector("What is LoRA?", collection, top_k=2)
            assert result.counts["passages"] == 1, result.counts
            assert result.citations[0]["document_id"] == "doc_a", "provenance lost"
            assert result.citations[0]["distance"] == 0.6
            return "hits beyond the distance threshold are dropped, provenance kept"

        def vector_no_results_state():
            collection = FakeCollection([
                {"id": "c0", "text": "far", "distance": 3.0,
                 "meta": {"document_id": "doc_a", "document_title": "Guide.pdf", "chunk_index": 0}},
            ])
            result = rag.retrieve_vector("nothing relevant", collection, top_k=1)
            assert result.exhausted and result.notes, result
            message = rag.answer(result, fake_llm("should not be used"))
            assert "Try Global synthesis" in message, message
            empty = rag.retrieve_vector("q", FakeCollection([]), top_k=4)
            assert empty.exhausted and "no passages yet" in empty.notes[0].lower()
            return "no passage above threshold states so and offers Global synthesis"

        def empty_graph_states():
            empty = rag.GraphData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
            assert empty.is_empty
            result = rag.retrieve_global("anything", empty)
            assert result.exhausted
            message = rag.answer(result, fake_llm("unused"))
            assert "has not been built yet" in message, message
            assert rag.suggested_prompts(empty) == [], "empty graph must advertise no prompts"
            return "missing artifacts produce the empty state, not a crash"

        def compare_renumbering():
            collection = FakeCollection([
                {"id": "doc_a::chunk_0", "text": "LoRA freezes weights.", "distance": 0.5,
                 "meta": {"document_id": "doc_a", "document_title": "Guide.pdf", "chunk_index": 0}},
            ])
            sides = rag.retrieve_compare("LoRA memory", data=graph, collection=collection, top_k=2)
            merged = rag.renumber(sides["graph"], sides["vector"])
            assert [x["index"] for x in merged] == list(range(1, len(merged) + 1))
            assert {x["side"] for x in merged} == {"drift", "vector"}, {x["side"] for x in merged}
            # each side keeps its own numbering so its own answer stays valid
            assert sides["vector"].citations[0]["index"] == 1
            assert merged[-1]["local_index"] == 1
            return f"{len(merged)} merged citations renumbered, per-side numbering preserved"

        def suggested_prompts_use_real_corpus():
            prompts = rag.suggested_prompts(graph)
            assert prompts, "prompts should be derived from the indexed corpus"
            assert len(prompts) <= 4
            joined = " ".join(p["text"] for p in prompts).lower()
            assert "parameter-efficient fine-tuning" in joined, joined
            assert all(p["mode"] in rag.MODE_KEYS for p in prompts)
            assert "rag and fine-tuning" not in joined, \
                "prompts must not be the hard-coded sample set"
            return f"{len(prompts)} prompts built from real report and node titles"

        def answer_stream_prefixes_notes():
            collection = FakeCollection([
                {"id": "c0", "text": "x", "distance": 0.4,
                 "meta": {"document_id": "d", "document_title": "T", "chunk_index": 0}},
                {"id": "c1", "text": "y", "distance": 2.9,
                 "meta": {"document_id": "d", "document_title": "T", "chunk_index": 1}},
            ])
            result = rag.retrieve_vector("q", collection, top_k=2)
            assert result.notes == [], "one good hit should not raise a note"
            text = rag.answer(result, fake_llm("Answer [1]."))
            assert text.strip() == "Answer [1]."
            return "generation streams the model text, notes only when coverage is short"

        for name, fn in [
            ("Local lookup ranks by relevance", local_picks_relevant_not_first),
            ("Local lookup relationship scoping", local_includes_touching_relationships),
            ("Local lookup fallback", local_fallback_when_nothing_matches),
            ("Global ranks relevance then importance", global_ranks_by_relevance_then_rank),
            ("DRIFT two-stage drill-down", drift_drills_into_chosen_domains),
            ("Source filter restricts citation kinds", source_filter_restricts_kinds),
            ("Citation numbering is contiguous", citation_numbering_is_contiguous),
            ("Vector distance threshold", vector_drops_far_hits),
            ("Vector no-results state", vector_no_results_state),
            ("Empty graph states", empty_graph_states),
            ("Compare merges and renumbers", compare_renumbering),
            ("Suggested prompts come from the corpus", suggested_prompts_use_real_corpus),
            ("Answer generation", answer_stream_prefixes_notes),
        ]:
            self.check("Retrieval", name, fn)

    # -------------------------------------------------------------------------
    # 4. Indexing pipeline
    # -------------------------------------------------------------------------
    def audit_pipeline(self):
        print("\n--- 4. Indexing pipeline ---")

        def extractor_parses_typed_relationships():
            parser = EntityRelationshipExtractor.__new__(EntityRelationshipExtractor)
            raw = (
                '("entity"<|delimiter|>LORA<|delimiter|>METHOD<|delimiter|>Low-rank adaptation)\n'
                '("relationship"<|delimiter|>LORA<|delimiter|>GPU MEMORY<|delimiter|>reduces'
                '<|delimiter|>Cuts memory by 3x<|delimiter|>8)\n'
                '("relationship"<|delimiter|>A<|delimiter|>B<|delimiter|>Legacy untyped form'
                '<|delimiter|>5)\n'
            )
            entities, relationships = parser._parse_response(raw, "doc::chunk_0")
            assert len(entities) == 1 and entities[0]["title"] == "LORA"
            assert len(relationships) == 2, relationships
            assert relationships[0]["type"] == "reduces", relationships[0]
            assert relationships[0]["weight"] == 8.0
            assert relationships[1]["type"] == "related", relationships[1]
            assert relationships[1]["description"] == "Legacy untyped form"
            return "6-field typed form and the legacy 5-field form both parse"

        def entities_dedupe():
            raw = [
                {"title": "LORA", "type": "METHOD", "description": "First take.", "chunk_id": "c0"},
                {"title": "LORA", "type": "METHOD", "description": "Second take.", "chunk_id": "c1"},
                {"title": "LORA", "type": "METHOD", "description": "First take.", "chunk_id": "c2"},
                {"title": "LORA", "type": "CONCEPT", "description": "Different type.", "chunk_id": "c0"},
            ]
            frame = GraphRAGEngine._build_entities(raw)
            assert len(frame) == 2, f"expected 2 rows by (title, type), got {len(frame)}"
            method = frame[frame["type"] == "METHOD"].iloc[0]
            assert method["description"] == "First take. Second take.", method["description"]
            assert sorted(method["text_unit_ids"]) == ["c0", "c1", "c2"]
            return "deduped by (title, type), descriptions merged once, chunk ids unioned"

        def relationships_dedupe_and_prune():
            raw = [
                {"source": "LORA", "target": "GPU MEMORY", "type": "reduces",
                 "description": "Cuts memory.", "weight": 8.0, "chunk_id": "c0"},
                {"source": "GPU MEMORY", "target": "LORA", "type": "reduces",
                 "description": "Cuts memory again.", "weight": 6.0, "chunk_id": "c1"},
                {"source": "LORA", "target": "GHOST", "type": "extends",
                 "description": "Endpoint was never extracted.", "weight": 9.0, "chunk_id": "c0"},
                {"source": "LORA", "target": "LORA", "type": "self",
                 "description": "Self loop.", "weight": 5.0, "chunk_id": "c0"},
            ]
            frame = GraphRAGEngine._build_relationships(raw, {"LORA", "GPU MEMORY"})
            assert len(frame) == 1, f"expected one merged pair, got {len(frame)}"
            row = frame.iloc[0]
            assert row["weight"] == 7.0, f"weights should average, got {row['weight']}"
            assert row["type"] == "reduces"
            assert row["combined_degree"] == 2
            assert "GHOST" not in set(frame["source"]) | set(frame["target"])
            return "undirected pairs merged, weights averaged, dangling and self edges dropped"

        def nodes_carry_community_and_degree():
            entities = pd.DataFrame([
                {"id": "e0", "human_readable_id": 0, "title": "LORA", "type": "METHOD",
                 "description": "d", "text_unit_ids": ["c0"]},
            ])
            frame = GraphRAGEngine._build_nodes(entities, {"LORA": 3}, {"LORA": 7}, {"LORA": [0.5, -0.5]})
            row = frame.iloc[0]
            assert row["community"] == 3 and row["degree"] == 7
            assert row["x"] == 0.5 and row["y"] == -0.5
            return "nodes carry community, degree and layout coordinates"

        def report_json_parsing():
            parse = CommunityReportSynthesizer._parse
            fenced = """```json
            {"title": "PEFT methods", "summary": "LoRA cuts memory 3x.",
             "findings": [{"summary": "s", "explanation": "e"}],
             "rating": 8.4, "rating_explanation": "central"}
            ```"""
            out = parse(fenced)
            assert out["title"] == "PEFT methods" and out["rating"] == 8.4
            assert out["findings"][0]["summary"] == "s"

            prose = 'Here you go: {"title":"T","summary":"S","findings":["a"],' \
                    '"rating":"not a number","rating_explanation":""} Hope that helps.'
            out = parse(prose)
            assert out["title"] == "T"
            assert out["rating"] is None, "an unusable rating must become None so the fallback runs"
            assert out["findings"] == [{"summary": "a", "explanation": ""}]

            clamped = parse('{"title":"T","summary":"S","rating":99}')
            assert clamped["rating"] == 10.0, clamped["rating"]
            return "fenced JSON, prose-wrapped JSON, bad and out-of-range ratings all handled"

        def fallback_rank_is_ordered():
            big, _ = _fallback_rank(20, 30, 20)
            small, _ = _fallback_rank(2, 1, 20)
            assert 1.0 <= small < big <= 10.0, (small, big)
            assert _fallback_rank(0, 0, 0)[0] == 1.0
            return f"structural rank orders {small} < {big} and stays within 1..10"

        def concept_map_styling():
            nodes = sample_graph().nodes
            relationships = sample_graph().relationships
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "map.html"
                write_concept_map(nodes, relationships, target, size_by="degree")
                html = target.read_text(encoding="utf-8")
                assert "#FFFFFF" in html, "map is not on a white background"
                for hue in COMMUNITY_PALETTE[:2]:
                    assert hue in html, f"palette hue {hue} missing"
                assert "#D6D6D0" in html, "edge color is not line-strong"
                assert "Geist" in html, "label font is not Geist"
                assert "selectNode" in html, "node selection handler missing"
                assert "prefers-reduced-motion" in html, "physics is not gated on reduced motion"
                assert '"enabled": true' in html.replace("'", '"') or '"physics"' in html

                filtered = Path(tmp) / "filtered.html"
                write_concept_map(nodes, relationships, filtered, communities=[0])
                filtered_html = filtered.read_text(encoding="utf-8")
                assert "REVENUE" not in filtered_html, "community filter did not apply"
                assert "LORA" in filtered_html
            return "white canvas, §2 palette, #D6D6D0 edges, selection handler, working filter"

        def chroma_metadata_contract():
            source = (self.root_dir / "core/pipeline.py").read_text(encoding="utf-8")
            assert '"document_id": u["document_id"]' in source, \
                "chunks must be tagged with document_id for citations to link back"
            assert '"document_title"' in source and '"chunk_index"' in source
            assert 'f"{doc_id}::chunk_{local_idx}"' in source, \
                "chunk ids must be namespaced per document so deletes stay targeted"
            return "vector chunks carry document_id, title, chunk index and namespaced ids"

        for name, fn in [
            ("Extraction parsing", extractor_parses_typed_relationships),
            ("Entity deduplication", entities_dedupe),
            ("Relationship merge and pruning", relationships_dedupe_and_prune),
            ("Node construction", nodes_carry_community_and_degree),
            ("Community report JSON parsing", report_json_parsing),
            ("Structural rank fallback", fallback_rank_is_ordered),
            ("Concept map styling and filter", concept_map_styling),
            ("Vector metadata contract", chroma_metadata_contract),
        ]:
            self.check("Pipeline", name, fn)

    # -------------------------------------------------------------------------
    # 5. Vault lifecycle
    # -------------------------------------------------------------------------
    def audit_vault(self):
        print("\n--- 5. Vault lifecycle ---")

        markdown_doc = (
            "# Benchmarks\n\nSome narrative text about fine-tuning.\n\n"
            "| Method | Compute | MMLU |\n"
            "| --- | --- | --- |\n"
            "| Full FT | 1.0x | 64.1 |\n"
            "| LoRA | 0.8x | 62.9 |\n"
            "\nClosing paragraph.\n\n"
            "| Metric | Value |\n"
            "| --- | --- |\n"
            "| Latency | 180 ms |\n"
        ).encode("utf-8")

        def lifecycle():
            with tempfile.TemporaryDirectory() as tmp:
                vault = DocumentVault(vault_dir=tmp)
                assert vault.get_catalog() == []

                stored = vault.store_document("bench.md", markdown_doc, source_type="txt")
                assert stored["id"].startswith("doc_") and len(stored["id"]) == 16
                assert stored["status"] == "Not indexed", stored["status"]
                assert stored["chunks_indexed"] == 0
                assert stored["page_count"] is None, \
                    "text documents must not invent a page count"
                assert Path(stored["storage_path"]).exists()

                tables = vault.get_tables(stored["id"])
                assert len(tables) == 2, f"expected both markdown tables, got {len(tables)}"
                assert tables[0]["columns"] == ["Method", "Compute", "MMLU"]
                assert tables[0]["rows_count"] == 2, tables[0]["rows_count"]
                assert tables[0]["records"][1]["MMLU"] == "62.9"
                assert tables[0]["row_facts"], "row facts not synthesised"
                assert tables[1]["columns"] == ["Metric", "Value"], \
                    "a table at end of file must still be captured"

                again = vault.store_document("bench.md", markdown_doc, source_type="txt")
                assert again["id"] == stored["id"], "sha256 dedup failed"
                assert len(vault.get_catalog()) == 1, "re-upload duplicated the catalog entry"

                text = vault.get_document_text(stored["id"])
                assert "Benchmarks" in text and "LoRA" in text

                vault.mark_indexed(stored["id"], 5, 12)
                document = vault.get_document(stored["id"])
                assert document["status"] == "Indexed"
                assert (document["chunks_indexed"], document["chunks_total"]) == (5, 12)
                assert document["indexed_at"]

                vault.mark_needs_reindex()
                assert vault.get_document(stored["id"])["status"] == "Needs re-index"

                assert vault.verify_integrity() == []
                Path(stored["storage_path"]).write_bytes(b"tampered")
                issues = vault.verify_integrity()
                assert issues and issues[0]["problem"] == "sha256 mismatch", issues

                assert vault.delete_document(stored["id"]) is True
                assert vault.get_catalog() == []
                assert not Path(stored["storage_path"]).exists()
                assert not (Path(tmp) / "tables" / f"{stored['id']}_tables.json").exists()
                assert vault.delete_document("doc_missing") is False
            return "store, dedupe, extract 2 tables, index state, integrity, delete all verified"

        def table_relevance():
            with tempfile.TemporaryDirectory() as tmp:
                vault = DocumentVault(vault_dir=tmp)
                vault.store_document("bench.md", markdown_doc, source_type="txt")
                hits = vault.find_relevant_tables("compute and MMLU by method", top_k=2)
                assert hits, "no table matched an on-topic query"
                assert hits[0]["columns"] == ["Method", "Compute", "MMLU"], hits[0]["columns"]
                assert vault.find_relevant_tables("xylophone submarine", top_k=2) == []
            return "on-topic query ranks the matching table first, off-topic returns none"

        def vector_purge_is_targeted():
            source = (self.root_dir / "core/vault.py").read_text(encoding="utf-8")
            assert 'where={"document_id": doc_id}' in source, \
                "delete must purge only the deleted document's vectors"
            assert "purge_vectors" in source
            screen = (self.root_dir / "screens/vault.py").read_text(encoding="utf-8")
            assert "chroma_dir=str(data.CHROMA_DIR)" in screen, \
                "the delete action does not pass the vector store path"
            return "delete purges the document's own vectors, then marks the graph stale"

        def ingest_is_cumulative():
            screen = (self.root_dir / "screens/vault.py").read_text(encoding="utf-8")
            assert "for entry in vault.get_catalog()" in screen, \
                "indexing must read every retained document, not just the upload"
            assert "build_from_documents" in screen
            assert "mark_indexed" in screen
            pipeline = (self.root_dir / "core/pipeline.py").read_text(encoding="utf-8")
            assert "def build_from_documents" in pipeline
            return "indexing rebuilds from the whole vault so documents share one graph"

        def ingest_progress_row():
            screen = (self.root_dir / "screens/vault.py").read_text(encoding="utf-8")
            assert "_pending_row" in screen and "k-progress" in THEME_CSS
            assert 'Extracting {stats.get' in screen, "progress row lacks the k / n label"
            assert screen.count("STEPS = [") == 1
            from screens.vault import STEPS, format_size, format_uploaded
            assert STEPS == ["Persist", "Extract text", "Extract tables",
                            "Build graph", "Sync vectors"], STEPS
            assert format_size(15974) == "15.6 MB", format_size(15974)
            assert format_size(315) == "315 KB"
            assert format_size(None) == "—"
            recent = pd.Timestamp.now() - pd.Timedelta(minutes=5)
            assert format_uploaded(recent.strftime("%Y-%m-%d %H:%M:%S")) == "5 min ago"
            old = pd.Timestamp.now() - pd.Timedelta(days=3)
            assert "ago" not in format_uploaded(old.strftime("%Y-%m-%d %H:%M:%S"))
            return "progress renders as a registry row; sizes and relative times format correctly"

        for name, fn in [
            ("Document lifecycle", lifecycle),
            ("Table relevance ranking", table_relevance),
            ("Targeted vector purge on delete", vector_purge_is_targeted),
            ("Indexing is cumulative over the vault", ingest_is_cumulative),
            ("Ingest progress row", ingest_progress_row),
        ]:
            self.check("Vault", name, fn)

    # -------------------------------------------------------------------------
    # 6. Governance
    # -------------------------------------------------------------------------
    def audit_governance(self):
        print("\n--- 6. Governance ---")

        def checks_are_real():
            report = self.change_agent.run_qa_checks()
            checks = report["checks"]
            assert len(checks) >= 10, f"expected the full suite, got {len(checks)}"
            labels = [x["label"] for x in checks]
            assert len(labels) == len(set(labels)), "duplicate check labels"
            for check in checks:
                assert set(check) >= {"label", "ok", "detail", "error", "duration"}
                assert isinstance(check["duration"], float)
            assert report["passed_count"] + len([x for x in checks if not x["ok"]]) == len(checks)
            source = (self.root_dir / "core/change_manager.py").read_text(encoding="utf-8")
            assert '"chromadb_vault": "PASSED"' not in source, \
                "chroma health must be measured, not hard-coded"
            return f"{len(checks)} checks, each timed, none hard-coded to PASSED"

        def missing_artifacts_fail_loudly():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "ragtest/output").mkdir(parents=True)
                (root / "vault").mkdir()
                (root / "empty.py").write_text("x = 1\n", encoding="utf-8")
                agent = ChangeManagementAgent(root_dir=str(root))
                report = agent.run_qa_checks()
                failed = {x["label"]: x for x in report["checks"] if not x["ok"]}
                assert any("create_final_entities" in label for label in failed), failed.keys()
                assert any("Vault catalog" in label for label in failed), failed.keys()
                assert any("Concept map" in label for label in failed), failed.keys()
                assert report["passed"] is False
                entities = next(x for x in report["checks"]
                                if "create_final_entities" in x["label"])
                assert entities["detail"] == "not built yet", entities
            return "a bare root reports the missing parquet, catalog and concept map"

        def syntax_check_catches_errors():
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "broken.py").write_text("def f(:\n", encoding="utf-8")
                agent = ChangeManagementAgent(root_dir=str(root))
                report = agent.run_qa_checks()
                syntax = next(x for x in report["checks"] if "Python syntax" in x["label"])
                assert syntax["ok"] is False, "a syntax error was not detected"
                assert "broken.py" in syntax["error"], syntax["error"]
            return "a deliberately broken file fails the syntax check with the file name"

        def changelog_timeline():
            from ui.data import parse_changelog
            releases = parse_changelog()
            assert releases, "no releases parsed from CHANGELOG.md"
            first = releases[0]
            assert re.match(r"^\d+\.\d+\.\d+$", first["version"]), first["version"]
            assert first["commits"], "no commit rows parsed"
            for commit in first["commits"]:
                assert len(commit["sha"]) == 7, commit
                assert not re.match(r"^(feat|fix|docs|chore|refactor|ui)(\(|:)",
                                    commit["message"]), \
                    f"conventional prefix not stripped: {commit['message']}"
            html = c.changelog_timeline(releases)
            assert "k-tl__dot" in html and "current" in html
            return f"{len(releases)} releases parsed, prefixes stripped, timeline rendered"

        def changelog_has_no_emoji():
            source = (self.root_dir / "core/change_manager.py").read_text(encoding="utf-8")
            found = find_emoji(source)
            assert not found, f"the generated changelog or README still emits emoji: {found}"
            return "generated changelog and README template are emoji-free"

        for name, fn in [
            ("QA checks are measured", checks_are_real),
            ("Missing artifacts fail", missing_artifacts_fail_loudly),
            ("Syntax check catches errors", syntax_check_catches_errors),
            ("Changelog timeline", changelog_timeline),
            ("Generated artifacts are emoji-free", changelog_has_no_emoji),
        ]:
            self.check("Governance", name, fn)

    # -------------------------------------------------------------------------
    # 7. Live artifacts (informational)
    # -------------------------------------------------------------------------
    def report_live_state(self):
        print("\n--- 7. Live workspace state ---")
        out_dir = self.root_dir / "ragtest/output"
        for name in ["entities", "relationships", "nodes", "community_reports"]:
            path = out_dir / f"create_final_{name}.parquet"
            if path.exists():
                print(f"       {name}: {len(pd.read_parquet(path))} rows")
            else:
                print(f"       {name}: not built")
        catalog = self.vault.get_catalog()
        print(f"       vault: {len(catalog)} documents, {len(self.vault.get_tables())} tables")
        for document in catalog:
            print(f"         - {document['id']} {document.get('status')} "
                  f"chunks {document.get('chunks_indexed', 0)}/{document.get('chunks_total', 0)}")

    # -------------------------------------------------------------------------
    def run_all(self) -> bool:
        print("=" * 72)
        print("GraphRAG knowledge workspace · UI and functional QA suite")
        print("=" * 72)
        self.audit_design_tokens()
        self.audit_components()
        self.audit_retrieval()
        self.audit_pipeline()
        self.audit_vault()
        self.audit_governance()
        self.report_live_state()

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed

        print("\n" + "=" * 72)
        print(f"SUMMARY: {passed} / {total} checks passed ({failed} failures)")
        print("=" * 72)
        if failed:
            print("\nFailures:")
            for result in self.results:
                if not result["passed"]:
                    print(f"  - {result['category']} | {result['test']}: {result['error']}")
            return False
        return True


if __name__ == "__main__":
    agent = UIQAAgent()
    sys.exit(0 if agent.run_all() else 1)
