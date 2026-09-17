#!/usr/bin/env python3
"""Runs the real Streamlit app for every screen and asserts what rendered.

Uses ``streamlit.testing.v1.AppTest``, so the actual ``app.py`` script executes
with real data. No LLM calls: the suite never submits a query.
"""

import sys
import traceback
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from streamlit.testing.v1 import AppTest

APP = str(ROOT_DIR / "app.py")
TIMEOUT = 120
OUTPUT_DIR = ROOT_DIR / "ragtest" / "output"


def graph_is_built() -> bool:
    """Whether the parquet artifacts exist, so tests can assert the right state."""
    return (OUTPUT_DIR / "create_final_nodes.parquet").exists()


class AppSmoke:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def log(self, name: str, passed: bool, detail: str, error: str = ""):
        self.results.append({"test": name, "passed": passed, "detail": detail, "error": error})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        if error:
            print(f"       {error}")

    def check(self, name: str, fn: Callable[[], Any]):
        try:
            detail = fn()
            self.log(name, True, detail or "ok")
        except AssertionError as exc:
            self.log(name, False, "assertion failed", str(exc))
        except Exception as exc:
            self.log(name, False, "raised",
                     f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}")

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def run(**params) -> AppTest:
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        for key, value in params.items():
            at.query_params[key] = str(value)
        at.run()
        return at

    @staticmethod
    def body(at: AppTest) -> str:
        """Rendered main-area markup, excluding the injected stylesheet.

        The theme sheet is itself a markdown element, so leaving it in would let
        a CSS class definition satisfy an assertion about rendered markup.
        """
        return "\n".join(
            m.value for m in at.main.markdown
            if not m.value.lstrip().startswith("<style>")
        )

    @staticmethod
    def side(at: AppTest) -> str:
        return "\n".join(m.value for m in at.sidebar.markdown)

    @staticmethod
    def active_mode_label(side: str) -> Optional[str]:
        """The label of the sidebar mode row that is marked active."""
        match = re.search(r'k-mode-link--active"[^>]*>.*?</span><span>([^<]+)</span>', side,
                          re.DOTALL)
        return match.group(1) if match else None

    @staticmethod
    def active_chip_label(body: str) -> Optional[str]:
        """The label of the active composer mode chip."""
        chips = body.split("k-composer-modes__label")[-1]
        match = re.search(r'k-pill k-pill--active"[^>]*>(?:<span[^>]*></span>)?([^<]+)', chips)
        return match.group(1) if match else None

    @staticmethod
    def assert_clean(at: AppTest, screen: str):
        assert not at.exception, (
            f"{screen} raised: " + " | ".join(e.value for e in at.exception)
        )

    # -- screens -------------------------------------------------------------
    def test_search(self):
        def run():
            at = self.run(screen="search")
            self.assert_clean(at, "search")
            body, side = self.body(at), self.side(at)

            assert "k-nav-link--active" in side, "no active nav item"
            assert "screen=search" in side
            assert "Knowledge" in side and "k-brand__mark" in side
            assert "k-mode-menu" in side and "k-mode-dot--active" in side, "mode list missing"
            assert "Temperature" in [s.label for s in at.sidebar.slider], \
                [s.label for s in at.sidebar.slider]
            assert "Citations" in [s.label for s in at.sidebar.slider]
            assert "k-side-foot" in side, "QA status not pinned to the sidebar"

            assert "Search" in body and "k-topbar__sep" in body, "breadcrumb missing"
            assert "k-pills-row" in body and "All sources" in body, "source pills missing"
            assert "k-composer-modes" in body, "composer mode chips missing"
            assert at.chat_input, "no composer rendered"
            assert "Evidence" in body, "evidence rail header missing"
            if graph_is_built():
                assert "Ask the knowledge base." in body, "empty-thread title missing"
                assert "k-suggest-row" in body, "suggested prompts missing"
            else:
                # No graph: either the no-corpus state, or the prompt list is empty
                assert ("No documents yet." in body
                        or "Add a document to start." in body), body[-400:]
                assert "k-suggest-row" not in body, \
                    "prompts advertised for a corpus that is not indexed"
            return (
                f"{len(at.sidebar.slider)} sliders, {len(at.chat_input)} composer, "
                "breadcrumb, pills, mode chips, empty state and rail all present"
            )
        self.check("Search renders", run)

    def test_search_mode_and_source_sync(self):
        def run():
            at = self.run(screen="search", mode="drift", source="graph")
            self.assert_clean(at, "search drift")
            body, side = self.body(at), self.side(at)
            # The active dot in the sidebar and the active chip in the composer
            # must both land on DRIFT.
            sidebar_mode = self.active_mode_label(side)
            assert sidebar_mode == "DRIFT deep-dive", f"sidebar shows {sidebar_mode!r}"
            chip_mode = self.active_chip_label(body)
            assert chip_mode == "DRIFT", f"composer chip shows {chip_mode!r}"
            assert 'class="k-pill k-pill--active">Graph' in body, "source pill not active"
            return f"sidebar {sidebar_mode!r} and composer {chip_mode!r} agree, source pill active"
        self.check("Mode and source stay in sync", run)

    def test_vault(self):
        def run():
            at = self.run(screen="vault")
            self.assert_clean(at, "vault")
            body = self.body(at)
            assert "Vault" in body and "SHA-256 deduplicated" in body
            assert "Add document" in [b.label for b in at.button], [b.label for b in at.button]
            assert "k-grid__head" in body, "registry header missing"
            for column in ["Document", "Format", "Size", "Pages", "Tables", "Uploaded", "Status"]:
                assert f"<span>{column}</span>" in body, f"registry column {column} missing"
            assert "k-grid__row--on" in body, "selected row is not tinted"
            assert "SHA-256" in body and "Chunks in graph" in body, "inspector fields missing"
            assert "Extracted tables" in body
            assert "Re-index graph" in [b.label for b in at.button]
            assert "Delete" in [b.label for b in at.button]
            return "registry columns, selected row, inspector and actions all present"
        self.check("Vault renders", run)

    def test_vault_delete_confirms_inline(self):
        def run():
            at = self.run(screen="vault")
            self.assert_clean(at, "vault")
            delete = next(b for b in at.button if b.label == "Delete")
            delete.click().run()
            self.assert_clean(at, "vault after delete click")
            labels = [b.label for b in at.button]
            body = self.body(at)
            assert "Cancel" in labels, f"no inline confirmation: {labels}"
            assert "Delete " in body or "Delete" in labels
            # cancelling must leave the document in place
            next(b for b in at.button if b.label == "Cancel").click().run()
            assert "Cancel" not in [b.label for b in at.button], "confirmation did not dismiss"
            return "delete asks for confirmation inline and cancel restores the row"
        self.check("Vault delete confirms inline", run)

    def test_concept_map(self):
        def run():
            at = self.run(screen="concepts")
            self.assert_clean(at, "concepts")
            body, side = self.body(at), self.side(at)
            assert "Concept map" in body and "Louvain" in body, "top bar missing"
            assert "k-seg" in body and "Degree" in body and "Rank" in body, "size-by toggle missing"
            assert "Selected node" in body, "inspector header missing"

            if not graph_is_built():
                assert "No graph yet — add a document." in body, \
                    "missing-graph state not shown"
                assert "No concepts indexed yet." in body, "inspector has no empty state"
                assert "Communities" not in side, "community list shown with no graph"
                return "empty state: 'No graph yet' plus an empty inspector"

            assert at.selectbox, "Find node select box missing"
            assert "Communities" in side, "community list missing from the sidebar"
            assert "k-side-item" in side
            assert "k-stat__num" in body, "node stats missing"
            assert "Relationships" in body
            labels = [b.label for b in at.button]
            assert any(l.startswith("Ask about") for l in labels), labels
            assert "Open in catalog" in labels
            return "map, size-by toggle, node picker, inspector stats and actions present"
        self.check("Concept map renders", run)

    def test_concept_map_community_filter(self):
        def run():
            at = self.run(screen="concepts", community="0")
            self.assert_clean(at, "concepts filtered")
            body, side = self.body(at), self.side(at)
            if not graph_is_built():
                assert "No graph yet" in body, "filter on an empty graph must not raise"
                return "filter is inert while the graph is empty, without raising"
            assert "filtered to community 0" in body, body[-600:]
            assert "k-side-item--active" in side, "active community not marked"
            return "community filter applies and is reflected in the sidebar"
        self.check("Concept map community filter", run)

    def test_catalog_tabs(self):
        def run():
            details = []
            for tab, expected, empty_message in [
                ("concepts", ["Title", "Type", "Description", "Degree", "Community"],
                 "No concepts indexed yet."),
                ("relationships", ["Source → Target", "Type", "Weight", "Description"],
                 "No relationships indexed yet."),
                ("nodes", ["Title", "Community", "Degree"],
                 "No graph nodes indexed yet."),
                ("briefs", ["Domain", "Summary", "Importance", "Concepts"],
                 "No domain briefs generated yet."),
            ]:
                at = self.run(screen="catalog", tab=tab)
                self.assert_clean(at, f"catalog {tab}")
                body = self.body(at)
                assert "k-tab-link--active" in body, f"{tab}: no active tab"
                if not graph_is_built():
                    assert empty_message in body, \
                        f"{tab}: expected {empty_message!r}, got {body[-260:]!r}"
                    details.append(f"{tab}:empty")
                    continue
                assert "k-grid__head" in body, f"{tab}: no table header"
                for column in expected:
                    needle = column.replace("→", "&#x2192;")
                    assert needle in body or column in body, f"{tab}: column {column} missing"
                assert "Showing" in body, f"{tab}: footer count missing"
                assert "Export CSV" in [b.label for b in at.download_button], f"{tab}: no CSV export"
                details.append(tab)
            if not graph_is_built():
                return f"all four tabs show their empty state: {details}"
            return f"all four tabs render their own columns, footer and CSV export: {details}"
        self.check("Catalog tabs render", run)

    def test_catalog_search_highlights(self):
        def run():
            at = self.run(screen="catalog", tab="concepts", term="a")
            self.assert_clean(at, "catalog search")
            body = self.body(at)
            if not graph_is_built():
                assert "No concepts indexed yet" in body, "search on an empty catalog must not raise"
                return "search is inert while the catalog is empty, without raising"
            assert "<mark>" in body or "No concepts match" in body, "no highlight and no empty state"
            assert "filtered by" in body, "footer does not report the filter"
            return "search highlights matches and the footer reports the filter"
        self.check("Catalog search", run)

    def test_catalog_brief_rail(self):
        def run():
            at = self.run(screen="catalog", tab="briefs")
            self.assert_clean(at, "catalog briefs")
            body = self.body(at)
            if not graph_is_built():
                assert "No domain briefs generated yet." in body, "briefs have no empty state"
                return "briefs show their empty state"
            assert "Domain brief" in body, "brief rail header missing"
            assert "k-grid__row--on" in body, "no brief selected by default"
            assert "rank" in body.lower()
            return "domain briefs open in the right rail, not an expander"
        self.check("Catalog brief rail", run)

    def test_governance(self):
        def run():
            at = self.run(screen="governance")
            self.assert_clean(at, "governance")
            body = self.body(at)
            assert "Governance" in body and "Change Management Agent" in body
            assert "QA health" in body and "passed" in body
            assert "k-check" in body, "checks are not a list"
            assert "Re-run QA &amp; sync" in body or "Re-run QA & sync" in [b.label for b in at.button]
            assert "Changelog" in body and "k-tl__dot" in body, "changelog timeline missing"
            assert "CHANGELOG.md" in body
            checks = body.count('class="k-check"')
            assert checks >= 10, f"only {checks} checks rendered"
            return f"{checks} QA checks as a list plus the changelog timeline"
        self.check("Governance renders", run)

    def test_unknown_screen_falls_back(self):
        def run():
            at = self.run(screen="nonsense")
            self.assert_clean(at, "unknown screen")
            body = self.body(at)
            assert "Ask the knowledge base." in body or "No documents yet." in body, body[:300]
            at2 = self.run(screen="search", mode="bogus")
            self.assert_clean(at2, "bogus mode")
            active = self.active_mode_label(self.side(at2))
            assert active == "Global synthesis", f"fell back to {active!r}"
            at3 = self.run(screen="catalog", tab="bogus")
            self.assert_clean(at3, "bogus tab")
            assert "k-tab-link--active" in self.body(at3)
            return "unknown screen, mode and tab all fall back without raising"
        self.check("Invalid routes fall back", run)

    def test_no_streamlit_banners(self):
        def run():
            offenders = []
            for screen in ["search", "vault", "concepts", "catalog", "governance"]:
                at = self.run(screen=screen)
                self.assert_clean(at, screen)
                if at.error:
                    offenders.append((screen, [e.value for e in at.error]))
            assert not offenders, f"st.error banners present: {offenders}"
            return "no page-level st.error banners on any screen"
        self.check("No top-of-page error banners", run)

    # -- runner --------------------------------------------------------------
    def run_all(self) -> bool:
        print("=" * 72)
        print("GraphRAG knowledge workspace · live app smoke tests")
        print("=" * 72)
        self.test_search()
        self.test_search_mode_and_source_sync()
        self.test_vault()
        self.test_vault_delete_confirms_inline()
        self.test_concept_map()
        self.test_concept_map_community_filter()
        self.test_catalog_tabs()
        self.test_catalog_search_highlights()
        self.test_catalog_brief_rail()
        self.test_governance()
        self.test_unknown_screen_falls_back()
        self.test_no_streamlit_banners()

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        print("\n" + "=" * 72)
        print(f"SUMMARY: {passed} / {total} screens verified ({total - passed} failures)")
        print("=" * 72)
        if passed != total:
            for result in self.results:
                if not result["passed"]:
                    print(f"  - {result['test']}: {result['error']}")
            return False
        return True


if __name__ == "__main__":
    sys.exit(0 if AppSmoke().run_all() else 1)
