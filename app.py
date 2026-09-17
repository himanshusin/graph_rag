"""GraphRAG knowledge workspace — entry point, sidebar and routing.

Screens live in ``screens/``; shared markup in ``ui/components.py``; cached data
access in ``ui/data.py``; tokens and the stylesheet in ``ui/tokens.py`` and
``ui/theme.css``.

Navigation is query-param driven rather than ``st.navigation`` so the sidebar can
keep the exact nav markup from the mockups while still giving every screen its
own URL.
"""

import streamlit as st

st.set_page_config(
    page_title="Knowledge · GraphRAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

from core import rag
from screens import catalog as screen_catalog
from screens import concept_map as screen_concept_map
from screens import governance as screen_governance
from screens import search as screen_search
from screens import vault as screen_vault
from ui import components as c
from ui import data
from ui.tokens import community_color, query_string, screen_style

load_dotenv()

SCREENS = {
    "search": screen_search.render,
    "vault": screen_vault.render,
    "concepts": screen_concept_map.render,
    "catalog": screen_catalog.render,
    "governance": screen_governance.render,
}

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("running", None)
st.session_state.setdefault("temperature", 0.20)
st.session_state.setdefault("citations_k", 4)
st.session_state.setdefault("chunk_limit", 8)

# -----------------------------------------------------------------------------
# Shared data
# -----------------------------------------------------------------------------
graph = data.graph_data()
vault = data.get_vault()
vault_documents = vault.get_catalog()
qa = data.current_qa_report()
version = data.app_version()

screen = st.query_params.get("screen", "search")
if screen not in SCREENS:
    screen = "search"

mode = st.query_params.get("mode", "global")
if mode not in rag.MODE_KEYS:
    mode = "global"

source = st.query_params.get("source", "all")
if source not in rag.SOURCE_KINDS:
    source = "all"

st.markdown(c.stylesheet(), unsafe_allow_html=True)
st.markdown(screen_style(screen), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
nav_counts = {"vault": len(vault_documents)}

with st.sidebar:
    st.markdown(c.brand(version), unsafe_allow_html=True)
    st.markdown(c.nav_bar(screen, nav_counts), unsafe_allow_html=True)

    if screen == "search":
        st.markdown(c.reasoning_modes(mode), unsafe_allow_html=True)
        with st.container(key="controls"):
            st.session_state.temperature = st.slider(
                "Temperature", min_value=0.0, max_value=1.0,
                value=float(st.session_state.temperature), step=0.05,
            )
            st.session_state.citations_k = st.slider(
                "Citations", min_value=2, max_value=8,
                value=int(st.session_state.citations_k),
            )

    if screen == "concepts" and not graph.nodes.empty and "community" in graph.nodes.columns:
        active = st.query_params.get("community")
        sizes = graph.nodes["community"].value_counts()
        listed = list(sizes.head(8).items())
        # Keep the filtered community on the list even when it is not one of the
        # largest, so the active filter is always visible and clearable.
        if active not in (None, "", "all") and not any(str(k) == str(active) for k, _v in listed):
            match = [(k, v) for k, v in sizes.items() if str(k) == str(active)]
            listed = match + listed[:7]

        items = [{
            "label": "All communities",
            "count": len(graph.nodes),
            "href": query_string(screen="concepts", size=st.query_params.get("size")),
            "active": active in (None, "", "all"),
        }]
        for community, size in listed:
            items.append({
                "label": graph.community_title(community),
                "count": int(size),
                "color": community_color(community),
                "href": query_string(
                    screen="concepts", community=community,
                    size=st.query_params.get("size"),
                ),
                "active": str(active) == str(community),
            })
        st.markdown(c.sidebar_list("Communities", items), unsafe_allow_html=True)

    st.markdown(
        c.sidebar_status(qa.get("passed_count", 0), qa.get("total_count", 0)),
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Route
# -----------------------------------------------------------------------------
state = {
    "graph": graph,
    "vault": vault,
    "vault_documents": vault_documents,
    "qa": qa,
    "version": version,
    "screen": screen,
    "mode": mode,
    "source": source,
    "temperature": float(st.session_state.temperature),
    "citations_k": int(st.session_state.citations_k),
    "chunk_limit": int(st.session_state.chunk_limit),
}

SCREENS[screen](state)
