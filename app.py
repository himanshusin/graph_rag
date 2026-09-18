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

from core import ledger as ledger_mod
from core import providers
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
st.session_state.setdefault("chunk_limit", 4)

# -----------------------------------------------------------------------------
# Shared data
# -----------------------------------------------------------------------------
graph = data.graph_data()
vault = data.get_vault()

# A build runs inside one script run, so a restart can leave a run stranded.
# Condemn those once per session and flag the documents they were covering.
if not st.session_state.get("reaped_stale_runs"):
    st.session_state.reaped_stale_runs = True
    for stranded in data.get_run_log().reap_stale():
        vault.mark_index_outcome(
            [d["id"] for d in stranded.get("documents", []) if d.get("id")],
            vault.INTERRUPTED,
            stranded.get("error", ""),
        )

vault_documents = vault.get_catalog()
qa = data.current_qa_report()
version = data.app_version()

# Provider routing, read from disk. healthy_routing falls back to a working
# route for any stage whose provider has no key or no endpoint, so a missing
# key degrades the app instead of breaking it mid-run.
provider_settings = data.provider_settings()
active_routing, routing_warnings = providers.healthy_routing(provider_settings["routing"])
lifetime = ledger_mod.lifetime_totals(str(data.VAULT_DIR))

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

    # -- Engine ---------------------------------------------------------------
    # Routing is a workspace setting, not a chat setting, so it is persisted to
    # disk rather than session state: a restart must not silently move the app
    # back onto a different (and differently priced) model.
    with st.container(key="engine"):
        st.markdown('<div class="k-rail__head">Engine</div>', unsafe_allow_html=True)
        preset_names = list(providers.PRESETS) + ["custom"]
        current_preset = provider_settings.get("preset", "balanced")
        chosen_preset = st.selectbox(
            "Preset", preset_names,
            index=preset_names.index(current_preset) if current_preset in preset_names else 1,
            key="engine_preset", label_visibility="collapsed",
            format_func=lambda k: {
                "frugal": "Frugal · all local, free",
                "balanced": "Balanced · cheap index, best answers",
                "best": "Best · highest quality",
                "custom": "Custom",
            }.get(k, k),
        )
        if chosen_preset != current_preset:
            new_settings = dict(provider_settings, preset=chosen_preset)
            if chosen_preset in providers.PRESETS:
                new_settings["routing"] = dict(providers.PRESETS[chosen_preset])
            data.save_provider_settings(new_settings)
            data.clear_caches()
            st.rerun()

        rows = []
        for provider in providers.available():
            ok, status = providers.health(provider)
            stages = [s for s, k in active_routing.items() if k == provider.key]
            if provider.cost_per_mtok:
                estimate = provider.cost_of(122_150, 57_763)
                trade = f"~${estimate:.3f}/index · fast · sends data"
            else:
                trade = "free · slow · fully offline"
            rows.append({
                "label": provider.label,
                "ok": ok,
                "status": ", ".join(stages) if stages else status if not ok else "idle",
                "trade": trade,
                "active": bool(stages),
            })
        st.markdown(c.provider_rows(rows), unsafe_allow_html=True)

        for warning in routing_warnings:
            st.markdown(
                f'<div class="k-faint" style="font-size:10.5px;color:#B7791F">{warning}</div>',
                unsafe_allow_html=True,
            )

        if lifetime.get("runs"):
            st.markdown(
                c.spend_badge("spent", f"${lifetime['cost']:.3f}")
                + c.spend_badge("avoided", f"{lifetime['calls_avoided']} calls", "ok"),
                unsafe_allow_html=True,
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
    "routing": active_routing,
    "provider_settings": provider_settings,
    "lifetime": lifetime,
}

SCREENS[screen](state)
