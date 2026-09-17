repo: himanshusin/graph_rag
branch: main

## Last sync
date: 2026-09-17T15:05:45Z

### Updated in this project
- Read README.md, app.py, CHANGELOG.md, vault/catalog.json to ground the redesign
- Built GraphRAG Mockups.dc.html (3 directions + 4 secondary screens)
- Built GraphRAG Developer Guide.dc.html (tokens, components, requirements, Streamlit notes)

## Screen map
| Screen | Repo files |
| --- | --- |
| Search / Compare (1a, 1b, 1c) | app.py (tab_search_chat, query_* functions, render_citations_drawer) |
| Vault (1d) | app.py (tab_vault_ui), core/vault.py, vault/catalog.json |
| Concept map (1e) | app.py (tab_concept_map), notebook/interactive_graph.html |
| Catalog (1f) | app.py (tab_catalog_ui) |
| Governance (1g) | app.py (tab_governance), core/change_manager.py, CHANGELOG.md |
