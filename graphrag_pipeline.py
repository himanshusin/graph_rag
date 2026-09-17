"""Compatibility shim.

The pipeline now lives in :mod:`core.pipeline`. This module re-exports it so
``streamlit_app.ipynb`` and any existing imports keep working against a single
implementation.
"""

from core.pipeline import (  # noqa: F401
    CHROMA_COLLECTION,
    COMMUNITY_PALETTE,
    CommunityReportSynthesizer,
    DocumentIngestor,
    EntityRelationshipExtractor,
    GraphRAGEngine,
    TextChunker,
    write_concept_map,
)

__all__ = [
    "CHROMA_COLLECTION",
    "COMMUNITY_PALETTE",
    "CommunityReportSynthesizer",
    "DocumentIngestor",
    "EntityRelationshipExtractor",
    "GraphRAGEngine",
    "TextChunker",
    "write_concept_map",
]
