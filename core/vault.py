import os
import json
import time
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pymupdf


class DocumentVault:
    """
    Enterprise Document Retention & Vault Manager.
    Persists uploaded files, tracks metadata catalog, and generates Glean-style source citations.
    """

    def __init__(self, vault_dir: str = "./vault"):
        self.vault_dir = Path(vault_dir)
        self.docs_dir = self.vault_dir / "documents"
        self.catalog_file = self.vault_dir / "catalog.json"

        self.docs_dir.mkdir(parents=True, exist_ok=True)
        if not self.catalog_file.exists():
            self._save_catalog([])

    def _load_catalog(self) -> List[Dict[str, Any]]:
        try:
            with open(self.catalog_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_catalog(self, catalog: List[Dict[str, Any]]):
        with open(self.catalog_file, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

    def get_catalog(self) -> List[Dict[str, Any]]:
        """Return all retained documents in the vault."""
        return self._load_catalog()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a specific document."""
        catalog = self._load_catalog()
        for doc in catalog:
            if doc["id"] == doc_id:
                return doc
        return None

    def store_document(
        self,
        filename: str,
        content_bytes: bytes,
        source_type: str = "pdf",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Persist document bytes to vault, compute SHA-256, extract text, and update catalog.
        """
        doc_hash = hashlib.sha256(content_bytes).hexdigest()
        doc_id = f"doc_{doc_hash[:12]}"
        
        # Determine stored filepath
        clean_filename = Path(filename).name
        target_path = self.docs_dir / f"{doc_id}_{clean_filename}"
        with open(target_path, "wb") as f:
            f.write(content_bytes)

        # Extract text based on file format
        if clean_filename.lower().endswith(".pdf") or source_type.lower() == "pdf":
            try:
                pdf_doc = pymupdf.open(stream=content_bytes, filetype="pdf")
                pages_text = [page.get_text() for page in pdf_doc]
                raw_text = "\n".join(pages_text)
                page_count = len(pdf_doc)
            except Exception:
                raw_text = content_bytes.decode("utf-8", errors="ignore")
                page_count = 1
        else:
            raw_text = content_bytes.decode("utf-8", errors="ignore")
            page_count = max(1, len(raw_text.splitlines()) // 50)

        # Update metadata catalog
        catalog = self._load_catalog()
        # Remove existing if duplicate
        catalog = [d for d in catalog if d["id"] != doc_id]

        entry = {
            "id": doc_id,
            "title": clean_filename,
            "filename": clean_filename,
            "storage_path": str(target_path),
            "file_size_bytes": len(content_bytes),
            "file_size_kb": round(len(content_bytes) / 1024, 1),
            "sha256": doc_hash,
            "source_type": source_type.upper(),
            "page_count": page_count,
            "char_count": len(raw_text),
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Indexed in Vault",
            "metadata": metadata or {}
        }
        catalog.append(entry)
        self._save_catalog(catalog)

        entry["extracted_text"] = raw_text
        return entry

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the vault."""
        catalog = self._load_catalog()
        doc_to_delete = None
        for doc in catalog:
            if doc["id"] == doc_id:
                doc_to_delete = doc
                break

        if doc_to_delete:
            file_path = Path(doc_to_delete["storage_path"])
            if file_path.exists():
                file_path.unlink()
            catalog = [d for d in catalog if d["id"] != doc_id]
            self._save_catalog(catalog)
            return True
        return False

    @staticmethod
    def format_glean_citations(
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format source excerpts into Glean-style citation cards with direct links.
        """
        citations = []
        for idx, src in enumerate(sources, 1):
            doc_name = src.get("document_title", src.get("title", "Enterprise Document"))
            chunk_id = src.get("chunk_id", f"chunk_{idx}")
            excerpt = src.get("excerpt", src.get("text", ""))
            relevance = src.get("relevance", "High Confidence")
            
            citations.append({
                "index": idx,
                "citation_tag": f"[{idx}]",
                "document_title": doc_name,
                "chunk_id": chunk_id,
                "excerpt": excerpt,
                "relevance": relevance
            })
        return citations
