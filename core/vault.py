import re
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import pymupdf


class DocumentVault:
    """
    Enterprise Document Retention & Vault Manager.
    Persists uploaded files, extracts structured tables & text, tracks metadata catalog,
    and generates verified source citations with table provenance.
    """

    def __init__(self, vault_dir: str = "./vault"):
        self.vault_dir = Path(vault_dir)
        self.docs_dir = self.vault_dir / "documents"
        self.tables_dir = self.vault_dir / "tables"
        self.catalog_file = self.vault_dir / "catalog.json"

        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
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

    # -------------------------------------------------------------------------
    # Structured Table Extraction
    # -------------------------------------------------------------------------
    def _extract_tables_from_pdf(self, doc_id: str, content_bytes: bytes) -> Tuple[str, List[Dict[str, Any]], int]:
        """
        Extract pages text and structured tables from PDF bytes using PyMuPDF table finder.
        Returns: (augmented_text_with_tables, extracted_tables_list, page_count)
        """
        pdf_doc = pymupdf.open(stream=content_bytes, filetype="pdf")
        page_count = len(pdf_doc)
        pages_content = []
        extracted_tables = []

        for p_idx, page in enumerate(pdf_doc):
            page_num = p_idx + 1
            page_blocks = []
            
            # 1. Search for structured tables on this page
            try:
                tabs = page.find_tables()
                if tabs and tabs.tables:
                    for t_idx, tab in enumerate(tabs.tables):
                        df = tab.to_pandas()
                        if df.empty or len(df.columns) < 2:
                            continue
                        
                        # Clean cell newlines and spaces
                        df = df.map(lambda x: str(x).replace("\n", " ").strip() if pd.notna(x) else "")
                        df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
                        
                        table_id = f"{doc_id}_p{page_num}_t{t_idx + 1}"
                        md_table = tab.to_markdown()

                        # Synthesize explicit row-level facts linking headers to values
                        row_facts = []
                        for r_idx, row in df.iterrows():
                            items = [f"[{col}] = {val}" for col, val in row.items() if val and str(val).lower() not in ["none", "nan", ""]]
                            if items:
                                row_facts.append(f"Row {r_idx + 1}: " + ", ".join(items))
                        
                        row_facts_text = "\n".join(row_facts)
                        table_title = f"Table {t_idx + 1} (Page {page_num}): {', '.join(df.columns[:4])}"

                        # Augmented block for semantic LLM chunking
                        table_repr = (
                            f"\n\n[STRUCTURED_TABLE_START id='{table_id}' page='{page_num}']\n"
                            f"**Table: {table_title}**\n\n"
                            f"{md_table}\n\n"
                            f"**Explicit Numerical & Relational Facts:**\n"
                            f"{row_facts_text}\n"
                            f"[STRUCTURED_TABLE_END]\n\n"
                        )
                        page_blocks.append(table_repr)

                        extracted_tables.append({
                            "table_id": table_id,
                            "doc_id": doc_id,
                            "page": page_num,
                            "title": table_title,
                            "columns": [str(c) for c in df.columns],
                            "rows_count": len(df),
                            "markdown": md_table,
                            "row_facts": row_facts,
                            "records": df.to_dict(orient="records")
                        })
            except Exception:
                pass

            # 2. Append narrative page text
            page_text = page.get_text().strip()
            if page_text:
                page_blocks.append(page_text)

            pages_content.append("\n\n".join(page_blocks))

        return "\n\n--- PAGE BREAK ---\n\n".join(pages_content), extracted_tables, page_count

    def _extract_tables_from_text(self, doc_id: str, text: str) -> Tuple[str, List[Dict[str, Any]], Optional[int]]:
        """Detect Markdown tables in raw text and extract structured representations.

        Plain text has no pagination, so the page count is reported as ``None``
        rather than estimated from line counts.
        """
        lines = text.splitlines()
        extracted_tables: List[Dict[str, Any]] = []
        table_lines: List[str] = []
        t_counter = 0

        def flush(buffer: List[str]):
            nonlocal t_counter
            if len(buffer) < 3:
                return
            try:
                headers = [h.strip() for h in buffer[0].split("|")[1:-1]]
                if not headers:
                    return
                t_counter += 1
                records = []
                for body_line in buffer[2:]:
                    cells = [c.strip() for c in body_line.split("|")[1:-1]]
                    if len(cells) == len(headers):
                        records.append(dict(zip(headers, cells)))
                row_facts = [
                    f"Row {i + 1}: " + ", ".join(f"[{k}] = {v}" for k, v in rec.items() if v)
                    for i, rec in enumerate(records)
                ]
                extracted_tables.append({
                    "table_id": f"{doc_id}_txt_t{t_counter}",
                    "doc_id": doc_id,
                    "page": 1,
                    "title": f"{', '.join(headers[:4])}",
                    "columns": headers,
                    "rows_count": len(records),
                    "markdown": "\n".join(buffer),
                    "row_facts": row_facts,
                    "records": records,
                })
            except Exception:
                pass

        for line in lines:
            stripped = line.strip()
            if "|" in stripped and stripped.startswith("|") and stripped.endswith("|"):
                table_lines.append(line)
            else:
                flush(table_lines)
                table_lines = []
        flush(table_lines)

        return text, extracted_tables, None

    # -------------------------------------------------------------------------
    # Document Ingestion
    # -------------------------------------------------------------------------
    def store_document(
        self,
        filename: str,
        content_bytes: bytes,
        source_type: str = "pdf",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Persist document bytes to vault, compute SHA-256, extract structured tables and text,
        and update catalog.
        """
        doc_hash = hashlib.sha256(content_bytes).hexdigest()
        doc_id = f"doc_{doc_hash[:12]}"
        
        # Determine stored filepath
        clean_filename = Path(filename).name
        target_path = self.docs_dir / f"{doc_id}_{clean_filename}"
        with open(target_path, "wb") as f:
            f.write(content_bytes)

        # Extract text & structured tables based on file format
        if clean_filename.lower().endswith(".pdf") or source_type.lower() == "pdf":
            try:
                raw_text, tables, page_count = self._extract_tables_from_pdf(doc_id, content_bytes)
            except Exception:
                raw_text = content_bytes.decode("utf-8", errors="ignore")
                tables = []
                page_count = 1
        else:
            decoded = content_bytes.decode("utf-8", errors="ignore")
            raw_text, tables, page_count = self._extract_tables_from_text(doc_id, decoded)

        # Persist extracted tables JSON
        if tables:
            tables_file = self.tables_dir / f"{doc_id}_tables.json"
            with open(tables_file, "w", encoding="utf-8") as f:
                json.dump(tables, f, indent=2)

        # Update metadata catalog
        catalog = self._load_catalog()
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
            "table_count": len(tables),
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Not indexed",
            "chunks_total": 0,
            "chunks_indexed": 0,
            "indexed_at": None,
            "metadata": metadata or {}
        }
        catalog.append(entry)
        self._save_catalog(catalog)

        entry["extracted_text"] = raw_text
        entry["tables"] = tables
        return entry

    # -------------------------------------------------------------------------
    # Index state
    # -------------------------------------------------------------------------
    def get_document_text(self, doc_id: str) -> str:
        """Re-extract the text of a stored document from disk.

        Extracted text is not cached in the catalog, so a cumulative rebuild
        reads every retained file back through the same extractor that ran at
        upload time.
        """
        document = self.get_document(doc_id)
        if not document:
            return ""
        path = Path(document["storage_path"])
        if not path.exists():
            return ""
        content = path.read_bytes()
        if str(document.get("source_type", "")).lower() == "pdf" or path.suffix.lower() == ".pdf":
            try:
                text, _tables, _pages = self._extract_tables_from_pdf(doc_id, content)
                return text
            except Exception:
                return content.decode("utf-8", errors="ignore")
        return content.decode("utf-8", errors="ignore")

    def mark_indexed(self, doc_id: str, chunks_indexed: int, chunks_total: int) -> None:
        """Record that a document's chunks are represented in the graph."""
        catalog = self._load_catalog()
        for document in catalog:
            if document["id"] == doc_id:
                document["chunks_indexed"] = int(chunks_indexed)
                document["chunks_total"] = int(chunks_total)
                document["indexed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                document["status"] = "Indexed" if chunks_indexed else "Not indexed"
        self._save_catalog(catalog)

    def mark_needs_reindex(self, doc_id: Optional[str] = None) -> None:
        """Flag one document, or all of them, as out of step with the graph."""
        catalog = self._load_catalog()
        for document in catalog:
            if doc_id is None or document["id"] == doc_id:
                if document.get("chunks_indexed"):
                    document["status"] = "Needs re-index"
        self._save_catalog(catalog)

    def delete_document(self, doc_id: str, chroma_dir: Optional[str] = None) -> bool:
        """Remove a document, its tables, its catalog entry and its vectors."""
        catalog = self._load_catalog()
        doc_to_delete = next((d for d in catalog if d["id"] == doc_id), None)
        if not doc_to_delete:
            return False

        file_path = Path(doc_to_delete["storage_path"])
        if file_path.exists():
            file_path.unlink()

        tables_file = self.tables_dir / f"{doc_id}_tables.json"
        if tables_file.exists():
            tables_file.unlink()

        if chroma_dir:
            self.purge_vectors(doc_id, chroma_dir)

        self._save_catalog([d for d in catalog if d["id"] != doc_id])
        return True

    @staticmethod
    def purge_vectors(doc_id: str, chroma_dir: str, collection_name: str = "paper_collection") -> int:
        """Delete a document's chunks from the vector store. Returns rows removed."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = client.get_or_create_collection(name=collection_name)
            existing = collection.get(where={"document_id": doc_id})
            ids = existing.get("ids") or []
            if ids:
                collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def verify_integrity(self) -> List[Dict[str, Any]]:
        """Re-hash every retained file and report SHA-256 mismatches or missing files."""
        issues = []
        for document in self._load_catalog():
            path = Path(document.get("storage_path", ""))
            if not path.exists():
                issues.append({"id": document["id"], "problem": "file missing", "path": str(path)})
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != document.get("sha256"):
                issues.append({"id": document["id"], "problem": "sha256 mismatch", "path": str(path)})
        return issues

    def get_tables(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all structured tables from vault or for a specific document."""
        all_tables = []
        if doc_id:
            tables_file = self.tables_dir / f"{doc_id}_tables.json"
            if tables_file.exists():
                try:
                    with open(tables_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return []
            return []
        
        for p in self.tables_dir.glob("*_tables.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    all_tables.extend(json.load(f))
            except Exception:
                continue
        return all_tables

    def find_relevant_tables(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find tables whose titles, columns, or cell facts match query keywords."""
        tables = self.get_tables()
        if not tables:
            return []

        q_terms = set(re.findall(r"\w+", query.lower()))
        if not q_terms:
            return []

        scored = []
        for tab in tables:
            score = 0
            title_terms = set(re.findall(r"\w+", tab.get("title", "").lower()))
            score += len(q_terms.intersection(title_terms)) * 3

            cols_text = " ".join(tab.get("columns", [])).lower()
            cols_terms = set(re.findall(r"\w+", cols_text))
            score += len(q_terms.intersection(cols_terms)) * 2

            facts_text = tab.get("markdown", "").lower()
            facts_terms = set(re.findall(r"\w+", facts_text))
            score += len(q_terms.intersection(facts_terms))

            if score > 0:
                scored.append((score, tab))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]
