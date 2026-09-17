#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.change_manager import ChangeManagementAgent


def main():
    print("=" * 60)
    print("🛡️  Enterprise QA & Code Quality Verification")
    print("=" * 60)

    agent = ChangeManagementAgent(root_dir=str(ROOT_DIR))
    results = agent.run_qa_checks()

    print(f"Timestamp:       {results['timestamp']}")
    print(f"Syntax Check:    {results['syntax_check']}")
    print(f"Parquet Tables:  {results['parquet_tables']}")
    print(f"ChromaDB Vault:  {results['chromadb_vault']}")
    print(f"Vault Catalog:   {results['vault_catalog']}")

    if results["errors"]:
        print("\n⚠️  Warnings / Errors:")
        for err in results["errors"]:
            print(f"  - {err}")

    if not results["passed"]:
        print("\n❌ QA Verification FAILED. Please resolve errors before pushing.")
        sys.exit(1)
    else:
        print("\n✅ QA Verification PASSED. Codebase is clean and ready for production.")
        sys.exit(0)


if __name__ == "__main__":
    main()
