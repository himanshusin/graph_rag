#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.change_manager import ChangeManagementAgent
from scripts.ui_qa_agent import UIQAAgent


def main():
    print("=" * 60)
    print("Enterprise QA & UI Mockups Compliance Verification")
    print("=" * 60)

    agent = ChangeManagementAgent(root_dir=str(ROOT_DIR))
    results = agent.run_qa_checks()

    print(f"Timestamp:       {results['timestamp']}")
    print(f"Syntax Check:    {results['syntax_check']}")
    print(f"Parquet Tables:  {results['parquet_tables']}")
    print(f"ChromaDB Vault:  {results['chromadb_vault']}")
    print(f"Vault Catalog:   {results['vault_catalog']}")
    print(f"Structured Tabs: {results.get('structured_tables', 'PASSED')}")

    if results["errors"]:
        print("\nWarnings / Errors:")
        for err in results["errors"]:
            print(f"  - {err}")

    if not results["passed"]:
        print("\nQA Verification FAILED. Please resolve errors before release.")
        sys.exit(1)

    print("\n--- Running UI Mockup & End-to-End QA Suite ---")
    ui_agent = UIQAAgent(root_dir=ROOT_DIR)
    ui_passed = ui_agent.run_all()

    if not ui_passed:
        print("\nUI Verification FAILED. Please resolve errors before release.")
        sys.exit(1)

    print("\nQA & UI Verification PASSED. Codebase and UI design are verified for release.")
    sys.exit(0)


if __name__ == "__main__":
    main()
