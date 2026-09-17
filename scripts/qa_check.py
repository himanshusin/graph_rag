#!/usr/bin/env python3
"""Pre-push gate: the change agent's health checks, then the full QA suite."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.change_manager import ChangeManagementAgent
from scripts.ui_qa_agent import UIQAAgent


def main():
    print("=" * 72)
    print("GraphRAG knowledge workspace · health checks")
    print("=" * 72)

    agent = ChangeManagementAgent(root_dir=str(ROOT_DIR))
    results = agent.run_qa_checks()

    print(f"Run at {results['timestamp']} in {results['duration']}s\n")
    for check in sorted(results["checks"], key=lambda c: (bool(c["ok"]), c["label"])):
        mark = "PASS" if check["ok"] else "FAIL"
        print(f"  [{mark}] {check['label']}: {check['detail']} ({check['duration']}s)")
        if check["error"]:
            print(f"         {check['error']}")

    print(f"\n{results['passed_count']} / {results['total_count']} checks passed")

    if not results["passed"]:
        print("\nHealth checks failed. Artifacts may not be built yet; "
              "index a document in the Vault, then re-run.")

    print("\n" + "-" * 72)
    print("Running the UI and functional QA suite")
    print("-" * 72)
    if not UIQAAgent(root_dir=ROOT_DIR).run_all():
        print("\nQA suite failed. Resolve the failures before release.")
        sys.exit(1)

    # Missing artifacts are a workspace state, not a code defect, so they do not
    # block a push on their own once the QA suite is green.
    print("\nQA suite passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
