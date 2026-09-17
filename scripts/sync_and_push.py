#!/usr/bin/env python3
"""Change management and pre-push orchestration."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.change_manager import ChangeManagementAgent
from scripts.ui_qa_agent import UIQAAgent


def main():
    parser = argparse.ArgumentParser(description="Change management and pre-push orchestration")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], default=None,
                        help="Bump this part of the version")
    parser.add_argument("--msg", type=str, default=None, help="Commit message")
    parser.add_argument("--no-push", action="store_true", help="Sync and commit without pushing")
    parser.add_argument("--skip-qa", action="store_true",
                        help="Skip the QA suite (health checks still run)")
    args = parser.parse_args()

    print("=" * 72)
    print("GraphRAG knowledge workspace · change management")
    print("=" * 72)

    agent = ChangeManagementAgent(root_dir=str(ROOT_DIR))

    print("\nStep 1: health checks")
    qa = agent.run_qa_checks()
    print(f"  {qa['passed_count']} / {qa['total_count']} passed in {qa['duration']}s")
    for issue in qa["errors"]:
        print(f"  - {issue}")

    if not args.skip_qa:
        print("\nStep 2: QA suite")
        if not UIQAAgent(root_dir=ROOT_DIR).run_all():
            print("\nQA suite failed. Nothing was committed.")
            sys.exit(1)

    print("\nStep 3: version, changelog and README")
    result = agent.run_full_sync(bump=args.bump)
    version = result["version"]
    print(f"  synchronised to v{version}")

    print("\nStep 4: staging")
    subprocess.run(["git", "add", "."], cwd=str(ROOT_DIR), check=True)

    message = args.msg or f"release: sync v{version} and refresh the changelog"
    print(f"\nStep 5: commit · {message}")
    try:
        subprocess.run(["git", "commit", "-m", message], cwd=str(ROOT_DIR), check=True)
    except subprocess.CalledProcessError:
        print("  nothing to commit")

    if args.no_push:
        print("\nSkipping push (--no-push).")
        return

    print("\nStep 6: push to origin main")
    subprocess.run(["git", "push", "origin", "main"], cwd=str(ROOT_DIR), check=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
