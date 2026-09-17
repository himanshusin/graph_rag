#!/usr/bin/env python3
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from core.change_manager import ChangeManagementAgent


def main():
    parser = argparse.ArgumentParser(description="Automated Change Management & Pre-Push Orchestrator")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], default=None, help="Bump version part")
    parser.add_argument("--msg", type=str, default=None, help="Custom commit message")
    parser.add_argument("--no-push", action="store_true", help="Perform change management without git push")
    args = parser.parse_args()

    print("=" * 65)
    print("🤖 Enterprise Change Management & Git Automation Agent")
    print("=" * 65)

    agent = ChangeManagementAgent(root_dir=str(ROOT_DIR))
    
    # 1. Run QA checks
    print("🔍 Step 1: Running QA Verification Suite...")
    qa_res = agent.run_qa_checks()
    if not qa_res["passed"]:
        print(f"❌ QA Check Failed: {qa_res['errors']}")
        sys.exit(1)
    print("✅ QA Passed.")

    # 2. Run documentation and changelog sync
    print("📜 Step 2: Updating VERSION, CHANGELOG.md, and README.md...")
    sync_res = agent.run_full_sync(bump=args.bump)
    version = sync_res["version"]
    print(f"✅ Synchronized to version v{version}")

    # 3. Stage changes
    print("📦 Step 3: Staging changes for git commit...")
    subprocess.run(["git", "add", "."], cwd=str(ROOT_DIR), check=True)

    # 4. Commit
    commit_msg = args.msg or f"release: sync version v{version}, update changelog, and pass QA checks"
    print(f"✍️ Step 4: Creating commit: '{commit_msg}'...")
    try:
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=str(ROOT_DIR), check=True)
    except subprocess.CalledProcessError:
        print("ℹ️ No new changes to commit.")

    # 5. Push
    if not args.no_push:
        print("🚀 Step 5: Pushing to remote origin main...")
        subprocess.run(["git", "push", "origin", "main"], cwd=str(ROOT_DIR), check=True)
        print("🎉 Successfully pushed all changes with full change management audit trail!")
    else:
        print("ℹ️ Skipping push as --no-push flag was set.")


if __name__ == "__main__":
    main()
