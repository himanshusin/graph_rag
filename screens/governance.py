"""Governance screen: QA checks as a list, changelog as a timeline (mockup 1g)."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from ui import components as c
from ui import data
from ui.tokens import SUCCESS, WARN, esc


def render(state: Dict[str, Any]) -> None:
    report = state["qa"]
    checks = report.get("checks", [])
    passed = report.get("passed_count", 0)
    total = report.get("total_count", len(checks))
    last_run = str(report.get("timestamp", ""))[-8:-3] or "—"

    with st.container(key="topbar_governance", horizontal=True, vertical_alignment="center"):
        st.markdown(
            '<div class="k-topbar__title">Governance</div>'
            f'<div class="k-topbar__note">Change Management Agent · last run {esc(last_run)}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Re-run QA & sync", key="gov_sync"):
            _sync(state)

    with st.container(key="pad_governance"):
        left, right = st.columns([1, 1.2], gap="medium")

        with left:
            all_passed = passed == total and total > 0
            color = SUCCESS if all_passed else WARN
            pill_class = "k-tag" if all_passed else "k-tag k-tag--warn"
            st.markdown(
                '<div class="k-card"><div class="k-card__head">'
                '<span class="k-card__title">QA health</span>'
                f'<span class="{pill_class}" style="display:flex;align-items:center;gap:6px;'
                'border-radius:999px;padding:2px 8px">'
                f'<span class="k-dot" style="background:{color}"></span>'
                f"{passed} / {total} passed</span>"
                f'<span class="k-check__meta">{report.get("duration", 0)}s</span></div>'
                f"{c.check_list(checks)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="k-faint" style="font-size:11.5px;padding:10px 2px">'
                f'v{esc(state["version"])} · {esc(report.get("timestamp", ""))}</div>',
                unsafe_allow_html=True,
            )

        with right:
            releases = data.parse_changelog()
            if not releases:
                st.markdown(
                    c.empty_card("The changelog is generated on the first sync."),
                    unsafe_allow_html=True,
                )
                return
            st.markdown(
                '<div class="k-card"><div class="k-card__head">'
                '<span class="k-card__title">Changelog</span>'
                '<span class="k-mono k-faint" style="font-size:11px">CHANGELOG.md</span>'
                '<span class="k-muted" style="margin-left:auto;font-size:11.5px">'
                "Semantic versioning · patch on sync</span></div>"
                f'<div style="padding:16px 16px 6px">{c.changelog_timeline(releases)}</div></div>',
                unsafe_allow_html=True,
            )


def _sync(state: Dict[str, Any]) -> None:
    """Run the change management cycle with a step status, then refresh."""
    agent = data.get_change_agent()
    with st.status("Running the change management cycle", expanded=True) as status:
        status.write("Running QA checks")
        qa = agent.run_qa_checks()
        status.write(
            f"{qa['passed_count']} of {qa['total_count']} checks passed in {qa['duration']}s"
        )
        status.write("Regenerating the changelog from git history")
        agent.update_changelog()
        status.write("Synchronising the README badges")
        agent.sync_readme()
        state_label = "complete" if qa["passed"] else "error"
        status.update(
            label=f"Synced v{agent.get_version()} · {qa['passed_count']}/{qa['total_count']} checks passed",
            state=state_label,
        )
    data.qa_report.clear()
    st.rerun()
