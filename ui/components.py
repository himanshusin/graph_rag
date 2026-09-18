"""HTML fragments for the workspace UI.

Every helper returns a string so screens can compose markup and hand it to
``st.markdown(..., unsafe_allow_html=True)``. Anything interpolated from data
goes through ``esc``.
"""

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from core.rag import MODE_DESCRIPTIONS, MODE_LABELS, MODE_KEYS
from ui.tokens import (
    ACCENT, CATALOG_TABS, DANGER, FAINT, NAV, NAV_ICONS, SOURCE_FILTERS, SUCCESS, WARN,
    clip, esc, query_string,
)

THEME_PATH = Path(__file__).with_name("theme.css")

MODES = [(key, MODE_LABELS[key]) for key in MODE_KEYS]


def stylesheet() -> str:
    """Read the stylesheet once per call site and wrap it for injection."""
    return f"<style>{THEME_PATH.read_text(encoding='utf-8')}</style>"


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
def brand(version: str) -> str:
    return (
        '<div class="k-brand"><div class="k-brand__mark">G</div>'
        '<div class="k-brand__name">Knowledge</div>'
        f'<div class="k-brand__ver">v{esc(version)}</div></div>'
    )


def nav_bar(current_screen: str, nav_counts: Optional[Dict[str, int]] = None) -> str:
    """Sidebar navigation with stroke icons and the active row tinted."""
    counts = nav_counts or {}
    items = []
    for key, label in NAV:
        is_active = key == current_screen
        stroke = "#111110" if is_active else "#55554F"
        cls = "k-nav-link k-nav-link--active" if is_active else "k-nav-link"
        icon = (
            f'<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="{stroke}" '
            f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{NAV_ICONS[key]}</svg>'
        )
        badge = f'<span class="k-nav-badge">{esc(counts[key])}</span>' if key in counts else ""
        aria = ' aria-current="page"' if is_active else ""
        items.append(
            f'<a href="{query_string(screen=key)}" target="_self" class="{cls}"{aria}>'
            f"{icon}<span>{esc(label)}</span>{badge}</a>"
        )
    return f'<nav class="k-nav-menu" aria-label="Screens">{"".join(items)}</nav>'


def reasoning_modes(current_mode: str) -> str:
    """Sidebar mode list; radio semantics with a 6px indicator dot."""
    items = []
    for key, label in MODES:
        is_active = key == current_mode
        cls = "k-mode-link k-mode-link--active" if is_active else "k-mode-link"
        dot = "k-mode-dot k-mode-dot--active" if is_active else "k-mode-dot"
        items.append(
            f'<a href="{query_string(screen="search", mode=key)}" target="_self" class="{cls}" '
            f'role="radio" aria-checked="{"true" if is_active else "false"}" '
            f'title="{esc(MODE_DESCRIPTIONS[key])}">'
            f'<span class="{dot}" aria-hidden="true"></span><span>{esc(label)}</span></a>'
        )
    return (
        '<div class="k-side-label">Reasoning</div>'
        f'<div class="k-mode-menu" role="radiogroup" aria-label="Reasoning mode">{"".join(items)}</div>'
    )


def composer_modes(current_mode: str, source: str = "all") -> str:
    """Mode chips above the composer. Writes the same query key as the sidebar."""
    chips = []
    for key, label in MODES:
        is_active = key == current_mode
        cls = "k-pill k-pill--active" if is_active else "k-pill"
        dot = (
            f'<span class="k-dot" style="background:{ACCENT}" aria-hidden="true"></span>'
            if is_active else ""
        )
        chips.append(
            f'<a href="{query_string(screen="search", mode=key, source=source)}" target="_self" '
            f'class="{cls}" title="{esc(MODE_DESCRIPTIONS[key])}">{dot}{esc(label.split(" ")[0])}</a>'
        )
    return (
        '<div class="k-composer-modes"><span class="k-composer-modes__label">Mode</span>'
        f'{"".join(chips)}</div>'
    )


def sidebar_status(passed: int, total: int) -> str:
    ok = passed == total and total > 0
    color = SUCCESS if ok else WARN
    label = f"QA passed · {total} checks" if ok else f"QA {passed} / {total} checks"
    cls = "k-side-foot" if ok else "k-side-foot k-side-foot--warn"
    return (
        f'<div class="{cls}"><span class="k-dot" style="background:{color}" aria-hidden="true"></span>'
        f"<span>{esc(label)}</span></div>"
    )


def sidebar_list(title: str, items: Sequence[Dict[str, Any]]) -> str:
    """Labelled sidebar list of colored entries, e.g. communities."""
    if not items:
        return ""
    rows = []
    for item in items:
        cls = "k-side-item k-side-item--active" if item.get("active") else "k-side-item"
        swatch = (
            f'<span class="k-swatch" style="background:{item["color"]}" aria-hidden="true"></span>'
            if item.get("color") else ""
        )
        count = (
            f'<span class="k-side-item__count">{esc(item["count"])}</span>'
            if item.get("count") is not None else ""
        )
        rows.append(
            f'<a class="{cls}" href="{esc(item.get("href", "?"))}" target="_self">'
            f'{swatch}<span class="k-side-item__name">{esc(item["label"])}</span>{count}</a>'
        )
    return (
        f'<div class="k-side-label">{esc(title)}</div>'
        f'<div class="k-side-list">{"".join(rows)}</div>'
    )


# -----------------------------------------------------------------------------
# Top bar controls
# -----------------------------------------------------------------------------
def source_pills(current_filter: str, mode: str = "global") -> str:
    """Evidence source filter. Selecting one restricts eligible citation kinds."""
    current = (current_filter or "all").lower()
    items = []
    for key, label in SOURCE_FILTERS:
        cls = "k-pill k-pill--active" if key == current else "k-pill"
        items.append(
            f'<a href="{query_string(screen="search", mode=mode, source=key)}" target="_self" '
            f'class="{cls}">{esc(label)}</a>'
        )
    return f'<div class="k-pills-row" role="group" aria-label="Source filter">{"".join(items)}</div>'


def catalog_tabs(current_tab: str, counts: Optional[Dict[str, int]] = None) -> str:
    current = (current_tab or "concepts").lower()
    counts = counts or {}
    items = []
    for key, label in CATALOG_TABS:
        cls = "k-tab-link k-tab-link--active" if key == current else "k-tab-link"
        badge = f'<span class="k-tab-badge">{esc(counts[key])}</span>' if key in counts else ""
        items.append(
            f'<a href="{query_string(screen="catalog", tab=key)}" target="_self" class="{cls}">'
            f"{esc(label)}{badge}</a>"
        )
    return f'<div class="k-tabs-row">{"".join(items)}</div>'


def segmented(options: Sequence[Tuple[str, str, str]], current: str) -> str:
    """Segmented control from (key, label, href) triples."""
    items = []
    for key, label, href in options:
        cls = "k-seg--on" if key == current else ""
        items.append(f'<a class="{cls}" href="{esc(href)}" target="_self">{esc(label)}</a>')
    return f'<div class="k-seg">{"".join(items)}</div>'


# -----------------------------------------------------------------------------
# Messages
# -----------------------------------------------------------------------------
def user_message(text: str, initials: str = "") -> str:
    return (
        f'<div class="k-msg"><div class="k-msg__av" aria-hidden="true">{esc(initials)}</div>'
        f'<div class="k-msg__q">{esc(text)}</div></div>'
    )


def assistant_meta(
    mode_label: str,
    sources: int,
    elapsed: Optional[float],
    grounding: Optional[float] = None,
    selected: bool = False,
    select_href: Optional[str] = None,
) -> str:
    """Meta line above an answer: mode · n sources · elapsed · grounding."""
    parts = [
        f'<span class="k-msg__mode">{esc(mode_label)}</span>',
        "<span>·</span>",
        f"<span>{sources} sources</span>",
    ]
    if elapsed is not None:
        parts += ["<span>·</span>", f'<span class="k-mono">{elapsed:.1f}s</span>']
    if grounding is not None:
        parts += ["<span>·</span>", grounding_tag(grounding)]
    if select_href and not selected:
        parts.append(
            f'<a class="k-msg__select" href="{esc(select_href)}" target="_self">Show evidence</a>'
        )
    elif selected:
        parts.append('<span class="k-msg__select">Evidence shown</span>')

    cls = "k-msg k-msg--ai k-selected" if selected else "k-msg k-msg--ai"
    return (
        f'<div class="{cls}"><div class="k-msg__av k-msg__av--ai" aria-hidden="true">G</div>'
        f'<div class="k-msg__meta">{"".join(parts)}</div></div>'
    )


CITATION_RE = re.compile(r"\[(\d+)\](?!\()")


def mark_citations(text: str, citations: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    """Turn ``[n]`` into a superscript link to ``#cite-n``.

    Numbers with no matching citation are left as plain text, so an answer can
    never point at a card that is not in the rail.
    """
    limit = len(citations) if citations else 0

    def replace(match: "re.Match[str]") -> str:
        index = int(match.group(1))
        if 1 <= index <= limit:
            citation = citations[index - 1]
            label = f'{citation.get("kind", "Source")} {citation.get("name", "")}'.strip()
            return (
                f'<sup class="k-cite"><a href="#cite-{index}" '
                f'aria-label="Source {index}: {esc(label)}">[{index}]</a></sup>'
            )
        return match.group(0)

    return CITATION_RE.sub(replace, text or "")


def grounding_tag(score: float) -> str:
    """Grounding pill. >=90% ok, 70-89% warn, below that danger."""
    pct = int(round(score * 100)) if score <= 1.0 else int(round(score))
    pct = max(0, min(100, pct))
    if pct >= 90:
        cls = "k-tag"
    elif pct >= 70:
        cls = "k-tag k-tag--warn"
    else:
        cls = "k-tag k-tag--danger"
    return f'<span class="{cls}">grounded {pct}%</span>'


def answer_skeleton() -> str:
    """Three bars at 92 / 80 / 60%, per the loading state in §6."""
    return (
        '<div class="k-skel k-skel--pulse" role="status" aria-label="Generating answer">'
        '<div class="k-skel__bar"></div><div class="k-skel__bar"></div>'
        '<div class="k-skel__bar"></div></div>'
    )


def evidence_skeleton(count: int = 2) -> str:
    cards = "".join(
        '<div class="k-skel k-skel--card k-skel--pulse">'
        '<div class="k-skel__bar"></div><div class="k-skel__bar"></div></div>'
        for _ in range(count)
    )
    return f'<div class="k-rail__body" role="status" aria-label="Loading evidence">{cards}</div>'


def error_block(message: str, detail: str = "", retry_hint: str = "") -> str:
    """Inline error in the assistant slot, never a page-level banner."""
    detail_html = f'<div class="k-error__detail">{esc(clip(detail, 400))}</div>' if detail else ""
    hint = f'<div class="k-faint" style="font-size:11.5px">{esc(retry_hint)}</div>' if retry_hint else ""
    return (
        '<div class="k-error"><div class="k-error__head">'
        f'<span class="k-dot" style="background:{DANGER}" aria-hidden="true"></span>'
        f"{esc(message)}</div>{detail_html}{hint}</div>"
    )


# -----------------------------------------------------------------------------
# Evidence
# -----------------------------------------------------------------------------
def mini_table(columns: Sequence[str], records: Sequence[Dict[str, Any]], total_rows: int = 0) -> str:
    """Four-row preview of an extracted table, with a 'n more' footer."""
    if not columns:
        return ""
    head = "".join(f"<th>{esc(clip(c, 16))}</th>" for c in list(columns)[:3])
    body = []
    for record in list(records)[:4]:
        cells = "".join(
            f"<td>{esc(clip(record.get(c, ''), 16))}</td>" for c in list(columns)[:3]
        )
        body.append(f"<tr>{cells}</tr>")
    more = ""
    remaining = max(0, int(total_rows or 0) - min(4, len(records)))
    if remaining:
        more = (
            f'<tr><td colspan="{min(3, len(columns))}" style="color:#8A8A83">'
            f"{remaining} more rows</td></tr>"
        )
    return (
        f'<table class="k-ev__minitable"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}{more}</tbody></table>'
    )


def evidence_card(citation: Dict[str, Any], focus: bool = False, muted: bool = False) -> str:
    """One evidence card. The body varies by citation kind, per §4."""
    kind = citation.get("kind", "Source")
    name = citation.get("name", "")
    meta = citation.get("meta", "")
    index = citation.get("index", "")

    if kind == "Table":
        body = mini_table(
            citation.get("columns", []),
            citation.get("records", []),
            citation.get("rows_count", 0),
        )
        if citation.get("document_title") or citation.get("document_id"):
            body += (
                f'<div class="k-ev__file">'
                f'{esc(citation.get("document_title") or citation.get("document_id"))}</div>'
            )
    elif kind == "Passage":
        body = f'<div class="k-ev__quote">…{esc(clip(citation.get("quote", ""), 200))}…</div>'
        if citation.get("distance") is not None:
            body += f'<div class="k-ev__file k-mono">distance {citation["distance"]}</div>'
    else:
        detail = citation.get("excerpt", "")
        body = f'<div class="k-ev__text">{esc(clip(detail, 260))}</div>' if detail else ""
        if kind == "Concept":
            bits = []
            if citation.get("entity_type"):
                bits.append(str(citation["entity_type"]))
            if citation.get("degree") is not None:
                bits.append(f"{citation['degree']} relationships")
            if bits:
                body += f'<div class="k-ev__file k-mono">{esc(" · ".join(bits))}</div>'

    href = _citation_href(citation)
    classes = ["k-ev"]
    if focus:
        classes.append("k-ev--focus")
    if muted:
        classes.append("k-ev--muted")

    card_id = f' id="cite-{esc(index)}"' if index != "" else ""
    aria = f'Source {index}: {kind} {name}'
    return (
        f'<a class="{" ".join(classes)}"{card_id} href="{href}" target="_self" '
        f'aria-label="{esc(aria)}">'
        f'<div class="k-ev__head"><span class="k-ev__idx">{esc(index)}</span>'
        f'<span class="k-ev__title">{esc(kind)} · {esc(name)}</span>'
        + (f'<span class="k-ev__meta">{esc(clip(meta, 18))}</span>' if meta else "")
        + f"</div>{body}</a>"
    )


def _citation_href(citation: Dict[str, Any]) -> str:
    """Clicking a card opens the screen that owns the source."""
    kind = citation.get("kind")
    if kind == "Passage" and citation.get("document_id"):
        return query_string(screen="vault", doc=citation["document_id"])
    if kind == "Table":
        if citation.get("document_id"):
            return query_string(screen="vault", doc=citation["document_id"],
                                table=citation.get("table_id"))
        return query_string(screen="vault")
    if kind == "Concept":
        return query_string(screen="concepts", node=citation.get("name"))
    if kind == "Domain":
        return query_string(screen="catalog", tab="briefs",
                            community=citation.get("community"))
    if kind == "Relationship":
        return query_string(screen="catalog", tab="relationships",
                            term=str(citation.get("name", "")).split(" → ")[0])
    return query_string(screen="search")


def evidence_rail(
    citations: Sequence[Dict[str, Any]],
    *,
    allowed_kinds: Optional[set] = None,
    limit: int = 8,
    empty_message: str = "Citations appear here once you run a query.",
) -> str:
    """Rail body. Kinds excluded by the source filter render struck through."""
    if not citations:
        return f'<div class="k-rail__body"><div class="k-ev__more">{esc(empty_message)}</div></div>'

    cards = []
    for citation in list(citations)[:limit]:
        muted = allowed_kinds is not None and citation.get("kind") not in allowed_kinds
        cards.append(evidence_card(citation, muted=muted))
    if len(citations) > limit:
        cards.append(f'<div class="k-ev__more">{len(citations) - limit} more</div>')
    return f'<div class="k-rail__body">{"".join(cards)}</div>'


def raw_context(text: str) -> str:
    if not text:
        return '<div class="k-faint" style="padding:6px 12px;font-size:11.5px">No context captured.</div>'
    return f'<div class="k-rail__context">{esc(text)}</div>'


# -----------------------------------------------------------------------------
# Compare
# -----------------------------------------------------------------------------
def compare_head(
    title: str,
    meta: str,
    grounding: Optional[float],
    swatch: str = ACCENT,
) -> str:
    tag = grounding_tag(grounding) if grounding is not None else ""
    return (
        '<div class="k-compare__head">'
        f'<span class="k-swatch" style="background:{swatch}" aria-hidden="true"></span>'
        f'<span class="k-card__title">{esc(title)}</span>'
        f'<span class="k-compare__meta">{esc(meta)}</span>'
        f'<span style="margin-left:auto">{tag}</span></div>'
    )


def compare_footnotes(citations: Sequence[Dict[str, Any]], limit: int = 4) -> str:
    if not citations:
        return ""
    rows = "".join(
        f'<div class="k-compare__note"><span class="k-cite">[{esc(c.get("index", ""))}]</span>'
        f'<span>{esc(c.get("kind", "Source"))} · {esc(clip(c.get("name", ""), 54))}</span></div>'
        for c in list(citations)[:limit]
    )
    return f'<div class="k-compare__foot">{rows}</div>'


# -----------------------------------------------------------------------------
# Generic panels
# -----------------------------------------------------------------------------
def empty_state(title: str, text: str = "") -> str:
    body = f'<div class="k-empty__text">{esc(text)}</div>' if text else ""
    return f'<div class="k-empty"><div class="k-empty__title">{esc(title)}</div>{body}</div>'


def empty_card(message: str) -> str:
    return f'<div class="k-card"><div class="k-card__body k-faint">{esc(message)}</div></div>'


def stat_box(value: Any, label: str) -> str:
    return (
        f'<div class="k-stat"><div class="k-stat__num">{esc(value)}</div>'
        f'<div class="k-stat__label">{esc(label)}</div></div>'
    )


def status_dot(kind: str, label: str) -> str:
    """Status is never colour-only; the label always travels with the dot."""
    colors = {"ok": SUCCESS, "progress": ACCENT, "warn": WARN, "danger": DANGER, "idle": FAINT}
    color = colors.get(kind, FAINT)
    text_color = {"ok": SUCCESS, "warn": WARN, "danger": DANGER}.get(kind, "var(--k-muted)")
    return (
        f'<span style="display:flex;align-items:center;gap:6px;color:{text_color}">'
        f'<span class="k-dot" style="background:{color}" aria-hidden="true"></span>{esc(label)}</span>'
    )


def progress_bar(fraction: float) -> str:
    pct = max(0, min(100, int(round(fraction * 100))))
    return f'<span class="k-progress"><span style="width:{pct}%"></span></span>'


def step_pills(steps: Sequence[str], current: Optional[str], done: Iterable[str] = ()) -> str:
    done_set = set(done)
    pills = []
    for step in steps:
        if step in done_set:
            cls = "k-step k-step--done"
        elif step == current:
            cls = "k-step k-step--active"
        else:
            cls = "k-step"
        pills.append(f'<span class="{cls}">{esc(step)}</span>')
    return f'<div class="k-steps">{"".join(pills)}</div>'


def suggested_prompts(prompts: Sequence[Dict[str, str]]) -> str:
    """Empty-thread state: title plus the numbered starter questions."""
    if not prompts:
        return (
            '<div class="k-empty-search"><div class="k-empty-title">Ask the knowledge base.</div>'
            '<div class="k-faint" style="font-size:13px">Add a document to start.</div></div>'
        )
    rows = []
    for position, prompt in enumerate(prompts, 1):
        label = MODE_LABELS.get(prompt["mode"], prompt["mode"]).split(" ")[0]
        rows.append(
            f'<a href="{query_string(screen="search", prompt=position)}" target="_self" '
            f'class="k-suggest-row">'
            f'<span class="k-suggest-num">{position}</span>'
            f'<span class="k-suggest-text">{esc(prompt["text"])}</span>'
            f'<span class="k-suggest-badge">{esc(label)}</span></a>'
        )
    return (
        '<div class="k-empty-search"><div class="k-empty-title">Ask the knowledge base.</div>'
        f'<div class="k-suggest-box">{"".join(rows)}</div></div>'
    )


# -----------------------------------------------------------------------------
# Data tables
# -----------------------------------------------------------------------------
def data_table(
    columns: Sequence[Tuple[str, str]],
    rows: Sequence[Sequence[str]],
    grid: str,
    row_hrefs: Optional[Sequence[Optional[str]]] = None,
    active_index: Optional[int] = None,
) -> str:
    """Panelled HTML table. ``columns`` are (label, extra style) pairs.

    Cells are pre-rendered strings so callers control escaping, highlighting
    and mono treatment per column.
    """
    head_cells = "".join(
        f'<span{f" style={chr(34)}{style}{chr(34)}" if style else ""}>{esc(label)}</span>'
        for label, style in columns
    )
    head = f'<div class="k-grid__head" style="grid-template-columns:{grid}">{head_cells}</div>'

    body = []
    for position, cells in enumerate(rows):
        classes = "k-grid__row"
        if active_index is not None and position == active_index:
            classes += " k-grid__row--on"
        inner = "".join(cells)
        href = row_hrefs[position] if row_hrefs and position < len(row_hrefs) else None
        if href:
            body.append(
                f'<a class="{classes}" href="{esc(href)}" target="_self" '
                f'style="grid-template-columns:{grid}">{inner}</a>'
            )
        else:
            body.append(
                f'<div class="{classes}" style="grid-template-columns:{grid}">{inner}</div>'
            )
    return f'<div class="k-card">{head}{"".join(body)}</div>'


def table_footer(shown: int, total: int, term: str = "") -> str:
    suffix = f" · filtered by “{term}”" if term else ""
    return f'<span class="k-foot__label">Showing {shown} of {total}{esc(suffix)}</span>'


# -----------------------------------------------------------------------------
# Governance
# -----------------------------------------------------------------------------
def check_list(checks: Sequence[Dict[str, Any]]) -> str:
    """QA checks as a list, failures first, with the error text beneath."""
    ordered = sorted(checks, key=lambda c: (bool(c.get("ok")), c.get("label", "")))
    out = []
    for check in ordered:
        ok = bool(check.get("ok"))
        mark = "✓" if ok else "✕"
        cls = "k-check__ok" if ok else "k-check__bad"
        out.append(
            f'<div class="k-check"><span class="{cls}" aria-hidden="true">{mark}</span>'
            f'<span>{esc(check.get("label", ""))}</span>'
            f'<span class="k-check__meta">{esc(check.get("detail", ""))}'
            + (f' · {check["duration"]}s' if check.get("duration") is not None else "")
            + "</span></div>"
        )
        if not ok and check.get("error"):
            out.append(f'<div class="k-check__error">{esc(clip(check["error"], 400))}</div>')
    return "".join(out)


def changelog_timeline(releases: Sequence[Dict[str, Any]], commit_limit: int = 4) -> str:
    entries = []
    for position, release in enumerate(releases):
        is_current = position == 0
        commits = "".join(
            f'<div class="k-tl__commit"><span class="k-tl__sha">{esc(c["sha"])}</span>'
            f'<span>{esc(c["message"])}</span></div>'
            for c in release.get("commits", [])[:commit_limit]
        )
        extra_count = max(0, len(release.get("commits", [])) - commit_limit)
        extra = (
            f'<div class="k-faint" style="font-size:11.5px;margin-top:6px">'
            f"{extra_count} more commits</div>"
            if extra_count else ""
        )
        connector = '<span class="k-tl__line"></span>' if position < len(releases) - 1 else ""
        current_tag = (
            '<span class="k-tag k-tag--accent" style="margin-left:auto">current</span>'
            if is_current else ""
        )
        entries.append(
            '<div class="k-tl"><div class="k-tl__rail">'
            f'<span class="k-tl__dot{"" if is_current else " k-tl__dot--past"}"></span>'
            f"{connector}</div>"
            '<div class="k-tl__body"><div style="display:flex;align-items:baseline;gap:8px">'
            f'<span class="k-tl__ver">v{esc(release["version"])}</span>'
            f'<span class="k-faint" style="font-size:11.5px">{esc(release["date"])}</span>'
            f"{current_tag}</div>{commits}{extra}</div></div>"
        )
    return "".join(entries)


# -----------------------------------------------------------------------------
# Concept map inspector
# -----------------------------------------------------------------------------
def node_inspector(
    title: str,
    entity_type: str,
    community: Any,
    community_label: str,
    degree: Any,
    passages: Any,
    description: str,
    relationships: Sequence[Dict[str, Any]],
) -> str:
    rows = "".join(
        f'<div style="display:flex;gap:8px;align-items:center">'
        f'<span style="color:var(--k-text);overflow:hidden;text-overflow:ellipsis;'
        f'white-space:nowrap">{esc(r["title"])}</span>'
        f'<span class="k-mono k-faint" style="margin-left:auto;font-size:11px;flex:none">'
        f'{esc(r.get("kind", "related"))} · {esc(r.get("weight", "—"))}/10</span></div>'
        for r in relationships
    ) or '<div class="k-faint">No relationships recorded.</div>'

    return (
        f'<div><div style="font:500 17px/1.2 var(--k-sans);letter-spacing:-.01em">{esc(title)}</div>'
        '<div style="margin-top:4px;display:flex;gap:6px;font:11px var(--k-mono);color:var(--k-faint)">'
        f'<span>{esc(entity_type)}</span><span>·</span>'
        f'<span style="color:{ACCENT}">{esc(clip(community_label, 32))}</span></div></div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">'
        f'{stat_box(degree, "degree")}{stat_box(community, "community")}'
        f'{stat_box(passages, "passages")}</div>'
        f'<div class="k-muted" style="line-height:1.55">{esc(description)}</div>'
        '<div><div class="k-side-label" style="margin:0 0 8px;padding:0">Relationships</div>'
        f'<div style="display:grid;gap:6px;color:var(--k-muted)">{rows}</div></div>'
    )


# -----------------------------------------------------------------------------
# Provider routing (sidebar)
# -----------------------------------------------------------------------------
def provider_rows(rows) -> str:
    """Engine list with a live readiness dot and the honest trade for each route.

    Cost, speed and privacy are shown together because they move in opposite
    directions: the free route is the slow one, and the fast cheap route is the
    one that sends the corpus off the machine.
    """
    from ui.tokens import esc

    out = []
    for row in rows:
        ok = row["ok"]
        colour = "#1F8A5B" if ok else "#8A8A83"
        weight = "600" if row["active"] else "400"
        border = "1px solid var(--k-accent)" if row["active"] else "1px solid transparent"
        out.append(
            f'<div style="display:flex;flex-direction:column;gap:2px;padding:6px 8px;'
            f'border-radius:6px;border:{border};margin-bottom:2px">'
            f'<div style="display:flex;align-items:center;gap:6px">'
            f'<span class="k-dot" style="background:{colour}"></span>'
            f'<span style="font-size:12px;font-weight:{weight}">{esc(row["label"])}</span>'
            f'<span class="k-mono k-faint" style="margin-left:auto;font-size:10px">'
            f'{esc(row["status"])}</span></div>'
            f'<div class="k-faint" style="font-size:10.5px;padding-left:14px">'
            f'{esc(row["trade"])}</div></div>'
        )
    return "".join(out)


def spend_badge(label: str, value: str, tone: str = "") -> str:
    from ui.tokens import esc

    colour = {"warn": "#B7791F", "ok": "#1F8A5B"}.get(tone, "var(--k-muted)")
    return (
        f'<span class="k-mono" style="font-size:10.5px;color:{colour};margin-right:8px">'
        f'{esc(label)} <b>{esc(value)}</b></span>'
    )
