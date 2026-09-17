"""Design tokens and small text helpers shared by the UI layer.

The values here mirror ``ui/theme.css``; Python needs them for the few places
that build inline styles (community swatches, status dots).
"""

import html
from typing import Any, Dict, List, Optional

CANVAS = "#F7F7F5"
PANEL = "#FFFFFF"
SUBTLE = "#FAFAF8"
HOVER = "#F1F1EE"
BORDER = "#E8E8E4"
BORDER_STRONG = "#E0E0DB"
BORDER_INPUT = "#D6D6D0"
BORDER_FAINT = "#EFEFEC"
TEXT = "#111110"
MUTED = "#55554F"
FAINT = "#8A8A83"
ACCENT = "#3056D3"
ACCENT_SOFT = "#EEF2FD"
ACCENT_BORDER = "#C9D3F5"
ACCENT_BG = "#FBFCFF"
SUCCESS = "#1F8A5B"
SUCCESS_BG = "#E8F5EE"
WARN = "#B7791F"
WARN_BG = "#FBF3E4"
DANGER = "#B03A2E"
HIGHLIGHT = "#FFF0B3"

# Developer guide §2: community colors in order, never more than 8 hues.
COMMUNITY_COLORS = [
    ACCENT, SUCCESS, WARN, FAINT, DANGER, "#6B4FBB", "#2A8FA8", "#C25E9A",
]

SIDEBAR_WIDTH = 212
RAIL_WIDTH = 300
THREAD_WIDTH = 760

NAV = [
    ("search", "Search"),
    ("vault", "Vault"),
    ("concepts", "Concept map"),
    ("catalog", "Catalog"),
    ("governance", "Governance"),
]

NAV_ICONS = {
    "search": '<circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5L14 14"/>',
    "vault": '<rect x="2" y="3" width="12" height="10" rx="1.5"/><path d="M2 7h12"/>',
    "concepts": (
        '<circle cx="4" cy="12" r="2"/><circle cx="12" cy="12" r="2"/>'
        '<circle cx="8" cy="4" r="2"/><path d="M7 5.5L5 10M9 5.5l2 4.5"/>'
    ),
    "catalog": (
        '<rect x="2" y="2" width="5" height="5" rx="1"/><rect x="9" y="2" width="5" height="5" rx="1"/>'
        '<rect x="2" y="9" width="5" height="5" rx="1"/><rect x="9" y="9" width="5" height="5" rx="1"/>'
    ),
    "governance": '<path d="M8 2l5 2v4c0 3-2.2 5-5 6-2.8-1-5-3-5-6V4l5-2z"/>',
}

SOURCE_FILTERS = [
    ("all", "All sources"),
    ("documents", "Documents"),
    ("graph", "Graph"),
    ("tables", "Tables"),
]

CATALOG_TABS = [
    ("concepts", "Concepts"),
    ("relationships", "Relationships"),
    ("nodes", "Nodes"),
    ("briefs", "Domain briefs"),
]


def esc(value: Any) -> str:
    """HTML-escape any value for interpolation into a fragment."""
    return html.escape(str(value), quote=True)


def clip(text: Any, limit: int) -> str:
    """Collapse whitespace and truncate with an ellipsis."""
    collapsed = " ".join(str(text or "").split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit].rstrip() + "…"


def community_color(community: Any) -> str:
    try:
        return COMMUNITY_COLORS[int(community) % len(COMMUNITY_COLORS)]
    except (TypeError, ValueError):
        return FAINT


def highlight(text: Any, term: Optional[str]) -> str:
    """Escape text, then wrap case-insensitive matches of ``term`` in ``<mark>``."""
    safe = esc(text)
    if not term:
        return safe
    needle = esc(term).lower()
    if not needle:
        return safe
    lowered = safe.lower()
    out: List[str] = []
    cursor = 0
    while True:
        found = lowered.find(needle, cursor)
        if found == -1:
            out.append(safe[cursor:])
            return "".join(out)
        out.append(safe[cursor:found])
        out.append(f"<mark>{safe[found:found + len(needle)]}</mark>")
        cursor = found + len(needle)


def query_string(**params: Any) -> str:
    """Build a same-page query string, dropping empty values."""
    from urllib.parse import quote as urlquote

    pairs = [
        f"{key}={urlquote(str(value))}"
        for key, value in params.items()
        if value not in (None, "")
    ]
    return "?" + "&".join(pairs) if pairs else "?"


def screen_style(screen: str, nav_counts: Optional[Dict[str, int]] = None) -> str:
    """Per-screen CSS that cannot live in the static sheet.

    The composer is only reserved space beside the evidence rail on Search;
    every other screen lets it span the full width.
    """
    if screen == "search":
        return (
            "<style>@media (min-width:1201px){"
            f'[data-testid="stBottom"]{{right:{RAIL_WIDTH}px !important;}}'
            "}</style>"
        )
    return ""
