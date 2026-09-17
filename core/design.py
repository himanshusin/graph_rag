"""Design tokens, stylesheet and HTML fragments for the Knowledge workspace UI."""

import html
from typing import Dict, List, Optional
from urllib.parse import quote

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

COMMUNITY_COLORS = [
    ACCENT,
    SUCCESS,
    WARN,
    FAINT,
    DANGER,
    "#6B4FBB",
    "#2A8FA8",
    "#C25E9A",
]

SIDEBAR_WIDTH = 212
# Evidence rail share of the main area, matching the st.columns([1, 0.42]) split.
RAIL_RATIO = round(0.42 / 1.42, 4)

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

NAV = [
    ("search", "Search"),
    ("vault", "Vault"),
    ("concepts", "Concept map"),
    ("catalog", "Catalog"),
    ("governance", "Governance"),
]

MODES = [
    ("global", "Global synthesis"),
    ("local", "Local lookup"),
    ("drift", "DRIFT deep-dive"),
    ("vector", "Vector passages"),
    ("compare", "Compare graph vs vector"),
]


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def _icon_mask(paths: str) -> str:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" '
        'stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
        f"{paths}</svg>"
    )
    return f'url("data:image/svg+xml,{quote(svg)}")'


def stylesheet(nav_counts: Optional[Dict[str, int]] = None) -> str:
    """Full app stylesheet with exact mockup design tokens and components."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap');

:root {
  --k-canvas:#F7F7F5; --k-panel:#FFFFFF; --k-subtle:#FAFAF8; --k-hover:#F1F1EE;
  --k-border:#E8E8E4; --k-border-strong:#E0E0DB; --k-border-input:#D6D6D0;
  --k-border-faint:#EFEFEC;
  --k-text:#111110; --k-muted:#55554F; --k-faint:#8A8A83;
  --k-accent:#3056D3; --k-accent-soft:#EEF2FD; --k-accent-border:#C9D3F5;
  --k-accent-bg:#FBFCFF;
  --k-success:#1F8A5B; --k-success-bg:#E8F5EE; --k-warn:#B7791F; --k-warn-bg:#FBF3E4;
  --k-danger:#B03A2E;
  --k-sans:'Geist',system-ui,-apple-system,'Segoe UI',sans-serif;
  --k-mono:'Geist Mono',ui-monospace,'SF Mono',monospace;
}

html, body, .stApp, [class*="st-"], button, input, textarea, select {
  font-family:var(--k-sans);
  -webkit-font-smoothing:antialiased;
}
.stApp { background:var(--k-canvas); color:var(--k-text); }
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer { display:none !important; }
[data-testid="stMainBlockContainer"], [data-testid="stAppViewBlockContainer"] {
  padding:0 !important; max-width:none !important;
}
[data-testid="stMain"], [data-testid="stAppScrollToBottomContainer"] { overflow:hidden !important; }
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] { gap:0; }
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"] { flex:0 0 auto; }
[data-testid="stHorizontalBlock"] { gap:0; }
[data-testid="stElementContainer"] { width:100%; }
/* Streamlit offsets its default 1rem block gap with a negative margin; we set gap:0 */
[data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
a { color:var(--k-accent); text-decoration:none; }
a:hover { text-decoration:none; }

/* each screen fills the viewport; only the thread / content area scrolls.
   The composer is fixed between the 212px sidebar and 300px evidence rail. */
[data-testid="stBottom"] {
  position:fixed !important; left:212px !important;
  right:300px !important;
  bottom:0 !important; width:auto !important; background:transparent !important;
  z-index:99 !important;
}
.st-key-thread, [class*="st-key-pad_"] { height:calc(100vh - 48px); overflow-y:auto; }

/* ---------------------------------------------------------------- sidebar */
[data-testid="stSidebar"] {
  width:212px !important; min-width:212px !important; max-width:212px !important;
  background:#FFFFFF !important; border-right:1px solid #E8E8E4 !important;
}
[data-testid="stSidebarContent"] { padding:0 !important; background:#FFFFFF !important; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding:14px 10px 12px !important; }
[data-testid="stSidebarHeader"], [data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] { display:none !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:0; }

.k-brand { display:flex; align-items:center; gap:8px; padding:2px 8px 16px; }
.k-brand__mark {
  width:22px; height:22px; flex:none; border-radius:6px; background:#111110; color:#fff;
  font:600 11px var(--k-mono); display:grid; place-items:center;
}
.k-brand__name { font-weight:600; font-size:14px; color:#111110; }
.k-brand__ver { margin-left:auto; font:500 10px var(--k-mono); color:#8A8A83; }

.k-side-label {
  padding:22px 8px 8px; margin:0; line-height:1.3;
  font:500 10.5px var(--k-mono); letter-spacing:.06em; color:#8A8A83; text-transform:uppercase;
}

/* Sidebar semantic navigation list */
.k-nav-menu { display:flex; flex-direction:column; gap:2px; width:100%; }
.k-nav-link {
  display:flex; align-items:center; gap:9px; padding:7px 8px; border-radius:6px;
  color:#55554F; font-size:13px; line-height:1.3; text-decoration:none !important;
  transition:background 100ms ease, color 100ms ease;
}
.k-nav-link:hover { background:#FAFAF8; color:#111110; text-decoration:none !important; }
.k-nav-link--active {
  background:#F1F1EE !important; color:#111110 !important; font-weight:500;
}
.k-nav-link--active svg { stroke:#111110 !important; }
.k-nav-badge {
  margin-left:auto; font:500 10.5px var(--k-mono); color:#8A8A83;
}

/* Sidebar reasoning modes */
.k-mode-menu { display:flex; flex-direction:column; gap:1px; width:100%; margin-top:2px; }
.k-mode-link {
  display:flex; align-items:center; gap:8px; padding:6px 8px; border-radius:6px;
  color:#55554F; font-size:12px; line-height:1.3; text-decoration:none !important;
  transition:background 100ms ease, color 100ms ease;
}
.k-mode-link:hover { background:#FAFAF8; color:#111110; text-decoration:none !important; }
.k-mode-link--active { color:#111110 !important; font-weight:500; }
.k-mode-dot {
  width:6px; height:6px; border-radius:50%; border:1px solid #D6D6D0; box-sizing:border-box; flex:none;
}
.k-mode-dot--active {
  background:#3056D3 !important; border-color:#3056D3 !important;
}

/* sidebar sliders */
.st-key-controls { margin-top:18px; padding:0 8px; }
.st-key-controls [data-testid="stWidgetLabel"] p { font-size:12px; color:#55554F; }
[data-testid="stSliderThumbValue"] p { font:500 11px var(--k-mono); color:#111110; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { display:none; }

.k-side-foot {
  margin-top:auto; padding:8px; display:flex; align-items:center; gap:7px;
  font-size:11.5px; color:#1F8A5B;
}
.k-side-foot--warn { color:#B7791F; }

/* ------------------------------------------------------------------ chrome */
[class*="st-key-topbar_"] {
  height:48px; flex:none; box-sizing:border-box; align-items:center; gap:6px;
  padding:0 24px; border-bottom:1px solid var(--k-border); background:var(--k-panel);
  font-size:13px; flex-wrap:nowrap; overflow:hidden;
}
[class*="st-key-topbar_"] [data-testid="stMarkdown"],
[class*="st-key-topbar_"] [data-testid="stMarkdown"] > div,
[class*="st-key-topbar_"] [data-testid="stMarkdownContainer"] {
  display:flex; align-items:center; gap:6px; width:auto; white-space:nowrap; font-size:13px;
}
[class*="st-key-topbar_"] > [data-testid="stElementContainer"],
[class*="st-key-topbar_"] > [data-testid="stLayoutWrapper"] { width:auto; flex:none; }
[class*="st-key-topbar_"] > *:last-child { margin-left:auto; flex:none; }
.k-topbar__title { font-weight:500; }
.k-topbar__sep { color:#B4B4AE; padding:0 4px; }
.k-topbar__crumb { color:var(--k-muted); overflow:hidden; text-overflow:ellipsis; }
.k-topbar__note { font-size:12px; color:var(--k-faint); margin-left:8px; }

[class*="st-key-pad_"] { padding:24px; gap:20px; }
.k-card {
  background:var(--k-panel); border:1px solid var(--k-border); border-radius:10px;
  overflow:hidden; font-size:12.5px;
}
.k-card__head {
  display:flex; align-items:center; gap:10px; padding:14px 16px;
  border-bottom:1px solid var(--k-border);
}
.k-card__title { font-weight:500; }
.k-card__body { padding:16px 18px; }
.k-muted { color:var(--k-muted); }
.k-faint { color:var(--k-faint); }
.k-mono { font-family:var(--k-mono); }
.k-num { font-family:var(--k-mono); font-size:12px; }
.k-clip { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.k-swatch { width:8px; height:8px; border-radius:2px; display:inline-block; }
.k-dot { width:6px; height:6px; border-radius:50%; display:inline-block; }
.k-tag {
  font:500 10.5px var(--k-mono); padding:2px 6px; border-radius:4px;
  background:var(--k-success-bg); color:var(--k-success);
}
.k-tag--warn { background:var(--k-warn-bg); color:var(--k-warn); }
.k-tag--accent { background:var(--k-accent-soft); color:var(--k-accent); }

.k-empty { padding:80px 0 0; text-align:center; }
.k-empty__title { font:500 20px/1.2 var(--k-sans); letter-spacing:-.02em; }
.k-empty__text {
  margin:10px auto 0; max-width:420px; font-size:13px; color:var(--k-muted); line-height:1.55;
}

/* ------------------------------------------------------------------ thread */
.st-key-thread { padding:28px 40px 96px; max-width:760px; margin:0 auto; width:100%; }
.k-msg { display:flex; gap:12px; margin-bottom:24px; }
.k-msg--ai { margin-bottom:10px; align-items:center; }
.k-msg__av {
  width:24px; height:24px; flex:none; display:grid; place-items:center;
  border-radius:50%; background:var(--k-border-strong); color:var(--k-muted); font:500 10px var(--k-sans);
}
.k-msg__av--ai { border-radius:6px; background:var(--k-text); color:#fff; font:600 10px var(--k-mono); }
.k-msg__q { font-size:15px; font-weight:500; letter-spacing:-.01em; padding-top:2px; }
.k-msg__meta { display:flex; align-items:center; gap:8px; font-size:11.5px; color:var(--k-faint); }
.k-msg__mode { color:var(--k-accent); font-weight:500; }

[class*="st-key-ans_msg_"], [class*="st-key-cmprow_"], [class*="st-key-link_actions_"] { padding-left:36px; }
[class*="st-key-ans_"] p, [class*="st-key-ans_"] li {
  font-size:14px; line-height:1.6; color:var(--k-text);
}
[class*="st-key-ans_"] h1, [class*="st-key-ans_"] h2,
[class*="st-key-ans_"] h3, [class*="st-key-ans_"] h4 {
  font-size:14px; font-weight:600; margin:14px 0 6px; padding:0; letter-spacing:0;
}
[class*="st-key-ans_"] code { font-family:var(--k-mono); font-size:13px; color:var(--k-text); }
[class*="st-key-ans_"] table { font-size:12px; }
.k-cite { font:500 10.5px var(--k-mono); color:var(--k-accent); margin-left:2px; vertical-align:super; }
.k-cite a { color:var(--k-accent); text-decoration:none; }
.k-cite a:hover { text-decoration:underline; }
[class*="st-key-link_actions_"] {
  gap:14px; align-items:center; margin-bottom:24px; padding-top:4px;
}
.k-actions__model { margin-left:auto; font-size:11px; color:var(--k-faint); }
[class*="st-key-cmprow_"] [data-testid="stHorizontalBlock"] { gap:16px; }
[class*="st-key-cmpcard_"] {
  border:1px solid var(--k-border); border-radius:10px; background:var(--k-panel);
  overflow:hidden; gap:0;
}
[class*="st-key-cmpcard_"] [class*="st-key-ans_cmp_"] { padding:14px 16px; }
[class*="st-key-cmpcard_"] [class*="st-key-ans_cmp_"] p { font-size:13.5px; }
.k-compare__head {
  display:flex; align-items:center; gap:8px; padding:12px 16px;
  border-bottom:1px solid var(--k-border); font-size:13px; white-space:nowrap;
}
.k-compare__meta {
  font-size:11.5px; color:var(--k-faint); overflow:hidden; text-overflow:ellipsis;
}
[data-testid="stBottomBlockContainer"]::after {
  content:"Answers cite vault documents and graph nodes. Verify numbers before publishing.";
  display:block; text-align:center; font-size:11px; color:var(--k-faint); padding:8px 0 10px;
}

/* ------------------------------------------------------------------- rail */
.k-rail {
  background:var(--k-panel); border-left:1px solid var(--k-border);
  height:100vh; overflow-y:auto;
}
.k-rail__head {
  height:48px; box-sizing:border-box; display:flex; align-items:center; gap:6px;
  padding:0 16px; border-bottom:1px solid var(--k-border); font-weight:500; font-size:13px;
}
.k-rail__count { font:500 11px var(--k-mono); color:var(--k-faint); }
.k-rail__body { padding:12px; display:flex; flex-direction:column; gap:8px; font-size:12.5px; }
.k-rail__body--node { padding:16px; gap:14px; }
.k-ev {
  border:1px solid var(--k-border); border-radius:8px; padding:10px 12px;
  scroll-margin-top:12px; transition:border-color 120ms ease, background 120ms ease;
  display:block; text-decoration:none; color:inherit;
}
.k-ev:hover, .k-ev:target, .k-ev--focus {
  border-color:var(--k-accent-border); background:var(--k-accent-bg); color:inherit;
}
.k-ev__head { display:flex; align-items:center; gap:8px; }
.k-ev__idx {
  font:500 10.5px var(--k-mono); color:var(--k-accent); background:var(--k-accent-soft);
  padding:1px 6px; border-radius:4px;
}
.k-ev__title { font-weight:500; font-size:12.5px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.k-ev__meta { margin-left:auto; font:500 10px var(--k-mono); color:var(--k-faint); white-space:nowrap; }
.k-ev__text { margin-top:6px; font-size:12px; color:var(--k-muted); line-height:1.5; }
.k-ev__quote {
  margin-top:6px; font:12px/1.5 var(--k-mono); color:var(--k-muted);
  background:var(--k-canvas); border-radius:6px; padding:8px 10px;
}
.k-ev__minitable { width:100%; border-collapse:collapse; margin-top:6px; font:11px/1.3 var(--k-mono); }
.k-ev__minitable th { background:var(--k-canvas); padding:4px 6px; text-align:left; font-weight:500; border-bottom:1px solid var(--k-border); }
.k-ev__minitable td { padding:4px 6px; border-bottom:1px solid var(--k-border-faint); color:var(--k-muted); }
.k-ev__more { text-align:center; font-size:11.5px; color:var(--k-faint); padding-top:4px; }
.k-graph__note { padding:10px 24px; font-size:11px; color:var(--k-faint); }

/* suggested prompts on empty search */
.k-suggested { display:flex; flex-direction:column; gap:8px; max-width:680px; margin:24px auto 0; }
.k-suggest-item {
  display:flex; align-items:center; gap:12px; padding:10px 14px; border:1px solid var(--k-border);
  border-radius:8px; background:var(--k-panel); cursor:pointer; text-decoration:none; color:var(--k-text);
  transition:border-color 120ms ease, background 120ms ease;
}
.k-suggest-item:hover { border-color:var(--k-border-strong); background:var(--k-subtle); color:var(--k-text); }
.k-suggest-num { font:500 11px var(--k-mono); color:var(--k-faint); width:14px; }
.k-suggest-text { font-size:13px; font-weight:500; flex:1; }
.k-suggest-mode {
  font:500 10.5px var(--k-mono); color:var(--k-accent); background:var(--k-accent-soft);
  padding:2px 7px; border-radius:4px;
}
.k-ev__more { text-align:center; font-size:11.5px; color:var(--k-faint); padding-top:4px; }
.k-graph__note { padding:10px 24px; font-size:11px; color:var(--k-faint); }

/* ------------------------------------------------------------------ tables */
.k-grid__head, .k-grid__row { display:grid; gap:12px; align-items:center; }
.k-grid__head {
  padding:9px 16px; color:var(--k-faint); font-size:11px;
  border-bottom:1px solid var(--k-border); background:var(--k-subtle);
}
.k-grid__row {
  padding:11px 16px; border-bottom:1px solid var(--k-border-faint);
  font-size:12.5px; color:var(--k-text);
}
.k-grid__row:last-child { border-bottom:none; }
.k-grid__row--on { background:var(--k-accent-bg); }
a.k-grid__row, a.k-grid__row:hover { color:var(--k-text); text-decoration:none; }
a.k-grid__row:hover { background:var(--k-subtle); }
.k-grid__name { font-weight:500; }
.k-grid__id { font:11px var(--k-mono); color:var(--k-faint); }
.k-foot__label { font-size:11.5px; color:var(--k-faint); }
.k-foot__page { font-size:11.5px; color:var(--k-text); }
mark { background:#FFF0B3; color:inherit; padding:0 1px; }

.k-kv { display:grid; grid-template-columns:110px 1fr; row-gap:8px; column-gap:12px; color:var(--k-muted); }
.k-kv > span:nth-child(even) { color:var(--k-text); }
.k-stat { border:1px solid var(--k-border); border-radius:7px; padding:8px 10px; }
.k-stat__num { font:500 15px var(--k-mono); }
.k-stat__label { font-size:11px; color:var(--k-faint); }

/* -------------------------------------------------------------- governance */
.k-check {
  display:flex; align-items:center; gap:10px; padding:10px 16px;
  border-bottom:1px solid var(--k-border-faint); font-size:12.5px;
}
.k-check:last-child { border-bottom:none; }
.k-check__ok { color:var(--k-success); }
.k-check__bad { color:var(--k-danger); }
.k-check__meta { margin-left:auto; font:11px var(--k-mono); color:var(--k-faint); }
.k-tl { display:flex; gap:14px; }
.k-tl__rail { display:flex; flex-direction:column; align-items:center; }
.k-tl__dot { width:8px; height:8px; border-radius:50%; background:var(--k-text); margin-top:5px; flex:none; }
.k-tl__dot--past { background:var(--k-panel); border:1px solid var(--k-border-input); }
.k-tl__line { width:1px; flex:1; background:var(--k-border); margin:4px 0; }
.k-tl__body { padding-bottom:18px; flex:1; }
.k-tl__ver { font:500 13px var(--k-mono); }
.k-tl__commit { display:flex; gap:8px; color:var(--k-muted); font-size:12.5px; margin-top:6px; }
.k-tl__sha { font:11px var(--k-mono); color:var(--k-faint); width:52px; flex:none; }

/* -------------------------------------------------------------- components */
[data-testid="stBottomBlockContainer"] {
  background:transparent; padding:0 40px 6px; max-width:760px; margin:0 auto;
}
[data-testid="stChatInput"] {
  background:var(--k-panel); border:1px solid var(--k-border-input); border-radius:10px;
  box-shadow:0 1px 2px rgba(0,0,0,.04);
}
[data-testid="stChatInput"] textarea { font-size:14px; color:var(--k-text); }
.stButton > button {
  border-radius:7px; border:1px solid var(--k-border-input); background:var(--k-panel);
  color:var(--k-text); font-size:12px; font-weight:500; padding:6px 12px; min-height:0;
}
.stButton > button:hover { border-color:var(--k-faint); color:var(--k-text); }
.stButton > button[kind="primary"] { background:var(--k-text); border-color:var(--k-text); color:#fff; }
.stButton > button[kind="primary"]:hover { background:#000; color:#fff; }
.stDownloadButton > button {
  border-radius:7px; border:1px solid var(--k-border-input); background:var(--k-panel);
  color:var(--k-text); font-size:12px; font-weight:500; padding:6px 12px; min-height:0;
}
[class*="st-key-link_"] { gap:12px; align-items:center; }
[class*="st-key-link_"] .stButton > button {
  border:none; background:none; color:var(--k-faint); padding:0; font-weight:400;
}
[class*="st-key-link_"] .stButton > button:hover { color:var(--k-text); background:none; }
[class*="st-key-link_"] [data-testid="stElementContainer"] { width:auto; }
[data-testid="stTextInputRootElement"] {
  border:1px solid var(--k-border-strong); border-radius:7px; background:var(--k-panel);
}
[data-testid="stTextInputRootElement"]:focus-within { border-color:var(--k-faint); }
[data-testid="stTextInputField"] { font-size:12px; color:var(--k-text); background:transparent; }
[data-testid="stTextInput"] input::placeholder { color:var(--k-faint); }
.st-key-search_catalog { width:240px; flex:none; margin-left:auto; }
.st-key-find_node { width:220px; flex:none; margin-left:auto; }
[class*="st-key-topbar_"] [data-testid="stTextInput"] input,
[class*="st-key-topbar_"] [data-baseweb="select"] > div { min-height:30px; }
[data-baseweb="select"] > div {
  border-radius:7px; border-color:var(--k-border-strong); background:var(--k-panel); font-size:12px;
}
[data-testid="stIFrame"] { display:block; width:100%; border:none; }
[data-testid="stExpander"] {
  border:1px solid var(--k-border); border-radius:10px; background:var(--k-panel); margin-top:10px;
}
[data-testid="stExpander"] summary { font-size:12.5px; font-weight:500; }
[data-testid="stFileUploaderDropzone"] {
  background:var(--k-subtle); border:1px dashed var(--k-border-input); border-radius:8px;
}
.st-key-card_upload {
  background:var(--k-panel); border:1px solid var(--k-border); border-radius:10px;
  padding:16px 18px; margin-bottom:20px; gap:12px;
}
[data-testid="stAlert"] { border-radius:8px; font-size:12.5px; }
[data-testid="stIFrame"] { background:var(--k-panel); }

/* ----------------------------------------------------------- mockup components */
/* horizontal pill row matching mockup 1a */
.k-pills-row { display:flex; gap:6px; align-items:center; }
.k-pill {
  padding:4px 10px; border-radius:999px; border:1px solid var(--k-border-strong);
  font-size:11.5px; color:var(--k-muted); text-decoration:none !important; line-height:1.35;
  transition:all 100ms ease; display:inline-flex; align-items:center;
}
.k-pill:hover { border-color:var(--k-faint); color:var(--k-text); text-decoration:none !important; }
.k-pill--active {
  background:var(--k-text) !important; border-color:var(--k-text) !important;
  color:#FFFFFF !important; font-weight:500;
}

/* horizontal tab row matching mockup 1f */
.k-tabs-row { display:flex; gap:18px; height:48px; font-size:12.5px; align-items:stretch; }
.k-tab-link {
  display:flex; align-items:center; color:var(--k-muted); text-decoration:none !important;
  transition:color 100ms ease; padding:0 2px;
}
.k-tab-link:hover { color:var(--k-text); text-decoration:none !important; }
.k-tab-link--active {
  color:var(--k-text) !important; font-weight:500;
  border-bottom:2px solid var(--k-text) !important;
}
.k-tab-badge { margin-left:6px; font:11px var(--k-mono); color:var(--k-faint); }

/* hairline suggested prompts matching mockup 1a/1c */
.k-empty-search { max-width:680px; margin:70px auto 0; text-align:center; padding:0 20px; }
.k-empty-title { font:500 24px/1.2 var(--k-sans); letter-spacing:-.025em; color:var(--k-text); margin-bottom:24px; }
.k-suggest-box {
  border:1px solid var(--k-border); border-radius:10px; background:#FFFFFF;
  overflow:hidden; text-align:left; box-shadow:0 1px 3px rgba(0,0,0,.03);
}
.k-suggest-row {
  display:flex; align-items:center; gap:12px; padding:12px 16px;
  border-bottom:1px solid var(--k-border-faint); color:var(--k-text);
  text-decoration:none !important; font-size:13.5px; transition:background 100ms ease;
}
.k-suggest-row:last-child { border-bottom:none; }
.k-suggest-row:hover { background:var(--k-subtle); color:var(--k-text); text-decoration:none !important; }
.k-suggest-num { font:500 11px var(--k-mono); color:var(--k-faint); width:16px; }
.k-suggest-text { flex:1; font-weight:450; color:var(--k-text); }
.k-suggest-badge {
  font:500 10.5px var(--k-mono); color:var(--k-accent); background:var(--k-accent-soft);
  padding:2px 7px; border-radius:4px; margin-left:auto;
}

/* layout columns: thread + 300px white evidence rail */
.st-key-split { height:100vh; overflow:hidden; }
.st-key-split > div[data-testid="stHorizontalBlock"] {
  display:flex !important; flex-direction:row !important; flex-wrap:nowrap !important;
  height:100vh !important; overflow:hidden !important; gap:0 !important;
}
.st-key-split > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {
  flex:1 1 auto !important; min-width:0 !important; max-width:none !important; width:auto !important;
  height:100vh !important; overflow-y:auto !important; background:var(--k-canvas) !important;
}
.st-key-split > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child {
  flex:0 0 300px !important; width:300px !important; min-width:300px !important; max-width:300px !important;
  height:100vh !important; background:#FFFFFF !important; border-left:1px solid var(--k-border) !important;
  overflow-y:auto !important;
}
</style>
"""


def brand(version: str) -> str:
    return (
        '<div class="k-brand"><div class="k-brand__mark">G</div>'
        '<div class="k-brand__name">Knowledge</div>'
        f'<div class="k-brand__ver">v{esc(version)}</div></div>'
    )


def nav_bar(current_screen: str, nav_counts: Optional[Dict[str, int]] = None) -> str:
    """Render the sidebar navigation with SVG icons and active state matching Mockup 1a/1d."""
    counts = nav_counts or {}
    items = []
    for key, label in NAV:
        is_active = key == current_screen
        stroke = "#111110" if is_active else "#55554F"
        cls = "k-nav-link k-nav-link--active" if is_active else "k-nav-link"
        icon_svg = f'<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="{stroke}" stroke-width="1.5">{NAV_ICONS[key]}</svg>'
        badge = f'<span class="k-nav-badge">{counts[key]}</span>' if key in counts else ""
        items.append(
            f'<a href="?screen={key}" target="_self" class="{cls}">'
            f'{icon_svg}<span>{label}</span>{badge}</a>'
        )
    return f'<div class="k-nav-menu">{"".join(items)}</div>'


def reasoning_modes(current_mode: str) -> str:
    """Render the reasoning mode list with 6px indicator dots matching Mockup 1a."""
    items = []
    for key, label in MODES:
        is_active = key == current_mode
        cls = "k-mode-link k-mode-link--active" if is_active else "k-mode-link"
        dot_cls = "k-mode-dot k-mode-dot--active" if is_active else "k-mode-dot"
        items.append(
            f'<a href="?screen=search&mode={key}" target="_self" class="{cls}">'
            f'<span class="{dot_cls}"></span><span>{label}</span></a>'
        )
    return (
        '<div class="k-side-label">Reasoning</div>'
        f'<div class="k-mode-menu">{"".join(items)}</div>'
    )


def source_pills(current_filter: str) -> str:
    """Render the top bar evidence source filter pills matching Mockup 1a."""
    options = [
        ("all", "All sources"),
        ("documents", "Documents"),
        ("graph", "Graph"),
        ("tables", "Tables"),
    ]
    cur = (current_filter or "all").lower()
    items = []
    for key, label in options:
        is_active = key == cur
        cls = "k-pill k-pill--active" if is_active else "k-pill"
        items.append(
            f'<a href="?screen=search&source={key}" target="_self" class="{cls}">{label}</a>'
        )
    return f'<div class="k-pills-row">{"".join(items)}</div>'


def catalog_tabs(current_tab: str, counts: Optional[Dict[str, int]] = None) -> str:
    """Render the catalog category tabs with active bottom borders matching Mockup 1f."""
    tabs = [
        ("concepts", "Concepts"),
        ("relationships", "Relationships"),
        ("nodes", "Nodes"),
        ("briefs", "Domain briefs"),
    ]
    cur = (current_tab or "concepts").lower()
    items = []
    c = counts or {}
    for key, label in tabs:
        is_active = key == cur
        cls = "k-tab-link k-tab-link--active" if is_active else "k-tab-link"
        badge = f'<span class="k-tab-badge">{c[key]}</span>' if key in c else ""
        items.append(
            f'<a href="?screen=catalog&tab={key}" target="_self" class="{cls}">'
            f'{label}{badge}</a>'
        )
    return f'<div class="k-tabs-row">{"".join(items)}</div>'


def suggested_prompts_box() -> str:
    """Render the 4 empty-state prompt recommendations matching Mockup 1a/1c."""
    rows = []
    for i, (prompt_text, p_mode) in enumerate(SUGGESTED_PROMPTS, 1):
        mode_badge = dict(MODES).get(p_mode, p_mode).split()[0]
        rows.append(
            f'<a href="?screen=search&prompt={i}" target="_self" class="k-suggest-row">'
            f'<span class="k-suggest-num">{i}</span>'
            f'<span class="k-suggest-text">{esc(prompt_text)}</span>'
            f'<span class="k-suggest-badge">{mode_badge}</span>'
            f'</a>'
        )
    return (
        '<div class="k-empty-search">'
        '<div class="k-empty-title">Ask the knowledge base.</div>'
        f'<div class="k-suggest-box">{"".join(rows)}</div>'
        '</div>'
    )


def sidebar_status(passed: bool, check_count: int) -> str:
    if passed:
        return (
            '<div class="k-side-foot">'
            f'<span class="k-dot" style="background:{SUCCESS}"></span>'
            f"QA passed · {check_count} checks</div>"
        )
    return (
        '<div class="k-side-foot k-side-foot--warn">'
        f'<span class="k-dot" style="background:{WARN}"></span>'
        f"QA warnings · {check_count} checks</div>"
    )


SUGGESTED_PROMPTS = [
    ("Cost, ROI, and resource trade-offs between RAG and fine-tuning", "global"),
    ("Techniques that reduce compute overhead and deployment risk", "drift"),
    ("Core thematic knowledge domains across the document", "global"),
    ("Parameter-efficient fine-tuning methods and memory footprints", "local"),
]


def grounding_tag(score: float) -> str:
    """Return an HTML tag pill for grounding percentage, e.g., 'grounded 96%'."""
    pct = int(round(score * 100)) if score <= 1.0 else int(round(score))
    pct = max(0, min(100, pct))
    if pct >= 90:
        cls = "k-tag"
    elif pct >= 70:
        cls = "k-tag k-tag--warn"
    else:
        cls = "k-tag k-tag--danger"
    return f'<span class="{cls}">grounded {pct}%</span>'


def compute_grounding(text: str) -> float:
    """Compute share of answer sentences that carry at least one [n] citation."""
    if not text:
        return 0.0
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r'\[\d+\]', s))
    return round(cited / len(sentences), 2)


def evidence_card(citation: Dict, focus: bool = False) -> str:
    kind = citation.get("kind", "Source")
    name = citation.get("name", citation.get("document_title", ""))
    meta = citation.get("meta", "")
    idx = citation.get("index", "")
    body = ""

    # Check for table mini-table
    if kind == "Table" and citation.get("table_preview"):
        body = citation["table_preview"]
    elif citation.get("quote"):
        body = f'<div class="k-ev__quote">…{esc(_clip(citation["quote"], 180))}…</div>'
    elif citation.get("excerpt"):
        body = f'<div class="k-ev__text">{esc(_clip(citation["excerpt"], 200))}</div>'

    target_screen = "search"
    query_param = ""
    if kind in ("Document", "Passage") and citation.get("document_id"):
        target_screen = "vault"
        query_param = f'&doc={quote(str(citation["document_id"]))}'
    elif kind in ("Concept", "Node"):
        target_screen = "concepts"
        query_param = f'&find={quote(str(name))}'
    elif kind == "Domain":
        target_screen = "catalog"
        query_param = f'&term={quote(str(name))}'
    elif kind == "Table":
        target_screen = "vault"

    href = f"?screen={target_screen}{query_param}"
    card_id = f"cite-{idx}" if idx else ""

    return (
        f'<a class="k-ev{" k-ev--focus" if focus else ""}" id="{card_id}" href="{href}" target="_self">'
        f'<div class="k-ev__head"><span class="k-ev__idx">{esc(idx)}</span>'
        f'<span class="k-ev__title">{esc(kind)} · {esc(name)}</span>'
        + (f'<span class="k-ev__meta">{esc(_clip(meta, 18))}</span>' if meta else "")
        + f"</div>{body}</a>"
    )


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def stat_box(value, label: str) -> str:
    return (
        f'<div class="k-stat"><div class="k-stat__num">{esc(value)}</div>'
        f'<div class="k-stat__label">{esc(label)}</div></div>'
    )


def community_color(community) -> str:
    try:
        return COMMUNITY_COLORS[int(community) % len(COMMUNITY_COLORS)]
    except (TypeError, ValueError):
        return FAINT


def highlight(text: str, term: str) -> str:
    """Escape text, then wrap case-insensitive matches of term in <mark>."""
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
