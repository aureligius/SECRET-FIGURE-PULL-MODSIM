"""
app.py — Labubu Blind Box Simulator
2 Machines: Budget Predictor + Real-Time Box Opener
"""

import os
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from stock_manager import (
    restock, open_one_box, get_pool_metadata,
    get_session_history, FIGURE_NAMES, FIGURE_EMOJIS,
    BOX_PRICE, P_SECRET, P_REGULAR_EACH, _theoretical_rates,
)
from simulation_engine import (
    simulate_budget_confidence,
    sensitivity_depletion_level,
    sensitivity_confidence_curve,
    compute_standard_error,
)

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Labubu Simulator",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}
.stApp { background: #0d0b14; color: #e8e0f0; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #130f20;
    border-right: 1px solid #2a2040;
}
section[data-testid="stSidebar"] * {
    color: #c8b8e8 !important;
    font-family: 'Montserrat', sans-serif !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
}

/* Custom interactive styling rules targeting our custom confidence level block column buttons */
div[data-testid="stColumn"] div.stButton > button {{
    background-color: #241b40 !important;
    border: 1px solid #3d2f66 !important;
    color: #bfa6ff !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    padding: 6px 12px !important;
    transition: all 0.2s ease-in-out !important;
}}
div[data-testid="stColumn"] div.stButton > button:hover {{
    border-color: #a78bfa !important;
    color: #ffffff !important;
    background-color: #2e2054 !important;
}}
div[data-testid="stColumn"] div.stButton > button:active {{
    transform: scale(0.96) !important;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1c1630;
    border: 1px solid #2e2550;
    border-radius: 14px;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: #8a7aaa !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0e8ff !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #8b5cf6, #6d28d9);
    color: white; border: none; border-radius: 100px;
    font-family: 'Montserrat', sans-serif;
    font-weight: 600; padding: 0.55rem 1.8rem;
    transition: all 0.2s; letter-spacing: 0.04em;
    font-size: 13px;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
.stButton > button:disabled { opacity: 0.35; transform: none; }

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #f0e8ff !important;
}
h4, h5, h6 {
    font-family: 'Montserrat', sans-serif !important;
    color: #c8b8e8 !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}
p, li, caption { color: #a090c0 !important; font-family: 'Montserrat', sans-serif !important; }
hr { border-color: #2a2040; }
.stAlert { border-radius: 12px; border: none; }

/* caption / small text */
.stCaption, [data-testid="stCaptionContainer"] {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 11px !important;
    color: #7060a0 !important;
}

/* ── Figure card grid — 3 per row ── */
.fig-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    padding: 8px 0 20px 0;
}

.fig-card-wrap {
    display: flex;
    flex-direction: column;
    gap: 0;
}

.fig-card {
    background: #1c1630;
    border: 2px solid #2e2550;
    border-radius: 18px;
    padding: 22px 16px 18px;
    text-align: center;
    transition: all 0.22s cubic-bezier(.34,1.2,.64,1);
    position: relative;
    overflow: hidden;
}
.fig-card:hover {
    border-color: #6d28d9;
    transform: translateY(-5px);
    box-shadow: 0 12px 32px rgba(109,40,217,0.22);
}
.fig-card.sel-regular {
    border-color: #a78bfa;
    background: #221840;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.28),
                0 12px 32px rgba(139,92,246,0.28);
    transform: translateY(-6px);
}
.fig-card.sel-secret {
    border-color: #f6c94e;
    background: #201408;
    box-shadow: 0 0 0 3px rgba(246,201,78,0.35),
                0 12px 32px rgba(246,201,78,0.25);
    transform: translateY(-6px);
}
.fig-card img {
    width: 110px;
    height: 110px;
    object-fit: contain;
    border-radius: 12px;
    margin-bottom: 12px;
    display: block;
    margin-left: auto;
    margin-right: auto;
}
.fig-card .fig-name {
    font-family: 'Montserrat', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: #c8b8e8;
    letter-spacing: 0.02em;
    line-height: 1.3;
    margin-bottom: 4px;
}
.fig-card.sel-regular .fig-name { color: #e8d8ff; }
.fig-card.sel-secret  .fig-name { color: #ffe566; }
.fig-card .fig-prob {
    font-family: 'Montserrat', sans-serif;
    font-size: 10px;
    font-weight: 500;
    color: #6050a0;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.fig-card.sel-regular .fig-prob { color: #a78bfa; }
.fig-card.sel-secret  .fig-prob { color: #f6c94e; }

/* Selected badge */
.fig-card .sel-badge {
    display: none;
    position: absolute;
    top: 10px; right: 10px;
    background: #a78bfa;
    color: #0d0b14;
    font-family: 'Montserrat', sans-serif;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 100px;
}
.fig-card.sel-regular .sel-badge { display: block; background: #a78bfa; }
.fig-card.sel-secret  .sel-badge { display: block; background: #f6c94e; color: #1a0d00; }

/* Select button under card — flush, no gap */
.fig-sel-btn > div > button {
    border-radius: 0 0 18px 18px !important;
    margin-top: -2px !important;
    background: linear-gradient(135deg, rgba(139,92,246,0.5), rgba(109,40,217,0.8)) !important;
    border: 2px solid #8b5cf6 !important;
    border-top: none !important;
    color: #ffffff !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.25s cubic-bezier(.34,1.2,.64,1) !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 4px 12px rgba(139,92,246,0.35) !important;
    position: relative !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
}
.fig-sel-btn > div > button:hover {
    background: linear-gradient(135deg, rgba(167,139,250,0.8), rgba(139,92,246,0.9)) !important;
    color: #ffff00 !important;
    border-color: #a78bfa !important;
    box-shadow: 0 6px 20px rgba(167,139,250,0.5), 0 0 30px rgba(139,92,246,0.4), inset 0 1px 0 rgba(255,255,255,0.3) !important;
    transform: translateY(-2px) !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.6) !important;
}
.fig-sel-btn > div > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 3px 8px rgba(139,92,246,0.3) !important;
}

/* ── Budget display stats row ── */
.stat-row {
    display: flex;
    align-items: stretch;
    background: #1c1630;
    border: 1px solid #2e2550;
    border-radius: 16px;
    overflow: hidden;
    margin: 12px 0;
}
.stat-cell {
    flex: 1;
    padding: 20px 16px;
    text-align: center;
    border-right: 1px solid #2e2550;
}
.stat-cell:last-child { border-right: none; }
.stat-cell .stat-label {
    font-family: 'Montserrat', sans-serif;
    font-size: 10px;
    font-weight: 600;
    color: #6050a0;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.stat-cell .stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    font-weight: 700;
    color: #f0e8ff;
    line-height: 1;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────
PLOT_BG  = "#0d0b14"
FONT_COL = "#c8b8e8"
GRID_COL = "#2a2040"

def base_layout(title="", height=380):
    return dict(
        title=dict(text=title,
                   font=dict(family="Playfair Display", size=16, color=FONT_COL)),
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans", color=FONT_COL),
        height=height,
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=44, r=20, t=50, b=44),
    )

# ── Image loader ──────────────────────────────────────────
IMAGE_DIR = "images"

IMAGE_FILES = {
    "Cherry Blossom":  "cherry_blossom.png",
    "Mint Dream":      "mint_dream.png",
    "Lavender Haze":   "lavender_haze.png",
    "Peach Glow":      "peach_glow.png",
    "Sky Pop":         "sky_pop.png",
    "Coral Bloom":     "coral_bloom.png",
    "Golden Labubu ✦": "golden_labubu.png",
}

def get_image_b64(figure_name: str) -> str:
    path = os.path.join(IMAGE_DIR, IMAGE_FILES.get(figure_name, ""))
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = path.rsplit(".", 1)[-1].lower()
            mime = "image/png" if ext == "png" else "image/jpeg"
            return f"data:{mime};base64,{b64}"
        except Exception:
            pass

    PLACEHOLDER_COLORS = {
        "Cherry Blossom":  "#f7c5c5",
        "Mint Dream":      "#b8f0e6",
        "Lavender Haze":   "#d9c9f5",
        "Peach Glow":      "#ffd9b3",
        "Sky Pop":         "#c7eeff",
        "Coral Bloom":     "#ffb3c6",
        "Golden Labubu ✦": "#ffe566",
    }
    color  = PLACEHOLDER_COLORS.get(figure_name, "#2e2550")
    emoji  = FIGURE_EMOJIS.get(figure_name, "🎁")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80">'
        f'<rect width="80" height="80" rx="10" fill="{color}"/>'
        f'<text x="40" y="52" text-anchor="middle" font-size="32">{emoji}</text>'
        f'</svg>'
    )
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"

# ── Session state ─────────────────────────────────────────
for k, v in {
    "selected_figure":  "Golden Labubu ✦",
    "selected_confidence": "95%",
    "machine1_result":  "None",
    "session_pulls":    [],
    "last_result":      None,
    "canvas_state":     "idle",
    "user_budget":      0.0,
    "boxes_used":       0,
    "sens_dep":         None,
    "sens_conf":        None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Animation renderer ────────────────────────────────────
def render_animation(state="idle", figure="", emoji="🎁",
                     is_secret=False, remaining=72):
    try:
        with open("animation.html", "r", encoding="utf-8") as f:
            html_raw = f.read()
    except FileNotFoundError:
        st.error("animation.html not found.")
        return

    inject = f"""<script>
    window._LABUBU_PARAMS = {{
      state:     "{state}",
      figure:    "{figure}",
      emoji:     "{emoji}",
      is_secret: {"true" if is_secret else "false"},
      remaining: {remaining}
    }};
    </script>"""

    patched = html_raw.replace(
        "function getParam(key, fallback) {",
        """function getParam(key, fallback) {
          if (window._LABUBU_PARAMS && window._LABUBU_PARAMS[key] !== undefined)
            return String(window._LABUBU_PARAMS[key]);"""
    )
    components.html(inject + patched, height=320, scrolling=False)

# ── Figure card selector (HTML component) ─────────────────
def render_figure_selector(selected: str) -> None:
    cards_html = ""
    for fig in FIGURE_NAMES:
        img_src   = get_image_b64(fig)
        is_secret = fig == "Golden Labubu ✦"
        is_sel    = fig == selected
        sel_class = ("selected-secret" if (is_sel and is_secret)
                     else "selected" if is_sel else "")
        prob_str  = "1/72 ≈ 1.39%" if is_secret else "71/432 ≈ 16.44%"
        cards_html += f"""
        <div class="figure-card {sel_class}" onclick="selectFigure('{fig}')">
          <img src="{img_src}" alt="{fig}" />
          <div class="fig-name">{fig}</div>
          <div class="fig-prob">{prob_str}</div>
        </div>
        """

    html = f"""
    <div class="figure-card-grid" id="figGrid">
      {cards_html}
    </div>
    <input type="hidden" id="selectedFig" value="{selected}" />
    <script>
    function selectFigure(name) {{
      document.getElementById('selectedFig').value = name;
      window.parent.postMessage({{
        type: 'streamlit:setComponentValue',
        value: name
      }}, '*');
    }}
    </script>
    """
    val = components.html(html, height=180, scrolling=False)


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 8px;'>
      <div style='font-family:Playfair Display,serif;font-size:18px;
                  font-weight:700;color:#f0e8ff;'>Labubu Simulator</div>
      <div style='font-size:10px;color:#6050a0;letter-spacing:.12em;
                  text-transform:uppercase;margin-top:2px;'>
        Monte Carlo
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio("", [
        "Budget Predictor",
        "Open Boxes",
    ], label_visibility="collapsed")

    st.markdown("---")

    st.markdown("#### Monte Carlo Trials")
    N_mc = st.select_slider(
        "N",
        options=[1_000, 10_000, 50_000, 100_000],
        value=100_000,
        label_visibility="collapsed",
        help="More trials = tighter confidence. 100k ≈ <1s on modern hardware."
    )
    st.caption(f"SE ≈ σ/√{N_mc:,} per Dr. Syukron Week 10")

    st.markdown("---")

    st.markdown("#### Factory Batch")
    meta = get_pool_metadata()
    pct  = meta["total_remaining"] / meta["total_original"]
    st.metric("Boxes Remaining",
              f"{meta['total_remaining']} / {meta['total_original']}")
    st.progress(pct, text=f"{pct*100:.0f}% left in batch")
    st.caption(f"Batch #{meta['batch_id']} · "
               f"{meta['boxes_opened']} opened by all users")

    st.markdown("---")

    # Isolated operational button layout without custom header and background container cards
    if st.button("Restock Batch Now", use_container_width=True):
        restock()
        st.session_state.session_pulls  = []
        st.session_state.last_result    = None
        st.session_state.canvas_state   = "idle"
        st.session_state.machine1_result= None
        st.session_state.boxes_used     = 0
        st.session_state.user_budget    = 0.0
        st.success("Fresh batch loaded successfully!")
        st.rerun()

# ── Page header ───────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:1.4rem 0 .8rem;'>
  <div style='font-family:"Montserrat",sans-serif;font-size:10px;
              letter-spacing:.20em;text-transform:uppercase;
              color:#8b5cf6;margin-bottom:10px;font-weight:600;'>
  </div>
  <div style='font-family:"Playfair Display",serif;
              font-size:clamp(2rem,5vw,3.4rem);
              font-weight:900;color:#f0e8ff;line-height:1.05;'>
    Blind Box
    <span style='font-style:italic;
                 background:linear-gradient(135deg,#ff6b9d,#a78bfa,#3ec6e0);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text;'>
      Simulator
    </span>
  </div>
  <p style='font-family:"Montserrat",sans-serif;
            color:#5040a0;margin-top:8px;font-size:11px;
            letter-spacing:.06em;font-weight:500;'>
    Live Shared Box Batch · Real Arcade Odds · Fair Play Guaranteed
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ==========================================================
# MACHINE 1 — BUDGET PREDICTOR
# ==========================================================
if "Budget Predictor" in page:

    st.markdown("""
    <div style='margin-bottom:6px;'>
      <div style='font-family:"Playfair Display",serif;font-size:clamp(1.4rem,3vw,2rem);
                  font-weight:700;color:#f0e8ff;line-height:1.1;'>
        Budget Predictor
      </div>
      <div style='font-family:"Montserrat",sans-serif;font-size:13px;
                  color:#7060a0;font-weight:400;margin-top:4px;'>
        Select the Labubu figure you want to hunt for. The simulator runs
        Monte Carlo trials from the <em>current pool state</em> and tells
        you how much to budget at different confidence levels.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:"Playfair Display",serif;font-size:1.1rem;
                font-weight:600;color:#c8b8e8;margin:18px 0 4px;'>
      Choose Your Target Figure
    </div>
    <div style='font-family:"Montserrat",sans-serif;font-size:11px;
                color:#6050a0;margin-bottom:14px;letter-spacing:.04em;'>
      Click <strong style="color:#8a7aaa;">Select</strong> on a card to target that figure.
      Images load from the <code>images/</code> folder.
    </div>
    """, unsafe_allow_html=True)

    rows = [FIGURE_NAMES[0:3], FIGURE_NAMES[3:6], FIGURE_NAMES[6:7]]

    for row_figs in rows:
        if len(row_figs) == 1:
            _, center_col, _ = st.columns([1, 1, 1])
            render_cols = [center_col]
        else:
            render_cols = st.columns(3)

        for col, fig in zip(render_cols, row_figs):
            with col:
                img_src   = get_image_b64(fig)
                is_secret = fig == "Golden Labubu ✦"
                is_sel    = st.session_state.selected_figure == fig

                if is_sel and is_secret:
                    card_cls  = "sel-secret"
                elif is_sel:
                    card_cls  = "sel-regular"
                else:
                    card_cls  = ""

                prob_str = "1/72 ≈ 1.39%" if is_secret else "16.44%"
                badge    = "SELECTED" if is_sel else ""

                st.markdown(
                    f"""<div class="fig-card {card_cls}">
                      <div class="sel-badge">{badge}</div>
                      <img src="{img_src}" alt="{fig}" />
                      <div class="fig-name">{fig}</div>
                      <div class="fig-prob">{prob_str}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                with st.container():
                    st.markdown('<div class="fig-sel-btn">', unsafe_allow_html=True)
                    if st.button(
                        "✦ Select" if not is_sel else "✓ Selected",
                        key=f"sel_{fig}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_figure = fig
                        st.session_state.machine1_result = None
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    target = st.session_state.selected_figure
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            f"<div style='font-family:Montserrat,sans-serif;font-size:13px;"
            f"color:#8a7aaa;padding-top:8px;'>"
            f"<strong style='color:#c8b8e8;'>{FIGURE_EMOJIS[target]} {target}</strong>"
            f" &nbsp;·&nbsp; N = <strong style='color:#a78bfa;'>{N_mc:,}</strong> Monte Carlo trials"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        run_m1 = st.button("Run Prediction", use_container_width=True)

    if run_m1:
        with st.spinner(f"Running {N_mc:,} trials from current pool state…"):
            st.session_state.machine1_result = simulate_budget_confidence(
                N=N_mc,
                target_figure=target,
                confidence_levels=[0.50, 0.70, 0.80, 0.90, 0.95, 0.99],
            )

    result = st.session_state.machine1_result

    if result is None or result == "None":
        st.info(
            "Select a figure above and click **Run Prediction** "
            "to see how much you should budget."
        )

    elif result.get("target_gone"):
        st.error(
            f"💀 **{result['message']}** "
            "This figure was already pulled by a previous user. "
            "Wait for the admin to restock."
        )

    else:
        st.markdown("---")
        st.markdown("""
        <div style='font-family:"Playfair Display",serif;font-size:1.3rem;
                    font-weight:700;color:#f0e8ff;margin-bottom:4px;'>
          How Much Should You Budget?
        </div>
        <div style='font-family:"Montserrat",sans-serif;font-size:11px;
                    color:#6050a0;margin-bottom:16px;letter-spacing:.04em;'>
          Results from Monte Carlo simulation on the current pool state. Click any confidence level card to select it.
        </div>
        """, unsafe_allow_html=True)

        bfc = result["budget_for_confidence"]
        
        simulated_costs = {
            "50%": bfc.get(0.50, 0),
            "70%": bfc.get(0.70, 0),
            "80%": bfc.get(0.80, 0),
            "90%": bfc.get(0.90, 0),
            "95%": bfc.get(0.95, 0),
            "99%": bfc.get(0.99, 0)
        }

        cols_conf = st.columns(6)
        confidence_keys = ["50%", "70%", "80%", "90%", "95%", "99%"]

        for idx, key in enumerate(confidence_keys):
            with cols_conf[idx]:
                is_selected = (st.session_state.selected_confidence == key)
                
                bg_color = "#2e1a50" if is_selected else "#140f26"
                border_color = "#a78bfa" if is_selected else "#2e2550"
                text_color = "#ffffff" if is_selected else "#8a7aaa"
                val_color = "#ffe566" if is_selected else "#b8a6e0"
                
                st.markdown(f"""
                    <div style="
                        background: {bg_color};
                        border: 2px solid {border_color};
                        border-radius: 12px;
                        padding: 14px 8px 10px 8px;
                        text-align: center;
                        margin-bottom: 8px;
                        box-shadow: { '0 4px 12px rgba(167,139,250,0.15)' if is_selected else 'none' };
                        transition: all 0.3s ease;">
                        <div style="font-family: 'Montserrat', sans-serif; font-size: 9px; color: {text_color}; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
                            {key} Confidence
                        </div>
                        <div style="font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 700; color: {val_color}; margin-top: 6px; margin-bottom: 4px;">
                            ${simulated_costs[key]:,.0f}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                btn_label = "✓ Selected" if is_selected else "✦ Select"
                
                if st.button(btn_label, key=f"btn_conf_{key}", use_container_width=True):
                    st.session_state.selected_confidence = key
                    st.session_state.user_budget = float(simulated_costs[key])
                    st.rerun()

        st.caption(
            f"Based on {result['N']:,} simulations · "
            f"Current pool: {result['total_remaining']} boxes remaining · "
            f"P(target now) = {result['p_target_now']:.4%}"
        )

        st.markdown("#### Confidence Curve")
        conf_vals = sorted(bfc.keys())
        budg_vals = [bfc[c] for c in conf_vals]

        fig_chart = go.Figure()
        fig_chart.add_trace(go.Scatter(
            x=[c * 100 for c in conf_vals],
            y=budg_vals,
            mode="lines+markers",
            name="Budget Required",
            line=dict(color="#a78bfa", width=3),
            marker=dict(size=8, color="#a78bfa"),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.07)",
        ))

        for kc, col in [(0.80, "#f6c94e"), (0.95, "#ff6b9d")]:
            if kc in bfc:
                fig_chart.add_vline(
                    x=kc * 100, line_dash="dash",
                    line_color=col, opacity=0.7,
                    annotation_text=f"{kc*100:.0f}%: ${bfc[kc]:,.0f}",
                    annotation_font_color=col,
                )

        fig_chart.update_layout(
            **base_layout("Budget Required vs. Confidence Level", height=360),
            xaxis_title="Confidence Level (%)",
            yaxis_title="Budget Required ($)",
        )
        st.plotly_chart(fig_chart, use_container_width=True)

        with st.expander("Statistical Validity — Why N = " + f"{N_mc:,}?"):
            se = compute_standard_error(result["cost_distribution"])
            st.latex(r"\text{95\% CI} = \bar{X} \pm 1.96 \cdot \frac{\sigma}{\sqrt{N}}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Standard Error", f"${se['se']:,.2f}")
            c2.metric("95% CI Width",   f"±${se['ci95_width']/2:,.2f}")
            c3.metric("Error % of Mean",f"{se['error_pct']:.3f}%")

        with st.expander("How Many Boxes Were Already Taken?"):
            st.markdown(
                "Shows how your expected budget changes depending on how many boxes "
                "were already taken before you arrived — the *Gandaria City shelf problem*."
            )
            if st.button("▶ Run Depletion Analysis", key="run_sa1"):
                with st.spinner("Running sensitivity analysis…"):
                    st.session_state.sens_dep = sensitivity_depletion_level(N=30_000)

            if st.session_state.sens_dep:
                rows = st.session_state.sens_dep
                df = pd.DataFrame([{
                    "Boxes Taken": r["n_taken"],
                    "Remaining":   r["pool_remaining"],
                    "Secret Here": "✅" if r["secret_in_pool"] else "❌",
                    "P(Secret)":  f"{r['p_secret']:.4%}",
                    "Mean Budget": f"${r['mean_budget']:,.0f}" if r["mean_budget"] else "N/A",
                    "80% Budget":  f"${r['budget_80pct']:,.0f}" if r["budget_80pct"] else "N/A",
                    "95% Budget":  f"${r['budget_95pct']:,.0f}" if r["budget_95pct"] else "N/A",
                } for r in rows])
                st.dataframe(df, use_container_width=True)

        with st.expander("Full Confidence Curve (1% → 99%)"):
            st.markdown(
                "Notice the nonlinear cost explosion chasing certainty beyond 95%."
            )
            if st.button("▶ Run Full Curve", key="run_sa2"):
                with st.spinner("Running full confidence curve…"):
                    st.session_state.sens_conf = sensitivity_confidence_curve(
                        target_figure=target, N=N_mc
                    )

            if st.session_state.sens_conf:
                sc = st.session_state.sens_conf
                if not sc.get("target_gone"):
                    fig_sc = go.Figure()
                    fig_sc.add_trace(go.Scatter(
                        x=[c * 100 for c in sc["confidence_levels"]],
                        y=sc["budget_required"],
                        mode="lines",
                        line=dict(color="#3ec6e0", width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(62,198,224,0.06)",
                    ))
                    kp = sc.get("key_points", {})
                    for kc, col in [(0.80,"#f6c94e"),(0.95,"#ff6b9d"),(0.99,"#ff4b4b")]:
                        if kc in kp:
                            fig_sc.add_vline(
                                x=kc*100, line_dash="dash",
                                line_color=col, opacity=0.7,
                                annotation_text=f"{kc*100:.0f}%: ${kp[kc]:,.0f}",
                                annotation_font_color=col,
                            )
                    fig_sc.update_layout(
                        **base_layout("Full Confidence Curve", height=340),
                        xaxis_title="Confidence (%)",
                        yaxis_title="Budget ($)",
                    )
                    st.plotly_chart(fig_sc, use_container_width=True)
                    st.info(
                        f"Going from 95% → 99% confidence costs "
                        f"**${kp.get(0.99,0) - kp.get(0.95,0):,.0f} extra**. "
                        "Chasing certainty is exponentially expensive."
                    )


# ==========================================================
# MACHINE 2 — REAL-TIME BOX OPENER
# ==========================================================
elif "Open Boxes" in page:

    st.markdown("""
    <div style='margin-bottom:6px;'>
      <div style='font-family:"Playfair Display",serif;font-size:clamp(1.4rem,3vw,2rem);
                  font-weight:700;color:#f0e8ff;line-height:1.1;'>
        Open Your Secret Box!
      </div>
      <div style='font-family:"Montserrat",sans-serif;font-size:13px;
                  color:#7060a0;font-weight:400;margin-top:4px;'>
        Set your budget, then shake the box. Each draw pulls from the
        <em>shared finite pool</em> using the Discrete Inverse Transform Method.
        Your shake allowance = budget ÷ $10 per box.
      </div>
    </div>
    """, unsafe_allow_html=True)

    meta = get_pool_metadata()

    st.markdown("""
    <div style='font-family:"Playfair Display",serif;font-size:1.1rem;
                font-weight:600;color:#c8b8e8;margin:18px 0 10px;'>
      Set Your Budget
    </div>
    """, unsafe_allow_html=True)

    col_bud, col_gap, col_info = st.columns([1.6, 0.2, 2.2])

    with col_bud:
        raw_budget = st.number_input(
            "How much are you staking? ($)",
            min_value=0.0,
            max_value=10_000.0,
            value=float(st.session_state.user_budget) if st.session_state.user_budget else 0.0,
            step=10.0,
            help="Minimum $10 per box. Your allowance = budget ÷ $10."
        )
        if st.button("Confirm Budget", use_container_width=True):
            if raw_budget < BOX_PRICE:
                st.error(f"Minimum budget is ${BOX_PRICE:.0f} (1 box).")
            else:
                st.session_state.user_budget   = raw_budget
                st.session_state.boxes_used    = 0
                st.session_state.session_pulls = []
                st.session_state.last_result   = None
                st.session_state.canvas_state  = "idle"
                st.rerun()

    with col_info:
        budget = st.session_state.user_budget
        b_used = st.session_state.boxes_used
        if budget > 0:
            allowance   = int(budget // BOX_PRICE)
            remaining_b = allowance - b_used
            spent       = b_used * BOX_PRICE
            shake_col   = "#ff6b9d" if remaining_b <= 3 else "#3ec6e0"
            st.markdown(
                f"""<div class="stat-row">
                  <div class="stat-cell">
                    <div class="stat-label">Total Budget</div>
                    <div class="stat-val">${budget:,.0f}</div>
                  </div>
                  <div class="stat-cell">
                    <div class="stat-label">Box Allowance</div>
                    <div class="stat-val" style="color:#a78bfa;">{allowance}</div>
                  </div>
                  <div class="stat-cell">
                    <div class="stat-label">Shakes Left</div>
                    <div class="stat-val" style="color:{shake_col};">{remaining_b}</div>
                  </div>
                  <div class="stat-cell">
                    <div class="stat-label">Spent</div>
                    <div class="stat-val" style="color:#f6c94e;">${spent:,.0f}</div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div style='background:#1c1630;border:1px solid #2e2550;
                    border-radius:16px;padding:32px 24px;text-align:center;
                    margin-top:4px;'>
                  <div style='font-size:28px;margin-bottom:10px;'>💰</div>
                  <div style='font-family:"Playfair Display",serif;font-size:1.1rem;
                       font-weight:600;color:#6050a0;margin-bottom:6px;'>
                    No Budget Initialized
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    budget    = st.session_state.user_budget
    b_used    = st.session_state.boxes_used
    allowance = int(budget // BOX_PRICE) if budget >= BOX_PRICE else 0
    shakes_left = allowance - b_used
    pool_empty  = meta["total_remaining"] <= 0

    can_open1   = (shakes_left >= 1) and not pool_empty and budget >= BOX_PRICE
    can_open12  = (shakes_left >= 12) and (meta["total_remaining"] >= 12) and budget >= BOX_PRICE

    col_o1, col_o12, col_reset = st.columns([1, 1, 1])
    with col_o1:
        open1 = st.button(
            f"Shake 1 Box  ($15)",
            disabled=not can_open1,
            use_container_width=True,
        )
    with col_o12:
        open12 = st.button(
            f"Shake Case  (12× · $180)",
            disabled=not can_open12,
            use_container_width=True,
        )
    with col_reset:
        if st.button("↺ Reset Session", use_container_width=True):
            st.session_state.session_pulls = []
            st.session_state.last_result   = None
            st.session_state.canvas_state  = "idle"
            st.session_state.boxes_used     = 0
            st.session_state.user_budget   = 0.0
            st.rerun()

    if open1:
        r = open_one_box()
        if not r["empty"]:
            st.session_state.session_pulls.append(r["figure"])
            st.session_state.last_result   = r
            st.session_state.canvas_state  = "secret" if r["is_secret"] else "reveal"
            st.session_state.boxes_used   += 1
            if r["is_secret"]:
                st.balloons()
        else:
            st.session_state.canvas_state = "empty"
            st.session_state.last_result  = r

    if open12:
        batch, secret_hits = [], []
        for _ in range(12):
            r = open_one_box()
            if r["empty"]: break
            batch.append(r)
            st.session_state.session_pulls.append(r["figure"])
            st.session_state.boxes_used += 1
            if r["is_secret"]: secret_hits.append(r)
        if batch:
            last_r = batch[-1]
            st.session_state.last_result  = last_r
            st.session_state.canvas_state = "secret" if last_r["is_secret"] else "reveal"
            if secret_hits:
                st.balloons()
                st.success(
                    f"🌟 SECRET in this case! "
                    f"{secret_hits[0]['emoji']} **{secret_hits[0]['figure']}**"
                )

    last   = st.session_state.last_result
    cstate = st.session_state.canvas_state
    meta2  = get_pool_metadata()

    if last and not last.get("empty"):
        render_animation(
            state=cstate,
            figure=last["figure"],
            emoji=last["emoji"],
            is_secret=last["is_secret"],
            remaining=meta2["total_remaining"],
        )
    else:
        render_animation(
            state="empty" if pool_empty else "idle",
            remaining=meta2["total_remaining"],
        )

    if budget > 0 and shakes_left <= 0 and not pool_empty:
        st.warning(
            "**Budget exhausted.** "
            "Go to Budget Predictor to plan your next attempt, "
            "or reset your session above."
        )

    if last and not last.get("empty") and cstate in ["reveal", "secret"]:
        st.markdown("---")
        realtime_img_src = get_image_b64(last["figure"])
        
        if last["is_secret"]:
            st.markdown(
                f"""<div style='text-align:center;padding:20px;
                    background:linear-gradient(135deg,#1a0d30,#2e1a50);
                    border:2px solid #f6c94e;border-radius:16px;
                    box-shadow:0 0 40px rgba(246,201,78,0.3);'>
                  <div style='font-size:10px;letter-spacing:.2em;
                       color:#f6c94e;'>✦ ULTRA RARE ✦</div>
                  <img src="{realtime_img_src}" style="width:110px;height:110px;object-fit:contain;margin:12px auto;display:block;" />
                  <div style='font-size:22px;font-weight:700;color:#ffe566;
                       font-family:Playfair Display,serif;'>{last['figure']}</div>
                  <div style='font-size:12px;color:rgba(255,229,102,.55);
                       margin-top:4px;'>P = 1/72 ≈ 1.39%</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div style='text-align:center;padding:16px;
                    background:#1c1630;border:1px solid #2e2550;
                    border-radius:14px;'>
                  <div style='font-size:10px;letter-spacing:.14em;
                       color:#8a7aaa;'>YOU GOT</div>
                  <img src="{realtime_img_src}" style="width:110px;height:110px;object-fit:contain;margin:12px auto;display:block;" />
                  <div style='font-size:18px;font-weight:600;
                       color:#f0e8ff;'>{last['figure']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### Batch Progress")
    st.caption("Exact figure counts are hidden — that's what makes it a blind box.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Remaining in Batch", meta2["total_remaining"])
    c2.metric("Opened (All Users)", meta2["boxes_opened"])
    c3.metric("Your Pulls This Session", len(st.session_state.session_pulls))

    if allowance > 0:
        st.progress(
            min(b_used / allowance, 1.0),
            text=f"Your budget: {b_used}/{allowance} boxes used (${b_used*BOX_PRICE:,.0f} of ${budget:,.0f})"
        )

    st.progress(
        1 - meta2["pct_remaining"] / 100,
        text=f"Factory batch: {meta2['boxes_opened']}/72 boxes opened by all users"
    )

    pulls = st.session_state.session_pulls
    if pulls:
        with st.expander(f"Your Pull History ({len(pulls)} boxes · ${len(pulls)*BOX_PRICE:,.0f} spent)"):
            hist_cols = st.columns(6)
            for i, fig in enumerate(reversed(pulls[-30:])):
                is_s = fig == "Golden Labubu ✦"
                hist_img_src = get_image_b64(fig)
                hist_cols[i % 6].markdown(
                    f"""<div style='text-align:center;padding:8px 4px;
                        background:{'#2e1a50' if is_s else '#1c1630'};
                        border:1px solid {'#f6c94e' if is_s else '#2e2550'};
                        border-radius:8px;margin:2px;'>
                      <img src="{hist_img_src}" style="width:40px;height:40px;object-fit:contain;margin:4px auto;display:block;" />
                      <div style='font-size:9px;
                           color:{'#ffe566' if is_s else '#8a7aaa'};'>
                        {fig[:10]}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )