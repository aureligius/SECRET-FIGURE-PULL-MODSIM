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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0b14; color: #e8e0f0; }

section[data-testid="stSidebar"] {
    background: #130f20;
    border-right: 1px solid #2a2040;
}
section[data-testid="stSidebar"] * { color: #c8b8e8 !important; }

[data-testid="metric-container"] {
    background: #1c1630;
    border: 1px solid #2e2550;
    border-radius: 14px;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: #8a7aaa !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0e8ff !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 24px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #8b5cf6, #6d28d9);
    color: white; border: none; border-radius: 100px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600; padding: 0.55rem 1.8rem;
    transition: all 0.2s; letter-spacing: 0.03em;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
.stButton > button:disabled { opacity: 0.35; transform: none; }

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #f0e8ff !important;
}
p, li { color: #a090c0 !important; }
hr { border-color: #2a2040; }
.stAlert { border-radius: 12px; border: none; }

/* Figure card grid */
.figure-card-grid {
    display: flex;
    gap: 14px;
    overflow-x: auto;
    padding: 8px 4px 16px 4px;
    scrollbar-width: thin;
    scrollbar-color: #2e2550 transparent;
}
.figure-card {
    flex: 0 0 130px;
    background: #1c1630;
    border: 2px solid #2e2550;
    border-radius: 16px;
    padding: 14px 10px 12px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    user-select: none;
}
.figure-card:hover {
    border-color: #8b5cf6;
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(139,92,246,0.25);
}
.figure-card.selected {
    border-color: #a78bfa;
    background: #2e2550;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.3),
                0 8px 24px rgba(139,92,246,0.3);
    transform: translateY(-4px);
}
.figure-card.selected-secret {
    border-color: #f6c94e;
    background: #2e1a10;
    box-shadow: 0 0 0 3px rgba(246,201,78,0.4),
                0 8px 28px rgba(246,201,78,0.3);
    transform: translateY(-4px);
}
.figure-card img {
    width: 80px;
    height: 80px;
    object-fit: contain;
    border-radius: 10px;
    margin-bottom: 8px;
    background: #130f20;
    padding: 4px;
}
.figure-card .fig-name {
    font-size: 11px;
    font-weight: 600;
    color: #c8b8e8;
    line-height: 1.3;
}
.figure-card.selected .fig-name { color: #e8d8ff; }
.figure-card .fig-prob {
    font-size: 10px;
    color: #7060a0;
    margin-top: 3px;
}
.figure-card.selected-secret .fig-name { color: #ffe566; }
.figure-card.selected-secret .fig-prob { color: #f6c94e; }

/* Budget input box */
.budget-display {
    background: #1c1630;
    border: 1px solid #2e2550;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 12px 0;
}

/* Pull history pill */
.pull-pill {
    display: inline-block;
    background: #1c1630;
    border: 1px solid #2e2550;
    border-radius: 8px;
    padding: 6px 10px;
    margin: 3px;
    font-size: 12px;
    color: #c8b8e8;
    text-align: center;
}
.pull-pill.secret {
    background: #2e1a10;
    border-color: #f6c94e;
    color: #ffe566;
}

/* Confidence metric card row */
.conf-grid {
    display: flex;
    gap: 10px;
    margin: 16px 0;
    flex-wrap: wrap;
}
.conf-card {
    flex: 1;
    min-width: 100px;
    background: #1c1630;
    border: 1px solid #2e2550;
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
}
.conf-card .conf-pct {
    font-size: 11px;
    color: #8a7aaa;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.conf-card .conf-val {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 700;
    color: #f0e8ff;
}
.conf-card.highlight {
    border-color: #a78bfa;
    background: #2e2550;
}
.conf-card.highlight .conf-val { color: #e8d8ff; }
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

# Map figure name → filename inside images/ folder
# EDIT THESE FILENAMES to match whatever you name your image files
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
    """
    Load a figure image from the images/ folder and return as base64 data URI.
    If the file doesn't exist, returns a coloured placeholder SVG.
    """
    path = os.path.join(IMAGE_DIR, IMAGE_FILES.get(figure_name, ""))
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        return f"data:{mime};base64,{b64}"

    # Placeholder — a soft coloured box with the emoji
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
    "machine1_result":  None,
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
    """
    Renders a horizontal scrollable card strip.
    Each card shows the figure image + name + probability.
    Clicking a card calls st.session_state update via a hidden form trick.
    We use st.radio hidden behind the cards for actual state management.
    """
    cards_html = ""
    for fig in FIGURE_NAMES:
        img_src   = get_image_b64(fig)
        is_secret = fig == "Golden Labubu ✦"
        is_sel    = fig == selected
        sel_class = ("selected-secret" if (is_sel and is_secret)
                     else "selected" if is_sel else "")
        prob_str  = "1/72 ≈ 1.39%" if is_secret else "71/432 ≈ 16.44%"
        # Each card is wrapped in a form that submits with a hidden input
        # We use a data attribute to pass the figure name
        cards_html += f"""
        <div class="figure-card {sel_class}" onclick="selectFigure('{fig}')">
          <!-- INSERT IMAGE: images/{IMAGE_FILES.get(fig, '')} -->
          <img src="{img_src}" alt="{fig}" />
          <div class="fig-name">{fig}</div>
          <div class="fig-prob">{prob_str}</div>
        </div>
        """

    # The JS sets a hidden input and submits a form back to Streamlit
    # We use the streamlit-js-eval trick via postMessage
    html = f"""
    <div class="figure-card-grid" id="figGrid">
      {cards_html}
    </div>
    <input type="hidden" id="selectedFig" value="{selected}" />
    <script>
    function selectFigure(name) {{
      document.getElementById('selectedFig').value = name;
      // Send to Streamlit via query string navigation
      const url = new URL(window.location.href);
      // We piggyback on Streamlit's component value mechanism
      window.parent.postMessage({{
        type: 'streamlit:setComponentValue',
        value: name
      }}, '*');
    }}
    </script>
    """
    # Render as a component that returns selected figure
    val = components.html(html, height=180, scrolling=False)


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 8px;'>
      <div style='font-size:22px;'>🌟</div>
      <div style='font-family:Playfair Display,serif;font-size:18px;
                  font-weight:700;color:#f0e8ff;'>Labubu Simulator</div>
      <div style='font-size:10px;color:#6050a0;letter-spacing:.12em;
                  text-transform:uppercase;margin-top:2px;'>
        Finite Pool · Monte Carlo
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio("", [
        "🎯  Machine 1 — Budget Predictor",
        "📦  Machine 2 — Open Boxes",
    ], label_visibility="collapsed")

    st.markdown("---")

    # Monte Carlo N slider — lives in sidebar as lecturer wants
    st.markdown("#### 🔢 Monte Carlo Trials")
    N_mc = st.select_slider(
        "N",
        options=[1_000, 10_000, 50_000, 100_000],
        value=100_000,
        label_visibility="collapsed",
        help="More trials = tighter confidence. 100k ≈ <1s on modern hardware."
    )
    st.caption(f"SE ≈ σ/√{N_mc:,} per Dr. Syukron Week 10")

    st.markdown("---")

    # Batch status
    st.markdown("#### 🏭 Factory Batch")
    meta = get_pool_metadata()
    pct  = meta["total_remaining"] / meta["total_original"]
    st.metric("Boxes Remaining",
              f"{meta['total_remaining']} / {meta['total_original']}")
    st.progress(pct, text=f"{pct*100:.0f}% left in batch")
    st.caption(f"Batch #{meta['batch_id']} · "
               f"{meta['boxes_opened']} opened by all users")

    st.markdown("---")

    with st.expander("🔧 Admin — Restock"):
        st.warning("Resets the pool for ALL users.")
        if st.button("🏭 Restock Now"):
            restock()
            st.session_state.session_pulls  = []
            st.session_state.last_result    = None
            st.session_state.canvas_state   = "idle"
            st.session_state.machine1_result= None
            st.session_state.boxes_used     = 0
            st.session_state.user_budget    = 0.0
            st.success("Fresh batch loaded!")
            st.rerun()

# ── Page header ───────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:1.2rem 0 .6rem;'>
  <div style='font-size:10px;letter-spacing:.18em;text-transform:uppercase;
              color:#8b5cf6;margin-bottom:6px;'>✦ UGM · TIF212247 · Stochastic Simulation</div>
  <h1 style='font-family:Playfair Display,serif;
             font-size:clamp(1.8rem,4vw,2.8rem);
             font-weight:900;color:#f0e8ff;margin:0;line-height:1.1;'>
    Labubu
    <span style='background:linear-gradient(135deg,#ff6b9d,#a78bfa,#3ec6e0);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
    Blind Box</span>
  </h1>
  <p style='color:#7060a0;margin-top:6px;font-size:13px;'>
    Shared Finite Pool · Sampling Without Replacement · Discrete Inverse Transform
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ══════════════════════════════════════════════════════════
# MACHINE 1 — BUDGET PREDICTOR
# ══════════════════════════════════════════════════════════
if "Machine 1" in page:

    st.markdown("### 🎯 Machine 1 — Budget Predictor")
    st.markdown(
        "Select the Labubu figure you want to hunt for. "
        "The simulator runs **Monte Carlo trials from the current pool state** "
        "and tells you how much to budget at different confidence levels."
    )

    # ── Figure selector ───────────────────────────────────
    st.markdown("#### Choose Your Target Figure")
    st.caption(
        "Click a card to select. "
        "Images load from `images/` folder — see comments in code for filenames."
    )

    # We use st.radio invisibly to track selected figure
    # and render the pretty HTML cards as a display layer
    cols = st.columns(7)
    for i, fig in enumerate(FIGURE_NAMES):
        with cols[i]:
            img_src   = get_image_b64(fig)
            is_secret = fig == "Golden Labubu ✦"
            is_sel    = st.session_state.selected_figure == fig

            # Border styling
            if is_sel and is_secret:
                border = "2px solid #f6c94e"
                bg     = "#2e1a10"
                shadow = "0 0 0 3px rgba(246,201,78,0.35), 0 8px 24px rgba(246,201,78,0.2)"
                name_col = "#ffe566"
            elif is_sel:
                border = "2px solid #a78bfa"
                bg     = "#2e2550"
                shadow = "0 0 0 3px rgba(167,139,250,0.35), 0 8px 24px rgba(139,92,246,0.2)"
                name_col = "#e8d8ff"
            else:
                border = "2px solid #2e2550"
                bg     = "#1c1630"
                shadow = "none"
                name_col = "#c8b8e8"

            prob_str  = "1/72 ≈ 1.39%" if is_secret else "16.44%"

            st.markdown(
                f"""<div style='background:{bg};border:{border};border-radius:14px;
                    padding:12px 6px 10px;text-align:center;
                    box-shadow:{shadow};transition:all .2s;'>
                  <!-- INSERT IMAGE: images/{IMAGE_FILES.get(fig, '')} -->
                  <img src="{img_src}" style='width:72px;height:72px;
                       object-fit:contain;border-radius:8px;' alt="{fig}"/>
                  <div style='font-size:10px;font-weight:600;color:{name_col};
                       margin-top:6px;line-height:1.3;'>{fig}</div>
                  <div style='font-size:9px;color:#7060a0;margin-top:2px;'>
                    {prob_str}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            # Invisible button under each card to handle click
            if st.button("select", key=f"sel_{i}",
                         help=f"Target: {fig}",
                         use_container_width=True):
                st.session_state.selected_figure  = fig
                st.session_state.machine1_result  = None
                st.rerun()

    st.markdown("---")

    # ── Run button ────────────────────────────────────────
    target = st.session_state.selected_figure
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            f"**Selected:** {FIGURE_EMOJIS[target]} **{target}** · "
            f"N = {N_mc:,} Monte Carlo trials"
        )
    with c2:
        run_m1 = st.button("🚀 Run Prediction", use_container_width=True)

    if run_m1:
        with st.spinner(f"Running {N_mc:,} trials from current pool state…"):
            st.session_state.machine1_result = simulate_budget_confidence(
                N=N_mc,
                target_figure=target,
                confidence_levels=[0.50, 0.70, 0.80, 0.90, 0.95, 0.99],
            )

    result = st.session_state.machine1_result

    if result is None:
        st.info(
            "👆 Select a figure above and click **Run Prediction** "
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
        st.markdown("#### 💡 How Much Should You Budget?")

        bfc = result["budget_for_confidence"]

        # Confidence metric cards
        conf_items = [
            ("50%",  bfc.get(0.50, 0), False),
            ("70%",  bfc.get(0.70, 0), False),
            ("80%",  bfc.get(0.80, 0), False),
            ("90%",  bfc.get(0.90, 0), False),
            ("95%",  bfc.get(0.95, 0), True),
            ("99%",  bfc.get(0.99, 0), False),
        ]

        cols_conf = st.columns(6)
        for i, (pct_label, budget, highlight) in enumerate(conf_items):
            with cols_conf[i]:
                bg  = "#2e2550" if highlight else "#1c1630"
                bdr = "#a78bfa" if highlight else "#2e2550"
                st.markdown(
                    f"""<div style='background:{bg};border:1.5px solid {bdr};
                        border-radius:12px;padding:14px 8px;text-align:center;'>
                      <div style='font-size:10px;color:#8a7aaa;letter-spacing:.08em;
                           text-transform:uppercase;margin-bottom:4px;'>
                        {pct_label} confidence</div>
                      <div style='font-family:Playfair Display,serif;font-size:20px;
                           font-weight:700;color:#f0e8ff;'>${budget:,.0f}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.caption(
            f"Based on {result['N']:,} simulations · "
            f"Current pool: {result['total_remaining']} boxes remaining · "
            f"P(target now) = {result['p_target_now']:.4%}"
        )

        # Confidence curve chart
        st.markdown("#### 📈 Confidence Curve")
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

        # Annotate the 80% and 95% points
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

        # SE validity proof
        with st.expander("📐 Statistical Validity — Why N = " + f"{N_mc:,}?"):
            se = compute_standard_error(result["cost_distribution"])
            st.latex(r"\text{95\% CI} = \bar{X} \pm 1.96 \cdot \frac{\sigma}{\sqrt{N}}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Standard Error", f"${se['se']:,.2f}")
            c2.metric("95% CI Width",   f"±${se['ci95_width']/2:,.2f}")
            c3.metric("Error % of Mean",f"{se['error_pct']:.3f}%")

        # Sensitivity Analysis 1 — Depletion
        with st.expander("🔬 SA-1: How Many Boxes Were Already Taken?"):
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

        # Sensitivity Analysis 2 — Confidence curve
        with st.expander("🔬 SA-2: Full Confidence Curve (1% → 99%)"):
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


# ══════════════════════════════════════════════════════════
# MACHINE 2 — REAL-TIME BOX OPENER
# ══════════════════════════════════════════════════════════
elif "Machine 2" in page:

    st.markdown("### 📦 Machine 2 — Real-Time Factory Loop")
    st.markdown(
        "Input your budget, then click the box to open it. "
        "Each click draws from the **shared finite pool** using the "
        "**Discrete Inverse Transform Method**. "
        "Your shake allowance = your budget ÷ $15 per box."
    )

    meta = get_pool_metadata()

    # ── Budget input ──────────────────────────────────────
    st.markdown("#### 💰 Set Your Budget")
    col_bud, col_info = st.columns([2, 3])

    with col_bud:
        raw_budget = st.number_input(
            "How much are you staking? ($)",
            min_value=0.0,
            max_value=10_000.0,
            value=float(st.session_state.user_budget) if st.session_state.user_budget else 0.0,
            step=15.0,
            help="Minimum $15 per box. Your allowance = budget ÷ $15."
        )

        if st.button("✅ Set Budget", use_container_width=True):
            if raw_budget < BOX_PRICE:
                st.error(f"Minimum budget is ${BOX_PRICE:.0f} (1 box).")
            else:
                st.session_state.user_budget = raw_budget
                st.session_state.boxes_used  = 0
                st.session_state.session_pulls = []
                st.session_state.last_result   = None
                st.session_state.canvas_state  = "idle"
                st.rerun()

    with col_info:
        budget  = st.session_state.user_budget
        b_used  = st.session_state.boxes_used
        if budget > 0:
            allowance   = int(budget // BOX_PRICE)
            remaining_b = allowance - b_used
            spent       = b_used * BOX_PRICE

            st.markdown(
                f"""<div style='background:#1c1630;border:1px solid #2e2550;
                    border-radius:14px;padding:18px 20px;'>
                  <div style='display:flex;justify-content:space-between;
                              flex-wrap:wrap;gap:12px;'>
                    <div style='text-align:center;'>
                      <div style='font-size:10px;color:#8a7aaa;
                           text-transform:uppercase;letter-spacing:.08em;'>
                        Total Budget</div>
                      <div style='font-family:Playfair Display,serif;
                           font-size:22px;font-weight:700;color:#f0e8ff;'>
                        ${budget:,.0f}</div>
                    </div>
                    <div style='text-align:center;'>
                      <div style='font-size:10px;color:#8a7aaa;
                           text-transform:uppercase;letter-spacing:.08em;'>
                        Boxes Allowance</div>
                      <div style='font-family:Playfair Display,serif;
                           font-size:22px;font-weight:700;color:#a78bfa;'>
                        {allowance}</div>
                    </div>
                    <div style='text-align:center;'>
                      <div style='font-size:10px;color:#8a7aaa;
                           text-transform:uppercase;letter-spacing:.08em;'>
                        Shakes Left</div>
                      <div style='font-family:Playfair Display,serif;
                           font-size:22px;font-weight:700;
                           color:{"#ff6b9d" if remaining_b <= 3 else "#3ec6e0"};'>
                        {remaining_b}</div>
                    </div>
                    <div style='text-align:center;'>
                      <div style='font-size:10px;color:#8a7aaa;
                           text-transform:uppercase;letter-spacing:.08em;'>
                        Spent</div>
                      <div style='font-family:Playfair Display,serif;
                           font-size:22px;font-weight:700;color:#f6c94e;'>
                        ${spent:,.0f}</div>
                    </div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("👆 Set your budget above to unlock the box opener.")

    st.markdown("---")

    # ── Box opener controls ───────────────────────────────
    budget    = st.session_state.user_budget
    b_used    = st.session_state.boxes_used
    allowance = int(budget // BOX_PRICE) if budget >= BOX_PRICE else 0
    shakes_left = allowance - b_used
    pool_empty  = meta["total_remaining"] <= 0

    can_open1  = (shakes_left >= 1) and not pool_empty and budget >= BOX_PRICE
    can_open12 = (shakes_left >= 12) and (meta["total_remaining"] >= 12) and budget >= BOX_PRICE

    col_o1, col_o12, col_reset = st.columns([1, 1, 1])
    with col_o1:
        open1 = st.button(
            f"🎁 Shake 1 Box  ($15)",
            disabled=not can_open1,
            use_container_width=True,
        )
    with col_o12:
        open12 = st.button(
            f"📦 Shake Case  (12× · $180)",
            disabled=not can_open12,
            use_container_width=True,
        )
    with col_reset:
        if st.button("↺ Reset Session", use_container_width=True):
            st.session_state.session_pulls = []
            st.session_state.last_result   = None
            st.session_state.canvas_state  = "idle"
            st.session_state.boxes_used    = 0
            st.session_state.user_budget   = 0.0
            st.rerun()

    # Handle open 1
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

    # Handle open 12
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

    # ── p5.js canvas ──────────────────────────────────────
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

    # Budget exhausted message
    if budget > 0 and shakes_left <= 0 and not pool_empty:
        st.warning(
            "🚫 **Budget exhausted.** "
            "Go to Machine 1 to plan your next attempt, "
            "or reset your session above."
        )

    # ── Last result card ──────────────────────────────────
    if last and not last.get("empty") and cstate in ["reveal", "secret"]:
        st.markdown("---")
        if last["is_secret"]:
            st.markdown(
                f"""<div style='text-align:center;padding:20px;
                    background:linear-gradient(135deg,#1a0d30,#2e1a50);
                    border:2px solid #f6c94e;border-radius:16px;
                    box-shadow:0 0 40px rgba(246,201,78,0.3);'>
                  <div style='font-size:10px;letter-spacing:.2em;
                       color:#f6c94e;'>✦ ULTRA RARE ✦</div>
                  <div style='font-size:52px;margin:8px 0;'>{last['emoji']}</div>
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
                  <div style='font-size:44px;margin:6px 0;'>{last['emoji']}</div>
                  <div style='font-size:18px;font-weight:600;
                       color:#f0e8ff;'>{last['figure']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Progress bar & batch status ───────────────────────
    st.markdown("---")
    st.markdown("#### 🏭 Batch Progress")
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

    # ── Pull history ──────────────────────────────────────
    pulls = st.session_state.session_pulls
    if pulls:
        with st.expander(f"📜 Your Pull History ({len(pulls)} boxes · ${len(pulls)*BOX_PRICE:,.0f} spent)"):
            hist_cols = st.columns(6)
            for i, fig in enumerate(reversed(pulls[-30:])):
                is_s = fig == "Golden Labubu ✦"
                hist_cols[i % 6].markdown(
                    f"""<div style='text-align:center;padding:8px 4px;
                        background:{'#2e1a50' if is_s else '#1c1630'};
                        border:1px solid {'#f6c94e' if is_s else '#2e2550'};
                        border-radius:8px;margin:2px;'>
                      <div style='font-size:20px;'>{FIGURE_EMOJIS[fig]}</div>
                      <div style='font-size:9px;
                           color:{'#ffe566' if is_s else '#8a7aaa'};'>
                        {fig[:10]}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )