"""
app.py
======
Labubu Blind Box — Full Stochastic Simulation Dashboard
Streamlit + Plotly + NumPy vectorised engine (N = 100,000)

Pages / Tabs:
  Tab 1 — Strategy Comparison Dashboard
  Tab 2 — Cost Distributions (histogram, box plot, CDF)
  Tab 3 — GBM Market Price Chart
  Tab 4 — Shelf Conditional Probability (Gandaria City)
  Tab 5 — Case Inspector + Probability Derivation
"""

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import urllib.parse

from simulation_engine import (
    run_all_strategies,
    run_gbm_display,
    simulate_shelf_scenarios,
    shelf_conditional,
    generate_sealed_case,
)
from stock_manager import (
    BOX_PRICE, CASE_PRICE, FIGURE_NAMES, FIGURE_EMOJIS,
    P_SECRET, P_REGULAR_EACH,
    open_one_box, open_one_case, restock,
    get_session_history, get_pool_metadata,
)

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Labubu Simulator",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# CUSTOM CSS — dark editorial theme
# ──────────────────────────────────────────────────────────
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
[data-testid="metric-container"] label { color: #8a7aaa !important; font-size: 12px !important; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f0e8ff !important; font-family: 'Playfair Display', serif !important; font-size: 28px !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 12px !important; }

.streamlit-expanderHeader {
    background: #1c1630 !important; border-radius: 10px !important;
    color: #c8b8e8 !important; font-family: 'DM Sans', sans-serif;
}

.stTabs [data-baseweb="tab-list"] { background: #130f20; border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #7060a0; border-radius: 8px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #2e2550 !important; color: #e8d8ff !important; }

.stButton > button {
    background: linear-gradient(135deg, #8b5cf6, #6d28d9);
    color: white; border: none; border-radius: 100px;
    font-family: 'DM Sans', sans-serif; font-weight: 600;
    letter-spacing: 0.03em; padding: 0.5rem 1.5rem; transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

.stDataFrame { background: #1c1630 !important; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #f0e8ff !important; }
h4, h5, h6 { font-family: 'DM Sans', sans-serif !important; color: #c8b8e8 !important; }
p, li { color: #a090c0 !important; }
.stAlert { border-radius: 12px; border: none; }
hr { border-color: #2a2040; }
.stNumberInput input, .stSelectbox select {
    background: #1c1630 !important; border: 1px solid #2e2550 !important;
    color: #e8e0f0 !important; border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# PLOTLY THEME HELPERS
# ──────────────────────────────────────────────────────────
PLOT_BG    = "#0d0b14"
PAPER_BG   = "#0d0b14"
GRID_COLOR = "#2a2040"
FONT_COLOR = "#c8b8e8"
PALETTE    = {
    "A": "#ff6b9d",
    "B": "#3ec6e0",
    "D": "#a78bfa",
    "C": "#f6c94e",
}

def base_layout(title="", height=420):
    return dict(
        title=dict(text=title, font=dict(family="Playfair Display", size=18, color=FONT_COLOR)),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans", color=FONT_COLOR),
        height=height,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_COLOR)),
        margin=dict(l=40, r=20, t=55, b=40),
    )


# ──────────────────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────────────────
for key, default in [
    ("results",    None),
    ("elapsed",    None),
    ("gbm_paths",  None),
    ("session_pulls", []),
    ("last_draw",  None),
    ("anim_state", "idle"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ──────────────────────────────────────────────────────────
# SIDEBAR — CONTROL PANEL
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")
    st.markdown("---")

    N = st.select_slider(
        "Monte Carlo Trials (N)",
        options=[1_000, 10_000, 50_000, 100_000],
        value=50_000,
        help="More trials = tighter confidence intervals."
    )

    st.markdown("### 🎲 Strategy Parameters")
    batch_size = st.slider("Strategy D — Batch Size (b)", 1, 36, 8, 1,
                            help="Boxes per attempt in Strategy D")

    st.markdown("### 📈 GBM Market Parameters")
    gbm_S0    = st.number_input("Initial Reseller Price ($)", 20.0, 500.0, 80.0, 5.0)
    gbm_mu    = st.slider("Annual Drift μ", -1.0, 2.0, 0.4, 0.05,
                           help="Positive = hype rising")
    gbm_sigma = st.slider("Annual Volatility σ", 0.05, 2.0, 0.6, 0.05)
    gbm_T     = st.slider("Observation Window (days)", 7, 90, 30, 1)
    threshold = st.number_input("Strategy C — Buy Threshold ($)", 5.0, float(gbm_S0),
                                 min(60.0, gbm_S0 * 0.75), 5.0)
    resale_discount = st.slider("Strategy B — Resale Discount", 0.1, 0.9, 0.4, 0.05)

    st.markdown("---")
    seed_on  = st.checkbox("Fix random seed (reproducible)", value=False)
    seed_val = st.number_input("Seed", value=42, step=1) if seed_on else None

    run_btn = st.button("🚀 Run Simulation", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏭 Pool Management")
    meta = get_pool_metadata()
    st.metric("Boxes Remaining", meta["total_remaining"])
    st.metric("Boxes Opened",    meta["boxes_opened"])
    if st.button("🔄 Restock Factory Batch", use_container_width=True):
        restock(seed=seed_val)
        st.success("Pool restocked! Fresh 72-box batch ready.")
        st.rerun()


# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem;'>
  <div style='font-size:11px; letter-spacing:0.18em; text-transform:uppercase;
              color:#8b5cf6; margin-bottom:10px;'>✦ Stochastic Simulation Engine</div>
  <h1 style='font-family:Playfair Display,serif; font-size:clamp(2rem,5vw,3.2rem);
             font-weight:900; color:#f0e8ff; margin:0; line-height:1.1;'>
    Labubu <span style='background:linear-gradient(135deg,#ff6b9d,#a78bfa,#3ec6e0);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Blind Box</span> Simulator
  </h1>
  <p style='color:#7060a0; margin-top:10px; font-size:15px;'>
    4 Strategies · Monte Carlo · Finite Pool · Fully vectorised NumPy
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ──────────────────────────────────────────────────────────
# RUN SIMULATION
# ──────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Running {N:,} trials across 4 strategies…"):
        t0 = time.perf_counter()
        results = run_all_strategies(
            N=N,
            batch_size=batch_size,
            gbm_S0=gbm_S0,
            gbm_mu=gbm_mu,
            gbm_sigma=gbm_sigma,
            gbm_T=gbm_T,
            threshold=threshold,
            resale_discount=resale_discount,
            seed=seed_val,
        )
        gbm_paths = run_gbm_display(
            S0=gbm_S0, mu=gbm_mu, sigma=gbm_sigma, T=gbm_T,
            n_display=300, seed=seed_val,
        )
        elapsed = time.perf_counter() - t0

    st.session_state.results   = results
    st.session_state.elapsed   = elapsed
    st.session_state.gbm_paths = gbm_paths


# ──────────────────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────────────────
if st.session_state.results is None:
    # ── Idle state: show probability reference ──────────────
    st.info("👈 Configure parameters in the sidebar and click **Run Simulation** to begin.")

    st.markdown("### 📐 Probability Reference")
    c1, c2, c3 = st.columns(3)
    c1.metric("P(Secret | 1 box)",         f"{P_SECRET:.4%}")
    c2.metric("P(Regular | 1 box)",         f"{P_REGULAR_EACH:.4%}")
    c3.metric("Expected boxes to secret",   "72")

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected cost (Strategy A)", "$1,080")
    c2.metric(f"P(secret in {batch_size} boxes)", f"{(1-(71/72)**batch_size):.2%}")
    c3.metric("Box price",                  f"${BOX_PRICE:.0f}")

    st.markdown("---")

    # ── Machine 2: Live Unboxing ────────────────────────────
    st.markdown("### 🎁 Machine 2 — Live Unboxing")

    draw_col, case_col, _ = st.columns([1, 1, 2])
    open_box_btn  = draw_col.button("📦 Open 1 Box",  use_container_width=True)
    open_case_btn = case_col.button("🗃️ Open Case (×12)", use_container_width=True)

    if open_box_btn:
        result = open_one_box()
        if result["empty"]:
            st.session_state.anim_state = "empty"
            st.session_state.last_draw  = None
        else:
            st.session_state.last_draw  = result
            st.session_state.anim_state = "secret" if result["is_secret"] else "reveal"
            st.session_state.session_pulls.append(result["figure"])

    if open_case_btn:
        results_case = open_one_case()
        for r in results_case:
            if not r["empty"] and r["figure"]:
                st.session_state.session_pulls.append(r["figure"])
        last = results_case[-1]
        st.session_state.last_draw  = last if not last["empty"] else None
        st.session_state.anim_state = "reveal" if not last["empty"] else "empty"

    # Build iframe URL
    last = st.session_state.last_draw
    anim = st.session_state.anim_state
    remaining = meta["total_remaining"]
    if last:
        fig_enc   = urllib.parse.quote(last["figure"])
        emoji_enc = urllib.parse.quote(last["emoji"])
        is_secret = str(last["is_secret"]).lower()
        iframe_src = (
            f"animation.html?state={anim}&figure={fig_enc}"
            f"&emoji={emoji_enc}&is_secret={is_secret}&remaining={remaining}"
        )
    else:
        iframe_src = f"animation.html?state={anim}&remaining={remaining}"

    st.components.v1.iframe(iframe_src, height=320, scrolling=False)

    if last and not st.session_state.anim_state == "empty":
        pull_emoji = "⭐" if last["is_secret"] else "📦"
        label      = "ULTRA SECRET" if last["is_secret"] else "You got"
        st.markdown(
            f"<div style='text-align:center; font-size:18px; color:#f0e8ff; "
            f"margin-top:8px;'>{pull_emoji} {label}: <b>{last['figure']}</b> "
            f"&nbsp;·&nbsp; {remaining} boxes remaining</div>",
            unsafe_allow_html=True
        )

    # ── Session Analytics ───────────────────────────────────
    pulls = st.session_state.session_pulls
    if pulls:
        st.markdown("---")
        st.markdown("### 📊 Your Session Analytics")
        stats = get_session_history(pulls)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Boxes Opened",   stats["total_pulled"])
        c2.metric("Money Spent",    f"${stats['money_spent']:.0f}")
        c3.metric("Secret Found",   "✅ YES" if stats["secret_found"] else "❌ Not yet")
        c4.metric("Secret Count",   stats["secret_count"])

        fig_df = pd.DataFrame({
            "Figure":        list(stats["figure_counts"].keys()),
            "Your Pulls":    list(stats["figure_counts"].values()),
            "Your Rate":     [f"{v:.1%}" for v in stats["empirical_rates"].values()],
            "Theoretical":   [f"{v:.1%}" for v in stats["theoretical_rates"].values()],
        })
        st.dataframe(fig_df.set_index("Figure"), use_container_width=True)

        if st.button("🗑️ Clear Session History"):
            st.session_state.session_pulls = []
            st.session_state.last_draw     = None
            st.session_state.anim_state    = "idle"
            st.rerun()

else:
    # ── Full simulation results ─────────────────────────────
    results   = st.session_state.results
    elapsed   = st.session_state.elapsed
    gbm_paths = st.session_state.gbm_paths   # np.ndarray (300, T+1)

    st.success(
        f"✅ {N:,} trials completed in **{elapsed*1000:.1f} ms** "
        f"({N/elapsed/1e6:.2f}M samples/sec)"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Strategy Comparison",
        "📈 Cost Distributions",
        "🏪 Market Price (GBM)",
        "🏬 Shelf Simulator",
        "📦 Case Inspector",
    ])

    strategies = ["A", "B", "D", "C"]

    label = {
        "A": "A — Gambler (1×)",
        "B": "B — Whale (12×)",
        "D": f"D — Calculated ({batch_size}×)",
        "C": "C — Patient Buyer",
    }

    # ════════════════════════════════════════════════════
    # TAB 1 — STRATEGY COMPARISON
    # ════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Key Statistics — All Strategies")

        cols = st.columns(4)
        for key, col in zip(strategies, cols):
            r = results[key]
            col.metric(label=label[key], value=f"${r.mean:,.0f}",
                       delta=f"σ = ${r.std:,.0f}", delta_color="off")

        st.markdown("---")

        fig_bar = go.Figure()
        metrics_map = {
            "Median":   [results[k].median for k in strategies],
            "95th Pct": [results[k].p95    for k in strategies],
            "99th Pct": [results[k].p99    for k in strategies],
        }
        bar_colors = ["#6d28d9", "#8b5cf6", "#c4b5fd"]

        for (mname, vals), color in zip(metrics_map.items(), bar_colors):
            fig_bar.add_trace(go.Bar(
                name=mname,
                x=[label[k] for k in strategies],
                y=vals,
                marker_color=color,
                text=[f"${v:,.0f}" for v in vals],
                textposition="outside",
                textfont=dict(color=FONT_COLOR, size=11),
            ))

        fig_bar.update_layout(
            **base_layout("Cost Distribution: Median · 95th · 99th Percentile", 420),
            barmode="group", yaxis_title="Total Cost ($)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### Full Statistics Table")
        rows = [results[k].summary_dict() for k in strategies]
        st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)

        st.markdown("### Probability of Exceeding Budget Thresholds")
        risk_data = {
            "Strategy":         [label[k] for k in strategies],
            "P(cost > $200)":   [f"{results[k].prob_over_200:.1%}" for k in strategies],
            "P(cost > $500)":   [f"{results[k].prob_over_500:.1%}" for k in strategies],
            "P(cost > $1,000)": [f"{results[k].prob_over_1000:.1%}" for k in strategies],
            "95th Pct Budget":  [f"${results[k].p95:,.0f}" for k in strategies],
        }
        st.dataframe(pd.DataFrame(risk_data).set_index("Strategy"), use_container_width=True)


    # ════════════════════════════════════════════════════
    # TAB 2 — COST DISTRIBUTIONS
    # ════════════════════════════════════════════════════
    with tab2:
        st.markdown("### Empirical Cost Distribution (Histogram + KDE overlay)")
        st.caption("Each distribution is built from Monte Carlo simulated trials.")

        cap = max(results[k].p99 for k in strategies)

        fig_hist = go.Figure()
        for key in strategies:
            r            = results[key]
            data_clipped = np.clip(r.costs, 0, cap)
            fig_hist.add_trace(go.Histogram(
                x=data_clipped, name=label[key],
                opacity=0.55, nbinsx=120,
                marker_color=PALETTE[key], histnorm="probability density",
            ))

        fig_hist.update_layout(
            **base_layout("Cost Probability Density — All Strategies", 460),
            barmode="overlay",
            xaxis_title="Total Cost to Acquire Secret ($)",
            yaxis_title="Probability Density",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("### Spread Comparison (Box Plots)")
        fig_box = go.Figure()
        rng_sample = np.random.default_rng(0)
        for key in strategies:
            r      = results[key]
            n_sample = min(5000, len(r.costs))
            sample = rng_sample.choice(r.costs, size=n_sample, replace=False)
            fig_box.add_trace(go.Box(
                y=np.clip(sample, 0, r.p99), name=label[key],
                marker_color=PALETTE[key], line_color=PALETTE[key],
                fillcolor=PALETTE[key] + "33", boxmean=True,
            ))

        fig_box.update_layout(
            **base_layout("Cost Spread — Box Plots (capped at 99th pct)", 420),
            yaxis_title="Cost ($)",
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("### Cumulative Cost Probability")
        st.caption("Read as: 'X% of the time I spend ≤ $Y to find the secret.'")
        fig_cdf = go.Figure()
        x_max   = np.percentile(results["A"].costs, 98)
        x_vals  = np.linspace(0, x_max, 800)

        for key in strategies:
            r   = results[key]
            cdf = np.searchsorted(np.sort(r.costs), x_vals) / len(r.costs)
            fig_cdf.add_trace(go.Scatter(
                x=x_vals, y=cdf, name=label[key], mode="lines",
                line=dict(color=PALETTE[key], width=2.5),
            ))

        for budget, dash in [(200, "dot"), (500, "dash"), (1000, "dashdot")]:
            fig_cdf.add_vline(x=budget, line_dash=dash, line_color="#4a3a6a",
                              annotation_text=f"${budget}",
                              annotation_font_color="#7060a0")

        fig_cdf.update_layout(
            **base_layout("Empirical CDF — Probability of Finding Secret Within Budget", 420),
            xaxis_title="Budget ($)",
            yaxis_title="Cumulative Probability",
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_cdf, use_container_width=True)


    # ════════════════════════════════════════════════════
    # TAB 3 — GBM MARKET PRICE
    # ════════════════════════════════════════════════════
    with tab3:
        st.markdown("### Reseller Market Price Simulation (GBM)")
        st.latex(
            r"S_{t+1} = S_t \cdot \exp\!\left[\left(\mu - \frac{\sigma^2}{2}\right)"
            r"\Delta t + \sigma\sqrt{\Delta t}\cdot Z_t\right],\quad Z_t \sim \mathcal{N}(0,1)"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Initial Price S₀", f"${gbm_S0:.0f}")
        c2.metric("Annual Drift μ",   f"{gbm_mu:+.2f}")
        c3.metric("Volatility σ",     f"{gbm_sigma:.2f}")

        # gbm_paths: ndarray (300, T+1)
        paths = gbm_paths
        days  = np.arange(paths.shape[1])

        fig_gbm = go.Figure()

        # 60 faint sample paths
        for i in range(min(60, paths.shape[0])):
            fig_gbm.add_trace(go.Scatter(
                x=days, y=paths[i], mode="lines",
                line=dict(color="rgba(167,139,250,0.12)", width=1),
                showlegend=False,
            ))

        mean_path = paths.mean(axis=0)
        fig_gbm.add_trace(go.Scatter(
            x=days, y=mean_path, mode="lines",
            name="Mean Path", line=dict(color="#a78bfa", width=2.5),
        ))

        p10 = np.percentile(paths, 10, axis=0)
        p90 = np.percentile(paths, 90, axis=0)
        fig_gbm.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([p90, p10[::-1]]),
            fill="toself", fillcolor="rgba(167,139,250,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="10th–90th Pct Band",
        ))

        fig_gbm.add_hline(y=threshold, line_dash="dash", line_color=PALETTE["C"],
                          annotation_text=f"Buy Threshold ${threshold:.0f}",
                          annotation_font_color=PALETTE["C"])
        fig_gbm.add_hline(y=gbm_S0, line_dash="dot", line_color="#4a3a6a",
                          annotation_text=f"S₀ = ${gbm_S0:.0f}",
                          annotation_font_color="#7060a0")

        fig_gbm.update_layout(
            **base_layout(f"300 GBM Price Paths — {gbm_T}-Day Window", 480),
            xaxis_title="Trading Day", yaxis_title="Reseller Price ($)",
        )
        st.plotly_chart(fig_gbm, use_container_width=True)

        st.markdown(f"### Distribution of Prices at Day {gbm_T}")
        final_prices = paths[:, -1]
        fig_fp = go.Figure()
        fig_fp.add_trace(go.Histogram(
            x=final_prices, nbinsx=40,
            marker_color="#a78bfa", opacity=0.75,
            name="Final Price Distribution",
        ))
        fig_fp.add_vline(x=gbm_S0,   line_dash="dot",  line_color="#4a3a6a")
        fig_fp.add_vline(x=threshold, line_dash="dash", line_color=PALETTE["C"],
                         annotation_text=f"Threshold ${threshold:.0f}",
                         annotation_font_color=PALETTE["C"])
        fig_fp.update_layout(
            **base_layout(f"Simulated Final Price Distribution at Day {gbm_T}", 320),
            xaxis_title="Price ($)", yaxis_title="Count",
        )
        st.plotly_chart(fig_fp, use_container_width=True)

        pct_below = np.mean(paths[:, -1] <= threshold)
        st.info(
            f"📊 With μ={gbm_mu:+.2f} and σ={gbm_sigma:.2f}, "
            f"**{pct_below:.1%}** of paths end below the buy threshold of "
            f"${threshold:.0f} by day {gbm_T}."
        )


    # ════════════════════════════════════════════════════
    # TAB 4 — SHELF CONDITIONAL PROBABILITY
    # ════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 🏬 The 'Gandaria City Shelf' Simulator")
        st.markdown(
            "Simulate walking into a store where some boxes are already taken. "
            "How does that change your real odds of finding the Secret?"
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            n_taken = st.slider("Boxes already taken from the shelf", 0, 71, 4)

        rng_shelf    = np.random.default_rng()
        result_shelf = shelf_conditional(n_taken, rng_shelf)

        with c2:
            st.markdown(f"**{result_shelf['verdict']}**")
            sc1, sc2 = st.columns(2)
            sc1.metric("Boxes remaining", result_shelf["n_remaining"])
            p_val = result_shelf["p_secret_remaining"]
            sc2.metric("P(secret still here)", f"{p_val:.2%}")

        st.markdown("---")
        st.markdown("### Distribution of Conditional P(secret) Across 10,000 Walk-Ins")
        st.caption(
            "Each point = one simulated customer walking into a random store state. "
            "Shows the full spectrum of luck asymmetry."
        )

        rng2   = np.random.default_rng(42)
        p_cond = simulate_shelf_scenarios(10_000, rng2)   # ndarray of floats

        p_zero    = p_cond[p_cond == 0.0]
        p_nonzero = p_cond[p_cond >  0.0]

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("P = 0% (pool empty)",       f"{len(p_zero)/len(p_cond):.1%}")
        sc2.metric("P > 0% (secret still here)", f"{len(p_nonzero)/len(p_cond):.1%}")
        sc3.metric("Mean conditional P",         f"{p_cond.mean():.4%}")

        fig_shelf = go.Figure()
        fig_shelf.add_trace(go.Histogram(
            x=p_nonzero * 100, nbinsx=30,
            marker_color="#f6c94e", opacity=0.8,
            name="P > 0% scenarios",
        ))
        fig_shelf.update_layout(
            **base_layout("Conditional P(secret) | Secret Still Present", 340),
            xaxis_title="Conditional Probability (%)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_shelf, use_container_width=True)

        st.info(
            f"📐 Interpretation: mean P(secret still in pool) across all random shelf states = "
            f"**{p_cond.mean():.4f}** ≈ 0.5.  "
            f"When you arrive at a fully random shelf (anywhere from 0 to 71 boxes already taken), "
            f"50% of the time the secret is still there — but your conditional P of drawing it "
            f"given it's present is 1/n_remaining."
        )


    # ════════════════════════════════════════════════════
    # TAB 5 — CASE INSPECTOR
    # ════════════════════════════════════════════════════
    with tab5:
        st.markdown("### 📦 Sealed Case Inspector")
        st.markdown(
            "Each click generates one physically-modelled sealed case of 12 boxes. "
            "The secret appears in exactly **1 in 6 cases** on average."
        )

        n_inspect = st.slider("Generate this many cases at once", 1, 50, 12)

        if st.button("🎁 Generate Cases", key="case_btn"):
            rng_case     = np.random.default_rng()
            cases        = []
            secret_count = 0

            for i in range(n_inspect):
                case       = generate_sealed_case(rng_case)   # rng as first arg
                has_secret = "Golden Labubu ✦" in case
                if has_secret:
                    secret_count += 1
                cases.append({
                    "Case #":         i + 1,
                    "Contains Secret": "⭐ YES" if has_secret else "—",
                    **{f"Box {j+1}": case[j] for j in range(12)},
                })

            df_cases = pd.DataFrame(cases).set_index("Case #")
            st.dataframe(df_cases, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Cases generated", n_inspect)
            c2.metric("Secret cases found", secret_count)
            c3.metric("Observed rate", f"{secret_count/n_inspect:.1%}",
                      delta=f"Expected: {1/6:.1%}", delta_color="off")

        st.markdown("---")
        st.markdown("### 📐 Probability Derivation Proof")
        st.markdown("""
| Rule | Value |
|------|-------|
| Boxes per case | 12 |
| Cases per secret case | 1 in 6 |
| P(this is a secret case) | 1/6 |
| P(you pick the secret box \\| secret case) | 1/12 |
| **P(secret \\| any box)** | **1/6 × 1/12 = 1/72** |
        """)

        st.latex(
            r"P(\text{secret}) = P(\text{secret case}) \times "
            r"P(\text{pick secret box} \mid \text{secret case}) "
            r"= \frac{1}{6} \times \frac{1}{12} = \frac{1}{72}"
        )

        st.markdown("### Strategy D — Batch Size vs. Variance")
        b_vals = np.arange(1, 73)
        p_b    = 1 - (71 / 72) ** b_vals
        ev     = (b_vals * BOX_PRICE) / p_b
        std_b  = np.sqrt((1 - p_b) / p_b ** 2) * b_vals * BOX_PRICE

        fig_batch = make_subplots(specs=[[{"secondary_y": True}]])
        fig_batch.add_trace(go.Scatter(
            x=b_vals, y=ev, name="Expected Cost",
            line=dict(color="#a78bfa", width=2.5),
        ), secondary_y=False)
        fig_batch.add_trace(go.Scatter(
            x=b_vals, y=std_b, name="Std Deviation",
            line=dict(color="#f6c94e", width=2, dash="dash"),
        ), secondary_y=True)

        fig_batch.add_vline(x=batch_size, line_dash="dot", line_color="#ff6b9d",
                            annotation_text=f"b = {batch_size}",
                            annotation_font_color="#ff6b9d")

        fig_batch.update_layout(
            **base_layout("Expected Cost & Std Dev vs. Batch Size", 380),
            xaxis_title="Batch Size b",
        )
        fig_batch.update_yaxes(title_text="Expected Total Cost ($)", secondary_y=False,
                               gridcolor=GRID_COLOR)
        fig_batch.update_yaxes(title_text="Std Deviation ($)", secondary_y=True,
                               gridcolor=GRID_COLOR)
        st.plotly_chart(fig_batch, use_container_width=True)

        st.info(
            "📊 Notice: expected cost stays flat at ≈$1,080 regardless of batch size. "
            "**Only the variance changes.** Larger batches = more predictable spending."
        )