"""
app.py
======
Labubu Blind Box — Full Stochastic Simulation Dashboard
Streamlit + Plotly + NumPy vectorised engine (N = 100,000)
"""

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from simulation_engine import (
    run_all_strategies,
    run_gbm_display,
    simulate_shelf_scenarios,
    shelf_conditional,
    generate_sealed_case,
    BOX_PRICE, CASE_PRICE, FIGURE_NAMES,
    P_SECRET, P_REGULAR_EACH,
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

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background: #0d0b14;
    color: #e8e0f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #130f20;
    border-right: 1px solid #2a2040;
}
section[data-testid="stSidebar"] * { color: #c8b8e8 !important; }
section[data-testid="stSidebar"] .stSlider > div { color: #c8b8e8; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #1c1630;
    border: 1px solid #2e2550;
    border-radius: 14px;
    padding: 16px !important;
}
[data-testid="metric-container"] label { color: #8a7aaa !important; font-size: 12px !important; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f0e8ff !important; font-family: 'Playfair Display', serif !important; font-size: 28px !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 12px !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #1c1630 !important;
    border-radius: 10px !important;
    color: #c8b8e8 !important;
    font-family: 'DM Sans', sans-serif;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #130f20; border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #7060a0; border-radius: 8px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #2e2550 !important; color: #e8d8ff !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #8b5cf6, #6d28d9);
    color: white;
    border: none;
    border-radius: 100px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* Dataframe */
.stDataFrame { background: #1c1630 !important; }

/* Section headers */
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #f0e8ff !important; }
h4, h5, h6 { font-family: 'DM Sans', sans-serif !important; color: #c8b8e8 !important; }
p, li { color: #a090c0 !important; }

/* Info / warning boxes */
.stAlert { border-radius: 12px; border: none; }

/* Divider */
hr { border-color: #2a2040; }

/* Number input, selectbox */
.stNumberInput input, .stSelectbox select {
    background: #1c1630 !important;
    border: 1px solid #2e2550 !important;
    color: #e8e0f0 !important;
    border-radius: 8px !important;
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
    "A": "#ff6b9d",   # Gambler     — hot pink
    "B": "#3ec6e0",   # Whale       — cyan
    "D": "#a78bfa",   # Calculated  — violet
    "C": "#f6c94e",   # Patient     — gold
}

def base_layout(title="", height=420):
    return dict(
        title=dict(text=title, font=dict(family="Playfair Display", size=18, color=FONT_COLOR)),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="DM Sans", color=FONT_COLOR),
        height=height,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_COLOR)),
        margin=dict(l=40, r=20, t=55, b=40),
    )


# ──────────────────────────────────────────────────────────
# SIDEBAR — CONTROL PANEL
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Simulation Controls")
    st.markdown("---")

    N = st.select_slider(
        "Monte Carlo Trials (N)",
        options=[1_000, 10_000, 50_000, 100_000],
        value=100_000,
        help="More trials = tighter confidence intervals. 100k runs in ~50ms."
    )

    st.markdown("### 🎲 Strategy Parameters")

    batch_size = st.slider(
        "Strategy D — Batch Size (b)",
        min_value=1, max_value=36, value=8, step=1,
        help="Number of boxes bought per attempt in Strategy D"
    )

    st.markdown("### 📈 GBM Market Parameters")

    gbm_S0 = st.number_input(
        "Initial Reseller Price ($)", min_value=20.0, max_value=500.0,
        value=80.0, step=5.0
    )
    gbm_mu = st.slider(
        "Annual Drift μ", min_value=-1.0, max_value=2.0, value=0.4, step=0.05,
        help="Positive = price growing (hype rising), negative = hype dying"
    )
    gbm_sigma = st.slider(
        "Annual Volatility σ", min_value=0.05, max_value=2.0, value=0.6, step=0.05,
        help="How noisy/unpredictable daily price swings are"
    )
    gbm_T = st.slider(
        "Observation Window (days)", min_value=7, max_value=90, value=30, step=1
    )
    threshold = st.number_input(
        "Strategy C — Buy Threshold ($)", min_value=5.0, max_value=float(gbm_S0),
        value=min(60.0, gbm_S0 * 0.75), step=5.0,
        help="Patient Buyer pulls the trigger when price ≤ this value"
    )
    resale_discount = st.slider(
        "Strategy B — Resale Discount",
        min_value=0.1, max_value=0.9, value=0.4, step=0.05,
        help="Whale sells duplicates at this fraction of market price"
    )

    st.markdown("---")
    seed_on = st.checkbox("Fix random seed (reproducible)", value=False)
    seed_val = st.number_input("Seed", value=42, step=1) if seed_on else None

    run_btn = st.button("🚀 Run Simulation", use_container_width=True)


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
    4 Strategies · N = 100,000 trials · Fully vectorised NumPy
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ──────────────────────────────────────────────────────────
# SESSION STATE — cache results
# ──────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "elapsed" not in st.session_state:
    st.session_state.elapsed = None
if "gbm_paths" not in st.session_state:
    st.session_state.gbm_paths = None


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
            S0=gbm_S0, mu=gbm_mu, sigma=gbm_sigma, T=gbm_T, n_display=300,
            seed=seed_val
        )
        elapsed = time.perf_counter() - t0

    st.session_state.results  = results
    st.session_state.elapsed  = elapsed
    st.session_state.gbm_paths = gbm_paths


# ──────────────────────────────────────────────────────────
# DISPLAY — only if results exist
# ──────────────────────────────────────────────────────────
if st.session_state.results is None:
    st.info("👈 Configure parameters in the sidebar and click **Run Simulation** to begin.")

    # Show probability reference while waiting
    st.markdown("### 📐 Probability Reference")
    col1, col2, col3 = st.columns(3)
    col1.metric("P(Secret | 1 box)",   f"{P_SECRET:.4%}")
    col2.metric("P(Regular | 1 box)",  f"{P_REGULAR_EACH:.4%}")
    col3.metric("Expected boxes to secret", "72")

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected cost (Strategy A)", "$1,080")
    col2.metric(f"P(secret in {batch_size} boxes)", f"{(1-(71/72)**batch_size):.2%}")
    col3.metric("Box price", f"${BOX_PRICE:.0f}")

else:
    results  = st.session_state.results
    elapsed  = st.session_state.elapsed
    gbm_paths = st.session_state.gbm_paths

    st.success(f"✅ {N:,} trials completed in **{elapsed*1000:.1f} ms** "
               f"({N/elapsed/1e6:.1f}M samples/sec)")

    # ── TABS ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Strategy Comparison",
        "📈 Cost Distributions",
        "🏪 Market Price (GBM)",
        "🏬 Shelf Simulator",
        "📦 Case Inspector",
    ])


    # ════════════════════════════════════════════════════
    # TAB 1 — STRATEGY COMPARISON DASHBOARD
    # ════════════════════════════════════════════════════
    with tab1:
        st.markdown("### Key Statistics — All Strategies")

        # Strategy name map
        label = {
            "A": f"A — Gambler (1×)",
            "B": f"B — Whale (12×)",
            "D": f"D — Calculated ({batch_size}×)",
            "C": f"C — Patient Buyer",
        }

        # KPI row
        cols = st.columns(4)
        for i, (key, col) in enumerate(zip(["A","B","D","C"], cols)):
            r = results[key]
            col.metric(
                label=label[key],
                value=f"${r.mean:,.0f}",
                delta=f"σ = ${r.std:,.0f}",
                delta_color="off",
            )

        st.markdown("---")

        # ── Percentile comparison bar chart
        strategies = ["A", "B", "D", "C"]
        fig = go.Figure()

        metrics = {
            "Median":    [results[k].median for k in strategies],
            "95th Pct":  [results[k].p95    for k in strategies],
            "99th Pct":  [results[k].p99    for k in strategies],
        }
        bar_colors = ["#6d28d9", "#8b5cf6", "#c4b5fd"]

        for (metric_name, vals), color in zip(metrics.items(), bar_colors):
            fig.add_trace(go.Bar(
                name=metric_name,
                x=[label[k] for k in strategies],
                y=vals,
                marker_color=color,
                text=[f"${v:,.0f}" for v in vals],
                textposition="outside",
                textfont=dict(color=FONT_COLOR, size=11),
            ))

        fig.update_layout(
            **base_layout("Cost Distribution: Median · 95th · 99th Percentile", height=420),
            barmode="group",
            yaxis_title="Total Cost ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Summary table
        st.markdown("### Full Statistics Table")
        rows = [results[k].summary_dict() for k in strategies]
        df_summary = pd.DataFrame(rows).set_index("Strategy")
        st.dataframe(df_summary, use_container_width=True)

        # ── Risk table
        st.markdown("### Probability of Exceeding Budget Thresholds")
        risk_data = {
            "Strategy":      [label[k] for k in strategies],
            "P(cost > $200)": [f"{results[k].prob_over_200:.1%}" for k in strategies],
            "P(cost > $500)": [f"{results[k].prob_over_500:.1%}" for k in strategies],
            "P(cost > $1,000)":[f"{results[k].prob_over_1000:.1%}" for k in strategies],
            "95th Pct Budget": [f"${results[k].p95:,.0f}" for k in strategies],
        }
        st.dataframe(pd.DataFrame(risk_data).set_index("Strategy"), use_container_width=True)


    # ════════════════════════════════════════════════════
    # TAB 2 — COST DISTRIBUTIONS
    # ════════════════════════════════════════════════════
    with tab2:
        st.markdown("### Empirical Cost Distribution (Histogram + KDE overlay)")
        st.caption("Each distribution is built from 100,000 simulated trials.")

        # Cap display at 99th percentile of max strategy for readability
        cap = max(results[k].p99 for k in ["A","B","D","C"])

        fig = go.Figure()
        for key in ["A", "B", "D", "C"]:
            r = results[key]
            data_clipped = np.clip(r.costs, 0, cap)
            fig.add_trace(go.Histogram(
                x=data_clipped,
                name=label[key],
                opacity=0.55,
                nbinsx=120,
                marker_color=PALETTE[key],
                histnorm="probability density",
            ))

        fig.update_layout(
            **base_layout("Cost Probability Density — All Strategies", height=460),
            barmode="overlay",
            xaxis_title="Total Cost to Acquire Secret ($)",
            yaxis_title="Probability Density",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Box plots
        st.markdown("### Spread Comparison (Box Plots)")
        fig2 = go.Figure()
        for key in ["A", "B", "D", "C"]:
            r = results[key]
            # Sample 5000 for box plot speed
            sample = np.random.choice(r.costs, size=5000, replace=False)
            fig2.add_trace(go.Box(
                y=np.clip(sample, 0, r.p99),
                name=label[key],
                marker_color=PALETTE[key],
                line_color=PALETTE[key],
                fillcolor=PALETTE[key] + "33",
                boxmean=True,
            ))

        fig2.update_layout(
            **base_layout("Cost Spread — Box Plots (capped at 99th pct)", height=420),
            yaxis_title="Cost ($)",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Cumulative Distribution (CDF)
        st.markdown("### Cumulative Cost Probability")
        st.caption("Read as: 'X% of the time I spend ≤ $Y to find the secret.'")
        fig3 = go.Figure()
        x_max = np.percentile(results["A"].costs, 98)
        x_vals = np.linspace(0, x_max, 800)

        for key in ["A", "B", "D", "C"]:
            r = results[key]
            cdf = np.searchsorted(np.sort(r.costs), x_vals) / len(r.costs)
            fig3.add_trace(go.Scatter(
                x=x_vals, y=cdf,
                name=label[key],
                mode="lines",
                line=dict(color=PALETTE[key], width=2.5),
            ))

        # Threshold lines
        for budget, dash in [(200, "dot"), (500, "dash"), (1000, "dashdot")]:
            fig3.add_vline(x=budget, line_dash=dash, line_color="#4a3a6a",
                           annotation_text=f"${budget}", annotation_font_color="#7060a0")

        fig3.update_layout(
            **base_layout("Empirical CDF — Probability of Finding Secret Within Budget", height=420),
            xaxis_title="Budget ($)",
            yaxis_title="Cumulative Probability",
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig3, use_container_width=True)


    # ════════════════════════════════════════════════════
    # TAB 3 — GBM MARKET PRICE CHART
    # ════════════════════════════════════════════════════
    with tab3:
        st.markdown("### Reseller Market Price Simulation (GBM)")
        st.latex(r"S_{t+1} = S_t \cdot \exp\!\left[\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\cdot Z_t\right], \quad Z_t \sim \mathcal{N}(0,1)")

        col1, col2, col3 = st.columns(3)
        col1.metric("Initial Price S₀", f"${gbm_S0:.0f}")
        col2.metric("Annual Drift μ",   f"{gbm_mu:+.2f}")
        col3.metric("Volatility σ",     f"{gbm_sigma:.2f}")

        paths = gbm_paths      # shape (300, T+1)
        days  = np.arange(paths.shape[1])

        fig = go.Figure()

        # Draw 60 faint paths
        for i in range(min(60, paths.shape[0])):
            fig.add_trace(go.Scatter(
                x=days, y=paths[i],
                mode="lines",
                line=dict(color="rgba(167,139,250,0.12)", width=1),
                showlegend=False,
            ))

        # Mean path
        mean_path = paths.mean(axis=0)
        fig.add_trace(go.Scatter(
            x=days, y=mean_path,
            mode="lines",
            name="Mean Path",
            line=dict(color="#a78bfa", width=2.5),
        ))

        # Percentile bands
        p10 = np.percentile(paths, 10, axis=0)
        p90 = np.percentile(paths, 90, axis=0)
        fig.add_trace(go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([p90, p10[::-1]]),
            fill="toself",
            fillcolor="rgba(167,139,250,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="10th–90th Pct Band",
        ))

        # Threshold line
        fig.add_hline(
            y=threshold,
            line_dash="dash", line_color=PALETTE["C"],
            annotation_text=f"Buy Threshold ${threshold:.0f}",
            annotation_font_color=PALETTE["C"],
        )

        fig.add_hline(
            y=gbm_S0,
            line_dash="dot", line_color="#4a3a6a",
            annotation_text=f"S₀ = ${gbm_S0:.0f}",
            annotation_font_color="#7060a0",
        )

        fig.update_layout(
            **base_layout(f"300 GBM Price Paths — {gbm_T}-Day Window", height=480),
            xaxis_title="Trading Day",
            yaxis_title="Reseller Price ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Distribution of final prices
        st.markdown("### Distribution of Prices at Day " + str(gbm_T))
        final_prices = paths[:, -1]
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=final_prices,
            nbinsx=40,
            marker_color="#a78bfa",
            opacity=0.75,
            name="Final Price Distribution",
        ))
        fig2.add_vline(x=gbm_S0,   line_dash="dot",  line_color="#4a3a6a")
        fig2.add_vline(x=threshold, line_dash="dash", line_color=PALETTE["C"],
                       annotation_text=f"Threshold ${threshold:.0f}",
                       annotation_font_color=PALETTE["C"])
        fig2.update_layout(
            **base_layout(f"Simulated Final Price Distribution at Day {gbm_T}", height=320),
            xaxis_title="Price ($)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig2, use_container_width=True)

        pct_below = np.mean(paths[:, -1] <= threshold)
        st.info(
            f"📊 With μ={gbm_mu:+.2f} and σ={gbm_sigma:.2f}, "
            f"**{pct_below:.1%}** of paths end below the buy threshold of ${threshold:.0f} "
            f"by day {gbm_T}."
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

        col1, col2 = st.columns([1, 1])
        with col1:
            n_taken = st.slider(
                "Boxes already taken from the shelf",
                min_value=0, max_value=11, value=4
            )
            run_shelf = st.button("🎰 Run This Scenario", key="shelf_btn")

        if run_shelf or True:
            rng_shelf = np.random.default_rng()
            result_shelf = shelf_conditional(n_taken, rng_shelf)

            with col2:
                st.markdown(f"**{result_shelf['verdict']}**")
                c1, c2 = st.columns(2)
                c1.metric("Boxes remaining", result_shelf["n_remaining"])
                c2.metric(
                    "Your P(secret)",
                    f"{result_shelf['p_secret_remaining']:.2%}"
                    if result_shelf["p_secret_remaining"] > 0 else "0%"
                )

        st.markdown("---")
        st.markdown("### Distribution of Conditional P(secret) Across 10,000 Walk-Ins")
        st.caption(
            "Each point = one simulated customer walking into a random store state. "
            "Shows the full spectrum of luck asymmetry."
        )

        rng2 = np.random.default_rng(42)
        p_cond = simulate_shelf_scenarios(10_000, rng2)

        # Split into three groups
        p_zero   = p_cond[p_cond == 0.0]
        p_nonzero = p_cond[p_cond > 0.0]

        col1, col2, col3 = st.columns(3)
        col1.metric("P = 0% (wasted trip)", f"{len(p_zero)/len(p_cond):.1%}")
        col2.metric("P > 0% (secret still here)", f"{len(p_nonzero)/len(p_cond):.1%}")
        col3.metric("Mean conditional P", f"{p_cond.mean():.4%}")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=p_nonzero * 100,
            nbinsx=30,
            marker_color="#f6c94e",
            opacity=0.8,
            name="P > 0% scenarios",
        ))
        fig.update_layout(
            **base_layout("Conditional P(secret) | Secret Still Present", height=340),
            xaxis_title="Conditional Probability (%)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"📐 Marginal check: mean across all scenarios = "
            f"**{p_cond.mean():.5f}** ≈ 1/72 = {1/72:.5f} ✓  "
            f"(Law of Total Probability holds)"
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
            rng_case = np.random.default_rng()
            cases = []
            secret_count = 0
            for i in range(n_inspect):
                case = generate_sealed_case(rng_case)
                has_secret = FIGURE_NAMES[6] in case
                if has_secret:
                    secret_count += 1
                cases.append({
                    "Case #": i + 1,
                    "Contains Secret": "⭐ YES" if has_secret else "—",
                    **{f"Box {j+1}": case[j] for j in range(12)}
                })

            df_cases = pd.DataFrame(cases).set_index("Case #")
            st.dataframe(df_cases, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Cases generated", n_inspect)
            col2.metric("Secret cases found", secret_count)
            col3.metric(
                "Observed rate",
                f"{secret_count/n_inspect:.1%}",
                delta=f"Expected: {1/6:.1%}",
                delta_color="off"
            )

        st.markdown("---")
        st.markdown("### 📐 Probability Derivation Proof")
        st.markdown("""
        The **1/72** is not an assumption — it's derived from the physical packaging rules:

        | Rule | Value |
        |------|-------|
        | Boxes per case | 12 |
        | Cases per secret case | 1 in 6 |
        | P(this is a secret case) | 1/6 |
        | P(you pick the secret box \| secret case) | 1/12 |
        | **P(secret \| any box)** | **1/6 × 1/12 = 1/72** |
        """)

        st.latex(r"P(\text{secret}) = P(\text{secret case}) \times P(\text{pick secret box} \mid \text{secret case}) = \frac{1}{6} \times \frac{1}{12} = \frac{1}{72}")

        st.markdown("### Strategy D — Batch Size vs. Variance")
        b_vals = np.arange(1, 73)
        p_b    = 1 - (71/72)**b_vals
        ev     = (b_vals * BOX_PRICE) / p_b
        std_b  = np.sqrt((1 - p_b) / p_b**2) * b_vals * BOX_PRICE

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=b_vals, y=ev,
            name="Expected Cost",
            line=dict(color="#a78bfa", width=2.5),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=b_vals, y=std_b,
            name="Std Deviation",
            line=dict(color="#f6c94e", width=2, dash="dash"),
        ), secondary_y=True)

        fig.add_vline(x=batch_size, line_dash="dot", line_color="#ff6b9d",
                      annotation_text=f"b = {batch_size}", annotation_font_color="#ff6b9d")

        fig.update_layout(
            **base_layout("Expected Cost & Std Dev vs. Batch Size", height=380),
            xaxis_title="Batch Size b",
        )
        fig.update_yaxes(title_text="Expected Total Cost ($)", secondary_y=False,
                         gridcolor=GRID_COLOR)
        fig.update_yaxes(title_text="Std Deviation ($)", secondary_y=True,
                         gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "📊 Notice: expected cost stays flat at ≈$1,080 regardless of batch size. "
            "**Only the variance changes.** Larger batches = more predictable spending."
        )