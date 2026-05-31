import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
N_RUNS = 100_000
DAYS_GBM = 30
P_SINGLE = 1 / 72          # flat pull probability per box
P_CASE = 1 / 6             # probability secret figure is in a given case
BOXES_PER_CASE = 12

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "jackpot_hit" not in st.session_state:
    st.session_state.jackpot_hit = False
if "results" not in st.session_state:
    st.session_state.results = None

# ─────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────

def simulate_strategy_a(single_cost: int) -> dict:
    """
    Strategy A — Single Box Gambler
    Each pull is an independent Bernoulli trial with p = 1/72.
    Number of pulls follows Geometric(p).
    """
    # vectorized geometric sampling: numpy uses 0-indexed so +1 for pull count
    pulls = np.random.geometric(p=P_SINGLE, size=N_RUNS)  # shape (100_000,)
    cost = pulls * single_cost
    return {"pulls": pulls, "cost": cost}


def simulate_strategy_b(case_cost: int) -> dict:
    """
    Strategy B — Case Whale
    Buy factory-sealed cases (12 boxes, guaranteed unique).
    Cases needed follows Geometric(p=1/6).
    Position within final case follows Discrete Uniform(1, 12).
    """
    cases_needed = np.random.geometric(p=P_CASE, size=N_RUNS)
    offset = np.random.randint(1, BOXES_PER_CASE + 1, size=N_RUNS)
    total_boxes = (cases_needed - 1) * BOXES_PER_CASE + offset
    cost = cases_needed * case_cost
    return {"cases": cases_needed, "total_boxes": total_boxes, "cost": cost}


def simulate_strategy_c(S0: float, mu: float, sigma: float) -> dict:
    """
    Strategy C — Secondary Market Arbitrageur
    Models reseller price as Geometric Brownian Motion over 30 days.
    S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_t)
    where Z_t ~ N(0,1), dt = 1/DAYS_GBM
    """
    dt = 1 / DAYS_GBM
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # gbm_matrix shape: (DAYS_GBM+1, N_RUNS)
    gbm_matrix = np.zeros((DAYS_GBM + 1, N_RUNS))
    gbm_matrix[0] = S0

    for t in range(1, DAYS_GBM + 1):
        Z = np.random.standard_normal(N_RUNS)
        gbm_matrix[t] = gbm_matrix[t - 1] * np.exp(drift + diffusion * Z)

    final_prices = gbm_matrix[-1]  # shape (100_000,)
    return {"gbm_matrix": gbm_matrix, "final_prices": final_prices}


# ─────────────────────────────────────────────
# METRIC AGGREGATOR
# ─────────────────────────────────────────────

def aggregate_metrics(a: dict, b: dict, c: dict) -> dict:
    return {
        # Strategy A
        "a_mean_pulls": a["pulls"].mean(),
        "a_std_pulls": a["pulls"].std(),
        "a_mean_cost": a["cost"].mean(),
        "a_std_cost": a["cost"].std(),
        "a_p95_cost": np.percentile(a["cost"], 95),
        "a_jackpot_count": int((a["pulls"] == 1).sum()),

        # Strategy B
        "b_mean_boxes": b["total_boxes"].mean(),
        "b_mean_cost": b["cost"].mean(),
        "b_std_cost": b["cost"].std(),
        "b_p95_cost": np.percentile(b["cost"], 95),

        # Strategy C
        "c_mean_price": c["final_prices"].mean(),
        "c_std_price": c["final_prices"].std(),
        "c_p05_price": np.percentile(c["final_prices"], 5),
        "c_p95_price": np.percentile(c["final_prices"], 95),
    }


# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────

def build_cost_density_chart(a_cost, b_cost, c_prices):
    """Overlaid probability density histogram for all 3 strategies."""
    df = pd.DataFrame({
        "cost": np.concatenate([a_cost, b_cost, c_prices]),
        "strategy": (
            ["A: Single Box"] * N_RUNS +
            ["B: Sealed Case"] * N_RUNS +
            ["C: Reseller Market"] * N_RUNS
        )
    })
    fig = px.histogram(
        df, x="cost", color="strategy",
        barmode="overlay", nbins=120,
        histnorm="probability density",
        marginal="box",
        opacity=0.65,
        title="Financial Risk Density — All 3 Strategies (100,000 runs)",
        labels={"cost": "Total Cost (IDR)", "strategy": "Strategy"},
        color_discrete_map={
            "A: Single Box": "#FF6B9D",
            "B: Sealed Case": "#845EC2",
            "C: Reseller Market": "#00C9A7"
        }
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#EEEEEE"
    )
    return fig


def build_gbm_path_chart(gbm_matrix, S0):
    """Plot 50 representative GBM paths out of 100,000."""
    fig = go.Figure()
    sample_idx = np.linspace(0, N_RUNS - 1, 50, dtype=int)
    days = np.arange(DAYS_GBM + 1)

    for idx in sample_idx:
        fig.add_trace(go.Scatter(
            x=days, y=gbm_matrix[:, idx],
            mode="lines",
            line=dict(width=0.8, color="#00C9A7"),
            opacity=0.4,
            showlegend=False
        ))

    # mean path overlay
    mean_path = gbm_matrix.mean(axis=1)
    fig.add_trace(go.Scatter(
        x=days, y=mean_path,
        mode="lines",
        line=dict(width=2.5, color="#FFD700", dash="dash"),
        name="Mean Path"
    ))

    fig.add_hline(y=S0, line_dash="dot", line_color="#FF6B9D",
                  annotation_text="Initial Price")

    fig.update_layout(
        title="Secondary Market Price Paths — GBM (50 sampled / 100,000 runs)",
        xaxis_title="Day",
        yaxis_title="Price (IDR)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#EEEEEE"
    )
    return fig


# ─────────────────────────────────────────────
# JACKPOT ANIMATION
# ─────────────────────────────────────────────

def render_jackpot_modal():
    components.html("""
    <style>
      @keyframes pop {
        0%   { transform: scale(0) rotate(-10deg); opacity: 0; }
        70%  { transform: scale(1.2) rotate(5deg); opacity: 1; }
        100% { transform: scale(1) rotate(0deg); }
      }
      @keyframes shimmer {
        0%, 100% { background-position: -200% center; }
        50%       { background-position: 200% center; }
      }
      .modal-overlay {
        position: fixed; inset: 0;
        background: rgba(0,0,0,0.75);
        display: flex; align-items: center; justify-content: center;
        z-index: 9999;
      }
      .modal-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #FF6B9D;
        border-radius: 20px;
        padding: 48px 64px;
        text-align: center;
        animation: pop 0.6s ease forwards;
        box-shadow: 0 0 60px rgba(255,107,157,0.4);
      }
      .modal-emoji { font-size: 72px; display: block; margin-bottom: 16px; }
      .modal-title {
        font-size: 32px; font-weight: 900;
        background: linear-gradient(90deg, #FF6B9D, #FFD700, #FF6B9D);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 2s infinite;
        font-family: sans-serif;
      }
      .modal-sub {
        color: #aaa; font-size: 15px; margin-top: 12px;
        font-family: sans-serif;
      }
    </style>
    <div class="modal-overlay">
      <div class="modal-card">
        <span class="modal-emoji">🧸⭐</span>
        <div class="modal-title">JACKPOT — FIRST PULL!</div>
        <div class="modal-sub">
          At least one simulation run hit the secret figure on pull #1.<br>
          Probability: 1/72 ≈ 1.39% — and it happened.
        </div>
      </div>
    </div>
    """, height=340)


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(page_title="Labubu Pull Simulator", layout="wide")
st.title("🧸 Labubu Secret Figure — Pull Strategy Simulator")
st.caption(f"Monte Carlo Engine · {N_RUNS:,} simulation runs per strategy")

# ── Sidebar Inputs ──
with st.sidebar:
    st.header("⚙️ Parameters")
    st.subheader("Retail Market")
    single_cost = st.number_input("Single Box Cost (IDR)", value=150_000, step=5_000)
    case_cost = st.number_input("Sealed Case Cost (IDR)", value=1_700_000, step=50_000)

    st.subheader("Secondary Market (Strategy C)")
    S0 = st.number_input("Initial Reseller Price (IDR)", value=2_500_000, step=100_000)
    mu = st.slider("Market Drift (μ)", min_value=-0.10, max_value=0.10,
                   value=0.0, step=0.01, format="%.2f")
    sigma = st.slider("Market Volatility (σ)", min_value=0.01, max_value=0.50,
                      value=0.15, step=0.01, format="%.2f")

    run_btn = st.button("🎲 Run Simulation", use_container_width=True, type="primary")

# ── Simulation Trigger ──
if run_btn:
    with st.spinner("Running 100,000 Monte Carlo iterations..."):
        rng_seed = np.random.seed(42)
        res_a = simulate_strategy_a(single_cost)
        res_b = simulate_strategy_b(case_cost)
        res_c = simulate_strategy_c(S0, mu, sigma)
        metrics = aggregate_metrics(res_a, res_b, res_c)

        st.session_state.results = {
            "a": res_a, "b": res_b, "c": res_c, "metrics": metrics
        }
        st.session_state.jackpot_hit = metrics["a_jackpot_count"] > 0

# ── Results Display ──
if st.session_state.results:
    res = st.session_state.results
    m = res["metrics"]

    # ── Key Metrics Row ──
    st.subheader("📊 Expected Cost Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🎰 Strategy A — Single Box",
            f"Rp {m['a_mean_cost']:,.0f}",
            delta=f"±Rp {m['a_std_cost']:,.0f} std dev"
        )
        st.caption(f"Avg pulls: {m['a_mean_pulls']:.1f} | P95 cost: Rp {m['a_p95_cost']:,.0f}")

    with col2:
        st.metric(
            "📦 Strategy B — Sealed Case",
            f"Rp {m['b_mean_cost']:,.0f}",
            delta=f"±Rp {m['b_std_cost']:,.0f} std dev"
        )
        st.caption(f"Avg boxes: {m['b_mean_boxes']:.1f} | P95 cost: Rp {m['b_p95_cost']:,.0f}")

    with col3:
        st.metric(
            "📈 Strategy C — Reseller (Day 30)",
            f"Rp {m['c_mean_price']:,.0f}",
            delta=f"±Rp {m['c_std_price']:,.0f} std dev"
        )
        st.caption(
            f"P5: Rp {m['c_p05_price']:,.0f} | P95: Rp {m['c_p95_price']:,.0f}"
        )

    st.divider()

    # ── Charts ──
    st.subheader("📉 Financial Risk Density Map")
    fig_density = build_cost_density_chart(
        res["a"]["cost"], res["b"]["cost"], res["c"]["final_prices"]
    )
    st.plotly_chart(fig_density, use_container_width=True)

    st.subheader("📈 Secondary Market Price Paths (GBM)")
    fig_gbm = build_gbm_path_chart(res["c"]["gbm_matrix"], S0)
    st.plotly_chart(fig_gbm, use_container_width=True)

    st.divider()

    # ── Jackpot State Check ──
    if st.session_state.jackpot_hit:
        st.warning(
            f"⭐ Jackpot detected: {m['a_jackpot_count']:,} runs "
            f"hit the secret figure on pull #1 out of {N_RUNS:,}."
        )
        render_jackpot_modal()
    else:
        st.info(
            f"No first-pull jackpot across {N_RUNS:,} runs. "
            f"Theoretical probability: {P_SINGLE:.4f} per run."
        )