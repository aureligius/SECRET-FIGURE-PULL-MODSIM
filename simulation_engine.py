"""
simulation_engine.py
====================
Labubu Blind Box — Stochastic Simulation Engine (Finite Pool Edition)

Key contracts with app.py:
  - run_all_strategies()   → dict with keys "A","B","C","D" → SimulationResult
  - run_gbm_display()      → np.ndarray shape (n_display, T+1)  ← critical
  - simulate_shelf_scenarios(N, rng) → np.ndarray of p_cond values
  - shelf_conditional(n_taken, rng)  → dict with keys used in Tab 4
  - generate_sealed_case(rng)        → list of 12 figure name strings
  - BOX_PRICE, CASE_PRICE, FIGURE_NAMES, P_SECRET, P_REGULAR_EACH exported
"""

import numpy as np
from typing import Optional

from stock_manager import (
    BOX_PRICE, CASE_PRICE, FIGURE_NAMES,
    P_SECRET, P_REGULAR_EACH,
    load_stock, _generate_fresh_batch,
)

# ─────────────────────────────────────────────
# SIMULATION RESULT WRAPPER
# ─────────────────────────────────────────────
class SimulationResult(dict):
    """
    Dict subclass with dot-notation access to statistical properties.
    Requires 'costs' key (np.ndarray) to be set for all computed properties.
    """

    @property
    def costs(self) -> np.ndarray:
        return self.get("_costs", np.array([0.0]))

    @costs.setter
    def costs(self, arr: np.ndarray):
        self["_costs"] = arr

    @property
    def mean(self) -> float:
        return float(np.mean(self.costs))

    @property
    def median(self) -> float:
        return float(np.median(self.costs))

    @property
    def std(self) -> float:
        return float(np.std(self.costs))

    @property
    def p95(self) -> float:
        return float(np.percentile(self.costs, 95))

    @property
    def p99(self) -> float:
        return float(np.percentile(self.costs, 99))

    @property
    def prob_over_200(self) -> float:
        return float(np.mean(self.costs > 200))

    @property
    def prob_over_500(self) -> float:
        return float(np.mean(self.costs > 500))

    @property
    def prob_over_1000(self) -> float:
        return float(np.mean(self.costs > 1000))

    def summary_dict(self) -> dict:
        return {
            "Strategy":        self.get("label", "?"),
            "Mean ($)":        f"${self.mean:,.0f}",
            "Median ($)":      f"${self.median:,.0f}",
            "Std Dev ($)":     f"${self.std:,.0f}",
            "95th Pct ($)":    f"${self.p95:,.0f}",
            "99th Pct ($)":    f"${self.p99:,.0f}",
            "P(>$200)":        f"{self.prob_over_200:.1%}",
            "P(>$500)":        f"{self.prob_over_500:.1%}",
            "P(>$1,000)":      f"{self.prob_over_1000:.1%}",
        }

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'SimulationResult' has no attribute '{name}'")


# ─────────────────────────────────────────────
# CORE: FINITE POOL SINGLE DRAW (Discrete Inverse Transform)
# ─────────────────────────────────────────────
def _draw_from_pool(stock: dict, rng: np.random.Generator) -> Optional[str]:
    """
    Draw one figure from a local stock snapshot using Discrete Inverse Transform.
    Mutates stock IN PLACE (sampling without replacement).

    Per Dr. Syukron Week 10 Slide 13:
      1. Build weights from current counts
      2. CDF = cumsum(weights)
      3. U ~ Uniform(0,1)
      4. Return figure at first i where U ≤ CDF[i]
    """
    figures = [f for f in FIGURE_NAMES if stock.get(f, 0) > 0]
    if not figures:
        return None

    counts = np.array([stock[f] for f in figures], dtype=float)
    probs  = counts / counts.sum()
    cdf    = np.cumsum(probs)

    U   = rng.uniform(0.0, 1.0)
    idx = int(np.searchsorted(cdf, U, side="left"))
    idx = min(idx, len(figures) - 1)

    chosen = figures[idx]
    stock[chosen] -= 1
    return chosen


# ─────────────────────────────────────────────
# STRATEGY A — Single Box (Gambler)
# ─────────────────────────────────────────────
def _simulate_strategy_a(stock_snapshot: dict, total_remaining: int,
                          N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Buy 1 box at a time until secret found.
    Each trial starts from the same depleted snapshot (sampling without replacement).
    """
    target = "Golden Labubu ✦"
    boxes_needed = np.zeros(N, dtype=np.int64)

    for trial in range(N):
        trial_stock = stock_snapshot.copy()
        pulls = 0
        while True:
            fig = _draw_from_pool(trial_stock, rng)
            if fig is None:
                pulls = total_remaining
                break
            pulls += 1
            if fig == target:
                break
        boxes_needed[trial] = pulls

    return boxes_needed * BOX_PRICE


# ─────────────────────────────────────────────
# STRATEGY B — Buy Full Case (Whale)
# ─────────────────────────────────────────────
def _simulate_strategy_b(stock_snapshot: dict, total_remaining: int,
                          N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Buy one full case (12 boxes) at a time until secret found.
    Resale value of duplicates is not modelled here (net cost = boxes opened × price).
    """
    target = "Golden Labubu ✦"
    costs  = np.zeros(N, dtype=float)

    for trial in range(N):
        trial_stock = stock_snapshot.copy()
        total_spent = 0.0
        found = False
        while not found:
            # Open up to 12 boxes per round
            for _ in range(12):
                fig = _draw_from_pool(trial_stock, rng)
                if fig is None:
                    found = True
                    break
                total_spent += BOX_PRICE
                if fig == target:
                    found = True
                    break
        costs[trial] = total_spent

    return costs


# ─────────────────────────────────────────────
# STRATEGY C — Patient Buyer (wait for price dip)
# Modelled as: buy only when a simulated GBM price ≤ threshold.
# For cost distribution we simulate how many boxes needed (same as A)
# but weight cost by average resale multiple at time of purchase.
# ─────────────────────────────────────────────
def _simulate_strategy_c(stock_snapshot: dict, total_remaining: int,
                          N: int, rng: np.random.Generator,
                          gbm_S0: float, gbm_mu: float, gbm_sigma: float,
                          gbm_T: int, threshold: float) -> np.ndarray:
    """
    Patient buyer: each trial draws a GBM final price.
    If price ≤ threshold, they buy (cost based on box count, same as A).
    If price > threshold, they wait and pay a premium = current_price/S0 × base_cost.
    """
    target = "Golden Labubu ✦"
    dt     = 1.0 / 252  # daily steps
    costs  = np.zeros(N, dtype=float)

    for trial in range(N):
        trial_stock = stock_snapshot.copy()
        pulls = 0
        while True:
            fig = _draw_from_pool(trial_stock, rng)
            if fig is None:
                pulls = total_remaining
                break
            pulls += 1
            if fig == target:
                break

        # Simulate GBM price on purchase day
        days   = rng.integers(1, max(gbm_T, 2))
        shocks = rng.standard_normal(days)
        log_r  = (gbm_mu - 0.5 * gbm_sigma**2) * dt + gbm_sigma * np.sqrt(dt) * shocks
        price  = float(gbm_S0 * np.exp(np.sum(log_r)))

        # Patient buyer saves when price ≤ threshold
        if price <= threshold:
            multiplier = threshold / gbm_S0
        else:
            multiplier = price / gbm_S0

        costs[trial] = pulls * BOX_PRICE * max(0.5, min(multiplier, 3.0))

    return costs


# ─────────────────────────────────────────────
# STRATEGY D — Calculated Batch Buyer
# ─────────────────────────────────────────────
def _simulate_strategy_d(stock_snapshot: dict, total_remaining: int,
                          N: int, rng: np.random.Generator,
                          batch_size: int) -> np.ndarray:
    """
    Buy batch_size boxes per round, stop when secret found or pool exhausted.
    """
    target = "Golden Labubu ✦"
    costs  = np.zeros(N, dtype=float)

    for trial in range(N):
        trial_stock = stock_snapshot.copy()
        total_spent = 0.0
        found = False
        while not found:
            for _ in range(batch_size):
                fig = _draw_from_pool(trial_stock, rng)
                if fig is None:
                    found = True
                    break
                total_spent += BOX_PRICE
                if fig == target:
                    found = True
                    break
        costs[trial] = total_spent

    return costs


# ─────────────────────────────────────────────
# MACHINE 1: MONTE CARLO BUDGET CONFIDENCE
# ─────────────────────────────────────────────
def simulate_budget_confidence(
    N:                 int   = 100_000,
    target_figure:     str   = "Golden Labubu ✦",
    confidence_levels: list  = None,
    seed:              Optional[int] = None,
) -> dict:
    """
    Run N Monte Carlo trials starting from the CURRENT depleted pool state.
    Returns budget needed for each confidence level and full cost distribution.
    """
    if confidence_levels is None:
        confidence_levels = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]

    rng         = np.random.default_rng(seed)
    pool_data   = load_stock()
    snapshot    = pool_data["stock"].copy()
    total_left  = pool_data["total_remaining"]

    if snapshot.get(target_figure, 0) == 0:
        return {
            "target_gone":           True,
            "target_figure":         target_figure,
            "total_remaining":       total_left,
            "budget_for_confidence": {c: 0.0 for c in confidence_levels},
            "cost_distribution":     np.array([0.0]),
            "message":               f"'{target_figure}' is no longer in the pool.",
        }

    costs = _simulate_strategy_a(snapshot, total_left, N, rng)

    budget_for_confidence = {
        c: float(np.percentile(costs, c * 100))
        for c in confidence_levels
    }

    total = sum(snapshot.values())
    p_target = snapshot.get(target_figure, 0) / total if total > 0 else 0.0

    return {
        "target_gone":           False,
        "target_figure":         target_figure,
        "total_remaining":       total_left,
        "budget_for_confidence": budget_for_confidence,
        "cost_distribution":     costs,
        "mean_cost":             float(np.mean(costs)),
        "median_cost":           float(np.median(costs)),
        "std_cost":              float(np.std(costs)),
        "p95_cost":              float(np.percentile(costs, 95)),
        "theoretical_mean":      (BOX_PRICE / p_target) if p_target > 0 else float("inf"),
        "p_target_now":          p_target,
        "N":                     N,
    }


# ─────────────────────────────────────────────
# RUN ALL STRATEGIES — returns dict keyed A/B/C/D
# app.py accesses results[key].costs, .mean, .std, .p95, .p99 etc.
# ─────────────────────────────────────────────
def run_all_strategies(
    N:               int   = 100_000,
    batch_size:      int   = 8,
    gbm_S0:          float = 80.0,
    gbm_mu:          float = 0.4,
    gbm_sigma:       float = 0.6,
    gbm_T:           int   = 30,
    threshold:       float = 60.0,
    resale_discount: float = 0.4,
    seed:            Optional[int] = None,
    **kwargs,
) -> dict:
    """
    Run all four purchase strategies and return SimulationResult objects.
    Each result exposes .costs (ndarray), .mean, .median, .std, .p95, .p99,
    .prob_over_200/500/1000, and .summary_dict().
    """
    rng       = np.random.default_rng(seed)
    pool_data = load_stock()
    snapshot  = pool_data["stock"].copy()
    total_left = pool_data["total_remaining"]

    label_map = {
        "A": f"A — Gambler (1×)",
        "B": f"B — Whale (12×)",
        "C": f"C — Patient Buyer",
        "D": f"D — Calculated ({batch_size}×)",
    }

    results = {}

    for key in ["A", "B", "C", "D"]:
        r = SimulationResult()
        r["label"] = label_map[key]

        if key == "A":
            r.costs = _simulate_strategy_a(snapshot, total_left, N, rng)
        elif key == "B":
            r.costs = _simulate_strategy_b(snapshot, total_left, N, rng)
        elif key == "C":
            r.costs = _simulate_strategy_c(
                snapshot, total_left, N, rng,
                gbm_S0, gbm_mu, gbm_sigma, gbm_T, threshold
            )
        elif key == "D":
            r.costs = _simulate_strategy_d(snapshot, total_left, N, rng, batch_size)

        results[key] = r

    return results


# ─────────────────────────────────────────────
# GBM DISPLAY — returns np.ndarray shape (n_display, T+1)
# app.py does: paths.shape, paths.mean(axis=0), np.percentile(paths, p, axis=0)
# ─────────────────────────────────────────────
def run_gbm_display(
    S0:        float = 80.0,
    mu:        float = 0.4,
    sigma:     float = 0.6,
    T:         int   = 30,
    n_display: int   = 300,
    seed:      Optional[int] = None,
) -> np.ndarray:
    """
    Simulate n_display independent GBM price paths each of length T+1.
    Returns ndarray of shape (n_display, T+1).

    GBM: S_{t+1} = S_t * exp[(μ - σ²/2)Δt + σ√Δt · Z],  Z ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    dt  = 1.0 / 252  # daily steps (trading-day convention)

    # Shape: (n_display, T) — one shock per path per day
    shocks   = rng.standard_normal((n_display, T))
    log_rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks

    # Build price paths: prepend S0 column, then cumulative product
    paths        = np.empty((n_display, T + 1), dtype=float)
    paths[:, 0]  = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_rets, axis=1))

    return paths


# ─────────────────────────────────────────────
# SHELF CONDITIONAL — called as shelf_conditional(n_taken, rng)
# app.py: result_shelf = shelf_conditional(n_taken, rng_shelf)
#         result_shelf['verdict'], ['n_remaining'], ['p_secret_remaining']
# ─────────────────────────────────────────────
def shelf_conditional(n_taken: int, rng: np.random.Generator) -> dict:
    """
    Compute conditional P(secret still on shelf | n_taken boxes already gone).
    Analytically exact (no Monte Carlo needed for this simple case).

    Uses hypergeometric reasoning:
      - Full batch = 72 boxes, 1 secret
      - n_taken boxes removed (each uniformly at random from remaining pool)
      - P(secret still present) = (72 - n_taken) / 72  if n_taken ≤ 71
    """
    n_taken      = min(n_taken, 71)
    n_remaining  = 72 - n_taken
    p_secret     = 1 / 72  # unconditional

    # Conditional on secret not yet found:
    # P(secret in remaining | secret not in first n_taken) = 1/n_remaining
    # But we don't know if it was taken — P(secret still here) = (72-n_taken)/72
    p_secret_remaining = n_remaining / 72

    if n_taken == 0:
        verdict = "🟢 Fresh shelf — full odds in play."
    elif p_secret_remaining > 0.5:
        verdict = "🟡 Some boxes gone, but secret likely still here."
    elif p_secret_remaining > 0:
        verdict = "🔴 Heavily depleted — secret may already be gone."
    else:
        verdict = "❌ Pool exhausted — no boxes remain."

    return {
        "n_taken":              n_taken,
        "n_remaining":          n_remaining,
        "p_secret_remaining":   p_secret_remaining,
        "p_secret_conditional": (1 / n_remaining) if n_remaining > 0 else 0.0,
        "verdict":              verdict,
    }


# ─────────────────────────────────────────────
# SIMULATE SHELF SCENARIOS — called as simulate_shelf_scenarios(N, rng)
# app.py: p_cond = simulate_shelf_scenarios(10_000, rng2)
#         expects np.ndarray of float values (p_secret_remaining per scenario)
# ─────────────────────────────────────────────
def simulate_shelf_scenarios(N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate N random walk-in scenarios. For each scenario:
      - Draw a random number of boxes already taken (0–71)
      - Compute p_secret_remaining for that depletion level
    Returns np.ndarray of shape (N,) with p_secret_remaining values.

    Each value is P(secret still in pool | n_taken boxes already gone).
    Marginal mean = 0.5 over uniform n_taken ~ [0,71].
    """
    # Random number of boxes already taken for each scenario
    n_taken_arr = rng.integers(0, 72, size=N)  # 0 to 71 inclusive

    # p_secret_remaining = (72 - n_taken) / 72
    p_cond = (72 - n_taken_arr) / 72.0
    return p_cond


# ─────────────────────────────────────────────
# GENERATE SEALED CASE — called as generate_sealed_case(rng)
# app.py: case = generate_sealed_case(rng_case)  →  list of 12 figure-name strings
# ─────────────────────────────────────────────
def generate_sealed_case(rng: np.random.Generator) -> list:
    """
    Generate one physical sealed case of 12 boxes.
    - 1-in-6 chance it is a secret case (11 regulars + 1 Golden Labubu ✦)
    - 5-in-6 chance it is a normal case (base set of 6 + 6 random extras)
    Returns list of 12 figure-name strings.
    """
    is_secret_case = rng.random() < (1 / 6)

    if is_secret_case:
        boxes = list(rng.choice(FIGURE_NAMES[:6], size=11, replace=True)) + ["Golden Labubu ✦"]
    else:
        base  = FIGURE_NAMES[:6].copy()
        extra = list(rng.choice(FIGURE_NAMES[:6], size=6, replace=True))
        boxes = base + extra

    rng.shuffle(boxes)
    return boxes


# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS 1 — Starting Depletion Level
# ─────────────────────────────────────────────
def sensitivity_depletion_level(
    depletion_levels: list         = None,
    N:                int          = 50_000,
    seed:             Optional[int] = None,
) -> list:
    """
    Varies how many boxes were already taken before the user arrives.
    Shows how conditional probability and required budget change.
    """
    if depletion_levels is None:
        depletion_levels = [0, 12, 24, 36, 48, 60, 71]

    rng        = np.random.default_rng(seed)
    fresh      = _generate_fresh_batch(seed=seed)
    full_stock = fresh["stock"].copy()
    results    = []

    for n_taken in depletion_levels:
        taken_stock = full_stock.copy()
        pool_total  = sum(taken_stock.values())

        # Remove n_taken boxes proportionally at random
        for _ in range(min(n_taken, pool_total)):
            figs = [f for f in FIGURE_NAMES if taken_stock[f] > 0]
            if not figs:
                break
            counts = np.array([taken_stock[f] for f in figs], dtype=float)
            probs  = counts / counts.sum()
            chosen = rng.choice(figs, p=probs)
            taken_stock[chosen] -= 1

        pool_remaining = sum(taken_stock.values())
        secret_count   = taken_stock.get("Golden Labubu ✦", 0)
        p_secret       = secret_count / pool_remaining if pool_remaining > 0 else 0.0

        if secret_count == 0:
            results.append({
                "n_taken":        n_taken,
                "pool_remaining": pool_remaining,
                "secret_in_pool": False,
                "p_secret":       0.0,
                "mean_budget":    None,
                "budget_80pct":   None,
                "budget_95pct":   None,
            })
            continue

        costs = _simulate_strategy_a(taken_stock, pool_remaining, N, rng)
        results.append({
            "n_taken":        n_taken,
            "pool_remaining": pool_remaining,
            "secret_in_pool": True,
            "p_secret":       p_secret,
            "mean_budget":    float(np.mean(costs)),
            "budget_80pct":   float(np.percentile(costs, 80)),
            "budget_95pct":   float(np.percentile(costs, 95)),
        })

    return results


# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS 2 — Confidence Threshold Curve
# ─────────────────────────────────────────────
def sensitivity_confidence_curve(
    target_figure: str         = "Golden Labubu ✦",
    N:             int         = 100_000,
    seed:          Optional[int] = None,
) -> dict:
    """
    Derive the full budget-vs-confidence curve from empirical CDF.
    Returns confidence_levels list, budget_required list, key_points dict.
    """
    confidence_list = [round(c, 2) for c in np.arange(0.01, 1.00, 0.01)]

    result = simulate_budget_confidence(
        N=N,
        target_figure=target_figure,
        confidence_levels=confidence_list,
        seed=seed,
    )

    if result["target_gone"]:
        return result

    budget_arr = np.array([result["budget_for_confidence"][c] for c in confidence_list])
    key_points = {
        c: float(np.percentile(result["cost_distribution"], c * 100))
        for c in [0.50, 0.80, 0.90, 0.95, 0.99]
    }

    return {
        "target_gone":       False,
        "target_figure":     target_figure,
        "confidence_levels": confidence_list,
        "budget_required":   budget_arr.tolist(),
        "key_points":        key_points,
        "mean_cost":         result["mean_cost"],
        "p_target_now":      result["p_target_now"],
        "total_remaining":   result["total_remaining"],
        "N":                 N,
        "cost_distribution": result["cost_distribution"],
    }


# ─────────────────────────────────────────────
# STANDARD ERROR (for paper defence)
# ─────────────────────────────────────────────
def compute_standard_error(costs: np.ndarray) -> dict:
    """Compute 95% CI for the mean cost estimate: mean ± 1.96 × (σ/√N)"""
    N    = len(costs)
    mean = float(np.mean(costs))
    std  = float(np.std(costs))
    se   = std / np.sqrt(N)
    ci_w = 1.96 * se
    return {
        "N":          N,
        "mean":       mean,
        "std":        std,
        "se":         se,
        "ci95_lower": mean - ci_w,
        "ci95_upper": mean + ci_w,
        "ci95_width": ci_w * 2,
        "error_pct":  (se / mean * 100) if mean > 0 else 0,
    }