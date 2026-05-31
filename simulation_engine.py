"""
simulation_engine.py
====================
Labubu Blind Box — Stochastic Simulation Engine
All strategies are fully vectorised with NumPy.
N = 100,000 trials runs in < 100 ms on any modern machine.

Mathematical foundations
------------------------
  P(secret in one box)          = 1/72
  P(regular figure i, one box)  = 71/432  (i = 0..5)
  Geometric inverse-CDF trick   : X = ceil(ln U / ln(1-p))
  GBM discrete exact solution   : S_{t+1} = S_t * exp((μ - σ²/2)dt + σ√dt * Z)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
BOX_PRICE       = 15.0          # retail price per single box  ($)
CASE_SIZE       = 12            # boxes per sealed case
CASES_PER_SECRET = 6            # 1 secret case per 6 cases  → P = 1/72
CASE_PRICE      = BOX_PRICE * CASE_SIZE   # $180
P_SECRET        = 1.0 / 72.0
P_REGULAR_EACH  = (71.0 / 72.0) / 6.0    # = 71/432

FIGURE_NAMES = [
    "Cherry Blossom", "Mint Dream", "Lavender Haze",
    "Peach Glow",     "Sky Pop",    "Coral Bloom",
    "Golden Labubu ✦"   # index 6 = secret
]


# ─────────────────────────────────────────────
# HELPER — inverse-CDF for Geometric distribution
# ─────────────────────────────────────────────
def _geom_icdf(p: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return n samples from Geom(p) using the inverse-CDF method.
    X = ceil( ln(U) / ln(1-p) ),  U ~ Uniform(0,1)
    This avoids any Python-level loop — pure NumPy SIMD.
    """
    u = rng.uniform(0.0, 1.0, n)
    # Clip to avoid log(0)
    u = np.clip(u, 1e-15, 1.0 - 1e-15)
    return np.ceil(np.log(u) / np.log(1.0 - p)).astype(np.int64)


# ─────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────
@dataclass
class StrategyResult:
    name:          str
    costs:         np.ndarray          # shape (N,)  — cost per trial
    mean:          float = field(init=False)
    median:        float = field(init=False)
    std:           float = field(init=False)
    p5:            float = field(init=False)   # 5th  percentile  (best case)
    p95:           float = field(init=False)   # 95th percentile  (worst case)
    p99:           float = field(init=False)   # 99th percentile
    prob_over_200: float = field(init=False)   # P(cost > $200)
    prob_over_500: float = field(init=False)
    prob_over_1000:float = field(init=False)

    def __post_init__(self):
        self.mean          = float(np.mean(self.costs))
        self.median        = float(np.median(self.costs))
        self.std           = float(np.std(self.costs))
        self.p5            = float(np.percentile(self.costs, 5))
        self.p95           = float(np.percentile(self.costs, 95))
        self.p99           = float(np.percentile(self.costs, 99))
        self.prob_over_200 = float(np.mean(self.costs > 200))
        self.prob_over_500 = float(np.mean(self.costs > 500))
        self.prob_over_1000= float(np.mean(self.costs > 1000))

    def summary_dict(self) -> dict:
        return {
            "Strategy":      self.name,
            "Mean Cost":     f"${self.mean:,.2f}",
            "Median Cost":   f"${self.median:,.2f}",
            "Std Dev":       f"${self.std:,.2f}",
            "5th Pct":       f"${self.p5:,.2f}",
            "95th Pct":      f"${self.p95:,.2f}",
            "99th Pct":      f"${self.p99:,.2f}",
            "P(cost>$200)":  f"{self.prob_over_200:.1%}",
            "P(cost>$500)":  f"{self.prob_over_500:.1%}",
            "P(cost>$1000)": f"{self.prob_over_1000:.1%}",
        }


# ─────────────────────────────────────────────
# MODULE 1 — SEALED CASE ENGINE
# ─────────────────────────────────────────────
def generate_sealed_case(rng: np.random.Generator) -> list[str]:
    """
    Generate one physical sealed case of 12 boxes.

    Rules (mirroring Pop Mart production logic):
      - 12 slots, one of each regular figure guaranteed (6 regulars × 2 = 12,
        but here we use 6 regulars with 2 slots each for the non-secret case,
        or 5 regulars × 2 + 1 rare-extra + 1 secret for the secret case).
      - With probability 1/6 this is a 'secret case':
            11 regular figures (assorted) + 1 secret figure, shuffled.
      - With probability 5/6 this is a 'normal case':
            12 regular figures (assorted, at least one of each type), shuffled.

    Returns a list of 12 figure name strings.
    """
    is_secret_case = rng.random() < (1.0 / 6.0)

    if is_secret_case:
        # 11 regular slots + 1 secret
        regulars = list(rng.choice(FIGURE_NAMES[:6], size=11, replace=True))
        boxes = regulars + [FIGURE_NAMES[6]]
    else:
        # 12 regular slots — guarantee at least one of each of the 6 types
        base = FIGURE_NAMES[:6].copy()                            # 6 guaranteed
        extra = list(rng.choice(FIGURE_NAMES[:6], size=6, replace=True))  # 6 filler
        boxes = base + extra

    rng.shuffle(boxes)
    return boxes


def simulate_case_batch(n_cases: int, rng: np.random.Generator) -> list[list[str]]:
    """Generate a batch of sealed cases (used by Strategy B)."""
    return [generate_sealed_case(rng) for _ in range(n_cases)]


# ─────────────────────────────────────────────
# MODULE 2 — PICKED-THROUGH SHELF (Conditional Probability)
# ─────────────────────────────────────────────
def shelf_conditional(
    boxes_already_taken: int,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate the 'Gandaria City shelf' scenario.

    Given that `boxes_already_taken` boxes are already gone,
    compute:
      - Whether this is a secret case  (1/6 chance)
      - Whether the secret was already taken
      - Updated P(secret | remaining boxes)
      - How many boxes remain

    Returns a dict with the conditional state.
    """
    n_remaining = CASE_SIZE - boxes_already_taken
    is_secret_case = rng.random() < (1.0 / 6.0)

    if not is_secret_case:
        return {
            "is_secret_case":      False,
            "secret_already_gone": False,
            "p_secret_remaining":  0.0,
            "n_remaining":         n_remaining,
            "boxes_taken":         boxes_already_taken,
            "verdict": "❌ Normal case — no secret was ever here.",
        }

    # Secret case: secret is in a uniformly random position 0..11
    secret_position = rng.integers(0, CASE_SIZE)
    secret_gone = secret_position < boxes_already_taken

    if secret_gone:
        p_remaining = 0.0
        verdict = f"💀 Secret case, but secret was in position {secret_position} — already taken."
    else:
        p_remaining = 1.0 / n_remaining
        verdict = (
            f"✨ Secret case! Secret is still here. "
            f"Your odds: 1/{n_remaining} = {p_remaining:.1%}"
        )

    return {
        "is_secret_case":      True,
        "secret_already_gone": secret_gone,
        "p_secret_remaining":  p_remaining,
        "n_remaining":         n_remaining,
        "boxes_taken":         boxes_already_taken,
        "verdict":             verdict,
    }


def simulate_shelf_scenarios(n_scenarios: int, rng: np.random.Generator) -> np.ndarray:
    """
    Vectorised shelf simulation over n_scenarios.
    Returns array of effective P(secret) values for each scenario,
    showing the distribution of conditional probabilities a real shopper faces.
    """
    # Randomly sample how many boxes are already gone (0..11)
    taken = rng.integers(0, CASE_SIZE, size=n_scenarios)

    # Is it a secret case?
    is_secret = rng.random(n_scenarios) < (1.0 / 6.0)

    # Where is the secret inside the case?
    secret_pos = rng.integers(0, CASE_SIZE, size=n_scenarios)

    # Secret still available?
    still_here = is_secret & (secret_pos >= taken)

    # Conditional probability
    n_remaining = CASE_SIZE - taken
    p_conditional = np.where(still_here, 1.0 / n_remaining, 0.0)

    return p_conditional


# ─────────────────────────────────────────────
# MODULE 3 — GEOMETRIC BROWNIAN MOTION
# ─────────────────────────────────────────────
def simulate_gbm(
    S0:    float,
    mu:    float,
    sigma: float,
    T:     int,
    N:     int,
    rng:   np.random.Generator,
) -> np.ndarray:
    """
    Exact discrete GBM for N paths over T trading days.

    S_{t+1} = S_t * exp( (μ - σ²/2)·dt + σ·√dt·Z_t )
    Z_t ~ N(0,1) i.i.d.,  dt = 1/252

    Returns
    -------
    paths : np.ndarray, shape (N, T+1)
        paths[:, 0] = S0 for all paths
        paths[:, t] = price on day t
    """
    dt    = 1.0 / 252.0
    drift = (mu - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)

    # Draw all random shocks at once — shape (N, T)
    Z = rng.standard_normal((N, T))

    # Log-return increments
    log_increments = drift + vol * Z          # shape (N, T)

    # Cumulative log returns — prepend 0 for t=0
    log_paths = np.concatenate(
        [np.zeros((N, 1)), np.cumsum(log_increments, axis=1)],
        axis=1
    )                                          # shape (N, T+1)

    return S0 * np.exp(log_paths)


def first_passage_time(paths: np.ndarray, threshold: float) -> np.ndarray:
    """
    For each GBM path, find the first day the price drops at or below `threshold`.
    Returns array of shape (N,) with the crossing day index (T+1 if never crossed).
    """
    N, T1 = paths.shape
    T = T1 - 1
    # Boolean: price <= threshold
    below = paths <= threshold                 # (N, T+1)
    # argmax returns index of first True; if never True, returns 0
    # so we mask with never-crossed flag
    crossed    = below.any(axis=1)             # (N,)
    first_day  = np.where(crossed, np.argmax(below, axis=1), T)
    return first_day


# ─────────────────────────────────────────────
# STRATEGY A — The Gambler (single boxes)
# ─────────────────────────────────────────────
def strategy_a(N: int, rng: np.random.Generator) -> StrategyResult:
    """
    Buy single boxes one at a time until the secret is pulled.
    Uses inverse-CDF for Geom(1/72) — zero Python loops.

    Cost = boxes_needed × $15
    """
    boxes = _geom_icdf(P_SECRET, N, rng)
    costs = boxes * BOX_PRICE
    return StrategyResult("A — Gambler (1×)", costs)


# ─────────────────────────────────────────────
# STRATEGY B — The Whale (sealed cases)
# ─────────────────────────────────────────────
def strategy_b(
    N:               int,
    rng:             np.random.Generator,
    resale_discount: float = 0.4,    # sell duplicates at 40% of GBM spot price
    gbm_S0:          float = 15.0,   # resale price of regular figures
    gbm_mu:          float = -0.1,
    gbm_sigma:       float = 0.3,
) -> StrategyResult:
    """
    Buy sealed cases (12 boxes, $180) until a secret case is drawn.
    Then sell all duplicate regular figures at the reseller market price.

    Cases needed ~ Geom(1/6)
    Gross cost   = cases_needed × $180
    Duplicates   = (cases_needed - 1) × 11  regular figures from failed cases
                 + 11 regular figures from the winning case
                 = cases_needed × 11
    Net cost     = gross - duplicates × spot_price × discount
    """
    # Cases until first secret case
    cases_needed  = _geom_icdf(1.0 / 6.0, N, rng)
    gross_cost    = cases_needed * CASE_PRICE

    # Total duplicate regular figures accumulated
    # Each case yields 11 regulars (secret case) or 12 regulars (normal case).
    # Simplified: all failed cases give 12 regulars, winning gives 11.
    duplicates    = (cases_needed - 1) * 12 + 11

    # Resale spot price — sample GBM endpoint for each trial
    # Use a 30-day window; take the price at day 30
    T = 30
    Z = rng.standard_normal(N)
    dt = T / 252.0
    spot = gbm_S0 * np.exp(
        (gbm_mu - 0.5 * gbm_sigma**2) * dt
        + gbm_sigma * np.sqrt(dt) * Z
    )

    resale_revenue = duplicates * spot * resale_discount
    net_cost       = np.maximum(gross_cost - resale_revenue, 0.0)

    return StrategyResult("B — Whale (12×)", net_cost)


# ─────────────────────────────────────────────
# STRATEGY D — The Calculated Buyer (batch of b)
# ─────────────────────────────────────────────
def strategy_d(
    N:          int,
    batch_size: int,
    rng:        np.random.Generator,
) -> StrategyResult:
    """
    Buy `batch_size` boxes per attempt until at least one secret is found.
    Uses inverse-CDF for Geom(p_b) where p_b = 1 - (71/72)^b.

    Cost = batches_needed × batch_size × $15
    """
    p_b      = 1.0 - (71.0 / 72.0) ** batch_size
    batches  = _geom_icdf(p_b, N, rng)
    costs    = batches * batch_size * BOX_PRICE
    return StrategyResult(f"D — Calculated ({batch_size}×)", costs)


# ─────────────────────────────────────────────
# STRATEGY C — The Patient Buyer (GBM price watch)
# ─────────────────────────────────────────────
def strategy_c(
    N:         int,
    rng:       np.random.Generator,
    S0:        float = 80.0,    # initial reseller price of secret figure
    mu:        float = 0.4,     # annual drift
    sigma:     float = 0.6,     # annual volatility
    T:         int   = 30,      # observation window (trading days)
    threshold: float = 60.0,    # buy-trigger price
) -> StrategyResult:
    """
    Watch the GBM reseller price for T days.
    Buy the secret figure the first time price ≤ threshold.
    If price never reaches threshold, buy at the end-of-window price.

    Cost = S_{τ*}  where τ* = inf{t : S_t ≤ threshold}, capped at T.

    Uses full (N × T) matrix simulation — no Python loops.
    """
    paths      = simulate_gbm(S0, mu, sigma, T, N, rng)   # (N, T+1)
    day_idx    = first_passage_time(paths, threshold)       # (N,)

    # Gather the price at the crossing day for each path
    row_idx    = np.arange(N)
    costs      = paths[row_idx, day_idx]

    return StrategyResult("C — Patient Buyer", costs)


# ─────────────────────────────────────────────
# MASTER RUNNER — run all strategies in one call
# ─────────────────────────────────────────────
def run_all_strategies(
    N:              int   = 100_000,
    batch_size:     int   = 8,
    gbm_S0:         float = 80.0,
    gbm_mu:         float = 0.4,
    gbm_sigma:      float = 0.6,
    gbm_T:          int   = 30,
    threshold:      float = 60.0,
    resale_discount:float = 0.4,
    seed:           Optional[int] = None,
) -> dict[str, StrategyResult]:
    """
    Run all four strategies with the same RNG seed for reproducibility.
    Returns a dict keyed by strategy letter.
    """
    rng = np.random.default_rng(seed)

    results = {
        "A": strategy_a(N, rng),
        "B": strategy_b(N, rng,
                        resale_discount=resale_discount,
                        gbm_S0=gbm_S0 * 0.2,   # regular figures ~20% of secret price
                        gbm_mu=gbm_mu,
                        gbm_sigma=gbm_sigma),
        "D": strategy_d(N, batch_size, rng),
        "C": strategy_c(N, rng,
                        S0=gbm_S0,
                        mu=gbm_mu,
                        sigma=gbm_sigma,
                        T=gbm_T,
                        threshold=threshold),
    }
    return results


def run_gbm_display(
    S0:    float = 80.0,
    mu:    float = 0.4,
    sigma: float = 0.6,
    T:     int   = 30,
    n_display: int = 200,
    seed:  Optional[int] = None,
) -> np.ndarray:
    """
    Generate a smaller set of GBM paths purely for visual display.
    Returns shape (n_display, T+1).
    """
    rng = np.random.default_rng(seed)
    return simulate_gbm(S0, mu, sigma, T, n_display, rng)
