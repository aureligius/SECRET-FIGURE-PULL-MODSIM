"""
stock_manager.py
================
Shared Finite Pool Manager — Labubu Blind Box Simulator
Handles cross-user persistent stock using a JSON file with file locking.

Design:
  - stock.json stores the current global pool state
  - Every box opened by any user writes back to stock.json
  - File locking prevents race conditions when two users open simultaneously
  - Admin restock resets to a fresh factory batch

Factory Batch Rules (mirroring Pop Mart production):
  - 1 batch = 6 cases × 12 boxes = 72 boxes total
  - Each case contains 12 unique figures (one of each regular × 2, shuffled)
  - 1 out of every 6 cases is a "secret case": 11 regulars + 1 secret
  - Therefore: 1 secret per batch → P(secret | any box) = 1/72
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
STOCK_FILE   = "stock.json"
LOCK_FILE    = "stock.lock"
BOX_PRICE    = 15.0
CASE_PRICE   = 180.0  #  FIX: 12 boxes × $15.00 = $180.00 per case
# 1 secret per factory batch of 72 boxes (6 cases × 12 boxes)
P_SECRET     = 1 / 72  #  FIX: Adds the baseline theoretical probability
P_REGULAR_EACH = 71 / 432

FIGURE_NAMES = [
    "Cherry Blossom",
    "Mint Dream",
    "Lavender Haze",
    "Peach Glow",
    "Sky Pop",
    "Coral Bloom",
    "Golden Labubu ✦",
]

FIGURE_EMOJIS = {
    "Cherry Blossom":  "🌸",
    "Mint Dream":      "🍃",
    "Lavender Haze":   "💜",
    "Peach Glow":      "🍑",
    "Sky Pop":         "🩵",
    "Coral Bloom":     "🌺",
    "Golden Labubu ✦": "⭐",
}

FULL_BATCH_REGULAR = 11
FULL_BATCH_SECRET  = 1


# 2. CLASSES AND UTILITIES SECOND
class FileLock:
    """Simple file-based mutex for preventing simultaneous writes."""
    def __init__(self, lock_path: str, timeout: float = 5.0):
        self.lock_path = lock_path
        self.timeout   = timeout

    def __enter__(self):
        start = time.time()
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return self
            except FileExistsError:
                if time.time() - start > self.timeout:
                    try:
                        os.remove(self.lock_path)
                    except FileNotFoundError:
                        pass
                time.sleep(0.05)

    def __exit__(self, *args):
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass


# 3. INTERNALS AND SIMULATIONS THIRD
def _generate_fresh_batch(seed: Optional[int] = None) -> dict:
    # This will now safely find FIGURE_NAMES defined above!
    rng = np.random.default_rng(seed)
    stock = {name: 0 for name in FIGURE_NAMES}
    # ... rest of your code

# ─────────────────────────────────────────────
# BATCH GENERATOR
# ─────────────────────────────────────────────
def _generate_fresh_batch(seed: Optional[int] = None) -> dict:
    """
    Generate one full factory batch of 72 boxes (6 cases).

    Returns a dict with figure counts and full metadata.
    The secret is injected into exactly 1 of the 6 cases.
    """
    rng = np.random.default_rng(seed)

    stock = {name: 0 for name in FIGURE_NAMES}
    case_log = []  # track which case had the secret

    secret_case_idx = int(rng.integers(0, 6))  # which of the 6 cases gets the secret

    for case_idx in range(6):
        if case_idx == secret_case_idx:
            # Secret case: 11 random regulars + 1 secret
            regulars = list(rng.choice(FIGURE_NAMES[:6], size=11, replace=True))
            boxes = regulars + ["Golden Labubu ✦"]
            rng.shuffle(boxes)
            case_log.append({"case": case_idx + 1, "has_secret": True})
        else:
            # Normal case: 12 regulars (at least one of each)
            base  = FIGURE_NAMES[:6].copy()
            extra = list(rng.choice(FIGURE_NAMES[:6], size=6, replace=True))
            boxes = base + extra
            rng.shuffle(boxes)
            case_log.append({"case": case_idx + 1, "has_secret": False})

        for box in boxes:
            stock[box] += 1

    return {
        "stock":          stock,
        "total_original": sum(stock.values()),
        "total_remaining":sum(stock.values()),
        "boxes_opened":   0,
        "secret_case_idx":secret_case_idx,   # hidden — never exposed to UI
        "case_log":       case_log,           # hidden — for admin only
        "history":        [],                 # list of {figure, timestamp}
        "batch_id":       int(time.time()),
    }


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def load_stock() -> dict:
    """
    Read the current stock from disk.
    If the file doesn't exist, generate and save a fresh batch.
    """
    if not Path(STOCK_FILE).exists():
        return restock()

    with FileLock(LOCK_FILE):
        with open(STOCK_FILE, "r") as f:
            return json.load(f)


def _save_stock(data: dict) -> None:
    """Write stock dict to disk. Must be called inside a FileLock."""
    with open(STOCK_FILE, "w") as f:
        json.dump(data, f, indent=2)


def restock(seed: Optional[int] = None) -> dict:
    """
    Reset the global pool to a fresh factory batch.
    Called by admin restock button.
    Returns the new stock dict.
    """
    data = _generate_fresh_batch(seed)
    with FileLock(LOCK_FILE):
        _save_stock(data)
    return data


def open_one_box() -> dict:
    """
    Draw one box from the shared finite pool using the
    Discrete Inverse Transform Method on the current stock CDF.

    Returns:
      {
        "figure":    str,    — name of figure pulled
        "is_secret": bool,
        "emoji":     str,
        "stock":     dict,   — updated stock (counts hidden from UI)
        "remaining": int,    — total boxes still in pool
        "opened":    int,    — total boxes opened this batch
        "cdf":       list,   — updated CDF weights for p5.js animation
        "empty":     bool,   — True if pool was already empty
      }
    """
    with FileLock(LOCK_FILE):
        with open(STOCK_FILE, "r") as f:
            data = json.load(f)

        stock     = data["stock"]
        remaining = data["total_remaining"]

        # Guard: pool exhausted
        if remaining <= 0:
            return {
                "figure":    None,
                "is_secret": False,
                "emoji":     "❌",
                "remaining": 0,
                "opened":    data["boxes_opened"],
                "cdf":       [],
                "empty":     True,
            }

        # ── Discrete Inverse Transform ──────────────────
        # Build probability weights from current stock
        figures   = [f for f in FIGURE_NAMES if stock[f] > 0]
        counts    = np.array([stock[f] for f in figures], dtype=float)
        probs     = counts / counts.sum()

        # CDF boundaries
        cdf       = np.cumsum(probs)

        # Draw U ~ Uniform(0,1)
        U         = np.random.default_rng().uniform(0.0, 1.0)

        # Find first i such that U <= CDF[i]
        drawn_idx = int(np.searchsorted(cdf, U, side="left"))
        drawn_idx = min(drawn_idx, len(figures) - 1)  # safety clamp
        figure    = figures[drawn_idx]

        # ── Update stock ─────────────────────────────────
        stock[figure]          -= 1
        data["stock"]           = stock
        data["total_remaining"] = remaining - 1
        data["boxes_opened"]   += 1
        data["history"].append({
            "figure":    figure,
            "timestamp": time.time(),
        })

        _save_stock(data)

    # Build CDF list for p5.js (probabilities after draw)
    remaining_after = data["total_remaining"]
    if remaining_after > 0:
        new_counts = np.array([stock[f] for f in FIGURE_NAMES], dtype=float)
        new_probs  = new_counts / new_counts.sum()
        cdf_export = [
            {"figure": f, "prob": float(p)}
            for f, p in zip(FIGURE_NAMES, new_probs)
        ]
    else:
        cdf_export = []

    return {
        "figure":    figure,
        "is_secret": figure == "Golden Labubu ✦",
        "emoji":     FIGURE_EMOJIS[figure],
        "remaining": remaining_after,
        "opened":    data["boxes_opened"],
        "cdf":       cdf_export,
        "empty":     False,
    }


def get_session_history(session_pulls: list) -> dict:
    """
    Given a list of figure names pulled this session,
    return summary stats for the analytics page.
    Only uses what the user personally pulled — no stock peeking.
    """
    if not session_pulls:
        return {
            "total_pulled":      0,
            "money_spent":       0.0,
            "secret_found":      False,
            "secret_count":      0,
            "figure_counts":     {f: 0 for f in FIGURE_NAMES},
            "empirical_rates":   {f: 0.0 for f in FIGURE_NAMES},
            "theoretical_rates": _theoretical_rates(),
        }

    counts = {f: 0 for f in FIGURE_NAMES}
    for fig in session_pulls:
        counts[fig] += 1

    n = len(session_pulls)
    empirical = {f: counts[f] / n for f in FIGURE_NAMES}

    return {
        "total_pulled":      n,
        "money_spent":       n * BOX_PRICE,
        "secret_found":      counts["Golden Labubu ✦"] > 0,
        "secret_count":      counts["Golden Labubu ✦"],
        "figure_counts":     counts,
        "empirical_rates":   empirical,
        "theoretical_rates": _theoretical_rates(),
    }


def get_pool_metadata() -> dict:
    """
    Return non-sensitive pool metadata for display.
    Never reveals which figures are remaining or their counts.
    """
    data = load_stock()
    return {
        "total_original":  data["total_original"],
        "total_remaining": data["total_remaining"],
        "boxes_opened":    data["boxes_opened"],
        "batch_id":        data["batch_id"],
        "pct_remaining":   data["total_remaining"] / data["total_original"] * 100,
    }


def _theoretical_rates() -> dict:
    """Theoretical pull probabilities from a fresh full pool."""
    rates = {f: (71 / 432) for f in FIGURE_NAMES[:6]}  # 71/432 each regular
    rates["Golden Labubu ✦"] = 1 / 72
    return rates
