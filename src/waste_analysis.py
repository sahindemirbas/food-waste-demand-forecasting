"""
waste_analysis.py: Part 2 of the project, turning forecast accuracy into
less food waste.

Motivation (mirrors the bakery-chain reality): pastry freshness is critical,
so unsold items are written off at closing time. A bakery therefore balances
two costly errors:
    * over-production  -> unsold stock that becomes food waste (cost + CO2)
    * under-production -> empty shelves, lost revenue, unhappy customers

This module:
  1. Verifies a structural fact about the data: `unsold` tracks the gap
     `ordered - sales` almost exactly, so `unsold` really is over-production
     and is the lever food-waste reduction must pull.
  2. Quantifies how often the chain over-produces (ordered > sales).
  3. Runs an ordering-policy simulation on the time-based validation set:
     "order = predicted demand + safety buffer". It compares the naive and
     the ML forecast at every safety-buffer level and plots the trade-off
     between mean waste (over-production) and mean shortage (under-supply),
     showing that a better forecast lets the bakery order closer to real
     demand -- i.e. waste less at the same service level.

Values in this anonymized dataset are scaled per store, so results are
reported in RELATIVE terms (scaled units / % change), not absolute item counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def waste_columns_summary() -> dict:
    """Structural findings on the ordered / unsold columns of the raw data."""
    df = pd.read_csv(ROOT / "data" / "train_raw.csv")
    sub = df.dropna(subset=["ordered", "unsold"]).copy()
    n_sub = len(sub)
    corr = sub.eval("ordered - sales").corr(sub.unsold)
    # per-store correlation of the same identity
    per_store = sub.groupby("store").apply(
        lambda g: g.eval("ordered - sales").corr(g.unsold), include_groups=False
    ).mean()
    overprod_rate = float((sub["ordered"] > sub["sales"]).mean())
    underprod_rate = float((sub["ordered"] < sub["sales"]).mean())
    return {
        "rows_with_ordered_unsold": int(n_sub),
        "share_of_rows": round(n_sub / len(df), 3),
        "corr_unsold_vs_ordered_minus_sales": round(float(corr), 4),
        "per_store_mean_corr": round(float(per_store), 4),
        "overproduction_rate": round(overprod_rate, 3),
        "underproduction_rate": round(underprod_rate, 3),
    }


def simulate_policies(forecasts_csv: Path, buffer_grid, seed: int = 0) -> pd.DataFrame:
    """Ordering-policy simulation on validation days.

    policy order = forecast + safety_buffer. For each buffer level we compute
    mean waste (overshoot beyond actual demand) and mean shortage (demand that
    went unserved) for the naive forecast and the ML forecast.
    """
    f = pd.read_csv(forecasts_csv, parse_dates=["date"])
    rng = np.random.default_rng(seed)
    rows = []
    for b in buffer_grid:
        for name, pred in [("Naive", f["naive"].to_numpy()),
                           ("GradientBoosting", f["model"].to_numpy())]:
            order = pred + b
            demand = f["actual"].to_numpy()
            waste = np.clip(order - demand, 0, None)   # produced but unsold
            shortage = np.clip(demand - order, 0, None)  # demand unserved
            rows.append({"buffer": b, "policy": name,
                         "mean_waste": float(waste.mean()),
                         "mean_shortage": float(shortage.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    # ---- 1) structural waste findings ----
    s = waste_columns_summary()
    print("=== Over-production (food-waste lever) in the raw data ===")
    for k, v in s.items():
        print(f"  {k}: {v}")

    # ---- 2) ordering-policy simulation ----
    # buffers in scaled units around the naive policy's residual std
    buffer_grid = np.round(np.linspace(0.0, 1.1, 23), 3)
    sim = simulate_policies(results / "forecasts.csv", buffer_grid)

    # service level (share of demand fully served, shortage == 0) per policy/buffer
    f = pd.read_csv(results / "forecasts.csv")
    serv = []
    for b in buffer_grid:
        for name, pred in [("Naive", f["naive"]), ("GradientBoosting", f["model"])]:
            served = (pred + b >= f["actual"]).mean()
            serv.append({"buffer": b, "policy": name, "service_level": float(served)})
    serv = pd.DataFrame(serv)

    sim = sim.merge(serv, on=["buffer", "policy"])
    sim.to_csv(results / "waste_policy_simulation.csv", index=False)

    # ---- headline comparison at a matched ~90% service level ----
    def waste_at_service(policy: str, target: float = 0.90):
        p = sim[sim.policy == policy].copy()
        p["dist"] = (p.service_level - target).abs()
        row = p.loc[p.dist.idxmin()]
        return row

    for tgt in (0.85, 0.90, 0.95):
        n_row = waste_at_service("Naive", tgt)
        m_row = waste_at_service("GradientBoosting", tgt)
        cut = 100 * (1 - m_row["mean_waste"] / n_row["mean_waste"]) if n_row["mean_waste"] else np.nan
        print(f"\nAt ~{tgt:.0%} service level: naive waste {n_row['mean_waste']:.3f} vs "
              f"ML waste {m_row['mean_waste']:.3f}  -> {cut:.0f}% less over-production")

    # ---- plot: waste vs shortage efficiency frontier ----
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=130)
    for policy, color in [("Naive", "#e07b39"), ("GradientBoosting", "#2f6fae")]:
        p = sim[sim.policy == policy].sort_values("buffer")
        ax.plot(p["mean_shortage"], p["mean_waste"], "-o", color=color,
                ms=4, lw=1.6, label=policy)
        # annotate buffer direction
    ax.set_xlabel("Mean shortage (unserved demand), lower is better")
    ax.set_ylabel("Mean waste / over-production (scaled units), lower is better")
    ax.set_title("Ordering-policy frontier: waste vs shortage by forecast")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    # arrow: larger safety buffer moves up-left on the frontier (more waste, less shortage)
    ax.annotate("", xy=(0.10, 0.98), xytext=(0.10, 0.30),
                xycoords="data", arrowprops=dict(arrowstyle="->", color="#888888", lw=1.4))
    ax.text(0.115, 0.62, "larger safety buffer", fontsize=9, color="#555555", rotation=90,
            va="center")
    fig.tight_layout(); fig.savefig(results / "waste_frontier.png", bbox_inches="tight")
    plt.close(fig)
    print("\nsaved results/waste_frontier.png and results/waste_policy_simulation.csv")


if __name__ == "__main__":
    main()
