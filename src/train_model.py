"""
train_model.py: Part 1 of the project, demand forecasting.

Predicts daily (scaled) pastry sales per store for a bakery chain using
calendar/holiday/weather features and recent demand, evaluated on a
TIME-BASED holdout (the last 16 weeks) to mirror a real "order tomorrow
from the past" production decision.

Two models are compared:
  * Naive baseline  -> uses sales from the same store 7 days earlier
  * HistGradientBoostingRegressor (sklearn, no external deps)

Writes metrics and plots to ../results/ and validation predictions to
forecasts.csv (consumed by waste_analysis.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import build_demand_dataset, make_model_matrix  # noqa: E402

VAL_WEEKS = 16
RNG = 0


def rmse(a, b): return mean_squared_error(a, b) ** 0.5


def main() -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    df = build_demand_dataset(ROOT / "data" / "train_raw.csv")
    X, y = make_model_matrix(df)

    # ---- time-based split: hold out the last VAL_WEEKS for validation ----
    cutoff = df["date"].max() - pd.Timedelta(weeks=VAL_WEEKS)
    is_train = (df["date"] < cutoff).to_numpy()
    Xtr, Xva = X[is_train], X[~is_train]
    ytr, yva = y[is_train], y[~is_train]

    # ---- naive baseline: sales from the same store, 7 days earlier ----
    naive = df.loc[~is_train, "lag7"].to_numpy()

    # ---- gradient boosting ----
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06,
                                          l2_regularization=1.0, random_state=RNG)
    model.fit(Xtr, ytr)
    pred = model.predict(Xva)

    # ---- metrics ----
    metrics = {
        "validation_start": str(df["date"][~is_train].min().date()),
        "validation_end": str(df["date"].max().date()),
        "n_train": int(is_train.sum()),
        "n_validation": int((~is_train).sum()),
        "naive_rmse": float(rmse(yva, naive)),
        "naive_mae": float(mean_absolute_error(yva, naive)),
        "model_rmse": float(rmse(yva, pred)),
        "model_mae": float(mean_absolute_error(yva, pred)),
        "rmse_improvement_pct": float(100 * (1 - rmse(yva, pred) / rmse(yva, naive))),
    }
    pd.Series(metrics).to_csv(results_dir / "metrics.csv", header=False)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # ---- save validation predictions for the waste module ----
    forecasts = pd.DataFrame({
        "date": df["date"][~is_train].values,
        "store": df["store"][~is_train].values,
        "actual": yva,
        "naive": naive,
        "model": pred,
    })
    forecasts.to_csv(results_dir / "forecasts.csv", index=False)
    print(f"saved results/forecasts.csv ({len(forecasts)} rows)")

    # ---- plots ----
    plot_forecast_example(df, is_train, yva, naive, pred, results_dir)
    plot_store_rmse(df, is_train, yva, naive, pred, results_dir)
    plot_importance(model, Xva, yva, X, results_dir)


def plot_forecast_example(df, is_train, yva, naive, pred, out):
    """Actual vs naive vs model for the busiest store over the validation tail."""
    val = df[~is_train].copy()
    val["actual"] = yva; val["naive"] = naive; val["model"] = pred
    store = val.groupby("store")["actual"].count().idxmax()
    s = val[val["store"] == store].sort_values("date").tail(56)

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=130)
    ax.plot(s["date"], s["actual"], color="#111111", lw=1.6, label="Actual sales")
    ax.plot(s["date"], s["naive"], color="#e07b39", lw=1.2, ls="--",
            label="Naive (last week same day)")
    ax.plot(s["date"], s["model"], color="#2f6fae", lw=1.4, label="Gradient boosting")
    ax.set_title(f"Daily sales forecast vs actual: {store} (validation tail)")
    ax.set_xlabel("Date"); ax.set_ylabel("Scaled units sold")
    ax.grid(alpha=0.25); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "forecast_example.png", bbox_inches="tight")
    plt.close(fig)


def plot_store_rmse(df, is_train, yva, naive, pred, out):
    val = df[~is_train].copy()
    val["actual"] = yva; val["naive"] = naive; val["model"] = pred
    val["se_naive"] = (val["actual"] - val["naive"]) ** 2
    val["se_model"] = (val["actual"] - val["model"]) ** 2
    n = val.groupby("store").size()
    se_n = val.groupby("store")["se_naive"].sum()
    se_m = val.groupby("store")["se_model"].sum()
    naive_rmse = (se_n / n) ** 0.5
    model_rmse = (se_m / n) ** 0.5
    order = model_rmse.sort_values().index
    idx = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=130)
    ax.bar(idx - w / 2, naive_rmse.loc[order], w, color="#e07b39", label="Naive")
    ax.bar(idx + w / 2, model_rmse.loc[order], w, color="#2f6fae", label="Gradient boosting")
    ax.set_xticks(idx); ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("Validation RMSE"); ax.set_title("Per-store forecast error")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "store_rmse.png", bbox_inches="tight")
    plt.close(fig)


def plot_importance(model, Xva, yva, X, out):
    pi = permutation_importance(model, Xva, yva, n_repeats=5, random_state=RNG,
                                scoring="neg_root_mean_squared_error")
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=130)
    ax.barh(np.arange(len(top)), top.values, color="#2f6fae")
    ax.set_yticks(np.arange(len(top))); ax.set_yticklabels(top.index)
    ax.set_xlabel("Increase in RMSE when the feature is shuffled")
    ax.set_title("What drives daily demand? (permutation importance)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
