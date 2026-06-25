"""All figures for the project. Each function saves one PNG into results/figures
and returns its path. Kept deliberately plain (matplotlib only)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE, RED, GREEN, GREY = "#1f5fa8", "#c0392b", "#27867a", "#888888"


def _save(fig, name):
    path = os.path.join(C.FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_orderbook(df, idx=None):
    """Snapshot of the book depth at one instant."""
    if idx is None:
        idx = len(df) // 2
    row = df.iloc[idx]
    nl = C.N_LEVELS
    bid_px = [row["bid_px"] - i * C.TICK for i in range(nl)]
    ask_px = [row["ask_px"] + i * C.TICK for i in range(nl)]
    bid_sz = [row[f"bid_sz_{i+1}"] for i in range(nl)]
    ask_sz = [row[f"ask_sz_{i+1}"] for i in range(nl)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(bid_px, bid_sz, height=C.TICK*0.8, color=BLUE, label="bids")
    ax.barh(ask_px, ask_sz, height=C.TICK*0.8, color=RED, label="asks")
    ax.axhline(row["mid"], color=GREY, ls="--", lw=1)
    ax.set_xlabel("resting size"); ax.set_ylabel("price ($)")
    ax.set_title("Limit order book snapshot")
    ax.legend()
    return _save(fig, "01_orderbook_snapshot.png")


def fig_price_and_ofi(b, start=0, n=900):
    """Mid price vs cumulative OFI over a short window."""
    start = max(0, min(start, max(0, len(b) - 1)))   # keep the slice non-empty
    s = b.iloc[start:start+n]
    cum_ofi = s["ofi"].cumsum().values
    fig, ax1 = plt.subplots(figsize=(9, 4))
    t = (s["t_start"].values - s["t_start"].values[0]) / 60.0
    ax1.plot(t, s["mid_last"].values, color=BLUE, lw=1.3, label="mid price")
    ax1.set_xlabel("minutes"); ax1.set_ylabel("mid price ($)", color=BLUE)
    ax2 = ax1.twinx()
    ax2.plot(t, cum_ofi, color=RED, lw=1.1, alpha=0.8, label="cumulative OFI")
    ax2.set_ylabel("cumulative OFI", color=RED)
    ax2.grid(False)
    ax1.set_title("Price moves track accumulated order flow imbalance")
    return _save(fig, "02_price_vs_ofi.png")


def fig_impact_scatter(b, impact, max_pts=6000):
    d = b.dropna(subset=["ret", "ofi"])
    if len(d) > max_pts:
        d = d.sample(max_pts, random_state=C.SEED)
    x = d["ofi"].values; y = d["ret"].values / C.TICK
    beta = impact["beta"] / C.TICK
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(x, y, s=5, alpha=0.15, color=BLUE)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, beta * xs, color=RED, lw=2,
            label=f"fit: R²={impact['r2']:.2f}, t={impact['tstat']:.0f}")
    ax.set_xlabel("OFI over bucket"); ax.set_ylabel("price change (ticks)")
    ax.set_title("Contemporaneous price impact of OFI")
    ax.legend()
    return _save(fig, "03_impact_scatter.png")


def fig_decay(decay):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(decay)), decay["ic"].values, color=BLUE)
    ax.set_xticks(range(len(decay)))
    ax.set_xticklabels([f"{h}s" for h in decay["horizon"]])
    ax.set_xlabel("forecast horizon"); ax.set_ylabel("information coefficient (corr)")
    ax.set_title("Forecasting power of OFI by horizon")
    for i, v in enumerate(decay["ic"].values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    return _save(fig, "04_forecast_decay.png")


def fig_edge_vs_cost(costs):
    labels = ["predicted\nedge", "round-trip\ntaker cost", "maker\ncapture"]
    vals = [costs["mean_abs_edge_ticks"], costs["round_trip_cost_ticks"],
            costs["maker_capture_per_rt_ticks"]]
    colors = [GREEN, RED, BLUE]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("ticks")
    ax.set_title("The edge is real, but smaller than the cost of crossing")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=10)
    return _save(fig, "05_edge_vs_cost.png")


def fig_backtest(naive, selective):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(naive["t_hours"], naive["equity_gross"], color=GREEN, lw=1.5,
            label=f"gross  (+{naive['gross_per_trade_ticks']:.3f} ticks/trade)")
    ax.plot(naive["t_hours"], naive["equity_net"], color=RED, lw=1.5,
            label=f"net of costs  ({naive['net_per_trade_ticks']:.3f} ticks/trade)")
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlabel("hours (out-of-sample)"); ax.set_ylabel("cumulative PnL ($/unit)")
    ax.set_title("Taker backtest: a real signal that costs erase")
    note = (f"Selective policy (trade only when edge > cost):\n"
            f"{selective['n_trades']} trades — the edge never clears the spread")
    ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec=GREY))
    ax.legend(loc="upper right")
    return _save(fig, "06_taker_backtest.png")


def fig_market_maker(mm):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"naive": GREY, "as": BLUE, "as_ofi": GREEN}
    names = {"naive": "naive", "as": "Avellaneda-Stoikov", "as_ofi": "A-S + OFI skew"}
    for pol, r in mm.items():
        ax1.plot(r["t_hours"], r["pnl"], color=colors[pol], lw=1.4,
                 label=f"{names[pol]} (Sharpe {r['pnl_sharpe']:.0f})")
        ax2.plot(r["t_hours"], r["inventory"], color=colors[pol], lw=1.0, alpha=0.85,
                 label=f"{names[pol]} (|inv|max {r['inv_abs_max']:.0f})")
    ax1.set_title("Market-maker PnL"); ax1.set_xlabel("hours"); ax1.set_ylabel("PnL ($)")
    ax1.legend(fontsize=9)
    ax2.set_title("Inventory"); ax2.set_xlabel("hours"); ax2.set_ylabel("inventory (units)")
    ax2.axhline(0, color="k", lw=0.8); ax2.legend(fontsize=9)
    return _save(fig, "07_market_maker.png")
