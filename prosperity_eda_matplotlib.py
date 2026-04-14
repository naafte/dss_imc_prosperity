"""
Static exploratory plots for IMC Prosperity-style simulator exports (order books + trades).

Expects CSVs under ./data with semicolon separators, e.g. prices_round_0_day_-1.csv
and trades_round_0_day_-1.csv (same layout as interactivegraph.py).

Run:
  pip install pandas matplotlib
  python prosperity_eda_matplotlib.py

Outputs PNGs under ./figures/ by default.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

# Writable cache for CI / sandboxed environments (must be before pyplot import).
_MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
_MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DATA = Path(__file__).resolve().parent / "data"


def _day_from_filename(path: Path) -> int:
    m = re.search(r"day_(-?\d+)", path.name)
    if not m:
        raise ValueError(f"Could not parse simulator day from filename: {path.name}")
    return int(m.group(1))


def load_all_prices(data_dir: Path) -> pd.DataFrame:
    paths = sorted(data_dir.glob("prices_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No prices_*.csv files in {data_dir}")
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p, sep=";")
        df["source_file"] = p.name
        df["sim_day"] = _day_from_filename(p)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["spread"] = out["ask_price_1"] - out["bid_price_1"]
    touch_vol = out["bid_volume_1"] + out["ask_volume_1"]
    out["microprice"] = np.where(
        touch_vol > 0,
        (out["bid_price_1"] * out["ask_volume_1"] + out["ask_price_1"] * out["bid_volume_1"])
        / touch_vol,
        out["mid_price"],
    )
    out["imbalance_touch"] = (out["bid_volume_1"] - out["ask_volume_1"]) / touch_vol.replace(0, np.nan)
    return out


def load_all_trades(data_dir: Path) -> pd.DataFrame:
    paths = sorted(data_dir.glob("trades_*.csv"))
    if not paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p, sep=";")
        df["source_file"] = p.name
        df["sim_day"] = _day_from_filename(p)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "symbol" in out.columns and "product" not in out.columns:
        out = out.rename(columns={"symbol": "product"})
    return out


def _decimate(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def plot_mid_and_spread(prices: pd.DataFrame, out_dir: Path, max_points: int) -> None:
    products = sorted(prices["product"].unique())
    fig, axes = plt.subplots(len(products), 2, figsize=(14, 3.8 * len(products)), sharex="col")
    if len(products) == 1:
        axes = np.array([axes])
    day_palette = plt.cm.tab10.colors
    for i, prod in enumerate(products):
        sub = prices[prices["product"] == prod].sort_values(["sim_day", "timestamp"])
        ax_mid, ax_spread = axes[i, 0], axes[i, 1]
        for j, day in enumerate(sorted(sub["sim_day"].unique())):
            d = sub[sub["sim_day"] == day]
            d_plot = _decimate(d, max_points)
            color = day_palette[j % len(day_palette)]
            ax_mid.plot(
                d_plot["timestamp"],
                d_plot["mid_price"],
                lw=0.8,
                alpha=0.85,
                color=color,
                label=f"day {day}",
            )
            ax_spread.plot(
                d_plot["timestamp"],
                d_plot["spread"],
                lw=0.8,
                alpha=0.85,
                color=color,
                label=f"day {day}",
            )
        ax_mid.set_ylabel("mid")
        ax_mid.set_title(f"{prod}: mid price (top-of-book implied)")
        ax_mid.legend(loc="upper right", fontsize=8)
        ax_spread.set_ylabel("spread")
        ax_spread.set_title(f"{prod}: ask₁ − bid₁")
        ax_spread.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("timestamp")
    axes[-1, 1].set_xlabel("timestamp")
    fig.suptitle("Fair value pathing and quoted edge (IMC Prosperity-style books)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "01_mid_and_spread.png", dpi=160)
    plt.close(fig)


def plot_microprice_vs_mid(prices: pd.DataFrame, out_dir: Path, max_points: int) -> None:
    products = sorted(prices["product"].unique())
    fig, axes = plt.subplots(len(products), 1, figsize=(14, 3.5 * len(products)), sharex=True)
    if len(products) == 1:
        axes = [axes]
    for ax, prod in zip(axes, products):
        sub = prices[prices["product"] == prod].sort_values(["sim_day", "timestamp"])
        sub = _decimate(sub, max_points)
        ax.plot(sub["timestamp"], sub["mid_price"], label="mid", lw=0.9, alpha=0.9)
        ax.plot(sub["timestamp"], sub["microprice"], label="microprice @ touch", lw=0.9, alpha=0.75)
        ax.set_ylabel("price")
        ax.set_title(f"{prod}: microprice vs mid (queue imbalance signal)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timestamp")
    fig.tight_layout()
    fig.savefig(out_dir / "02_microprice_vs_mid.png", dpi=160)
    plt.close(fig)


def plot_touch_imbalance(prices: pd.DataFrame, out_dir: Path, max_points: int) -> None:
    products = sorted(prices["product"].unique())
    fig, axes = plt.subplots(len(products), 1, figsize=(14, 3.2 * len(products)), sharex=True)
    if len(products) == 1:
        axes = [axes]
    for ax, prod in zip(axes, products):
        sub = prices[prices["product"] == prod].sort_values(["sim_day", "timestamp"])
        sub = _decimate(sub, max_points)
        ax.plot(sub["timestamp"], sub["imbalance_touch"], lw=0.7)
        ax.axhline(0, color="k", lw=0.4, alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("imbalance")
        ax.set_title(f"{prod}: (bid_vol₁ − ask_vol₁) / (bid_vol₁ + ask_vol₁)")
    axes[-1].set_xlabel("timestamp")
    fig.suptitle("Order-book pressure at the inside quote", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "03_touch_imbalance.png", dpi=160)
    plt.close(fig)


def plot_trade_tape(trades: pd.DataFrame, out_dir: Path) -> None:
    if trades.empty:
        return
    products = sorted(trades["product"].unique())
    fig, axes = plt.subplots(len(products), 1, figsize=(14, 3.4 * len(products)), sharex=True)
    if len(products) == 1:
        axes = [axes]
    day_palette = plt.cm.Set2.colors
    for ax, prod in zip(axes, products):
        sub = trades[trades["product"] == prod]
        for j, day in enumerate(sorted(sub["sim_day"].unique())):
            d = sub[sub["sim_day"] == day]
            ax.scatter(
                d["timestamp"],
                d["price"],
                s=np.clip(d["quantity"] * 8, 12, 120),
                alpha=0.35,
                color=day_palette[j % len(day_palette)],
                label=f"day {day}",
            )
        ax.set_ylabel("trade px")
        ax.set_title(f"{prod}: tape (marker size ∝ quantity)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timestamp")
    fig.suptitle("Executed trades vs time", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "04_trade_scatter.png", dpi=160)
    plt.close(fig)


def plot_volume_by_day(trades: pd.DataFrame, out_dir: Path) -> None:
    if trades.empty or "quantity" not in trades.columns:
        return
    g = trades.groupby(["sim_day", "product"], as_index=False)["quantity"].sum()
    pivot = g.pivot(index="product", columns="sim_day", values="quantity")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("total traded qty")
    ax.set_title("Liquidity / activity by simulator day")
    ax.legend(title="sim_day", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "05_volume_by_day.png", dpi=160)
    plt.close(fig)


def plot_mid_volatility(prices: pd.DataFrame, out_dir: Path, window: int) -> None:
    products = sorted(prices["product"].unique())
    fig, axes = plt.subplots(len(products), 1, figsize=(14, 3.2 * len(products)), sharex=True)
    if len(products) == 1:
        axes = [axes]
    for ax, prod in zip(axes, products):
        for day in sorted(prices["sim_day"].unique()):
            sub = prices[(prices["product"] == prod) & (prices["sim_day"] == day)].sort_values("timestamp")
            vol = sub["mid_price"].diff().rolling(window, min_periods=max(2, window // 4)).std()
            ax.plot(sub["timestamp"], vol, lw=0.9, label=f"day {day}")
        ax.set_ylabel("rolling σ(Δmid)")
        ax.set_title(f"{prod}: rolling volatility of mid changes (window={window})")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timestamp")
    fig.tight_layout()
    fig.savefig(out_dir / "06_mid_change_volatility.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_DATA)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--max-points", type=int, default=8000, help="Max points per (product, day) line.")
    parser.add_argument("--vol-window", type=int, default=120, help="Rolling window for mid-change std.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prices = load_all_prices(args.data_dir)
    trades = load_all_trades(args.data_dir)

    plot_mid_and_spread(prices, args.out_dir, args.max_points)
    plot_microprice_vs_mid(prices, args.out_dir, args.max_points)
    plot_touch_imbalance(prices, args.out_dir, args.max_points)
    plot_trade_tape(trades, args.out_dir)
    plot_volume_by_day(trades, args.out_dir)
    plot_mid_volatility(prices, args.out_dir, args.vol_window)

    print(f"Wrote figures to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
