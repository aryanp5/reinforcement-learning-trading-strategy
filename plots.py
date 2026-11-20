"""Make the README figures from the CSVs that main.py saves in results/.

Run:   python main.py      (once, to create results/)
       python plots.py
Saves to figures/:
    equity_curves.png     growth of £1, validation and test, no stops
    sharpe.png            Sharpe ratio by strategy, validation vs test
    learned_policy.png    what the regime agent does in each state
    trades.png            AAPL price with each agent's buys, sells and time in the market
"""
import os
import matplotlib
matplotlib.use("Agg")      # just write PNG files; no on-screen window needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS, FIGURES = "results", "figures"

# Colours and styling, chosen to stay readable for colour-blind readers
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
TEXT, TEXT_2 = "#0b0b0b", "#52514e"
COLOURS = {
    "Buy & Hold": "#2a78d6",
    "Trend long-only": "#eb6834",
    "Q-learning": "#1baf7a",
    "Q-learning + regime": "#eda100",
}
PERIOD_COLOURS = {"validation": "#2a78d6", "test": "#eb6834"}
ACTION_COLOURS = {"Long": "#2a78d6", "Out": "#e1e0d9", "Short": "#eb6834"}
BUY, SELL = "#2a78d6", "#eb6834"
LONG_SHADE, SHORT_SHADE = "#cde2fb", "#fbd9c9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "legend.frameon": False,
})


def finish(fig, name):
    """Save the figure to figures/ and free its memory."""
    fig.savefig(f"{FIGURES}/{name}.png", dpi=150)
    plt.close(fig)


def label_line_ends(ax, equity):
    """Write each line's final return at its right end, nudged apart so labels don't overlap."""
    low, high = ax.get_ylim()
    gap = 0.05 * (high - low)
    last_y = None
    for value, name in sorted((equity[name].iloc[-1], name) for name in COLOURS):
        y = value if last_y is None else max(value, last_y + gap)
        last_y = y
        ax.annotate(f"{value - 1:+.0%}", xy=(equity.index[-1], y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT_2,
                    annotation_clip=False)


def equity_curves():
    """Growth of £1 for the main strategies, one panel per period."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, period, title in [(axes[0], "validation", "Validation 2021–22 (up, then down)"),
                              (axes[1], "test", "Test 2023–24 (rising market)")]:
        equity = pd.read_csv(f"{RESULTS}/equity_{period}.csv", index_col=0, parse_dates=True)
        for name, colour in COLOURS.items():
            # The regime agent ends up identical to the trend rule in places, so draw it
            # dashed on top; otherwise one line would hide the other
            style = "--" if name == "Q-learning + regime" else "-"
            ax.plot(equity.index, equity[name], color=colour, linewidth=2, linestyle=style,
                    label=name)
        label_line_ends(ax, equity)
        ax.axhline(1, color=AXIS, linewidth=1)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title(title)
        ax.set_ylabel("Value of £1 (after costs)")
        ax.margins(x=0.08)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Growth of £1 out of sample (no stops; agents averaged over 10 seeds)",
                 x=0.01, y=0.985, ha="left", fontsize=12, fontweight="bold")
    fig.legend(handles, labels, loc="upper left", ncol=4, bbox_to_anchor=(0.005, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    finish(fig, "equity_curves")


def sharpe_bars():
    """Sharpe by strategy, validation vs test, with ±1 sd across seeds for the agents."""
    order = ["Buy & Hold", "Trend L/S", "Trend long-only", "MA 50/200",
             "Q-learning", "Q-learning + regime"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.38
    for i, period in enumerate(["validation", "test"]):
        res = pd.read_csv(f"{RESULTS}/{period}_results.csv")
        res = res[res["setting"] == "no_stops"].set_index("strategy").loc[order]
        x = [j + (i - 0.5) * width for j in range(len(order))]
        sd = res["sharpe_sd"].fillna(0) if "sharpe_sd" in res else 0
        ax.bar(x, res["sharpe"], width=width - 0.04, color=PERIOD_COLOURS[period],
               label=period.capitalize(), yerr=sd, capsize=3,
               error_kw={"ecolor": TEXT_2, "linewidth": 1})
        for xi, v in zip(x, res["sharpe"]):
            ax.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 4 if v >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=8, color=TEXT_2,
                        bbox={"boxstyle": "square,pad=0.15", "fc": SURFACE, "ec": "none"})
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.margins(y=0.12)                    # room for the value labels above and below
    ax.set_xticks(range(len(order)), order)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("Sharpe ratio by strategy (no stops)", pad=24)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    ax.text(0, -0.16, "Error bars: ±1 sd across 10 seeds. With two years of data a Sharpe estimate "
            "has a standard error of roughly ±0.7, so small gaps are not meaningful.",
            transform=ax.transAxes, fontsize=8, color=TEXT_2)
    fig.tight_layout()
    finish(fig, "sharpe")


def learned_policy():
    """Grid of the regime agent's actions: rows = current position, columns = market state."""
    pol = pd.read_csv(f"{RESULTS}/learned_policies.csv")
    pol = pol[pol["agent"] == "Q-learning + regime"]

    columns = [("down", "down"), ("flat", "down"), ("flat", "none yet"), ("flat", "up"), ("up", "up")]
    col_labels = ["Trend down", "Trend flat\nlast breakout down", "Trend flat\nno breakout yet",
                  "Trend flat\nlast breakout up", "Trend up"]
    rows = ["short", "out", "long"]

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.grid(False)
    for c, (trend, breakout) in enumerate(columns):
        for r, position in enumerate(rows):
            match = pol[(pol["trend"] == trend) & (pol["last breakout"] == breakout)
                        & (pol["position"] == position)]
            if match.empty:
                action, agree = "unseen", ""
            else:
                action, agree = match["action"].iloc[0], match["seeds agreeing"].iloc[0]
            ax.add_patch(plt.Rectangle((c + 0.03, r + 0.05), 0.94, 0.9,
                                       color=ACTION_COLOURS.get(action, SURFACE)))
            ink = SURFACE if action in ("Long", "Short") else TEXT
            ax.text(c + 0.5, r + 0.42, action, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=ink)
            ax.text(c + 0.5, r + 0.72, f"{agree} seeds" if agree else "", ha="center",
                    va="center", fontsize=8, color=ink)
    ax.set_xlim(0, len(columns))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([c + 0.5 for c in range(len(columns))], col_labels)
    ax.set_yticks([r + 0.5 for r in range(len(rows))], [f"Currently {p}" for p in rows])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("What the Q-learning + regime agent learned (2015–2020)")
    fig.tight_layout()
    finish(fig, "learned_policy")


def trades():
    """AAPL price with each agent's buys and sells (seed 0, no stops). Shading = in the market."""
    agents = ["Q-learning", "Q-learning + regime"]
    periods = [("validation", "Validation 2021–22"), ("test", "Test 2023–24")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")

    for c, (period, period_title) in enumerate(periods):
        data = pd.read_csv(f"{RESULTS}/positions_{period}.csv", index_col=0, parse_dates=True)
        price = data["Close"]
        for r, agent in enumerate(agents):
            ax = axes[r, c]
            pos = data[agent].to_numpy()
            # Position on day t is held from that close to the next, so shade with step="post"
            for value, shade, label in [(1, LONG_SHADE, "Long"), (-1, SHORT_SHADE, "Short")]:
                if (pos == value).any():
                    ax.fill_between(data.index, 0, 1, where=pos == value, step="post",
                                    color=shade, transform=ax.get_xaxis_transform(),
                                    linewidth=0, label=f"Holding {label.lower()}")
            ax.plot(data.index, price, color=TEXT_2, linewidth=1.2, label="AAPL close")

            change = np.diff(np.concatenate([[0], pos]))      # +ve = buying, -ve = selling
            buys, sells = change > 0, change < 0
            ax.scatter(data.index[buys], price[buys], marker="^", s=55, color=BUY,
                       edgecolor=SURFACE, linewidth=1, zorder=3, label="Buy")
            ax.scatter(data.index[sells], price[sells], marker="v", s=55, color=SELL,
                       edgecolor=SURFACE, linewidth=1, zorder=3, label="Sell")

            days_in = (pos != 0).mean()
            ax.set_title(f"{agent}: {period_title}", fontsize=11)
            ax.text(0.01, 0.97, f"{buys.sum()} buys · {sells.sum()} sells · in the market "
                    f"{days_in:.0%} of days", transform=ax.transAxes, va="top",
                    fontsize=9, color=TEXT_2,
                    bbox={"boxstyle": "square,pad=0.2", "fc": SURFACE, "ec": "none", "alpha": 0.85})
            ax.grid(axis="x", visible=False)
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            if c == 0:
                ax.set_ylabel("AAPL price ($)")

    handles = {}                       # one legend entry per label across all four panels
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="upper right", ncol=len(handles),
               bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("When each agent was in the market (seed 0, no stops)", x=0.01, y=0.985,
                 ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    finish(fig, "trades")


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    equity_curves()
    sharpe_bars()
    learned_policy()
    trades()
    print(f"Saved 4 figures to {FIGURES}/")