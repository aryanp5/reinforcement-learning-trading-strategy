"""Train the Q-learning agents, evaluate every strategy, and save the results.

Run:   python main.py

Data:  AAPL daily, 2015-2024. Train 2015-2020, validate 2021-22, test 2023-24 (used once).
Saves to results/:
    training_results.csv                in-sample performance (2015-2020)
    learned_policies.csv                each agent's action in each state
    validation_results.csv, test_results.csv
                                        every strategy x every risk setting
    equity_validation.csv, equity_test.csv
                                        daily wealth curves (no stops), for the charts
    positions_validation.csv, positions_test.csv
                                        daily price and position (no stops), for the trade charts
"""
import os
import numpy as np
import pandas as pd

from data import download_prices, add_features
from env import TradingEnv
from agent import QLearningAgent
from evaluate import run_policy, metrics
from baselines import buy_and_hold, trend_long_short, trend_long_only, ma_crossover

# ---------------------------------------------------------------- settings
TICKER, START, END = "AAPL", "2015-01-01", "2025-01-01"
PRICES_CSV = f"data/{TICKER}_prices.csv"
TRAIN_END = "2020-12-31"
PERIODS = {
    "validation": ("2021-01-01", "2022-12-31"),
    "test":       ("2023-01-01", "2024-12-31"),    # final check: run once, report as is
}

N_SEEDS = 10
EPISODES = 300

# The agent is always trained WITHOUT stops; the risk layer is only applied when evaluating
RISK_SETTINGS = {
    "no_stops":    {},
    "stop10":      {"stop_loss": 0.10},
    "stop10_tp20": {"stop_loss": 0.10, "take_profit": 0.20},
}
RISK_LABELS = {"no_stops": "No stops", "stop10": "Stop 10%", "stop10_tp20": "Stop 10% + TP 20%"}

# Two versions of the agent's state; both are always reported
AGENT_VARIANTS = {
    "Q-learning": {},
    "Q-learning + regime": {"use_regime": True},
}

BENCHMARKS = [("Buy & Hold", buy_and_hold),
              ("Trend L/S", trend_long_short),
              ("Trend long-only", trend_long_only)]

ORDER = ["Buy & Hold", "Trend L/S", "Trend long-only", "MA 50/200",
         "Q-learning", "Q-learning + regime"]


# ---------------------------------------------------------------- data and training
def load_prices():
    """Download once, then reuse the saved CSV so every run uses identical data."""
    if os.path.exists(PRICES_CSV):
        return pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    prices = download_prices(TICKER, START, END)
    os.makedirs("data", exist_ok=True)
    prices.to_csv(PRICES_CSV)
    return prices


def train(agent, env, episodes=EPISODES):
    for _ in range(episodes):
        state, done = env.reset(), False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()
    return agent


# ---------------------------------------------------------------- evaluation
def evaluate(env, policy):
    """One episode: metrics (plus how often the risk layer fired), daily log rewards,
    and the position held each day."""
    rewards = run_policy(env, policy)
    m = metrics(rewards)
    m["stops"] = env.exits["stop_loss"]
    m["take_profits"] = env.exits["take_profit"]
    positions = [pos for _, pos, _ in env.history]
    return m, rewards, positions


def summarise(name, setting, results):
    """Average per-seed results into one row, keeping the spread (sd) of the key metrics."""
    res = pd.DataFrame(results)
    row = {"strategy": name, "setting": setting, **res.mean().to_dict()}
    for col in ["total_return", "sharpe", "max_drawdown"]:
        row[f"{col}_sd"] = res[col].std()
    return row


def evaluate_period(df, agents):
    """Every strategy under every risk setting on one period.

    Returns (results table, daily wealth curves, daily positions), the last two without stops.
    For the agents the wealth curve is the average across seeds; positions are seed 0's.
    """
    rows, curves = [], {}
    positions = {"Close": df["Close"].to_numpy()}

    def record(name, setting, rewards_list):
        if setting == "no_stops":
            curves[name] = np.mean([np.exp(np.cumsum(r)) for r in rewards_list], axis=0)

    for setting, risk in RISK_SETTINGS.items():
        for name, policy in BENCHMARKS:
            m, rewards, pos = evaluate(TradingEnv(df, **risk), policy)
            rows.append({"strategy": name, "setting": setting, **m})
            record(name, setting, [rewards])
            if setting == "no_stops":
                positions[name] = pos

        env = TradingEnv(df, **risk)     # MA crossover reads prices from its own env
        m, rewards, _ = evaluate(env, ma_crossover(env))
        rows.append({"strategy": "MA 50/200", "setting": setting, **m})
        record("MA 50/200", setting, [rewards])

        # The same trained agents under every setting, acting greedily
        for name, state_kw in AGENT_VARIANTS.items():
            runs = [evaluate(TradingEnv(df, **state_kw, **risk),
                             lambda s: agent.act(s, greedy=True))
                    for agent in agents[name]]
            rows.append(summarise(name, setting, [m for m, _, _ in runs]))
            record(name, setting, [r for _, r, _ in runs])
            if setting == "no_stops":
                positions[name] = runs[0][2]

    equity = pd.DataFrame(curves, index=df.index)[ORDER]
    return pd.DataFrame(rows), equity, pd.DataFrame(positions, index=df.index)


def training_results(train_df, agents):
    """In-sample (2015-2020) performance of the agents and the rule-based benchmarks."""
    rows = []
    for name, state_kw in AGENT_VARIANTS.items():
        results = [metrics(run_policy(TradingEnv(train_df, **state_kw),
                                      lambda s: agent.act(s, greedy=True)))
                   for agent in agents[name]]
        rows.append(summarise(name, "no_stops", results))
    for name, policy in BENCHMARKS:
        m = metrics(run_policy(TradingEnv(train_df), policy))
        rows.append({"strategy": name, "setting": "no_stops", **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- learned policies
TREND = {0: "down", 1: "flat", 2: "up"}
REGIME = {0: "down", 1: "none yet", 2: "up"}
POSITION = {-1: "short", 0: "out", 1: "long"}
ACTION = {0: "Short", 1: "Out", 2: "Long"}


def policy_table(agents):
    """For each agent variant and state: the most common action across seeds, and agreement."""
    rows = []
    for name, state_kw in AGENT_VARIANTS.items():
        policies = [agent.policy() for agent in agents[name]]
        for state in sorted(set().union(*policies)):
            actions = [p[state] for p in policies if state in p]
            best = max(set(actions), key=actions.count)
            rows.append({
                "agent": name,
                "trend": TREND[state[0]],
                "last breakout": REGIME[state[1]] if state_kw.get("use_regime") else "-",
                "position": POSITION[state[-1]],
                "action": ACTION[best],
                "seeds agreeing": f"{actions.count(best)}/{len(policies)}",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- printing
def pct(x):
    return f"{x:+.1%}"


def with_sd(row, col, fmt):
    """'+43.6%' for single runs, '+43.6% ± 9.3%' for strategies averaged over seeds."""
    text = fmt(row[col])
    sd = row.get(f"{col}_sd")
    if sd is not None and not pd.isna(sd):
        text += " ± " + (f"{sd:.1%}" if fmt is pct else f"{sd:.2f}")
    return text


def headline_table(results):
    """One row per strategy: return, Sharpe and max drawdown (no stops)."""
    rows = results[results["setting"] == "no_stops"].set_index("strategy")
    table = pd.DataFrame({
        "Strategy": rows.index,
        "Return": [with_sd(r, "total_return", pct) for _, r in rows.iterrows()],
        "Sharpe": [with_sd(r, "sharpe", lambda x: f"{x:.2f}") for _, r in rows.iterrows()],
        "Max drawdown": [f"{r['max_drawdown']:.1%}" for _, r in rows.iterrows()],
    }).set_index("Strategy")
    return table.loc[[s for s in ORDER if s in table.index]]


def risk_table(results):
    """Return (Sharpe) under each risk setting, plus how often the stop / take-profit fired."""
    table = pd.DataFrame(index=ORDER)
    for setting, label in RISK_LABELS.items():
        rows = results[results["setting"] == setting].set_index("strategy")
        table[label] = [f"{pct(rows.loc[s, 'total_return'])} ({rows.loc[s, 'sharpe']:.2f})"
                        for s in ORDER]
    rows = results[results["setting"] == "stop10_tp20"].set_index("strategy")
    table["Exits (stop / TP)"] = [f"{rows.loc[s, 'stops']:g} / {rows.loc[s, 'take_profits']:g}"
                                  for s in ORDER]
    return table


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    df = add_features(load_prices())
    train_df = df.loc[:TRAIN_END]
    os.makedirs("results", exist_ok=True)

    print(f"Training {N_SEEDS} seeds x {len(AGENT_VARIANTS)} agent versions "
          f"on {train_df.index[0]:%Y-%m-%d} to {train_df.index[-1]:%Y-%m-%d} ...")
    agents = {name: [] for name in AGENT_VARIANTS}
    for name, state_kw in AGENT_VARIANTS.items():
        for seed in range(N_SEEDS):
            agents[name].append(train(QLearningAgent(seed=seed), TradingEnv(train_df, **state_kw)))
        print(f"  done: {name}")

    # In-sample check: can the agents match the simple rules on their own training data?
    train_res = training_results(train_df, agents)
    train_res.to_csv("results/training_results.csv", index=False)
    heading("TRAINING 2015-2020 (in-sample)")
    print(headline_table(train_res).to_string())

    policies = policy_table(agents)
    policies.to_csv("results/learned_policies.csv", index=False)
    for name in AGENT_VARIANTS:
        heading(f"Learned policy: {name}")
        print(policies[policies["agent"] == name].drop(columns="agent").to_string(index=False))

    # Out-of-sample: validation, then the test period
    for period, (start, end) in PERIODS.items():
        results, equity, positions = evaluate_period(df.loc[start:end], agents)
        results.to_csv(f"results/{period}_results.csv", index=False)
        equity.to_csv(f"results/equity_{period}.csv")
        positions.to_csv(f"results/positions_{period}.csv")

        heading(f"{period.upper()} {start[:4]}-{end[2:4]}: no stops "
                f"(± = sd across {N_SEEDS} seeds)")
        print(headline_table(results).to_string())
        heading(f"{period.upper()} {start[:4]}-{end[2:4]}: risk layer, return (Sharpe)")
        print(risk_table(results).to_string())

    print("\nAll results saved in results/")