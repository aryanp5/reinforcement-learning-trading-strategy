"""Run a strategy through the environment and score it."""
import numpy as np


def run_policy(env, choose_action):
    """Play one full episode. Returns the daily log rewards (net of costs)."""
    state = env.reset()
    done = False
    rewards = []
    while not done:
        state, reward, done = env.step(choose_action(state))
        rewards.append(reward)
    return np.array(rewards)


def metrics(log_rewards):
    """Standard performance measures from daily log returns.

    Sharpe uses simple daily returns, a zero risk-free rate and sqrt(252) annualisation.
    Max drawdown is the largest fall from a previous peak in the wealth curve.
    """
    n = len(log_rewards)
    wealth = np.exp(np.cumsum(log_rewards))
    simple = np.exp(log_rewards) - 1
    sd = simple.std(ddof=1)
    return {
        "total_return": wealth[-1] - 1,
        "annual_return": np.exp(log_rewards.sum() / n * 252) - 1,
        # Undefined if the strategy never holds a position (zero volatility)
        "sharpe": simple.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
        "max_drawdown": np.max(1 - wealth / np.maximum.accumulate(wealth)),
    }


if __name__ == "__main__":
    # Check: buy-and-hold over the full sample
    from data import add_features
    from env import TradingEnv
    from main import load_prices

    df = add_features(load_prices())
    for name, value in metrics(run_policy(TradingEnv(df), lambda s: 2)).items():
        print(f"{name:14s} {value:8.4f}")