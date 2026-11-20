"""Trading environment: one stock, one decision per day (short, out or long)."""
import numpy as np


class TradingEnv:
    """Each day the strategy picks its position for the next day: short (-1), out (0) or long (+1).

    State:  (trend_bin, position), or (trend_bin, regime, position) if use_regime=True
        trend_bin  0 = price more than 2% below its 20-day average
                   1 = within +/-2%
                   2 = more than 2% above
        regime     the last trend_bin that was outside the band (0 = down, 2 = up,
                   1 = no breakout yet). It gives the agent a memory of the last breakout.

    Reward: log(1 + position * next_return) - cost * |change in position|
        Log returns add up over time, so an episode's total reward is its log final wealth.

    Optional risk layer (off by default), checked at each day's close:
        stop_loss    exit if the trade falls this far below its best level (trailing stop)
        take_profit  exit if the trade is up this much since entry
        After an exit the position is held flat for `cooldown` days, whatever the
        strategy asks for, so it can't sell and immediately buy back in.
    """

    ACTIONS = (-1, 0, 1)    # action index 0 = short, 1 = out, 2 = long

    def __init__(self, df, cost=0.001, use_regime=False, trend_edges=(-0.02, 0.02),
                 stop_loss=None, take_profit=None, cooldown=10):
        self.df = df.reset_index(drop=True)
        self.n = len(self.df)
        self.cost = cost
        self.use_regime = use_regime

        # Trade P&L is tracked as a log return, so convert the thresholds to match
        self.stop_level = np.log1p(-stop_loss) if stop_loss else None    # 10% -> -0.105
        self.tp_level = np.log1p(take_profit) if take_profit else None   # 20% -> +0.182
        self.cooldown = cooldown

        # Nothing here depends on the strategy's actions, so compute it once up front
        self.next_ret = self.df["Next_Return"].to_numpy()
        self.trend_bins = np.digitize(self.df["Trend"].to_numpy(), trend_edges)
        self.regime = self._regime(self.trend_bins)

        self.reset()

    @staticmethod
    def _regime(trend_bins):
        """Carry the last out-of-band trend bin forward. Only looks backwards in time."""
        regime = np.empty(len(trend_bins), dtype=int)
        last = 1
        for i, b in enumerate(trend_bins):
            if b != 1:
                last = b
            regime[i] = last
        return regime

    def reset(self):
        self.t = 0
        self.position = 0
        self.trade_pnl = 0.0       # log return of the current trade so far
        self.trade_peak = 0.0      # best trade_pnl reached, for the trailing stop
        self.cooldown_left = 0
        self.exits = {"stop_loss": 0, "take_profit": 0}
        self.history = []          # (day, position, reward)
        return self._state()

    def _state(self):
        state = [int(self.trend_bins[self.t])]
        if self.use_regime:
            state.append(int(self.regime[self.t]))
        state.append(self.position)
        return tuple(state)

    def step(self, action):
        new_position = self.ACTIONS[action]
        if self.cooldown_left > 0:             # risk layer overrides the strategy
            new_position = 0
            self.cooldown_left -= 1

        r = self.next_ret[self.t]
        reward = np.log1p(new_position * r) - self.cost * abs(new_position - self.position)

        if new_position != self.position:      # a new trade starts (or we go flat)
            self.trade_pnl = 0.0
            self.trade_peak = 0.0
        self.trade_pnl += np.log1p(new_position * r)
        self.trade_peak = max(self.trade_peak, self.trade_pnl)
        self.position = new_position
        self.history.append((self.t, new_position, reward))

        if self.position != 0:
            if self.stop_level is not None and self.trade_pnl - self.trade_peak <= self.stop_level:
                self.exits["stop_loss"] += 1
                self.cooldown_left = self.cooldown
            elif self.tp_level is not None and self.trade_pnl >= self.tp_level:
                self.exits["take_profit"] += 1
                self.cooldown_left = self.cooldown

        self.t += 1
        done = self.t >= self.n
        next_state = None if done else self._state()
        return next_state, reward, done


if __name__ == "__main__":
    # Sanity check: always-long with no stops must equal buy-and-hold minus one entry cost
    from data import add_features
    from main import load_prices

    df = add_features(load_prices())

    def run(action, **risk):
        env = TradingEnv(df, **risk)
        total, done = 0.0, False
        while not done:
            _, reward, done = env.step(action)
            total += reward
        return np.exp(total), env.exits

    growth = df["Close"].iloc[-1] * (1 + df["Next_Return"].iloc[-1]) / df["Close"].iloc[0]
    print(f"Buy-and-hold from prices (after entry cost): {growth * np.exp(-0.001):.4f}x")
    for label, action, risk in [("Always long, no stops", 2, {}),
                                ("Always short, no stops", 0, {}),
                                ("Always long, stop 10%", 2, {"stop_loss": 0.10}),
                                ("Always long, stop 10% + TP 20%", 2,
                                 {"stop_loss": 0.10, "take_profit": 0.20})]:
        wealth, exits = run(action, **risk)
        print(f"{label:32s} {wealth:8.4f}x   stops {exits['stop_loss']:3d}   "
              f"take-profits {exits['take_profit']:3d}")
    print("Example states:", TradingEnv(df).reset(), TradingEnv(df, use_regime=True).reset())