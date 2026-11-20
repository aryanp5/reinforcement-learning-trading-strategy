"""Benchmark strategies. Each one maps a state to an action index: 0 = short, 1 = out, 2 = long."""


def position_to_action(position):
    """Position (-1, 0, 1) -> action index (0, 1, 2). Used to 'keep doing what I'm doing'."""
    return position + 1


def buy_and_hold(state):
    return 2


def trend_long_short(state):
    """Long above the +/-2% band, short below it, otherwise keep the current position."""
    trend_bin, position = state[0], state[-1]
    if trend_bin == 2:
        return 2
    if trend_bin == 0:
        return 0
    return position_to_action(position)


def trend_long_only(state):
    """As above, but go flat instead of short below the band."""
    trend_bin, position = state[0], state[-1]
    if trend_bin == 2:
        return 2
    if trend_bin == 0:
        return 1
    return position_to_action(position)


def ma_crossover(env, allow_short=True):
    """50/200-day moving-average crossover. Needs the env because it reads prices, not the state."""
    fast = env.df["SMA50"].to_numpy()
    slow = env.df["SMA200"].to_numpy()

    def policy(state):
        if fast[env.t] > slow[env.t]:
            return 2                            # golden cross: long
        return 0 if allow_short else 1          # death cross: short (or out)
    return policy