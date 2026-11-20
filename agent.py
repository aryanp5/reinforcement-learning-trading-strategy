"""Tabular Q-learning agent."""
from collections import defaultdict
import numpy as np


class QLearningAgent:
    """Q-learning with a lookup table and epsilon-greedy exploration.

    Learning rate: each (state, action) pair uses 1 / n**omega, where n is the number of
    times that pair has been updated. With 0.5 < omega <= 1 this meets the usual
    convergence conditions (Even-Dar & Mansour, 2003), while early updates still move fast.
    Ties between equally good actions are broken at random, using the agent's own seed.
    """

    def __init__(self, n_actions=3, omega=0.6, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.985, epsilon_min=0.05, seed=0):
        self.n_actions = n_actions
        self.omega = omega
        self.gamma = gamma
        self.epsilon = epsilon                 # start fully random, then decay each episode
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.visits = defaultdict(lambda: np.zeros(n_actions, dtype=int))

    def act(self, state, greedy=False):
        """Pick an action. greedy=True switches exploration off (used for evaluation)."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(self.n_actions))
        q = self.q_table[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def learn(self, state, action, reward, next_state, done):
        """One Q-learning update towards reward + gamma * (best value of the next state)."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.visits[state][action] += 1
        lr = 1 / self.visits[state][action] ** self.omega
        self.q_table[state][action] += lr * (target - self.q_table[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def policy(self):
        """The learned policy: best action index for every state the agent has seen."""
        return {state: int(np.argmax(q)) for state, q in self.q_table.items()}


if __name__ == "__main__":
    # Quick demo: train one agent on 2015-2020 and show its Q-table
    from data import add_features
    from env import TradingEnv
    from main import load_prices, train, TRAIN_END

    df = add_features(load_prices())
    agent = train(QLearningAgent(seed=0), TradingEnv(df.loc[:TRAIN_END]))

    labels = ["Short", "Out", "Long"]
    print(f"States seen: {len(agent.q_table)}   final epsilon: {agent.epsilon:.3f}\n")
    print(f"{'state':10s} {'Q(Short)':>9s} {'Q(Out)':>9s} {'Q(Long)':>9s}   best   visits")
    for s in sorted(agent.q_table):
        q = agent.q_table[s]
        print(f"{str(s):10s} {q[0]:9.4f} {q[1]:9.4f} {q[2]:9.4f}   {labels[int(np.argmax(q))]:5s}  "
              f"{agent.visits[s].tolist()}")