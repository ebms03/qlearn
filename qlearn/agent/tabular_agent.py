import numpy as np

from qlearn.agent.agent import Agent


class TabularAgent(Agent):
    def __init__(
        self, n_states, n_actions, alpha, gamma, epsilon_min, epsilon_decay, rng
    ):
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def act(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.n_actions)
        else:
            return self.greedy_action(state)

    def greedy_action(self, state):
        best = np.max(self.Q[state])
        candidates = np.where(self.Q[state] == best)[0]
        return self.rng.choice(candidates)
