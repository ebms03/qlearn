import numpy as np

from qlearn.agent.agent import Agent


class LinearAgent(Agent):
    def __init__(
        self,
        n_actions,
        feature_fn,
        n_features,
        alpha,
        gamma,
        epsilon_min,
        epsilon_decay,
        rng,
    ):
        self.n_actions = n_actions
        self.feature_fn = feature_fn
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng

        self.theta = np.zeros((n_actions, n_features))

    def q_values(self, state):
        phi = self.feature_fn(state)
        return self.theta @ phi

    def update(self, state, action, reward, next_state, done):
        phi = self.feature_fn(state)
        q_sa = self.theta[action] @ phi

        if done:
            target = reward
        else:
            phi_next = self.feature_fn(next_state)
            q_next = self.theta @ phi_next
            target = reward + self.gamma * np.max(q_next)

        td_error = target - q_sa

        self.theta[action] += self.alpha * td_error * phi

    def greedy_action(self, state):
        Q_state = self.q_values(state)
        best = np.max(Q_state)
        candidates = np.where(Q_state == best)[0]
        return self.rng.choice(candidates)

