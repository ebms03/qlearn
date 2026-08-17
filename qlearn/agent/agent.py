import numpy as np


class Agent:
    def __init__(self, agent_config, env, rng):
        raise NotImplementedError

    def q_values(self, state):
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def act(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.n_actions)
        else:
            return self.greedy_action(state)

    def greedy_action(self, state):
        raise NotImplementedError

    def decay_epsilon(self, progress):
        self.epsilon = epsilon_schedule(progress, self.epsilon_min, self.epsilon_decay)


def epsilon_schedule(progress, eps_min, lam):
    if abs(lam) < 1e-8:
        return eps_min + (1 - eps_min) * (1 - progress)
    z = np.exp(-lam * progress) - np.exp(-lam)
    return eps_min + (1 - eps_min) * z / (1 - np.exp(-lam))
