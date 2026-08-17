import numpy as np

from qlearn.env import ENV_INFO


class Reporter:
    def __init__(self, reporter_config, env_config):
        self.log_every = reporter_config["log_every"]
        self.window = reporter_config.get("window") or self.log_every
        env_info = ENV_INFO[env_config["name"]]
        self.target_value = env_info["target_value"]
        self.optimal_value = env_info["optimal_value"]

        self.rewards = []
        self.shaped_rewards = []
        self.epsilon_history = []
        self.steps_history = []

    def log_eval(self, reward):
        self.rewards.append(reward)

    def log(self, raw_reward, shaped_reward, epsilon, steps, ep):
        self.rewards.append(raw_reward)
        self.shaped_rewards.append(shaped_reward)
        self.epsilon_history.append(epsilon)
        self.steps_history.append(steps)
        if ep % self.log_every == 0:
            result = self.summary(ep)
            print(result)

    def summary_eval(self, eval_episodes):
        line = f"rewards mean: {np.mean(self.rewards[-eval_episodes:])}"
        line += f" | rewards p10: {np.quantile(self.rewards[-eval_episodes:],0.1)}"
        line += f" | rewards p90: {np.quantile(self.rewards[-eval_episodes:],0.9)}"
        if self.target_value is not None:
            line += f" | target: {self.target_value}"
        if self.optimal_value is not None:
            line += f" | optimal: {self.optimal_value}"
        return line

    def summary(self, ep):
        line = f"Ep {ep:5d}"
        if self.epsilon_history:
            line += f" | ε={self.epsilon_history[-1]:.3f}"
        if self.shaped_rewards:
            line += f" | shaped: {np.mean( self.shaped_rewards[-self.window:]):.2f}"
        line += f" | rewards: {np.mean(
             self.rewards[-self.window:]
            ):.2f}"
        if self.target_value is not None:
            line += f" | target: {self.target_value}"
        if self.optimal_value is not None:
            line += f" | optimal: {self.optimal_value}"
        return line
