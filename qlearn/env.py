import numpy as np
import gymnasium as gym

from qlearn.seeding import SeedManager

ENV_INFO = {
    "Acrobot-v1": {
        "optimal_value": -70,
        "target_value": -100,
        "log_every": 50,
        "type": "classic",
        "bounds": (
            np.array([-1, -1, -1, -1, -4 * np.pi, -9 * np.pi]),
            np.array([1, 1, 1, 1, 4 * np.pi, 9 * np.pi]),
        ),
    },
    "MountainCar-v0": {
        "optimal_value": -85,
        "target_value": -110,
        "type": "classic",
        "bounds": (
            np.array([-1.2, -0.7]),
            np.array([0.6, 0.07]),
        ),
    },
    "Taxi-v4": {
        "optimal_value": 8.5,
        "target_value": 8.0,
        "type": "gridworld",
    },
}


def build_env(env_config, render_mode=None):
    env_name = env_config["name"]
    info = ENV_INFO[env_name]
    env_type = info["type"]
    match env_type:
        case "gridworld" | "classic":
            env = gym.make(env_name, render_mode=render_mode)
    bounds = info.get("bounds")
    if bounds is not None:
        env = Bounds(env, *bounds)
    seed_manager = SeedManager(env_config["seed"])
    return env, seed_manager


class Bounds(gym.ObservationWrapper):
    def __init__(self, env, low, high):
        super().__init__(env)
        self.low = low
        self.high = high

    def observation(self, obs):
        scaled = 2.0 * (obs - self.low) / (self.high - self.low) - 1.0
        return scaled
