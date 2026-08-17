# scripts/play.py
"""
Watch a trained agent play.

Usage:
    python -m scripts.play --config configs/cartpole_dqn.yaml
    python -m scripts.play --config configs/atari_pong.yaml --checkpoint results/checkpoints/pong/best.pt
    python -m scripts.play --config configs/frozenlake_qtable.yaml --episodes 5
"""

import argparse
import sys
import numpy as np
import gymnasium as gym

from qlearn import checkpoint
from qlearn.env import build_env


def main():
    ckpt = checkpoint.load(sys.argv[-1])
    config = ckpt.config
    agent = ckpt.agent
    env, seed_manager = build_env(config["env"], render_mode="human")
    state, _ = env.reset(seed=seed_manager.eval_env_reset_seed(3))
    done = truncated = False

    while not done and not truncated:
        action = agent.greedy_action(state)
        state, *_ = env.step(action)


if __name__ == "__main__":
    main()
