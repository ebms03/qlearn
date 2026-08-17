import numpy as np


import numpy as np
import torch
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class RNGState:
    """Snapshot of all RNG state for save/restore."""

    numpy_global: tuple
    torch_global: torch.Tensor
    python_random: tuple
    agent_rng: Optional[dict] = None
    env_rng: Optional[dict] = None
    episode: int = 0
    epsilon: float = 1.0


class SeedManager:
    """
    Centralized RNG management.

    Creates independent streams for:
    - Training env (per-episode seeds)
    - Evaluation env (fixed, comparable scenarios)
    - Agent (epsilon-greedy, replay sampling)

    Handles save/restore of complete RNG state for checkpointing.
    """

    def __init__(self, master_seed: int):
        self.master_seed = master_seed

        self.train_env_seed = master_seed
        self.eval_env_seed = master_seed + 1_000_000
        self.agent_seed = master_seed + 2_000_000
        self.torch_seed = master_seed + 3_000_000
        self.python_seed = master_seed + 4_000_000

        self.train_episode = 0

    def make_agent_rng(self) -> np.random.Generator:
        """Create an independent RNG for the agent."""
        return np.random.default_rng(self.agent_seed)

    def train_env_reset_seed(self, episode: int) -> int:
        """Deterministic seed for training episode `episode`."""
        self.train_episode = episode
        return hash(self.train_env_seed + episode)

    def eval_env_reset_seed(self, episode: int) -> int:
        """Deterministic seed for eval episode `episode`.

        Uses a separate seed space so eval is identical regardless
        of how many training episodes have run.
        """
        return hash(self.eval_env_seed + episode)

    def capture(self, agent, env, episode: int, epsilon: float) -> RNGState:
        """Snapshot all RNG state for checkpointing."""
        state = RNGState(
            numpy_global=np.random.get_state(),
            torch_global=torch.get_rng_state(),
            python_random=random.getstate(),
            episode=episode,
            epsilon=epsilon,
        )

        # Capture agent RNG if it has one
        if hasattr(agent, "rng") and agent.rng is not None:
            state.agent_rng = agent.rng.bit_generator.state

        # Capture env RNG if accessible
        if hasattr(env, "np_random") and env.np_random is not None:
            state.env_rng = env.np_random.bit_generator.state

        return state

    def restore(self, state: RNGState, agent, env):
        """Restore all RNG state from a checkpoint."""
        np.random.set_state(state.numpy_global)
        torch.set_rng_state(state.torch_global)
        random.setstate(state.python_random)
        self.train_episode = state.episode

        # Restore agent RNG
        if state.agent_rng is not None and hasattr(agent, "rng"):
            agent.rng.bit_generator.state = state.agent_rng

        # Restore env RNG
        if state.env_rng is not None and hasattr(env, "np_random"):
            env.np_random.bit_generator.state = state.env_rng
