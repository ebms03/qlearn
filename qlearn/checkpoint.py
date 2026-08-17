from dataclasses import dataclass
from pathlib import Path
import pickle

from qlearn.agent.agent import Agent


@dataclass
class Checkpoint:
    agent: Agent
    config: dict
    ep: int


class CheckpointManager:
    def __init__(self, config):
        self.config = config
        checkpoint_config = config.get("checkpoint")
        if not checkpoint_config:
            print("Checkpoint config not provided. Skipping checkpointing.")
            self.inactive = True
            return
        self.inactive = False
        self.save_dir = Path(checkpoint_config["save_dir"])
        self.save_every = checkpoint_config["save_every"]

    def save_sparse(self, agent, ep):
        if self.inactive:
            return
        if ep % self.save_every == 0:
            self._save(agent, ep)

    def _save(self, agent, ep):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir.joinpath(f"checkpoint_ep{ep:10d}")
        ckpt = Checkpoint(agent, self.config, ep)
        with open(path, "wb+") as f:
            pickle.dump(ckpt, f)


def load(resume_dir):
    files = sorted(Path(resume_dir).iterdir())
    latest = files[-1]
    with open(latest, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt
