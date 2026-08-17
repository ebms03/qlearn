import argparse

import numpy as np

from evaluate import evaluate
from qlearn.agent.build_agent import build_agent
from qlearn.config import load_config
from qlearn.env import build_env
from qlearn.reporter import Reporter
from qlearn.reward_shapers import REWARDS_SHAPERS
from qlearn.checkpoint import CheckpointManager
from qlearn import checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="", type=str)
    parser.add_argument("--resume", "-r", default="", type=str)
    args = parser.parse_args()

    if args.resume:
        assert not args.config
        ckpt = checkpoint.load(args.resume)
        start_ep = ckpt.ep
        config = ckpt.config
        agent = ckpt.agent
        env, seed_manager = build_env(config["env"])

    if args.config:
        assert not args.resume
        start_ep = 0
        config = load_config(args.config)
        env, seed_manager = build_env(config["env"])
        agent = build_agent(config["agent"], env, seed_manager.make_agent_rng())

    reporter = Reporter(config["reporting"], config["env"])
    checkpoint_manager = CheckpointManager(config)

    trainer_config = config["trainer"]
    reward_shaper = config.get("reward_shaper")
    if reward_shaper is not None:
        reward_shaper = REWARDS_SHAPERS[reward_shaper["type"]](config)

    train(
        env,
        agent,
        reporter,
        seed_manager,
        reward_shaper,
        start_ep,
        checkpoint_manager,
        trainer_config,
    )

    evaluate(env, agent, reporter, seed_manager, config["evaluator"])


def train(
    env,
    agent,
    reporter,
    seed_manager,
    reward_shaper,
    start_ep,
    checkpoint_manager,
    trainer_config,
):
    to_ep = trainer_config["episodes"]
    max_steps = trainer_config.get("max_steps", np.inf)
    reward_shaper = reward_shaper or (lambda r, d, n, ns: r)
    print("==== Training ====")
    for ep in range(start_ep, to_ep):
        state, _ = env.reset(seed=seed_manager.train_env_reset_seed(ep))
        done = False
        truncated = False
        total_reward = 0
        total_shaped = 0
        steps = 0

        while not done and not truncated and steps < max_steps:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            shaped = reward_shaper(reward, done, state, next_state)
            agent.update(state, action, shaped, next_state, done)
            state = next_state
            total_reward += reward
            total_shaped += shaped
            steps += 1

        reporter.log(total_reward, total_shaped, agent.epsilon, steps, ep + 1)
        agent.decay_epsilon((ep + 1) / to_ep)

        checkpoint_manager.save_sparse(agent, ep + 1)


if __name__ == "__main__":
    main()
