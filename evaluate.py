import sys

from qlearn.env import build_env
from qlearn import checkpoint
from qlearn.reporter import Reporter


def evaluate(env, agent, reporter, seed_manager, evaluator_config):
    episodes = evaluator_config["episodes"]
    for ep in range(episodes):
        state, _ = env.reset(seed=seed_manager.eval_env_reset_seed(ep))
        done = truncated = False
        steps = 0
        total_reward = 0
        while not done and not truncated:
            action = agent.greedy_action(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
        reporter.log_eval(total_reward)
    print("==== Evaluation ====")
    result = reporter.summary_eval(episodes)
    print(result)


def main():
    ckpt = checkpoint.load(sys.argv[-1])
    config = ckpt.config
    agent = ckpt.agent
    env, seed_manager = build_env(config["env"])
    reporter = Reporter(config["reporting"], config["env"])

    evaluate(env, agent, reporter, seed_manager, config["evaluator"])


if __name__ == "__main__":
    main()
