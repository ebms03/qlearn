from qlearn.agent.linear_agent import LinearAgent
from qlearn.agent.mlpdqn_agent import MLPDQNAgent
from qlearn.agent.tabular_agent import TabularAgent
from qlearn.config import kwargs
from qlearn.features import FEATURES


def build_agent(agent_config, env, rng):
    match agent_config["type"]:
        case "tabular":
            agent = TabularAgent(
                n_states=env.observation_space.n,
                n_actions=env.action_space.n,
                rng=rng,
                **kwargs(agent_config),
            )
        case "linear":
            feature_fn_config = agent_config["feature_fn"]
            feature_fn = FEATURES[feature_fn_config["type"]](
                state_dim=env.observation_space.shape[0],
                **kwargs(feature_fn_config),
            )

            agent = LinearAgent(
                n_actions=env.action_space.n,
                feature_fn=feature_fn,
                n_features=feature_fn.n_features,
                rng=rng,
                **kwargs(agent_config, additional_removes=("feature_fn")),
            )
        case "dqn":
            agent = MLPDQNAgent(
                state_shape=env.observation_space.shape,
                state_dtype=env.observation_space.dtype,
                n_actions=env.action_space.n,
                rng=rng,
                **kwargs(agent_config),
            )
        case _:
            raise ValueError(f"Unknown agent type {agent_config["type"]}")
    return agent
