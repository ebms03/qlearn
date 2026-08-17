class MountainCarPotentialRewardShaper:
    def __init__(self, config):
        self.gamma = config["agent"]["gamma"]
        self.scale = config["reward_shaper"]["scale"]

    def __call__(self, reward, done, state, next_state):
        return (
            reward + self.gamma * self.potential(*next_state) - self.potential(*state)
        )

    # at least kinda
    def potential(self, x, v):
        # x=-0.5 is bottom of sinusoid
        return (abs(x - (-0.5)) + 0.5 * v**2) * self.scale


REWARDS_SHAPERS = {"mountaincar_potential": MountainCarPotentialRewardShaper}
