import numpy as np


class FourierFeatures:
    def __init__(self, state_dim, order, max_freq=None):
        self.order = order

        all_freqs = np.indices((order + 1,) * state_dim).reshape(state_dim, -1).T

        if max_freq is not None:
            total_freq = np.sum(all_freqs, axis=1)
            mask = total_freq <= max_freq
            self.freqs = all_freqs[mask]
        else:
            self.freqs = all_freqs

        self.n_features = len(self.freqs)

    def __call__(self, state):
        fourier_normed = np.pi * (state + 1) / 2
        return np.cos(self.freqs @ fourier_normed)


class PolyFeatures:
    def __init__(self, state_dim, order, max_power=None):
        self.order = order

        powers = np.indices((order + 1,) * state_dim).reshape(state_dim, -1).T

        if max_power is not None:
            total_freq = np.sum(powers, axis=1)
            mask = total_freq <= max_power
            self.powers = powers[mask]
        else:
            self.powers = powers

        self.n_features = len(self.powers)

    def __call__(self, state):
        return self.powers @ state


FEATURES = {
    "fourier": FourierFeatures,
    "poly": PolyFeatures,
}
