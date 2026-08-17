import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
import random

from qlearn.agent.agent import Agent


class MLP(nn.Module):
    def __init__(self, n_states, hidden, n_actions):
        super().__init__()
        net = []

        for s_in, s_out, act in zip(
            [n_states, *hidden],
            [*hidden, n_actions],
            [nn.ReLU() for _ in hidden] + [nn.Identity()],
        ):
            net.extend([nn.Linear(s_in, s_out), act])

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class MLPDQNAgent(Agent):

    def __init__(
        self,
        state_shape,
        state_dtype,
        n_actions,
        network_hidden,
        alpha,
        gamma,
        epsilon_min,
        epsilon_decay,
        batch_size,
        buffer_size,
        target_update_tau,
        rng,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = target_update_tau
        self.rng = rng

        self.q_net = MLP(np.prod(state_shape), network_hidden, n_actions)
        self.q_net.compile()

        self.target_net = MLP(np.prod(state_shape), network_hidden, n_actions)
        self.target_net.compile()
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(buffer_size, state_shape, state_dtype)
        self.step_count = 0

    def update(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.rng, self.batch_size
        )

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            max_next_q = (
                self.target_net(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze()
            )
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.step_count += 1
        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def greedy_action(self, state):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return self.q_net(s).argmax(dim=1).item()


class ReplayBuffer:
    def __init__(self, capacity, state_shape, dtype):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.dtype = dtype
        self.states = np.zeros((capacity, *state_shape), dtype=dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=dtype)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        idx = self.pos
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, gen, batch_size, device="cpu"):
        idx = gen.integers(0, self.size, size=batch_size)
        # Convert to tensors on-the-fly
        return (
            torch.as_tensor(self.states[idx], dtype=torch.uint8, device=device),
            torch.as_tensor(self.actions[idx], dtype=torch.long, device=device),
            torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_states[idx], dtype=torch.uint8, device=device),
            torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
        )

    def __len__(self):
        return self.size
