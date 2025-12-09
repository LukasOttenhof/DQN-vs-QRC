import numpy as np
import os
import random, numpy as np, torch
from tbu_discrete import TruckBackerEnv_D
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


def set_global_seed(seed):
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch CPU RNG
    torch.manual_seed(seed)

    # PyTorch GPU RNG
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensures deterministic algorithms wherever possible
    torch.use_deterministic_algorithms(False)

class QRCAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=0.5,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=50000,
        batch_size=64,
        beta=1,     # weight decay for h-net (small default)
        device=None,
        h_lr=0.01
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.beta = beta
        self.h_lr = h_lr

        # Replay memory
        self.memory = deque(maxlen=buffer_size)

        # Device
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Networks
        self.q_net = self.build_nn(output_dim=self.action_dim).to(self.device)
        self.target_net = self.build_nn(output_dim=self.action_dim).to(self.device)
        self.h_net = self.build_nn(output_dim=self.action_dim).to(self.device)

        # init target
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.h_optimizer = optim.Adam(self.h_net.parameters(), lr=self.h_lr, weight_decay=self.beta)

        self.steps = 0

    def build_nn(self, output_dim):
        """Simple 2-layer MLP"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def agent_policy(self, state):
        """Epsilon-greedy"""
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)
        with torch.no_grad():
            q_values = self.q_net(state_t)  # (1, action_dim)
        return int(q_values.argmax(dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train_with_mem(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample mini-batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Q(s,a)
        q_values = self.q_net(states)
        q_sa = q_values.gather(1, actions)

        # ---- target net forward pass (with grad for correction) ----
        next_q_all = self.target_net(next_states)               # (B,A)
        next_q = next_q_all.max(dim=1, keepdim=True)[0]         # (B,1)

        # TD target (no grad)
        target = rewards + self.gamma * next_q.detach() * (1.0 - dones)
        delta = target - q_sa
        delta_detached_for_h = delta.detach()

        # h prediction
        h_values = self.h_net(states)
        h_sa = h_values.gather(1, actions)

        # Losses
        td_loss = 0.5 * (delta.pow(2)).mean()
        correction_term = torch.mean(self.gamma * h_sa.detach() * next_q)
        h_loss = 0.5 * ((delta_detached_for_h - h_sa).pow(2)).mean()

        # ---- Update q_net ----
        self.q_optimizer.zero_grad()
        self.target_net.zero_grad()

        td_loss.backward()
        correction_term.backward()

        # Transfer target_net gradients to q_net
        for p_q, p_t in zip(self.q_net.parameters(), self.target_net.parameters()):
            if p_t.grad is not None:
                if p_q.grad is None:
                    p_q.grad = p_t.grad.clone()
                else:
                    p_q.grad.add_(p_t.grad)

        self.q_optimizer.step()

        # Clean target_net grads after transferring
        self.target_net.zero_grad(set_to_none=True)

        # ---- Update h_net ----
        self.h_optimizer.zero_grad()
        h_loss.backward()
        self.h_optimizer.step()

        # ---- Epsilon decay ----
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 

    # Target network update (hard update)
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

class QRCAgent_NoTNU:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 50000,
        batch_size: int = 64,
        beta: float = 1.0,  # h-net weight decay
        device = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.beta = beta

        # Replay memory
        self.memory = deque(maxlen=buffer_size)

        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_net = self.build_nn(self.action_dim).to(self.device)
        self.h_net = self.build_nn(self.action_dim).to(self.device)

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.h_optimizer = optim.Adam(self.h_net.parameters(), lr=self.lr, weight_decay=self.beta)

        self.steps = 0

    def build_nn(self, output_dim: int) -> nn.Module:
        """Simple 2-layer MLP"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def policy(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def remember(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool):
        self.memory.append((s, a, r, s2, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # --- Compute Q(s,a) ---
        q_values = self.q_net(states)
        q_sa = q_values.gather(1, actions)

        # --- TD target ---
        with torch.no_grad():
            next_q = self.q_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        delta = target - q_sa
        delta_detached = delta.detach()

        # --- H-network ---
        h_values = self.h_net(states)
        h_sa = h_values.gather(1, actions)

        # --- Losses ---
        td_loss = 0.5 * (delta ** 2).mean()
        correction_term = torch.mean(self.gamma * h_sa.detach() * next_q)
        h_loss = 0.5 * ((delta_detached - h_sa) ** 2).mean()

        # --- Update Q-network ---
        self.q_optimizer.zero_grad()
        q_loss = td_loss + correction_term
        q_loss.backward()
        self.q_optimizer.step()

        # --- Update H-network ---
        self.h_optimizer.zero_grad()
        h_loss.backward()
        self.h_optimizer.step()

        # --- Epsilon decay ---
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Convenience function for interacting with environment
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.remember(state, action, reward, next_state, done)
        self.train()


def run_no_target_net(seed):
    set_global_seed(seed)

    env = TruckBackerEnv_D()
    env.seed(seed)
    env.action_space.seed(seed)
    state = env.reset()

    agent = QRCAgent_NoTNU(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=0.002261,
        epsilon=1.0,
        epsilon_decay=0.99997,
        epsilon_min=0.01,
        batch_size=256,
        buffer_size=50000,
        gamma=0.95,
        beta=0.951
    )

    episode_rewards = []
    episodes = 1000
    max_steps_per_episode = 500

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for t in range(max_steps_per_episode):
            action = agent.policy(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        episode_rewards.append(total_reward)
        print(f"Seed {seed} | Episode {episode} | Reward {total_reward:.2f} | Epsilon {agent.epsilon:.4f}")

    # save individual seed result
    os.makedirs("result_no_tnu", exist_ok=True)
    np.savetxt(f"result_no_tnu/qrc_seed_{seed}.txt", np.array(episode_rewards))
    print(f"[Seed = {seed}] saved to result_no_tnu/qrc_seed_{seed}.txt")

def run(seed):
    set_global_seed(seed)

    env = TruckBackerEnv_D()
    env.seed(seed)
    env.action_space.seed(seed)
    state = env.reset()

    agent = QRCAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=0.002261,
        epsilon=1.0,
        epsilon_decay=0.99997,
        epsilon_min=0.01,
        batch_size=256,
        buffer_size=50000,
        gamma=0.95,
        beta=0.951,
        h_lr=0.1
    )

    episode_rewards = []
    episodes = 1000
    max_steps_per_episode = 500
    target_update_freq = 5

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        for t in range(max_steps_per_episode):
                action = agent.agent_policy(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.train_with_mem()
                state = next_state
                total_reward += reward
                if done:
                    break

        if episode % target_update_freq == 0:
            agent.update_target()

        episode_rewards.append(total_reward)
        print(f"Seed {seed} | Episode {episode} | Reward {total_reward:.2f} | Epsilon {agent.epsilon:.4f}")

    # save individual seed result
    os.makedirs("result_h_lr_01", exist_ok=True)
    np.savetxt(f"result_h_lr_01/qrc_seed_{seed}.txt", np.array(episode_rewards))
    # print(f"[Seed = {seed}] saved to results/qrc_seed_{seed}.txt")


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1])
    # run_no_target_net(seed)
    run(seed)