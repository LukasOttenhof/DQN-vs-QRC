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
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,            # main optimizer stepsize (used like JAX stepsize)
        gamma: float = 0.99,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 50000,
        batch_size: int = 64,
        beta: float = 1.0,
        device=None,
        h_lr: float = 1e-3,          # h network optimizer lr (kept separate but decay uses `lr`)
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr                    # stepsize used for h-decay to match JAX `stepsize`
        self.h_lr = h_lr
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.memory = deque(maxlen=buffer_size)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # networks
        self.q_net = self.build_nn(self.action_dim).to(self.device)
        self.target_net = self.build_nn(self.action_dim).to(self.device)
        self.h_net = self.build_nn(self.action_dim).to(self.device)

        # initial target sync
        self.target_net.load_state_dict(self.q_net.state_dict())

        # optimizers
        self.q_optimizer = optim.AdamW(self.q_net.parameters(), lr=self.lr)
        # Important: do NOT pass beta into optimizer as weight_decay
        self.h_optimizer = optim.AdamW(self.h_net.parameters(), lr=self.h_lr)

        self.steps = 0

    def build_nn(self, output_dim: int):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def agent_policy(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qv = self.q_net(state_t)
        return int(qv.argmax(dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train_with_mem(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Tensors
        states = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)       # (B, S)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)     # (B,1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)      # (B,1)

        # Q(s, a)
        q_vals = self.q_net(states)                          # (B, A)
        q_sa = q_vals.gather(1, actions)                     # (B, 1)

        # target v_tp1 from target network
        with torch.no_grad():
            q_next_target = self.target_net(next_states)     # (B, A)
            v_tp1, _ = q_next_target.max(dim=1, keepdim=True)   # (B,1)
            v_tp1 = v_tp1 * (1.0 - dones)

            # target = r + gamma * v_tp1
            target = rewards + self.gamma * v_tp1
            target = target.detach()

        # TD error
        delta = target - q_sa                                  # (B,1)

        # h(s,a)
        h_vals = self.h_net(states)                           # (B, A)
        h_sa = h_vals.gather(1, actions)                      # (B,1)

        # v_loss per sample: 0.5 * delta^2 + gamma * stopgrad(h_sa) * v_tp1
        v_loss_per = 0.5 * delta.pow(2) + self.gamma * h_sa.detach() * v_tp1
        v_loss = v_loss_per.mean()

        # h_loss per sample: 0.5 * (stopgrad(delta) - h_sa)^2
        h_loss_per = 0.5 * (delta.detach() - h_sa).pow(2)
        h_loss = h_loss_per.mean()

        # Backprop Q network (v_loss)
        self.q_optimizer.zero_grad()
        v_loss.backward(retain_graph=True)   # retain_graph in case of shared parts; harmless otherwise
        self.q_optimizer.step()

        # Backprop h network (h_loss)
        self.h_optimizer.zero_grad()
        h_loss.backward()
        self.h_optimizer.step()

        if self.beta != 0.0 and self.lr != 0.0:
            decay_factor = 1.0 - (self.lr * self.beta)
            if decay_factor < 0.0:
                decay_factor = 0.0
            with torch.no_grad():
                for p in self.h_net.parameters():
                    p.mul_(decay_factor)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
        beta=1,
        device=None
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

        self.memory = deque(maxlen=buffer_size)
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.q_net = self.build_nn(action_dim).to(self.device)
        self.target_net = self.build_nn(action_dim).to(self.device)
        self.h_net = self.build_nn(action_dim).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())

        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.h_optimizer = optim.Adam(self.h_net.parameters(), lr=self.lr, weight_decay=self.beta)

    def build_nn(self, output_dim):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def agent_policy(self, state):
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train_with_mem(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.q_net(states)
        q_sa = q_values.gather(1, actions)

        next_q_all = self.target_net(next_states)
        next_q = next_q_all.max(dim=1, keepdim=True)[0]

        target = rewards + self.gamma * next_q.detach() * (1.0 - dones)
        delta = target - q_sa
        delta_detached = delta.detach()

        h_values = self.h_net(states)
        h_sa = h_values.gather(1, actions)

        td_loss = 0.5 * (delta.pow(2)).mean()
        correction_term = torch.mean(self.gamma * h_sa.detach() * next_q)
        h_loss = 0.5 * ((delta_detached - h_sa).pow(2)).mean()

        # Update q-net
        self.q_optimizer.zero_grad()
        self.target_net.zero_grad()

        td_loss.backward()
        correction_term.backward()

        for p_q, p_t in zip(self.q_net.parameters(), self.target_net.parameters()):
            if p_t.grad is not None:
                if p_q.grad is None:
                    p_q.grad = p_t.grad.clone()
                else:
                    p_q.grad.add_(p_t.grad)

        self.q_optimizer.step()
        self.target_net.zero_grad(set_to_none=True)

        # Update h-net
        self.h_optimizer.zero_grad()
        h_loss.backward()
        self.h_optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

class QRCAgent_NT:
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

# --------------------------
# Shared backbone + 2 heads
# --------------------------
class QRCNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Q-value head
        self.q_head = nn.Linear(128, action_dim)

        # h-value head
        self.h_head = nn.Linear(128, action_dim)

    def forward(self, x):
        z = self.backbone(x)
        q = self.q_head(z)
        h = self.h_head(z)
        return q, h

class QRCAgent_ON:
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
        beta=1e-4,
        h_lr=1e-3,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.beta = beta
        self.h_lr = h_lr

        self.memory = deque(maxlen=buffer_size)

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # shared network with two heads
        self.net = QRCNet(state_dim, action_dim).to(self.device)
        self.target_net = QRCNet(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())

        # separate optimizers:
        # Q-head: no weight decay
        self.q_optimizer = optim.Adam(
            list(self.net.backbone.parameters()) +
            list(self.net.q_head.parameters()),
            lr=lr
        )

        # H-head: with weight decay
        self.h_optimizer = optim.Adam(
            self.net.h_head.parameters(),
            lr=h_lr,
            weight_decay=beta
        )

    # --------------------------
    # epsilon-greedy policy
    # --------------------------
    def agent_policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q, _ = self.net(state_t)
        return int(q.argmax(dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    # --------------------------
    # training step
    # --------------------------
    def train_with_mem(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # forward pass
        q_values, h_values = self.net(states)
        q_sa = q_values.gather(1, actions)
        h_sa = h_values.gather(1, actions)

        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states)
            next_q = next_q_values.max(dim=1, keepdim=True)[0]

            target = rewards + self.gamma * next_q * (1 - dones)
            delta = target - q_sa

        # --------------------------
        # Q update
        # --------------------------
        td_loss = 0.5 * delta.pow(2).mean()
        correction = torch.mean(self.gamma * h_sa.detach() * next_q)

        self.q_optimizer.zero_grad()
        (td_loss + correction).backward()
        self.q_optimizer.step()

        # --------------------------
        # H update
        # --------------------------
        h_loss = 0.5 * (delta.detach() - h_sa).pow(2).mean()

        self.h_optimizer.zero_grad()
        h_loss.backward()
        self.h_optimizer.step()

        # --------------------------
        # epsilon decay
        # --------------------------
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

class QRCAgent_FB:
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

        self.device = device if device is not None else torch.device("cpu")

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

        # Networks
        self.q_net = self.build_nn(self.action_dim).to(self.device)
        self.h_net = self.build_nn(self.action_dim).to(self.device)
        self.target_net = self.build_nn(self.action_dim).to(self.device)
        self.update_target()  # initialize target

        # Optimizers
        self.q_optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=self.lr)
        self.h_optimizer = torch.optim.AdamW(self.h_net.parameters(), lr=self.h_lr)

    # -----------------------------
    # Build simple MLP
    # -----------------------------
    def build_nn(self, output_dim):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    # -----------------------------
    # Epsilon-greedy policy
    # -----------------------------
    def agent_policy(self, state):
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # -----------------------------
    # Store transition
    # -----------------------------
    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    # -----------------------------
    # Train on mini-batch
    # -----------------------------
    def train_with_mem(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        batch_indices = torch.arange(self.batch_size).unsqueeze(1).to(self.device)

        # --- Forward pass ---
        q_values = self.q_net(states)
        h_values = self.h_net(states)
        q_next_target = self.target_net(next_states)

        q_sa = q_values.gather(1, actions)  # Q(s,a)
        h_sa = h_values.gather(1, actions)  # H(s,a)

        # a' = argmax_a Q̂(s',a)
        with torch.no_grad():
            a_prime = q_next_target.argmax(dim=1, keepdim=True)
            q_target_sp_ap = q_next_target.gather(1, a_prime)

        # TD error δ = r + γ Q̂(s',a') - Q(s,a)
        delta = rewards + self.gamma * q_target_sp_ap * (1 - dones) - q_sa

        # ---------------------
        # Update H network
        # ---------------------
        h_loss = 0.5 * ((delta.detach() - h_sa) ** 2 + self.beta * h_sa ** 2).mean()
        self.h_optimizer.zero_grad()
        h_loss.backward()
        self.h_optimizer.step()

        # Recompute h after update
        h_values_updated = self.h_net(states)
        h_sa_updated = h_values_updated.gather(1, actions)

        with torch.no_grad():
            h_sp_ap = h_values_updated.gather(1, a_prime)

        # ---------------------
        # Update Q network
        # ---------------------
        q_loss = (-delta.detach() * q_sa + self.gamma * h_sp_ap.detach() * q_target_sp_ap).mean()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ---------------------
        # Hard target network update (default)
        # ---------------------
        self.update_target()

        # ---------------------
        # Epsilon decay
        # ---------------------
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # -----------------------------
    # Hard update target network
    # -----------------------------
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

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
    os.makedirs("result", exist_ok=True)
    np.savetxt(f"result/qrc_seed_{seed}.txt", np.array(episode_rewards))
    # print(f"[Seed = {seed}] saved to results/qrc_seed_{seed}.txt")


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1])
    run(seed)