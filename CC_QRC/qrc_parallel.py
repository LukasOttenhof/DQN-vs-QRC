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
    def __init__(self, 
                 state_dim,
                 action_dim,
                 lr=1e-3,
                 gamma=0.99,
                 epsilon=0.5,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 buffer_size=50000,
                 batch_size=64,
                 beta=1.0,
                 device=None):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.beta = beta
        
        # Handle Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize Memory
        self.memory = deque(maxlen=self.buffer_size)

        # Internal Architecture Param (Must match the middle layer of build_nn)
        self.hidden_dim = 128 

        # Build Networks
        self.q_net = self.build_nn(self.action_dim).to(self.device)
        self.target_net = self.build_nn(self.action_dim).to(self.device)
        self.update_target()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        # QRC Specific Parameters (Secondary weights optimization)
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.eps = 1e-8
        
        # Auxiliary weights h, and Adam stats v and m
        self.h = torch.zeros(self.action_dim, self.hidden_dim, requires_grad=False).to(self.device)
        self.v = torch.zeros(self.action_dim, self.hidden_dim, requires_grad=False).to(self.device)
        self.m = torch.zeros(self.action_dim, self.hidden_dim, requires_grad=False).to(self.device)
        
        self.updates = 0

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
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state_t)
        self.q_net.train()
        return int(q_values.argmax(dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train_with_mem(self):
        if len(self.memory) < self.batch_size:
            return

        self.updates += 1
        
        # Sample batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Convert to numpy first for speed, then tensor
        states_np = np.array([x[0] for x in minibatch])
        actions_np = np.array([x[1] for x in minibatch])
        rewards_np = np.array([x[2] for x in minibatch])
        next_states_np = np.array([x[3] for x in minibatch])
        dones_np = np.array([x[4] for x in minibatch])

        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor(actions_np).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards_np).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.FloatTensor(dones_np).to(self.device)

        # --- Forward Pass Split ---
        # We need the features (x) from the penultimate layer for QRC.
        # Since build_nn returns Sequential, we slice it.
        # features_net: first 4 layers (Linear, ReLU, Linear, ReLU)
        # head_net: last layer (Linear)
        features_net = self.q_net[:-1] 
        head_net = self.q_net[-1]
        
        # 1. Compute Q(s, a) and features x
        x = features_net(states)
        Qs = head_net(x)
        Qsa = Qs.gather(1, actions).squeeze()

        # 2. Compute Target: r + gamma * max Q_target(s', a')
        with torch.no_grad():
            # Get Q_target(s') using the target net
            target_features = self.target_net[:-1](next_states)
            target_Qs = self.target_net[-1](target_features)
            
            # Standard Q-Learning max
            max_next_q = target_Qs.max(1).values
            target = rewards + (1 - dones) * self.gamma * max_next_q

        # 3. TD Loss
        td_loss = 0.5 * nn.functional.mse_loss(target, Qsa)

        # 4. QRC Correction Terms
        # Compute E[\delta | x] ~= <h, x>
        with torch.no_grad():
            delta_hats = torch.matmul(x, self.h.t())
            delta_hat = delta_hats.gather(1, actions).squeeze()
            
            # For correction loss, we need Q value of next state from target net
            # but we need it to be differentiable for the backward pass on target_net
            # Note: We re-run forward on target net below with gradients enabled momentarily
        
        # Re-run target forward pass to allow gradient flow for correction term
        # (We only need gradients for the correction calculation, not to update target net weights directly)
        
        # Toggle requires_grad on for target net for this specific calculation
        for param in self.target_net.parameters():
            param.requires_grad = True
            
        t_feat = self.target_net[:-1](next_states)
        t_out = self.target_net[-1](t_feat)
        
        # We need the Q-value of the greedy action at s' (or just the value used in bootstrap)
        # To simplify, we use the value associated with the max action found earlier
        # Note: If done, the value is 0, so mask it
        max_actions = t_out.argmax(dim=1).unsqueeze(1)
        Qspap = t_out.gather(1, max_actions).squeeze()
        Qspap = Qspap * (1 - dones)

        # The gradient correction term: gamma * <h, x> * \nabla_w Q(s', a')
        correction_loss = torch.mean(self.gamma * delta_hat * Qspap)

        # 5. Backpropagation
        self.optimizer.zero_grad()
        self.target_net.zero_grad()

        # A. Backprop TD loss through Policy Net
        td_loss.backward()

        # B. Backprop Correction loss through Target Net
        correction_loss.backward()

        # C. Add Target Net gradients to Policy Net gradients
        for policy_param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
            if target_param.grad is not None:
                if policy_param.grad is None:
                    policy_param.grad = target_param.grad
                else:
                    policy_param.grad.add_(target_param.grad)

        # Re-disable grads for target net
        for param in self.target_net.parameters():
            param.requires_grad = False
            
        self.optimizer.step()

        # 6. Update Secondary Weights (h) using Adam-like update
        with torch.no_grad():
            delta = target - Qsa # Temporal Difference Error
            
            # dh = (delta - delta_hat) * x
            # We must do this for every action in the batch
            
            dh_base = (delta - delta_hat).unsqueeze(1) * x # Shape: [batch, hidden]
            
            for a in range(self.action_dim):
                # Mask for samples where action 'a' was taken
                mask = (actions == a).squeeze()
                
                if mask.sum() == 0:
                    continue
                
                # Get relevant gradients for this action's h vector
                # If batch size is 1, mask is a scalar boolean, otherwise tensor
                if mask.dim() == 0:
                    if not mask: continue
                    h_update = dh_base - self.beta * self.h[a]
                else:
                    h_update = dh_base[mask].mean(0) - self.beta * self.h[a]

                # Update Adam stats
                self.v[a] = self.beta_2 * self.v[a] + (1 - self.beta_2) * (h_update**2)
                self.m[a] = self.beta_1 * self.m[a] + (1 - self.beta_1) * h_update

                m_hat = self.m[a] / (1 - self.beta_1**self.updates)
                v_hat = self.v[a] / (1 - self.beta_2**self.updates)

                # Update h
                self.h[a] = self.h[a] + self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class QRCAgentV2:
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
        h_lr = 1.0
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
        self.q_optimizer = optim.AdamW(self.q_net.parameters(), lr=self.lr)
        self.h_optimizer = optim.AdamW(self.h_net.parameters(), lr=self.h_lr, weight_decay=self.beta)

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

        # --- Compute Q(s,a) ---
        q_values = self.q_net(states)
        q_sa = q_values.gather(1, actions)

        # --- TD target : Q_target(s', a') using target_net ---
        with torch.no_grad():
            next_q_all = self.target_net(next_states)              
            next_q = next_q_all.max(dim=1, keepdim=True)[0]       
            target = rewards + self.gamma * next_q * (1 - dones)

        delta = target - q_sa
        delta_detached = delta.detach()

        # --- H-network ---
        h_values = self.h_net(states)
        h_sa = h_values.gather(1, actions)

        # --- Losses ---
        td_loss = 0.5 * (delta.pow(2)).mean()
        correction = torch.mean(self.gamma * h_sa.detach() * next_q)  # QRC cross-term
        h_loss = 0.5 * ((delta_detached - h_sa).pow(2)).mean()

        # --- Update Q-network ---
        self.q_optimizer.zero_grad()
        q_loss = td_loss + correction
        q_loss.backward()
        self.q_optimizer.step()

        # --- Update H-network ---
        self.h_optimizer.zero_grad()
        h_loss.backward()
        self.h_optimizer.step()

        # --- Epsilon decay ---
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Target network update (hard update)
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
        beta=0.951
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
    os.makedirs("result_new_adam", exist_ok=True)
    np.savetxt(f"result_new_adam/qrc_seed_{seed}.txt", np.array(episode_rewards))
    # print(f"[Seed = {seed}] saved to results/qrc_seed_{seed}.txt")


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1])
    run(seed)