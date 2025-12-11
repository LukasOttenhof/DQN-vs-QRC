import numpy as np
from tqdm import tqdm
from rl_glue import RLGlue
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class QRCAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,               # Q-network step-size (α)
        h_lr=1.0,               # h-network step-size (α_h)
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=50000,
        batch_size=64,
        beta=1,                 # regularization coef β
        lambda_coef=0.8,        # λ coefficient for correction term
        device=None,
        seed=42
    ):
        # Save parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.h_lr = h_lr
        self.gamma = gamma
        self.lambda_coef = lambda_coef
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.beta = beta

        # Seeding
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_nn = self.build_nn().to(self.device)
        self.h_net = self.build_nn().to(self.device)

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_nn.parameters(), lr=self.lr)
        self.h_optimizer = optim.AdamW(self.h_net.parameters(), lr=self.h_lr, weight_decay=self.beta)

        self.steps = 0

    def build_nn(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )

    def agent_policy(self, state):
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_nn(state_t)
        return int(q_values.argmax(dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Q(s,a)
        q_values = self.q_nn(states)
        q_sa = q_values.gather(1, actions)

        # Q(s',a')
        next_q_vals = self.q_nn(next_states)
        next_q = next_q_vals.max(1)[0].unsqueeze(1)

        # Standard TD error
        delta = (rewards + self.gamma * (1.0 - dones) * next_q.detach()) - q_sa
        delta_detached_for_h = delta.detach()

        # h(s,a)
        h_values = self.h_net(states)
        h_sa = h_values.gather(1, actions)

        # -----------------------------
        # Loss components
        # -----------------------------
        td_loss = 0.5 * delta.pow(2).mean()

        # λ * γ * h(s,a) * Q(s',a')
        correction_term = self.lambda_coef * (self.gamma * h_sa.detach() * next_q).mean()

        # h-loss: regress h toward delta
        h_loss = 0.5 * ((delta_detached_for_h - h_sa).pow(2)).mean()

        return td_loss, correction_term, h_loss

    def train_with_mem(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        td_loss, correction_term, h_loss = self.compute_loss(
            (states, actions, rewards, next_states, dones)
        )

        # -----------------------------
        # Update Q-network
        # -----------------------------
        self.q_optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        correction_term.backward()
        self.q_optimizer.step()

        # -----------------------------
        # Update h-network
        # -----------------------------
        self.h_optimizer.zero_grad()
        h_loss.backward()
        self.h_optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return td_loss.item() + correction_term.item() + h_loss.item()

    def update_target(self):
        pass

class DQNAgent:
    def __init__(
            self, 
            state_dim, 
            action_dim, 
            lr=1e-3, 
            gamma=0.99, 
            epsilon=1.0, 
            epsilon_decay=0.995, 
            epsilon_min=0.01, 
            buffer_size=50000, 
            batch_size=64, 
            seed=42
        ):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (optional, may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        # device, not needed, but needed if going on canada compute to specify gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_nn = self.build_nn(seed).to(self.device) # build q network
        self.target_net = self.build_nn(seed).to(self.device) # build target network
        self.target_net.load_state_dict(self.q_nn.state_dict()) # make target net same as q net. initialization is random so need this
        self.optimizer = optim.Adam(self.q_nn.parameters(), lr=lr) 
        
    def build_nn(self, seed): # to build the q network, 2 hiddne layer with relu and 128 neuronsin each
        # torch.manual_seed(seed)
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
    
    def agent_policy(self, state): # act e greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim) # rand action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # convert state to tensor, then add batch dimension, then move to device
        with torch.no_grad(): # dont calculate gradients, no need
            q_values = self.q_nn(state)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # store transition sars, done is if terminal state
    
    def train_with_mem(self): # train with experience from memory using batch size set in agent
        if len(self.memory) < self.batch_size: # if not enought mem, could be changed to use what we have instead of skip
            return
        batch = random.sample(self.memory, self.batch_size) # get batch
        states, actions, rewards, next_states, dones = zip(*batch) # get batch features

        # convert data to tensors which can be used by pytorch
        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device) 
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # get current q values from q network
        q_values = self.q_nn(states).gather(1, actions)

        # use the samples from experience batch to calc target network q values
        with torch.no_grad(): # no need to calc gradients for target
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]  
        # target = r + gamma * max_a' Q(s', a') * (1 - done)
        target = rewards + self.gamma * next_q_values * (1 - dones)

        # mean squared error 
        errors = target - q_values
        squared_errors = errors ** 2
        mean_squared_error = torch.mean(squared_errors)

        self.optimizer.zero_grad()
        mean_squared_error.backward()
        self.optimizer.step()
 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 
    
    def update_target(self): # update target network replacing it with the current q network
        self.target_net.load_state_dict(self.q_nn.state_dict())

class QRCAgent_WithTarget:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=50000,
        batch_size=64,
        beta=1,     # weight decay for h-net (small default)
        device=None,
        seed=None
    ):
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            # Make CUDA operations deterministic (optional, may slow down training)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            self.seed = None
        # Basic configs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.beta = beta

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

        # Device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks (simple MLPs)
        self.q_nn = self.build_nn(output_dim=action_dim).to(self.device)
        self.target_net = self.build_nn(output_dim=action_dim).to(self.device)
        self.h_net = self.build_nn(output_dim=action_dim).to(self.device)

        # Initialize target net
        self.target_net.load_state_dict(self.q_nn.state_dict())
        self.target_net.eval()

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_nn.parameters(), lr=self.lr)
        self.h_optimizer = optim.AdamW(self.h_net.parameters(), lr=self.lr, weight_decay=self.beta)

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
            q_values = self.q_nn(state_t)  # (1, action_dim)
        return int(q_values.argmax(dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)            
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(self.device)  
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)          
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)       
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)           

        # Current Q(s,a)
        q_values = self.q_nn(states)                  
        q_sa = q_values.gather(1, actions)            

        # Next Q from target net — do NOT use no_grad() here,
        # because correction term needs gradients w.r.t. target_net.
        next_q_vals = self.target_net(next_states)    
        next_q = next_q_vals.max(1)[0].unsqueeze(1)   


        # Standard TD target
        delta = (rewards + self.gamma * (1.0 - dones) * next_q.detach()) - q_sa

        delta_detached_for_h = delta.detach()

        # Predict correction h(s,a)
        h_values = self.h_net(states)                      
        h_sa = h_values.gather(1, actions)                 

        # td_loss (0.5 * MSE)
        td_loss = 0.5 * (delta.pow(2)).mean()

        # delta_hat (we detach h_sa for correction so the correction backward affects only target_net)
        delta_hat_detached = h_sa.detach()

        # correction term: gamma * <h, x> * Q(s', a')  -> next_q is from target_net (has grad)
        correction_term = (self.gamma * delta_hat_detached * next_q).mean()

        # h loss: regress h(s,a) to delta.detach()  (this uses delta_detached_for_h so no q-network graph)
        h_loss = 0.5 * ((delta_detached_for_h - h_sa).pow(2)).mean()

        return td_loss, correction_term, h_loss

    def train_with_mem(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # compute losses (td_loss, correction_term, h_loss)
        td_loss, correction_term, h_loss = self.compute_loss(
            (states, actions, rewards, next_states, dones)
        )

        # zero grads on policy and target nets
        self.q_optimizer.zero_grad()
        # We need to zero grads of target_net because we'll backprop the correction term through target_net
        for p in self.target_net.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # 1) compute gradients for td_loss w.r.t policy network parameters
        td_loss.backward(retain_graph=False)   # no need to retain graph, since h_loss does not use q-graph now

        # 2) compute gradients for correction_term w.r.t target_net parameters
        # (correction_term depends on next_q which comes from target_net)
        correction_term.backward(retain_graph=False)

        # 3) add target_net grads to policy network grads (mirror reference code)
        # Important: q_nn and target_net must have same parameter ordering
        for (policy_param, target_param) in zip(self.q_nn.parameters(), self.target_net.parameters()):
            if target_param.grad is not None:
                if policy_param.grad is None:
                    policy_param.grad = target_param.grad.clone()
                else:
                    policy_param.grad.add_(target_param.grad)

        # 4) step policy optimizer (updates q_nn)
        self.q_optimizer.step()

        # ---- Update h_net (separately) ----
        # Because h_loss used delta.detach(), its computational graph only depends on h_net parameters.
        self.h_optimizer.zero_grad()
        h_loss.backward()
        self.h_optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        overall_loss = td_loss.item() + correction_term.item() + h_loss.item()
        return overall_loss

    # Target network update (hard update)
    def update_target(self):
        self.target_net.load_state_dict(self.q_nn.state_dict())