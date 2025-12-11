import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from agents import QRCAgent, DQNAgent
from tbu_gym.tbu_discrete import TruckBackerEnv_D

QRC_AGENT = "QRC_AGENT"
DQN_Agent = "DQN_Agent"

class Experiment:
    def __init__(
        self,
        num_episodes=1000,
        max_steps_per_episode=500,
        gamma=0.99,
        learning_rate=1e-3,
        epsilon_start=1.0,
        epsilon_decay=0.99997,
        epsilon_min=0.01,
        batch_size=64,
        target_update_freq=5,
        agent_name=QRC_AGENT
    ):
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.agent_name = agent_name
        self.episode_rewards = []
        self.recent_loss = 0.0  # initialize

    def set_seed(self, seed):
        """Set seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def run_single(self, seed=None):
        self.set_seed(seed)
        env = TruckBackerEnv_D(render_mode=None)
        if seed is not None:
            env.seed(seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        episode_rewards = []
        self.recent_loss = 0.0

        # Initialize agent with seed
        if self.agent_name == QRC_AGENT:
            agent = QRCAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=self.learning_rate,
                gamma=self.gamma,
                epsilon=self.epsilon_start,
                epsilon_decay=self.epsilon_decay,
                epsilon_min=self.epsilon_min,
                batch_size=self.batch_size,
                seed=seed
            )
        elif self.agent_name == DQN_Agent:
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=self.learning_rate,
                gamma=self.gamma,
                epsilon=self.epsilon_start,
                epsilon_decay=self.epsilon_decay,
                epsilon_min=self.epsilon_min,
                batch_size=self.batch_size,
                seed=seed
            )
        else:
            return episode_rewards

        for episode in range(1, self.num_episodes + 1):
            reset_output = env.reset()
            state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
            total_reward = 0

            for t in range(self.max_steps_per_episode):
                action = agent.agent_policy(state)
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = step_result

                total_reward += reward
                agent.remember(state, action, reward, next_state, done)
                loss = agent.train_with_mem()
                if loss is not None:
                    self.recent_loss = loss
                state = next_state
                if done:
                    break
            if episode % self.target_update_freq == 0:
                agent.update_target()

            episode_rewards.append(total_reward)
            if self.recent_loss:
                print(f"Episode {episode:4d}/{self.num_episodes}, "
                      f"Reward: {total_reward:7.2f}, "
                      f"Epsilon: {agent.epsilon:.5f}, "
                      f"Loss: {self.recent_loss:.5f}")
            else:
                print(f"Episode {episode:4d}/{self.num_episodes}, "
                      f"Reward: {total_reward:7.2f}, "
                      f"Epsilon: {agent.epsilon:.5f}")
        return episode_rewards

    def run_multiple(self, num_runs=5, seeds=None):
        all_rewards = []
        for run in range(num_runs):
            run_seed = seeds[run] if seeds and len(seeds) > run else None
            print(f"\n===== Running Experiment {run+1}/{num_runs} with seed {run_seed} =====")
            rewards = self.run_single(seed=run_seed)
            all_rewards.append(rewards)

        all_rewards = np.array(all_rewards)
        avg_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        return avg_rewards, std_rewards, all_rewards
    
    def run_single_visual(self, seed=None, save_path=None, smooth_window=50):
        """Run a single experiment and visualize rewards (with optional smoothing)."""
        rewards = self.run_single(seed=seed)
        
        plt.figure(figsize=(14, 6))
        plt.plot(rewards, label="Episode Reward", alpha=0.4)

        # Smooth rewards
        if len(rewards) >= smooth_window:
            window = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(rewards, window, mode='valid')
            x_smooth = np.arange(smooth_window - 1, len(rewards))
            plt.plot(x_smooth, smoothed, label=f"Smoothed (window={smooth_window})", linewidth=2.0)

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"{self.agent_name} Reward over Episodes")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return rewards


    def run_multiple_visual(self, seeds, smooth_window=50):
        """Run multiple experiments and visualize mean ± std rewards (with optional smoothing)."""
        all_rewards = []

        for seed in seeds:
            print(f"\n===== Running Experiment with seed {seed} =====")
            rewards = self.run_single(seed=seed)
            all_rewards.append(rewards)

        all_rewards = np.array(all_rewards)
        avg_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        plt.figure(figsize=(14, 6))
        plt.plot(avg_rewards, label="Average Reward", alpha=0.5)

        # Smooth average rewards
        if len(avg_rewards) >= smooth_window:
            window = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(avg_rewards, window, mode='valid')
            x_smooth = np.arange(smooth_window - 1, len(avg_rewards))
            plt.plot(x_smooth, smoothed, label=f"Smoothed Avg (window={smooth_window})", linewidth=2)

        # Plot ±1 std deviation
        plt.fill_between(range(len(avg_rewards)), avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.2, label="±1 Std Dev")

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"{self.agent_name} — Average Reward over {len(seeds)} Seeds")
        plt.legend()
        plt.grid(True)
        plt.show()

        return {
            "seeds": seeds,
            "avg_rewards": avg_rewards,
            "std_rewards": std_rewards,
            "all_rewards": all_rewards
        }

    def run_agents_sequential_multiple_seeds(self, seeds, smooth_window=50, qrc_file="qrc.txt", dqn_file="dqn.txt"):
        results = {"QRC_AGENT": [], "DQN_Agent": []}

        for run_idx, seed in enumerate(seeds):
            print(f"\n===== Running Seed {run_idx+1}/{len(seeds)}: {seed} =====")
            self.set_seed(seed)

            # Initialize environment
            env = TruckBackerEnv_D(render_mode=None)
            if seed is not None:
                env.seed(seed)

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            # Initialize agents
            qrc_agent = QRCAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=self.learning_rate,
                gamma=self.gamma,
                epsilon=self.epsilon_start,
                epsilon_decay=self.epsilon_decay,
                epsilon_min=self.epsilon_min,
                batch_size=self.batch_size,
                seed=seed
            )
            dqn_agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=self.learning_rate,
                gamma=self.gamma,
                epsilon=self.epsilon_start,
                epsilon_decay=self.epsilon_decay,
                epsilon_min=self.epsilon_min,
                batch_size=self.batch_size,
                seed=seed
            )

            # --- Run QRC ---
            qrc_rewards = []
            self.recent_loss = 0.0
            for episode in range(1, self.num_episodes + 1):
                reset_output = env.reset()
                state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
                total_reward = 0

                for t in range(self.max_steps_per_episode):
                    action = qrc_agent.agent_policy(state)
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    else:
                        next_state, reward, done, _ = step_result

                    total_reward += reward
                    qrc_agent.remember(state, action, reward, next_state, done)
                    loss = qrc_agent.train_with_mem()
                    if loss is not None:
                        self.recent_loss = loss
                    state = next_state
                    if done:
                        break

                qrc_rewards.append(total_reward)
                print(f"[Seed {seed}] QRC Episode {episode:4d}/{self.num_episodes} | "
                    f"Reward: {total_reward:7.2f} | Loss: {self.recent_loss:.5f} | "
                    f"Epsilon: {qrc_agent.epsilon:.5f}")

            results["QRC_AGENT"].append(qrc_rewards)

            env.reset()
            env.seed(seed)

            # --- Run DQN ---
            dqn_rewards = []
            self.recent_loss = 0.0
            for episode in range(1, self.num_episodes + 1):
                reset_output = env.reset()
                state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
                total_reward = 0

                for t in range(self.max_steps_per_episode):
                    action = dqn_agent.agent_policy(state)
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    else:
                        next_state, reward, done, _ = step_result

                    total_reward += reward
                    dqn_agent.remember(state, action, reward, next_state, done)
                    loss = dqn_agent.train_with_mem()
                    if loss is not None:
                        self.recent_loss = loss
                    state = next_state
                    if done:
                        break

                # DQN updates target network periodically
                if episode % self.target_update_freq == 0:
                    dqn_agent.update_target()

                dqn_rewards.append(total_reward)
                print(f"[Seed {seed}] DQN Episode {episode:4d}/{self.num_episodes} | "
                    f"Reward: {total_reward:7.2f} | Loss: {self.recent_loss:.5f} | "
                    f"Epsilon: {dqn_agent.epsilon:.5f}")

            results["DQN_Agent"].append(dqn_rewards)

        # Aggregate results
        qrc_array = np.array(results["QRC_AGENT"])
        dqn_array = np.array(results["DQN_Agent"])

        qrc_avg = np.mean(qrc_array, axis=0)
        qrc_std = np.std(qrc_array, axis=0)
        dqn_avg = np.mean(dqn_array, axis=0)
        dqn_std = np.std(dqn_array, axis=0)

        # Save results
        np.savetxt(qrc_file, qrc_array, fmt="%.5f")
        np.savetxt(dqn_file, dqn_array, fmt="%.5f")

        # Plot comparison
        plt.figure(figsize=(14, 6))
        for name, avg, std in [("QRC_AGENT", qrc_avg, qrc_std), ("DQN_Agent", dqn_avg, dqn_std)]:
            plt.plot(avg, label=f"{name} Avg Reward", alpha=0.5)
            plt.fill_between(range(len(avg)), avg - std, avg + std, alpha=0.2)
            if len(avg) >= smooth_window:
                smoothed = np.convolve(avg, np.ones(smooth_window)/smooth_window, mode='valid')
                x_smooth = np.arange(smooth_window - 1, len(avg))
                plt.plot(x_smooth, smoothed, label=f"{name} Smoothed", linewidth=2)

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"QRC vs DQN Sequential Run ({len(seeds)} seeds)")
        plt.legend()
        plt.grid(True)
        plt.show()

        return {
            "QRC_AGENT": {"all": qrc_array, "avg": qrc_avg, "std": qrc_std},
            "DQN_Agent": {"all": dqn_array, "avg": dqn_avg, "std": dqn_std}
        }