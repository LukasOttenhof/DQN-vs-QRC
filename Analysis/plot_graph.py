import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# os.makedirs("../data", exist_ok=True)

def plot_rewards(name="QRC", pt_file="data/qrc_reward_seeds.pt"):
    data = torch.load(pt_file)
    all_rewards = np.array(data['rewards'])  # shape: (num_seeds, num_episodes)

    # Ensure 2D shape
    if all_rewards.ndim == 1:
        all_rewards = all_rewards.reshape(1, -1)

    num_seeds, num_episodes = all_rewards.shape
    print("\nAgent Type :", name)
    print("num_seeds :", num_seeds)
    print("num_episodes :", num_episodes)

    # # Compute mean and std across seeds
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    # episodes = np.arange(1, num_episodes + 1)
    # print("mean_rewards :", mean_rewards)
    # for index, item in enumerate(mean_rewards):
    #     print(index+1, ": ",item)

    # # 90% confidence interval
    # ci_90 = 1.645 * (std_rewards / np.sqrt(num_seeds))
    # plt.figure(figsize=(12,6))
    # sns.lineplot(x=episodes, y=mean_rewards, label="Mean Reward", color="blue")
    # plt.fill_between(episodes, mean_rewards - ci_90, mean_rewards + ci_90,
    #                 alpha=0.3, color="blue", label="90% CI")
    # plt.title(f"QRC Training Rewards seeds 0-{num_seeds}, 90% CI")
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # # 95% confidence interval
    # ci_95 = 1.96 * (std_rewards / np.sqrt(num_seeds))
    # plt.figure(figsize=(14,6))
    # sns.lineplot(x=episodes, y=mean_rewards, label="Mean Reward", color="red")
    # plt.fill_between(episodes, mean_rewards - ci_95, mean_rewards + ci_95,
    #                 alpha=0.3, color="red", label="95% CI")
    # plt.title(f"QRC Training Rewards seeds 0-{num_seeds}, 95% CI")
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

# plot_rewards()
# plot_rewards(pt_file="data/qrc_reward_seeds_target_update.pt")
plot_rewards(name="DQN Target Network", pt_file="data/dqn_reward_seeds_target_update.pt")
plot_rewards(name="DQN Epsilon", pt_file="data/dqn_reward_seeds_epsilon.pt")
