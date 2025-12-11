import numpy as np
import matplotlib.pyplot as plt
import os


def load_and_process_data(filename):
    """
    Loads reward data from a file and returns avg + std across runs.
    Handles:
        - comma or space delimited
        - headers
        - single or multiple runs
    """

    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return None, None, None, None

    try:
        # Try comma-delimited first
        try:
            all_rewards = np.loadtxt(filename, delimiter=",", comments="#")
        except:
            # Fall back to whitespace-delimited
            all_rewards = np.loadtxt(filename, comments="#")

    except Exception as e:
        print(f"❌ Error loading '{filename}': {e}")
        return None, None, None, None

    if all_rewards.ndim == 1:
        # Single run → reshape
        all_rewards = all_rewards.reshape(1, -1)

    num_runs, num_episodes = all_rewards.shape
    
    avg_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    
    print(f"Loaded '{filename}' → {num_runs} runs, {num_episodes} episodes")

    return avg_rewards, std_rewards, num_episodes, num_runs



def plot_drl_comparison(agent_data_list, smooth_window=50):
    """
    Plots DRL agent performance curves (avg + std + smoothed).
    agent_data_list: list of (filename, label, color)
    """
    plt.figure(figsize=(16, 5))

    processed_data = []
    max_episodes = 0
    run_counts = []

    # Load data
    for filename, label, color in agent_data_list:
        avg, std, num_episodes, num_runs = load_and_process_data(filename)

        print(f"Processing {label}: {num_runs} runs, {num_episodes} episodes")

        if avg is None:
            print(f"⚠️ Skipping {label}, no valid data.")
            continue

        run_counts.append(num_runs)
        max_episodes = max(max_episodes, num_episodes)

        # Smoothing
        if num_episodes >= smooth_window:
            window = np.ones(smooth_window) / smooth_window
            smoothed_avg = np.convolve(avg, window, mode="valid")
            x_smooth = np.arange(smooth_window - 1, num_episodes)
        else:
            smoothed_avg = avg
            x_smooth = np.arange(num_episodes)

        processed_data.append({
            "label": label,
            "color": color,
            "avg": avg,
            "std": std,
            "episodes": num_episodes,
            "smoothed": smoothed_avg,
            "x_smooth": x_smooth,
            "num_runs": num_runs,
        })

    # No data at all
    if not processed_data:
        print("❌ No valid agent data to plot.")
        return

    # Plot each agent
    for data in processed_data:
        # print("Color : ", data["color"])
        # Std dev shading
        x = np.arange(data["episodes"])
        plt.fill_between(
            x,
            data["avg"] - data["std"],
            data["avg"] + data["std"],
            color=data["color"],
            alpha=0.20
        )

        # Raw average line
        # plt.plot(
        #     x, data["avg"],
        #     color=data["color"],
        #     linewidth=1.5,
        #     # linewidth=1,
        #     # alpha=0.8,
        #     label=f"{data['label']} Raw Avg"
        # )

        # Smoothed line
        plt.plot(
            data["x_smooth"],
            data["smoothed"],
            color=data["color"],
            linewidth=3,
            label=f"{data['label']} Smoothed (N={data['num_runs']})"
        )

    # Layout
    plt.title("DQN vs QRC Performance Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc='lower right')
    plt.xlim(0, max_episodes)
    plt.tight_layout()
    plt.show()



# ============================
#  EXECUTION
# ============================

dqn_file = "results/dqn.txt"
qrc_file = "results/qrc.txt"

# Blue for DQN, Red for QRC
agents_to_compare = [
    (dqn_file, "DQN Agent", "black"),
    (qrc_file, "QRC Agent", "red"),
]

plot_drl_comparison(agents_to_compare, smooth_window=50)