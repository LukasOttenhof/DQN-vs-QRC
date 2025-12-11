import random
from experiment import Experiment, QRC_AGENT, DQN_Agent

def generate_seeds(n, seed_range=(0, 1000000)):
    seeds = [random.randint(seed_range[0], seed_range[1]) for _ in range(n)]
    return seeds


experiment = Experiment()
seeds = generate_seeds(3)
print("Running experiment with seeds:", seeds)
experiment.run_agents_sequential_multiple_seeds(
    seeds=seeds,
    qrc_file="qrc_results_10.txt",
    dqn_file="dqn_results_10.txt"
    )