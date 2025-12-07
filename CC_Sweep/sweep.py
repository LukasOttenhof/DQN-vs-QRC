import argparse
import os
import sys
import random
import secrets
import time
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from matplotlib.widgets import EllipseSelector

from CC_DQN.dqn import DQNAgent, set_global_seed as set_seed_dqn
from CC_QRC.qrc import QRCAgent, set_global_seed as set_seed_qrc
from tbu_discrete import TruckBackerEnv_D

def sample_config(base, space, rng):
    config = dict(base)
    
    def log_uniform(lo, hi):
        return 10 ** (rng.uniform(math.log10(lo), math.log10(hi)))

    def pick(val):
        # If the space entry is a list, pick a discrete choice; otherwise sample log-uniform
        return rng.choice(val) if isinstance(val, list) else log_uniform(*val)
    
    config["learning_rate"] = pick(space["learning_rate"])
    config["epsilon_start"] = pick(space["epsilon_start"])
    config["epsilon_decay"] = pick(space["epsilon_decay"])
    config["epsilon_min"] = pick(space["epsilon_min"])
    config["batch_size"] = pick(space["batch_size"])
    config["buffer_size"] = pick(space["buffer_size"])
    config["gamma"] = pick(space["gamma"])
    config["target_update_freq"] = pick(space["target_update_freq"])
    
    if "beta" in space:
        config["beta"] = pick(space["beta"])
    
    return config

def print_device_once():
    # Guard to avoid repeated prints per worker
    if os.environ.get("CUDA_DEVICE_PRINTED") == "1":
        return
    os.environ["CUDA_DEVICE_PRINTED"] = "1"
    try:
        # This happens inside the spawned worker, which is safe
        using_cuda = torch.cuda.is_available()
        dev_name = torch.cuda.get_device_name(0) if using_cuda else "CPU"
        print(f"[worker {os.getpid()}] Device: {'CUDA' if using_cuda else 'CPU'} ({dev_name})")
    except Exception as e:
        print(f"[worker {os.getpid()}] Device check failed: {e}")

def run_single_dqn(config, save_dir):
    if config.get("verbose", True):
        print_device_once()
    set_seed_dqn(config["seed"])
    env = TruckBackerEnv_D(render_mode=None)
    env.seed(config["seed"])
    env.action_space.seed(config["seed"])
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=config["learning_rate"],
        epsilon=config["epsilon_start"],
        epsilon_decay=config["epsilon_decay"],
        epsilon_min=config["epsilon_min"],
        batch_size=config["batch_size"],
        buffer_size=config["buffer_size"],
        gamma=config["gamma"]
    )
    
    episode_rewards = []
    with tqdm(total=config["num_episodes"], desc=f"DQN seed {config['seed']}", leave=False, disable=not config.get("verbose", False)) as pbar:
        for episode in range(1, config["num_episodes"] + 1):
            env.seed(config["seed"] + episode)
            state = env.reset()
            total_reward = 0
            
            for t in range(config["max_steps_per_episode"]):
                action = agent.agent_policy(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.train_with_mem()
                state = next_state
                total_reward += reward
                if done:
                    break
                    
            if episode % config["target_update_freq"] == 0:
                agent.update_target()
            episode_rewards.append(total_reward)
            pbar.update(1)
        
    result = {
        "config": config,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
    }
    
    run_id = f"dqn_{config['seed']}_{int(config['learning_rate']*1e6)}_{config['batch_size']}"
    torch.save({"rewards": torch.tensor(episode_rewards, dtype=torch.float32)}, 
               os.path.join(save_dir, run_id + ".pt"))
    with open(os.path.join(save_dir, run_id + ".json"), "w") as f:
        json.dump(result, f, indent=2)
    return result

def run_single_qrc(config, save_dir):
    if config.get("verbose", True):
        print_device_once()
    set_seed_qrc(config["seed"])
    env = TruckBackerEnv_D(render_mode=None)
    env.seed(config["seed"])
    env.action_space.seed(config["seed"])
    
    agent = QRCAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=config["learning_rate"],
        epsilon=config["epsilon_start"],
        epsilon_decay=config["epsilon_decay"],
        epsilon_min=config["epsilon_min"],
        batch_size=config["batch_size"],
        buffer_size=config["buffer_size"],
        gamma=config["gamma"],
        beta=config["beta"]
    )
    
    episode_rewards = []
    with tqdm(total=config["num_episodes"], desc=f"QRC seed {config['seed']}", leave=False, disable=not config.get("verbose", False)) as pbar:
        for episode in range(1, config["num_episodes"] + 1):
            env.seed(config["seed"] + episode)
            state = env.reset()
            total_reward = 0
    
            for t in range(config["max_steps_per_episode"]):
                action = agent.agent_policy(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.train_with_mem()
                state = next_state
                total_reward += reward
                if done:
                    break
            if config["target_update_freq"] is not None and episode % config["target_update_freq"] == 0:
                agent.update_target()
            episode_rewards.append(total_reward)
            print("seed: ", config["seed"], "Episode:", episode, "Reward:", total_reward)
            pbar.update(1)
        
    result = {
        "config": config,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
    }
    
    run_id = f"qrc_{config['seed']}_{int(config['learning_rate']*1e6)}_{config['batch_size']}_{'no_target_update' if config['target_update_freq'] is None else config['target_update_freq']}"
    torch.save({"rewards": torch.tensor(episode_rewards, dtype=torch.float32)}, 
               os.path.join(save_dir, f"{run_id}.pt"))
    with open(os.path.join(save_dir, f"{run_id}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "qrc"], required=True)
    parser.add_argument("--output", default="data/sweeps_random")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=30) # seeds per ... seed, I guess
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=None) # random
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    resolved_seed = args.seed if args.seed is not None else secrets.randbits(32)
    print(f"No seed specified, using seed: {resolved_seed}")
    
    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    random.seed(resolved_seed)
    
    if args.verbose:
        os.environ["SWEEP_VERBOSE"] = "1"
    
    os.makedirs(args.output, exist_ok=True)
    print(f"Saving to: {args.output}")
    
    base = {
        "num_episodes": args.episodes,
        "max_steps_per_episode": args.steps,
        "seed": 0,
        "verbose": args.verbose
    }
    
    dqn_space = {
        "learning_rate": (1e-4, 5e-3),
        "epsilon_start": [0.3, 0.5, 0.8, 1.0],
        "epsilon_min": [0.01, 0.05, 0.1],
        "epsilon_decay": [0.99990, 0.99997, 0.99999],
        "batch_size": [64, 128, 256],
        "buffer_size": [50000, 100000],
        "gamma": [0.95, 0.99],
        "target_update_freq": [5, 10, 20],
    }
    
    qrc_space = {
        "learning_rate": (1e-4, 5e-3),
        "epsilon_start": [0.3, 0.5, 0.8, 1.0],
        "epsilon_min": [0.01, 0.05, 0.1],
        "epsilon_decay": [0.99990, 0.99997, 0.99999],
        "batch_size": [64, 128, 256],
        "buffer_size": [50000, 100000],
        "gamma": [0.95, 0.99],
        "target_update_freq": [None, 5, 10, 20],
        "beta": (0.9, 1.0)
    }

    print("Hyperparameter space defined.")
    
    space = dqn_space if args.agent == "dqn" else qrc_space
    print(f"Using agent: {args.agent}")
    run_function = run_single_dqn if args.agent == "dqn" else run_single_qrc

    
    def make_config(i):
        # golden ratio xor
        # and then mask to 32 bit
        rng_seed = (resolved_seed ^ (i + 0x9E3779B9)) & 0xFFFFFFFF
        rng = random.Random(rng_seed)
        config = sample_config(base, space, rng)
        config["seed"] = i

        id_fields = {
            "learning_rate": config["learning_rate"],
            "epsilon_start": config["epsilon_start"],
            "epsilon_decay": config["epsilon_decay"],
            "epsilon_min": config["epsilon_min"],
            "batch_size": config["batch_size"],
            "buffer_size": config["buffer_size"],
            "gamma": config["gamma"],
            "target_update_freq": config["target_update_freq"],
        }
        
        if "beta" in config:
            id_fields["beta"] = config["beta"]

        config["config_id"] = "cfg_" + "_".join(f"{k}={id_fields[k]}" for k in sorted(id_fields.keys()))
        
        return config

    start = time.time()
    all_results = []
    max_in_flight = args.jobs * 2
    
    def iter_tasks():
        for i in range(args.seeds):
            config = make_config(i)
            for rep in range(args.replicates):
                run_config = dict(config)
                run_config["rep_index"] = rep
                run_config["seed"] = i + rep * args.seeds
                yield run_config
    
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        in_flight = set()
        task_iter = iter_tasks()
        
        try:
            while len(in_flight) < max_in_flight:
                in_flight.add(executor.submit(run_function, next(task_iter), args.output))
        except StopIteration:
            pass
        
        while in_flight:
            done = next(as_completed(in_flight))
            all_results.append(done.result())
            in_flight.remove(done)
            try:
                in_flight.add(executor.submit(run_function, next(task_iter), args.output))
            except StopIteration:
                pass
    
    summary_by_config = {}
    rep_config = {}
    for r in all_results:
        cfg = r.get("config", {})
        cid = cfg.get("config_id")
        summary_by_config.setdefault(cid, []).append(r["mean_reward"])
        rep_config.setdefault(cid, {k: v for k, v in cfg.items()
                                    if k in ("learning_rate","epsilon_start","epsilon_decay","epsilon_min",
                                             "batch_size","buffer_size","gamma","target_update_freq","beta")})

    mean_reward_per_config = {cid: float(np.mean(v)) for cid, v in summary_by_config.items()}
    best_cid = max(mean_reward_per_config, key=mean_reward_per_config.get) if mean_reward_per_config else None
    best_mean_reward = mean_reward_per_config.get(best_cid, None)
    best_hyperparams = rep_config.get(best_cid, None)
    
    summary = {
        "agent": args.agent,
        "count": len(all_results),
        "replicates": args.replicates,
        "replicates_per_config": {cid: len(v) for cid, v in summary_by_config.items()},
        "mean_reward_per_config": mean_reward_per_config,
        "best_config_id": best_cid,
        "best_mean_reward": best_mean_reward,
        "best_hyperparameters": best_hyperparams,
        "results": all_results,
        "elapsed_time": time.time() - start,
    }
    
    summary_path = os.path.join(args.output, f"{args.agent}_random_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved summary to {summary_path}")
    
if __name__ == "__main__":
    main()