import os
import json
import argparse
import numpy as np

def is_run_json(fname):
    if not fname.endswith(".json"):
        return False
    base = os.path.basename(fname)
    if base.endswith("_summary.json"):
        return False
    return base.startswith("dqn_") or base.startswith("qrc_")

def detect_agent(files):
    agents = set()
    for f in files:
        b = os.path.basename(f)
        if b.startswith("dqn_"):
            agents.add("dqn")
        elif b.startswith("qrc_"):
            agents.add("qrc")
    if len(agents) == 1:
        return agents.pop()
    elif len(agents) == 0:
        return "unknown"
    else:
        return "mixed"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory containing run JSON files")
    parser.add_argument("--out", default=None, help="Optional output summary path")
    parser.add_argument("--min-replicates", type=int, default=30, help="Minimum number of replicates per config (default 30)")
    args = parser.parse_args()

    min_reps = max(0, args.min_replicates)

    run_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if is_run_json(f)]
    if not run_files:
        print("No run JSON files found.")
        return

    all_results = []
    for path in run_files:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if "config" in data and "mean_reward" in data:
                all_results.append(data)
        except Exception as e:
            print(f"Skipping '{path}' due to error: {e}")

    if not all_results:
        print("No valid results found.")
        return

    summary_by_config = {}
    rep_config = {}
    results_by_config = {}
    for r in all_results:
        config = r.get("config", {})
        cid = config.get("config_id", None)
        if cid is None:
            hp_keys = ["learning_rate", "epsilon_start", "epsilon_decay", "epsilon_min",
                       "batch_size", "buffer_size", "gamma", "target_update_freq", "beta"]
            id_fields = [f"{k}={config[k]}" for k in hp_keys if k in config]
            cid = "cfg_" + "_".join(sorted(id_fields)) if id_fields else "cfg_unknown"
        summary_by_config.setdefault(cid, []).append(r["mean_reward"])
        results_by_config.setdefault(cid, []).append(r)
        rep_config.setdefault(cid, {k: v for k, v in config.items()
                                    if k in ("learning_rate","epsilon_start","epsilon_decay","epsilon_min",
                                             "batch_size","buffer_size","gamma","target_update_freq","beta")})

    kept_configs = {cid for cid, vals in summary_by_config.items() if len(vals) >= min_reps}
    if not kept_configs:
        if min_reps == 0:
            print("No valid results found.")
        else:
            print(f"No configs with at least {min_reps} replicates found. Try a lower --min-replicates value.")
        return

    filtered_summary = {cid: summary_by_config[cid] for cid in kept_configs}
    filtered_results_dict = {cid: results_by_config[cid] for cid in kept_configs}
    filtered_rep_config = {cid: rep_config.get(cid) for cid in kept_configs}

    mean_reward_per_config = {cid: float(np.mean(v)) for cid, v in filtered_summary.items()}
    best_cid = max(mean_reward_per_config, key=mean_reward_per_config.get) if mean_reward_per_config else None
    best_mean_reward = mean_reward_per_config.get(best_cid, None)
    best_hyperparams = filtered_rep_config.get(best_cid, None)

    filtered_results = [r for cid in filtered_results_dict for r in filtered_results_dict[cid]]
    agent = detect_agent(run_files)
    summary = {
        "agent": agent,
        "count": len(filtered_results),
        "replicates_per_config": {cid: len(v) for cid, v in filtered_summary.items()},
        "mean_reward_per_config": mean_reward_per_config,
        "best_config_id": best_cid,
        "best_mean_reward": best_mean_reward,
        "best_hyperparams": best_hyperparams,
        "results": filtered_results,
        "min_replicates_used": min_reps
    }

    out_path = args.out or os.path.join(args.dir, f"{agent}_recalculated_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to '{out_path}'")

if __name__ == "__main__":
    main()