# CMPUT655-Project

Setup
```bash
conda create -n rl
conda activate rl
pip3 install -r requirements.txt
conda install jupyter
jupyter notebook
```

Hyperparameter Sweep
```bash
python3 CC_Sweep/sweep.py --agent qrc --seeds 9999 --jobs 12 --output 'data/qrc_sweep_results'
python3 CC_Sweep/sweep.py --agent dqn --seeds 9999 --jobs 12 --output 'data/dqn_sweep_results'
```

Hyperparameter Sweep Analysis
```bash
python3 ./CC_Sweep/recalc_summary.py --dir ./data/dqn_sweep_results/
python3 ./CC_Sweep/recalc_summary.py --dir ./data/qrc_sweep_results/
```