#!/bin/bash
#SBATCH --job-name=qrc_hyperparam_adam
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=output_qrc_change.log
#SBATCH --mail-user=rany@ualberta.ca
#SBATCH --mail-type=BEGIN,END

echo "starting job"

module load python/3.11.5
module load cuda/12.6

source cc/bin/activate
echo "starting python"
python CC_Sweep/sweep.py --agent qrc --seeds 9999 --jobs 12 --output 'data/qrc_sweep_results'

echo "finished"
