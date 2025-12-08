#!/bin/bash
#SBATCH --job-name=qrc-no-tnu
#SBATCH --account=def-cepp
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --array=0-251
#SBATCH --output=logs/result_no_tnu/output_%A_%a.log
#SBATCH --mail-user=rany@ualberta.ca
#SBATCH --mail-type=BEGIN,END

echo "starting job"

nvidia-smi

module load python/3.11.5
module load cuda/12.6
source cc/bin/activate

echo "Running seed ${SLURM_ARRAY_TASK_ID}"
python CC_QRC/qrc.py ${SLURM_ARRAY_TASK_ID}
echo "Finished seed ${SLURM_ARRAY_TASK_ID}"