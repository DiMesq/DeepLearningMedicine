#!/bin/bash
#
#SBATCH --output=slurm_eval_%j.out
#SBATCH --job-name=DLM_proj
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1

module load python3/intel/3.6.3

source /home/dam740/pytorch_venv/bin/activate

line_number=$SLURM_ARRAY_TASK_ID
params=$(sed -n ${line_number}p to_eval.txt)
read -r model_name run_id <<< $params

python main.py -m $model_name evaluate --run-id $run_id
