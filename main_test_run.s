#!/bin/bash
#
#SBATCH --output=slurm_train_%j.out
#SBATCH --job-name=DLM_proj
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

module load python3/intel/3.6.3

source /home/dam740/pytorch_venv/bin/activate

python main.py train -m resnet50 --test-run --max-stale 100 #--negative-only
