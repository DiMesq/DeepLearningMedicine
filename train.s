#!/bin/bash
#
#SBATCH --output=slurm_train_%j.out
#SBATCH --job-name=DLM_proj
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-03:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:p40:1

module load python3/intel/3.6.3

source /home/dam740/pytorch_venv/bin/activate

python main.py -m densenet161 train
# python main.py -m resnet50 --test-run train
