#!/bin/bash
#
#SBATCH --output=slurm_train_%j.out
#SBATCH --job-name=DLM_proj
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1

module load python3/intel/3.6.3

#source /home/dam740/pytorch_venv/bin/activate
source /scratch/dam740/DLM/pytorch/bin/activate

python main.py -m basic train --augmentation-level low
# python main.py -m resnet50 --test-run train
