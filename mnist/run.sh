#!/bin/bash
#!/usr/bin/env python

#SBATCH --job-name=seal
#SBATCH --output="%j-mnist.out"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-2:00:00
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8

source /home/jylab_intern001/.bashrc
eval "$(conda shell.bash hook)"
conda activate seal_ad

srun python main.py