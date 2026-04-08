#!/bin/bash
# slurm_sync_all.sh
# Submit: sbatch slurm_sync_all.sh
#SBATCH --job-name=sync_all
#SBATCH --output=/scratch/network/%u/thesis_week1/logs/sync_%j.out
#SBATCH --error=/scratch/network/%u/thesis_week1/logs/sync_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

mkdir -p /scratch/network/$USER/thesis_week1/logs

source /scratch/network/$USER/thesis_env_scratch/bin/activate

python /scratch/network/$USER/thesis_week1/code/27_sync_all.py
