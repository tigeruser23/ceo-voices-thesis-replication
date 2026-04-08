#!/bin/bash
# slurm_finbert.sh
# Submit: sbatch slurm_finbert.sh
#SBATCH --job-name=finbert
#SBATCH --output=/scratch/network/%u/thesis_week1/logs/finbert_%j.out
#SBATCH --error=/scratch/network/%u/thesis_week1/logs/finbert_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

mkdir -p /scratch/network/$USER/thesis_week1/logs

# Cache HuggingFace model to scratch (avoids home-dir quota issues)
export HF_HOME=/scratch/network/$USER/huggingface_cache
export TRANSFORMERS_CACHE=/scratch/network/$USER/huggingface_cache

source /scratch/network/$USER/thesis_env_scratch/bin/activate

python /scratch/network/$USER/thesis_week1/code/28_run_finbert.py
