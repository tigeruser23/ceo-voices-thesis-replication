#!/bin/bash
# slurm_audio_features.sh
# Submit: sbatch slurm_audio_features.sh
#SBATCH --job-name=audio_features
#SBATCH --output=/scratch/network/%u/thesis_week1/logs/audio_%j.out
#SBATCH --error=/scratch/network/%u/thesis_week1/logs/audio_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

mkdir -p /scratch/network/$USER/thesis_week1/logs

# FIX: use thesis_env_scratch (not ~/thesis_env)
source /scratch/network/$USER/thesis_env_scratch/bin/activate

python /scratch/network/$USER/thesis_week1/code/26_extract_all_audio.py
