#!/bin/bash
# slurm_financial_controls.sh
# Submit: sbatch slurm_financial_controls.sh
#SBATCH --job-name=fin_controls
#SBATCH --output=/scratch/network/%u/thesis_week1/logs/finctrl_%j.out
#SBATCH --error=/scratch/network/%u/thesis_week1/logs/finctrl_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

mkdir -p /scratch/network/$USER/thesis_week1/logs

# Ensure WRDS pgpass file has correct permissions
export PGPASSFILE=/home/oy3009/.pgpass
chmod 600 /home/oy3009/.pgpass

source /scratch/network/$USER/thesis_env_scratch/bin/activate

python /scratch/network/$USER/thesis_week1/code/29_financial_controls.py
