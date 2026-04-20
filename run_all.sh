#!/bin/bash
#SBATCH --job-name=safety_exp
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

FORCE=${FORCE:+--force}

# Gemma 3 anchor (paper validation)
python3 run.py --model google/gemma-3-270m-it $FORCE
python3 run.py --model google/gemma-3-1b-it $FORCE
python3 run.py --model google/gemma-3-4b-it $FORCE
python3 run.py --model google/gemma-3-12b-it $FORCE
python3 run.py --model google/gemma-3-27b-it $FORCE

# Qwen3.5
python3 run.py --model Qwen/Qwen3.5-0.8B $FORCE
python3 run.py --model Qwen/Qwen3.5-2B $FORCE
python3 run.py --model Qwen/Qwen3.5-4B $FORCE
python3 run.py --model Qwen/Qwen3.5-9B $FORCE
python3 run.py --model Qwen/Qwen3.5-27B $FORCE

# Gemma 4
python3 run.py --model google/gemma-4-E2B-it $FORCE
python3 run.py --model google/gemma-4-E4B-it $FORCE
python3 run.py --model google/gemma-4-31B-it $FORCE

echo "ALL DONE. Results in results/results.csv"
