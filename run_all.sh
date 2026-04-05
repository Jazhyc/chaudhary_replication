#!/bin/bash
set -e

# Gemma 3 anchor (paper validation)
python3 run.py --model google/gemma-3-270m-it
python3 run.py --model google/gemma-3-1b-it
python3 run.py --model google/gemma-3-4b-it
python3 run.py --model google/gemma-3-12b-it
python3 run.py --model google/gemma-3-27b-it

# Qwen3.5
python3 run.py --model Qwen/Qwen3.5-0.8B
python3 run.py --model Qwen/Qwen3.5-2B
python3 run.py --model Qwen/Qwen3.5-4B
python3 run.py --model Qwen/Qwen3.5-9B
python3 run.py --model Qwen/Qwen3.5-27B

# Gemma 4
python3 run.py --model google/gemma-4-E2B-it
python3 run.py --model google/gemma-4-E4B-it
python3 run.py --model google/gemma-4-31B-it

echo "ALL DONE. Results in results/results.csv"
