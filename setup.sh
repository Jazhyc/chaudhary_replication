#!/bin/bash
set -e
pip3 install torch transformers scikit-learn pandas tqdm
git clone https://github.com/LRudL/sad.git sad_repo
ln -s sad_repo/sad/stages sad_data
echo "Setup complete."
