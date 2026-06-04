#!/bin/bash
set -e

# Step 1: Create the conda environment
conda env create -f environment.yml
conda activate nemo_moe

# Step 2: Clone and install NeMo fork
git clone https://github.com/a-sasin/NeMo
cd NeMo
pip install -e ".[asr]"
cd ..

echo "Done. Activate with: conda activate nemo_moe"