#!/bin/bash
set -e

# Step 1: Clone and install NeMo fork
git clone https://github.com/a-sasin/NeMo
cd NeMo
pip install -e ".[asr]"
cd ..

# Step 2: Create the conda environment
conda env create -f environment.yml

echo "Done. Activate with: conda activate nemo_moe"