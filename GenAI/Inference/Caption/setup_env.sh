#!/bin/bash

# 1. Fresh start: Create the venv
echo "Creating venv 'caption' with Python 3.11..."
rm -rf caption  # Clean up the failed attempt
uv venv caption --python 3.11

# 2. Activate
source caption/bin/activate

# 3. Install the 'Golden Combo'
echo "Installing compatible AI stack..."

uv pip install git+https://github.com/huggingface/transformers.git
# uv pip install torch num2words torchvision datasets accelerate torchmetrics Pillow==9.4.0

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install num2words datasets accelerate torchmetrics Pillow==9.4.0

# 3. CRITICAL: Clean the build cache
# This prevents the installer from reusing the "broken" binary it just made
rm -rf build/
rm -rf ~/.cache/pip
rm -rf ~/.cache/uv

uv pip install ipykernel
uv pip install --upgrade huggingface_hub
uv pip install --upgrade pip setuptools wheel

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

nvcc --version

uv pip install flash-attn --no-build-isolation

# uv pip install "flash-attn==2.8.3" -f https://pytorch-geometric.com/whl/

# 4. Register the kernel for Jupyter
python -m ipykernel install --user --name=caption --display-name "Python 3.11 (caption)"

echo "------------------------------------------------"
echo "Success! Versions are now aligned."
echo "Run: source transformer/bin/activate"
echo "------------------------------------------------"
