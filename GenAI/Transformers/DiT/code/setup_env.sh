#!/bin/bash

# 1. Fresh start: Create the venv
echo "Creating venv 'transformer' with Python 3.11..."
rm -rf transformer  # Clean up the failed attempt
uv venv transformer --python 3.11

# 2. Activate
source transformer/bin/activate

# 3. Install the 'Golden Combo'
echo "Installing compatible AI stack..."
uv pip install \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "torchmetrics==1.6.1" \
    "diffusers==0.33.1" \
    "transformers==4.51.3" \
    "lightning" \
    "pandas" \
    "matplotlib" \
    "portalocker>=2.0.0" \
    "ipykernel"


# uv pip install torchtext==0.17.2

uv pip install accelerate
uv pip install datasets
uv pip install setuptools
uv pip install triton


# 4. Register the kernel for Jupyter
python -m ipykernel install --user --name=transformer --display-name "Python 3.11 (Transformer)"

echo "------------------------------------------------"
echo "Success! Versions are now aligned."
echo "Run: source transformer/bin/activate"
echo "------------------------------------------------"
