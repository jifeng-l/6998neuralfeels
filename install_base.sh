#!/bin/bash -e

echo "Installing neuralfeels dependencies into the current environment (base)..."

# Check if inside base environment
if [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    echo "⚠️ Warning: You're not in the 'base' environment. Current env: $CONDA_DEFAULT_ENV"
    echo "Continue anyway? (y/n)"
    read -r answer
    if [[ "$answer" != "y" ]]; then
        echo "Aborting."
        exit 1
    fi
fi

unset PYTHONPATH LD_LIBRARY_PATH

# Upgrade pip
python -m pip install --upgrade pip

# (Optional) Uninstall torch-related packages if needed
# pip uninstall torch torchvision functorch tinycudann -y

# You mentioned torch is already installed, so we skip installing torch

# Ensure CUDA toolkit (via conda-forge)
conda install -y -c conda-forge cudatoolkit=11.8

# Check torch + CUDA
python -c "import torch; assert torch.cuda.is_available(); print('CUDA is available via torch ✅')"

# Check nvcc
if command -v nvcc &>/dev/null; then
    echo "nvcc is installed and working ✅"
else
    echo "nvcc is not installed or not in PATH ❌"
    exit 1
fi

# Install project dependencies
pip install ninja \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \
    git+https://github.com/facebookresearch/segment-anything.git \
    git+https://github.com/suddhu/tacto.git@master

# Theseus-related dependencies
conda install -y -c conda-forge suitesparse
pip install theseus-ai

# Install local package (neuralfeels)
pip install -e .

# Make run script executable if it exists
if [ -f scripts/run ]; then
    chmod +x scripts/run
    echo "scripts/run is now executable ✅"
else
    echo "scripts/run not found ⚠️"
fi

echo "✅ Installation complete in base environment!"