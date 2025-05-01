#!/bin/bash -e

usage="$(basename "$0") [-h] [-e ENV_NAME] --
Install the neuralfeels environment
where:
    -h  show this help text
    -e  name of the environment, default=_neuralfeels
"

options=':he:'
while getopts $options option; do
    case "$option" in
    h)
        echo "$usage"
        exit
        ;;
    e) ENV_NAME=$OPTARG ;;
    :)
        printf "missing argument for -%s\n" "$OPTARG" >&2
        echo "$usage" >&2
        exit 1
        ;;
    \?)
        printf "illegal option: -%s\n" "$OPTARG" >&2
        echo "$usage" >&2
        exit 1
        ;;
    esac
done

# if ENV_NAME is not set, then set it to _neuralfeels
if [ -z "$ENV_NAME" ]; then
    ENV_NAME=_neuralfeels
fi

echo "Environment Name: $ENV_NAME"

unset PYTHONPATH LD_LIBRARY_PATH

# Remove existing conda environment if exists
conda deactivate || true
conda remove -y --name $ENV_NAME --all || true

# Create new environment from environment.yml
conda env create -n $ENV_NAME -f environment.yml
conda activate $ENV_NAME

# Upgrade pip and install required packages
python -m pip install --upgrade pip

# Remove pre-installed torch-related packages
pip uninstall torch torchvision functorch tinycudann -y

# Install specific versions of torch and torchvision with CUDA 11.8 support
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Ensure CUDA toolkit is installed (via conda-forge)
conda install -c conda-forge cudatoolkit=11.8

# Check torch + CUDA is working
python -c "import torch; assert torch.cuda.is_available()"

# Check if nvcc is available
if command -v nvcc &>/dev/null; then
    echo "nvcc is installed and working."
else
    echo "nvcc is not installed or not in PATH."
    exit 1
fi

# Install tinycudann and other required packages
pip install ninja \
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \
    git+https://github.com/facebookresearch/segment-anything.git \
    git+https://github.com/suddhu/tacto.git@master

# Install theseus and its dependencies
conda install -y -c conda-forge suitesparse
pip install theseus-ai

# Install neuralfeels package (editable mode)
pip install -e .

# Make the entrypoint executable
chmod +x scripts/run