# scvi-tools Installation Guide

## Quick Install

scvi-tools can be installed via `conda` or `pip`. We recommend installing into a fresh virtual environment to avoid conflicts with other packages and compatibility issues.

### Basic CPU Version

For the basic CPU version run:

```bash
pip install -U scvi-tools
```

or

```bash
conda install scvi-tools -c conda-forge
```

### GPU Support

#### Linux Systems (Ubuntu, RedHat, etc.) with Nvidia GPU CUDA support:

```bash
pip install -U scvi-tools[cuda]
```

#### Apple Silicon (MPS) support:

```bash
pip install -U scvi-tools[metal]
```

---

## Prerequisites

### Virtual Environment

A virtual environment can be created with either `conda` or `venv`. We recommend using a fresh `conda` environment. We currently support **Python 3.11 - 3.13**.

#### Option 1: Using Conda (Recommended)

For `conda`, we recommend using the [Miniforge](https://github.com/conda-forge/miniforge) or [Mamba](https://mamba.readthedocs.io/) distribution, which are generally lighter & faster than the official distribution and comes with conda-forge as the default channel (where scvi-tools is hosted).

```bash
conda create -n scvi-env python=3.13  # any python 3.11 to 3.13
conda activate scvi-env
```

#### Option 2: Using venv

For `venv`, we recommend using [uv](https://github.com/astral-sh/uv), which is a high-performance Python package manager and installer written in Rust.

```bash
pip install -U uv
uv venv .scvi-env
source .scvi-env/bin/activate  # for macOS and Linux
.scvi-env\Scripts\activate  # for Windows
```

---

## GPU Support with PyTorch and JAX

scvi-tools depends on PyTorch for accelerated computing (and optionally on JAX). 

### If You Don't Plan on Using GPU

If you don't plan on using an accelerated device, we recommend installing scvi-tools directly and letting these dependencies be installed automatically by your package manager of choice.

### If You Plan on Using GPU (Recommended)

If you plan on taking advantage of an accelerated device (e.g., Nvidia GPU or Apple Silicon), which is likely, scvi-tools supports it and you should install with the GPU support dependency of scvi-tools.

### Special Cases: Older GPU Hardware

However, there might be cases where the GPU hardware does not support the latest installation of PyTorch and JAX. In this case, we recommend installing PyTorch and JAX **before** installing scvi-tools. 

Please follow the respective installation instructions for:
- [PyTorch](https://pytorch.org/get-started/locally/) compatible with your system and device type
- [JAX](https://jax.readthedocs.io/en/latest/installation.html) compatible with your system and device type

---

## Installation Steps Summary

1. **Create and activate a virtual environment** (conda or venv)
2. **(Optional) Install PyTorch/JAX** if you have older GPU hardware
3. **Install scvi-tools** with appropriate options for your system

### Example: Linux with Nvidia GPU

```bash
# Step 1: Create environment
conda create -n scvi-env python=3.13
conda activate scvi-env

# Step 2: Install scvi-tools with CUDA support
pip install -U scvi-tools[cuda]
```

### Example: Apple Silicon Mac

```bash
# Step 1: Create environment
conda create -n scvi-env python=3.13
conda activate scvi-env

# Step 2: Install scvi-tools with Metal support
pip install -U scvi-tools[metal]
```

### Example: CPU Only

```bash
# Step 1: Create environment
conda create -n scvi-env python=3.13
conda activate scvi-env

# Step 2: Install scvi-tools
pip install -U scvi-tools
```

---

## Troubleshooting

### Virtual Environment Issues

Don't know how to get started with virtual environments or `conda`/`pip`? Check out:
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)
- [Python venv Documentation](https://docs.python.org/3/library/venv.html)

### GPU Compatibility Issues

If you encounter GPU-related errors:
1. Check your CUDA version: `nvcc --version` or `nvidia-smi`
2. Ensure PyTorch is properly installed with GPU support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Install compatible PyTorch version before scvi-tools if needed

---

## Additional Resources

- [scvi-tools Documentation](https://docs.scvi-tools.org/)
- [scvi-tools GitHub Repository](https://github.com/scverse/scvi-tools)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [JAX Installation Guide](https://jax.readthedocs.io/en/latest/installation.html)

---

## Support

If you encounter any issues during installation, please:
1. Check the [scvi-tools documentation](https://docs.scvi-tools.org/)
2. Search or post in [GitHub Issues](https://github.com/scverse/scvi-tools/issues)
3. Join the [scvi-tools community discussions](https://discourse.scverse.org/)
