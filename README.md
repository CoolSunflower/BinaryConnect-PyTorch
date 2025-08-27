# BinaryConnect in PyTorch

BinaryConnect-PyTorch is a comprehensive, well-structured PyTorch implementation of the BinaryConnect algorithm, facilitating efficient training of neural networks with binary weights. Developed for research and experimentation, it is ideal for benchmarking and studying discrete-weight deep learning on datasets like MNIST and CIFAR-10. This repository provides modular training, evaluation, logging, and analysis routines, with a focus on reproducibility, flexibility, and educational clarity.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Background](#background)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Traning & Evaluation](#analysis--visualization)
- [Analysis & Visualization](#analysis--visualization)
- [Results](#results)
- [Contributing](#contributing)

---

## Overview

BinaryConnect is a foundational method (Courbariaux et al., 2015) for training neural networks with binary (±1) weights during the forward and backward passes. This dramatically reduces the memory, bandwidth, and computational footprint of deep networks, especially useful for deployment on resource-constrained hardware.

This repository implements the full BinaryConnect workflow with PyTorch on MNIST and CIFAR-10. It also supports customizable hyperparameters, systematic logging, and insightful analysis scripts, making it suitable for both research and production prototyping.

---

## Features

- **End-to-End BinaryConnect:** Implements stochastically binarized weight updates while retaining full-precision accumulators.
- **Dataset Support:** Ready-to-go experiments on MNIST and CIFAR-10 datasets.
- **Configurable Hyperparameters:** Adjust learning rates, optimizers, batch sizes, and network architecture in a single location.
- **Logging & Reproducibility:** Training statistics, validation accuracy, losses, and timing are logged to enable reproduction and tuning.
- **Automated Visualization:** Routines for plotting error, loss, and timing curves with built-in log parsing.
- **Clean, Modular Codebase:** Clear separation between model definition, data processing, training logic, and analysis utilities.
- **Jupyter Notebook Integration:** main_test.ipynb for interactive, cell-based experimentation and demonstration.

---

## Background

BinaryConnect is based on the insight that binarizing weights during forward and backward passes can substantially reduce the complexity of deep networks without significant loss in accuracy—when real-valued weights are kept for the parameter updates. This technique led to further advances in binarized neural networks (BNNs) and quantization-aware training.

References:

- Courbariaux, M., Bengio, Y., & David, J. (2015). BinaryConnect: Training Deep Neural Networks with binary weights during propagations. Advances in Neural Information Processing Systems, 28.
- https://arxiv.org/pdf/1511.00363

---

## Architecture

- **Model:** Defined in `model.py`, supports binary and conventional linear/convolutional layers. BinaryConnect binarizes weights using sign() except during optimization.
- **Dataset Scripts:** `mnist.py` and `cifar10.py` provide dataset-specific training and evaluation routines using PyTorch datasets and dataloaders.
- **Hyperparameter Management:** Centralized in `hyperparams.py` for easy tuning.
- **Utilities:** Data loading, seed setting, preprocessing, and logging in `utils.py`.
- **Analysis:** The `analysis` directory contains scripts and plots for easy benchmarking and comparison.

---

## Project Structure

```
BinaryConnect-PyTorch/
├── .gitignore
├── cifar10.py                   # CIFAR-10 training/evaluation script
├── hyperparams.py               # Centralized hyperparameter configuration
├── main_test.ipynb              # Jupyter notebook for interactive experiments
├── mnist.py                     # MNIST training/evaluation script
├── model.py                     # BinaryConnect model definitions
├── README.md                    # Project documentation
├── mnist_standard.py            
├── utils.py                     # Utility functions (data loading, logging, etc.)
└── analysis/
    ├── analysis.py              # Log parsing & plotting scripts
    ├── mnist_error_rate_vs_epoch.png
    ├── mnist_logs.txt
    ├── mnist_loss_vs_epoch.png
    ├── mnist_time_vs_epoch.png
    ├── mnist_standard_error_rate_vs_epoch.png
    ├── mnist_standard_logs.txt
    ├── mnist_standard_loss_vs_epoch.png
    ├── mnist_standard_time_vs_epoch.png
    ├── cifar10_error_rate_vs_epoch.png
    ├── cifar10_logs.txt
    ├── cifar10_loss_vs_epoch.png
    └── cifar10_time_vs_epoch.png
```

---

## Installation

### Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- NumPy
- Matplotlib

### Setup

1. **Clone the repository**
   ```sh
   git clone https://github.com/CoolSunflower/BinaryConnect-PyTorch.git
   cd BinaryConnect-PyTorch
   ```

2. **Install dependencies**
   ```sh
   pip install torch numpy matplotlib
   # Optionally
   pip install jupyter
   ```

---

## Quick Start

### Train on MNIST

```sh
python mnist.py
```
- Logs saved to `analysis/mnist_logs.txt`.

### Train on CIFAR-10

```sh
python cifar10.py
```
- Logs saved to `analysis/cifar10_logs.txt`.

### Train Standard (Non-Binary) MNIST Model

```sh
python mnist_standard.py
```
- Logs saved to `analysis/mnist_standard_logs.txt`.

### Interactive Experimentation

Open `main_test.ipynb` in Jupyter Notebook for step-by-step experimentation.

---

## Configuration

Hyperparameters are managed in [`hyperparams.py`](hyperparams.py):

- `learning_rate:` Learning rate for optimizer (float)
- `batch_size:` Batch size for training (int)
- `epochs:` Number of epochs (int)
- `optimizer:` Optimizer type (SGD/Adam/Other)
- `model architecture:` Network layer settings and depth
- `seed:` Random seed for reproducibility

Edit `hyperparams.py` before running scripts, e.g., set batch_size=256 or change epochs=50.

---

## Training & Evaluation

Each training script (e.g., `mnist.py`) sets up the data, model, optimizer, and training loop. Models are trained using binary weights during propagation, with checkpoints and validation at each epoch.

- **Preprocessing:** Standardize data to zero mean/unit variance.
- **Augmentation:** (CIFAR-10) Includes random cropping and flipping.
- **Binarization:** Weights are stochastically binarized before fwd/bwd, but updates are computed on full-precision parameters.
- **Logging:** At each epoch, train/validation loss, error, and timing are recorded.
- **Checkpoints:** Optionally, add model checkpointing with torch.save() for large experiments.

---

## Analysis & Visualization

Generate performance plots from logs:

```sh
cd analysis
python analysis.py
```

---

## Results

Sample results and plots are available in the `analysis/` directory, including:
- Training time per epoch
- Validation error rate vs epoch
- Training and validation loss vs epoch

---

## Contributing

We welcome contributions! Please fork the repository, create a feature branch, and submit a pull request. For major changes, open an issue first to discuss your proposal.

---

