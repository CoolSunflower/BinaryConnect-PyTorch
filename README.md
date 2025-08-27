# BinaryConnect in PyTorch

BinaryConnect is a foundational method for training neural networks with binary (±1) weights during the forward and backward passes. This repository implements the full BinaryConnect workflow with PyTorch on MNIST and CIFAR-10. It also supports customizable hyperparameters, systematic logging, and analysis scripts.

This repository is a reimplementation of the original paper: https://arxiv.org/pdf/1511.00363 (BinaryConnect algorithm) in PyTorch.

---

## Features

- **End-to-End BinaryConnect:** Implements stochastically binarized weight updates while retaining full-precision accumulators.
- **Dataset Support:** Ready-to-go experiments on MNIST and CIFAR-10 datasets.
- **Configurable Hyperparameters:** Adjust learning rates, optimizers, batch sizes, and network architecture in a single location.
- **Logging & Reproducibility:** Training statistics, validation accuracy, losses, and timing are logged to enable reproduction and tuning.
- **Visualization Scripts:** Routines for plotting error, loss, and timing curves with log parsing.
- **Clean, Modular Codebase:** Clear separation between model definition, data processing, training logic, and analysis utilities.
- **Jupyter Notebook Integration:** main_test.ipynb for interactive, cell-based experimentation and demonstration.

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
├── mnist_standard.py            # Standard PyTorch MNIST model for comparision
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
   pip install torch numpy matplotlib # install with cuda if required, see guide for this on PyTorch website
   # Optionally
   pip install jupyter
   ```

---

## Quick Start

### Train on MNIST

```sh
python mnist.py
```
Post training, save the logs generated in the analysis folder as `mnist_logs.txt` and change the dataset name in the analysis.py file, then run `python analysis.py` to generate all the visualisations.

### Train on CIFAR-10

```sh
python cifar10.py
```
Post training, save the logs generated in the analysis folder as `cifar10_logs.txt` and change the dataset name in the analysis.py file, then run `python analysis.py` to generate all the visualisations.

### Train Standard (Non-Binary) MNIST Model

```sh
python mnist_standard.py
```
Post training, save the logs generated in the analysis folder as `mnist_standard_logs.txt` and change the dataset name in the analysis.py file, then run `python analysis.py` to generate all the visualisations.

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

## Contributing

We welcome contributions! Please fork the repository, create a feature branch, and submit a pull request. For major changes, open an issue first to discuss your proposal.

---

