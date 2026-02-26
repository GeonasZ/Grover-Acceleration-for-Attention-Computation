# Grover Acceleration for Attention Computation

This repository explores applying **Grover search** to accelerate the attention mechanism in transformer models. We implement a **Quantum Vision Transformer (QVIT)** that uses Grover search to approximate attention weight computation, and compare it with a standard **Vision Transformer (ViT)** on the MNIST dataset.

## Features

- **ViT**: Standard Vision Transformer with patch embedding and self-attention.
- **QVIT**: Quantum-augmented ViT with Grover-search-based attention approximation (optionally with filtering).
- **Patch tokenizers**:
  - **PatchTokenizerCNN** (`feature_extraction.py`): CNN-based patch embedding (hybrid pipeline; can be pretrained with `pretrain_patch_tokenizer`).
  - **SimplePatchTokenizer** (`simple_tokenizer.py`): Flatten each patch to pixel×channel and project with a single linear layer; drop-in replacement, no pretrained tokenizer.
- **Evaluation**: Shared evaluation for ViT and QVIT (accuracy, etc.) on MNIST.

## Requirements & Installation

- **Python**: ≥ 3.13  
- **Dependencies**: See `pyproject.toml`. Main ones: `torch`, `torchvision`, `qiskit`.  
- **Package manager**: [uv](https://github.com/astral-sh/uv) is recommended.

```bash
# With uv (recommended)
uv sync

# Or with pip (after creating a venv)
pip install -e .
```

Optional: use the PyTorch CUDA index (see `pyproject.toml` and `[[tool.uv.index]]`) for GPU support.

## Usage

Run training and evaluation from the project root:

```bash
python entrance.py
```

This will:

1. Load MNIST via `get_mnist_dataloaders`.
2. Train ViT with the chosen patch tokenizer.
3. Optionally train QVIT (with Grover and filtering) with the tokenizer frozen.
4. Evaluate both models and print metrics.

Configuration is controlled by `TrainConfig` in `entrance.py` (e.g. `batch_size`, `epochs`, `device`, `train_qvit`, `qvit_use_grover`, `qvit_enable_filter`, `qvit_filter_start_epoch`). The default tokenizer can be switched between `PatchTokenizerCNN` and `SimplePatchTokenizer` by changing the import and instantiation in `entrance.py`.

## Project Structure

```
.
├── entrance.py              # Main entry: training and evaluation for ViT / QVIT
├── pyproject.toml            # Project config and dependencies
├── uv.lock                   # Lock file (use uv for reproducible installs)
├── README.md
├── data/                     # Dataset directory (e.g. MNIST, created on first run)
│
└── qvit_test/                # Main package
    ├── __init__.py           # Exports: PatchConfig, PatchTokenizerCNN, SimplePatchTokenizer,
    │                         #          get_mnist_dataloaders, ViT, ViTConfig, QVIT,
    │                         #          evaluate_qvit, evaluate_vit
    ├── feature_extraction.py # PatchConfig, PatchTokenizerCNN, get_mnist_dataloaders,
    │                         # PatchEmbeddingClassifier, pretrain_patch_tokenizer
    ├── simple_tokenizer.py   # SimplePatchTokenizer (linear patch embedding, same interface as PatchTokenizerCNN)
    ├── vit.py                # ViT, ViTConfig (standard Vision Transformer)
    ├── qvit.py               # QVIT (quantum ViT with Grover-based attention)
    ├── qiskit_grover.py      # Grover search via Qiskit (used internally by qvit.py)
    └── evaluation.py         # evaluate_vit, evaluate_qvit
```

## TODOs

- Implement percentile record during training. Use the average of percentiles during training as the percentile when during prediction. (Solution for looking for an appropriate percentile threshold)
- Implement the local connection mask in parallel with Grover search.
- For report, discuss dequantization, and mention that QRAM is theoretically feasible, and maybe more.

## References


**Algorithms**
- [Grover Search for Acceleration of Attention Computation](https://arxiv.org/abs/arXiv:2307.08045) (arXiv:2307.08045)
- [Sublinear Time Quantum Algorithm for Attention Approximation](https://arxiv.org/abs/2602.00874) (arXiv:2602.00874)

**QRAM**

- [Quantum Random Access Memory](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.160501)

**Dequantization**
- [Robust Dequantization of the Quantum Singular value Transformation and Quantum Machine Learning Algorithms](https://arxiv.org/abs/2304.04932)
- [Robust Dequantization of the Quantum Singular value Transformation and Quantum Machine Learning Algorithms](https://arxiv.org/abs/1811.00414)

