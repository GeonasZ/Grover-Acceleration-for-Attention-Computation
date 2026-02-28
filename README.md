# Grover Acceleration for Attention Computation

This repository explores applying **Grover search** to accelerate the attention mechanism in transformer models. We implement a **Quantum Vision Transformer (QVIT)** that uses Grover search to approximate attention weight computation, and compare it with a standard **Vision Transformer (ViT)** on the MNIST dataset.

## Features

- **ViT**: Standard Vision Transformer with patch embedding and self-attention.
- **QVIT**: Quantum-augmented ViT with Grover-search-based attention approximation (optionally with filtering).
- **Dynamic thresholding**: During training, each attention row uses its own percentile (e.g. 0.9) as the selection threshold; a running **history threshold** (EMA of these values) is recorded and used at inference so no per-row quantile is needed at test time.
- **Image local mask**: After Grover or classical thresholding, the selected indices are expanded by a configurable **neighbor radius** so that patch tokens adjacent in the 2D grid are also kept (e.g. `neighbor_radius=1` adds the 8 surrounding patches).
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

Configuration is controlled by `TrainConfig` in `entrance.py` (e.g. `batch_size`, `epochs`, `device`, `train_qvit`, `qvit_use_grover`, `qvit_enable_filter`). The default tokenizer can be switched between `PatchTokenizerCNN` and `SimplePatchTokenizer` by changing the import and instantiation in `entrance.py`.

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

- Think about and implement how classic data is encoded into quantum states.
- Implement quantum top-k search.
- Dive into how to design oracles for thesholding Grover, top-k attention search and attetion computaion integrated top-k attetion search algorithm.
- The oracle now is hard coded by classic solutions, but in practice it should be a better black-box oracle.

## Report TODOs
- For report, discuss dequantization, and mention that QRAM is theoretically feasible, and maybe more.

- For report, discuss how dynamic modifying the oracle would affect the actual training and inference speed. It may be solved if there exists an black-box oracle that, given any few states and there values, can output the solution.

- For report, discuss the two proposed approaches of attetion acceleration. 
  1. Assume an oracle that, given $i$ as the $i^{\text{th}}$ row of query matrix and given $j$ as the $j^{\text{th}}$ row of key matrix, the oracle output whether their attention score is higher than one threshold.
  2. First compute attention scores for all elements in attetion matrix. Then make use of the quantum top-k algorithm to compute the top-k attetion scores for each row. In the future, a more efficient oracle can be designed such that computation of attetion socres is integrated into the oracle or the quantum algorithm itself instead of relying on classic computers.

- For report, mention the idea of combining classic static masking as well as quantum dynamic masking to make efficient and well-performed predictions.

## References


**Quantum Algorithm for Inference Speedup**
- [Grover Search for Acceleration of Attention Computation](https://arxiv.org/abs/arXiv:2307.08045) (arXiv:2307.08045)
- [Sublinear Time Quantum Algorithm for Attention Approximation](https://arxiv.org/abs/2602.00874) (arXiv:2602.00874)

**QRAM**

- [Quantum Random Access Memory](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.160501)
- [Quantum random access memory: a survey and critique](https://inspirehep.net/files/5d80c1c8461ae8f7896b24bfd170e1fe)


**Dequantization**
- [Robust Dequantization of the Quantum Singular value Transformation and Quantum Machine Learning Algorithms](https://arxiv.org/abs/2304.04932)
- [Robust Dequantization of the Quantum Singular value Transformation and Quantum Machine Learning Algorithms](https://arxiv.org/abs/1811.00414)


**Quantum Top-k**
- [A Quantum Algorithm for Finding the Minimum](https://arxiv.org/pdf/quant-ph/9607014)
- [Quantum Approximate -Minimum Finding](https://arxiv.org/abs/2412.16586)

**Quantum Algorithms for Basic Arithmatics**
- [Quantum Networks for Elementary Arithmetic Operations](https://arxiv.org/abs/quant-ph/9511018)
