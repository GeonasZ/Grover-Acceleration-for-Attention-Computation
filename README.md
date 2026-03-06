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

High-level run options (batch size, epochs, device, etc.) are set via `TrainConfig` in `entrance.py`. **QVIT attention filtering** (Grover vs top-k and all related parameters) is controlled by **`config.json`**; see [Configuration](#configuration) below. The default tokenizer can be switched between `PatchTokenizerCNN` and `SimplePatchTokenizer` by changing the import and instantiation in `entrance.py`.

## Configuration

QVIT reads attention-filter type and all Grover/top-k parameters from **`config.json`** in the project root. The file is loaded and merged with defaults by `qvit_test.config_loader`; training and evaluation use these settings automatically.

### Config file location

- **Default path**: `config.json` next to `entrance.py` (project root).
- **Override**: set the environment variable `QVIT_CONFIG_PATH` to the full path of your JSON file.

### Top-level structure

```json
{
  "training_option": { ... },
  "grover": { ... },
  "topk": { ... }
}
```

### `training_option`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `attention_filter` | `"grover"` \| `"topk"` | `"grover"` | Which attention filter QVIT uses: **grover** = threshold/Grover-based selection; **topk** = top-k by attention score (plus local mask). |

### `grover`

Used when `training_option.attention_filter` is `"grover"`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `threshold` | float | `0.0482` | Fixed threshold for score > threshold (used when not using row percentile). |
| `use_qiskit` | bool | `true` | If true, use Grover/threshold path; if false, pure classical threshold (vectorized). |
| `grover_backend` | string | `"shortcut"` | Grover implementation: **shortcut** = vectorized threshold (fastest); **numpy** = NumPy state-vector simulation; **qiskit** = Qiskit Aer circuit simulation. |
| `max_qubits` | int | `5` | Max qubits for Grover (e.g. 5 for 17 tokens). Sequences longer than 2^max_qubits fall back to classical threshold. |
| `shots` | int \| null | `null` | Grover measurement shots; null means 2× sequence length. |
| `enable_filter` | bool | `true` | If false, no filtering (mask all True). |
| `percentile` | float | `0.9` | Row-wise percentile for dynamic threshold in training. |
| `neighbor_radius` | int | `1` | 2D patch neighborhood radius (includes self). E.g. 1 = self + 8 neighbors (3×3). |

### `topk`

Used when `training_option.attention_filter` is `"topk"`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `k` | int | `8` | Number of remote (non-local) positions to keep per query row, by top attention score. Selection is per row (each of the B×H×S query positions), not per head or per batch. |
| `enable_filter` | bool | `true` | If false, no filtering (mask all True). |
| `neighbor_radius` | int | `1` | Same as Grover: 2D patch neighborhood radius (includes self). |

### Example: switch to top-k and tune Grover

To use **top-k** attention filter:

```json
"training_option": {
  "attention_filter": "topk"
}
```

To use **Grover** with faster vectorized backend and a higher threshold:

```json
"training_option": { "attention_filter": "grover" },
"grover": {
  "threshold": 0.05,
  "grover_backend": "shortcut",
  "max_qubits": 5,
  "enable_filter": true,
  "neighbor_radius": 1
}
```

Config is loaded once and cached; use `config_loader.reset_config_cache()` in code if you need to reload (e.g. after changing the file).

## Project Structure

```
.
├── config.json               # QVIT attention filter and grover/topk options (see Configuration)
├── entrance.py               # Main entry: training and evaluation for ViT / QVIT
├── pyproject.toml            # Project config and dependencies
├── uv.lock                   # Lock file (use uv for reproducible installs)
├── README.md
├── data/                     # Dataset directory (e.g. MNIST, created on first run)
│
└── qvit_test/                # Main package
    ├── config_loader.py      # Load config.json and expose get_attention_filter, get_grover_config, get_topk_config
    ├── __init__.py           # Exports: PatchConfig, PatchTokenizerCNN, SimplePatchTokenizer,
    │                         #          get_mnist_dataloaders, ViT, ViTConfig, QVIT,
    │                         #          evaluate_qvit, evaluate_vit
    ├── feature_extraction.py # PatchConfig, PatchTokenizerCNN, get_mnist_dataloaders,
    │                         # PatchEmbeddingClassifier, pretrain_patch_tokenizer
    ├── simple_tokenizer.py   # SimplePatchTokenizer (linear patch embedding, same interface as PatchTokenizerCNN)
    ├── vit.py                # ViT, ViTConfig (standard Vision Transformer)
    ├── qvit.py               # QVIT (quantum ViT with Grover/topk attention filter)
    ├── topk.py               # topk_search_filter (top-k attention mask)
    ├── qiskit_grover.py      # Grover search (shortcut / numpy / qiskit backends)
    └── evaluation.py         # evaluate_vit, evaluate_qvit
```

## TODOs

- Think about and implement how classic data is encoded into quantum states.
- Dive into how to design oracles for thesholding Grover, top-k attention search and attetion computaion integrated top-k attetion search algorithm.
- The oracle now is hard coded by classic solutions, but in practice it should be a better black-box oracle.

## Report TODOs

- For report, discuss how the complexity of reading classic data into quantum and write them back to classic after would affect the efficiency of algorithms.

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
