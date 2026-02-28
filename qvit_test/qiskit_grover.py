"""Grover search utilities for attention index selection.

Provides multiple backends for Grover search:
- \"numpy\": Pure NumPy state vector simulation (default, fastest, most stable).
- \"torch\": PyTorch state vector simulation (can be placed on GPU).
- \"qiskit\": Original Qiskit Aer circuit simulation (slow, but easy to compare with paper implementation).
- \"shortcut\": Pure classical thresholding, fastest, no quantum simulation.
"""

from __future__ import annotations

from typing import Iterable, List, Literal, Optional

import math

import numpy as np
import torch

try:  # 可选依赖：只有在 backend=\"qiskit\" 时才真正需要
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import GroverOperator
    from qiskit_aer import Aer

    _HAS_QISKIT = True
except Exception:  # pragma: no cover - optional dependency
    QuantumCircuit = None  # type: ignore[assignment]
    GroverOperator = None  # type: ignore[assignment]
    Aer = None  # type: ignore[assignment]
    transpile = None  # type: ignore[assignment]
    _HAS_QISKIT = False


def _int_to_bitstring(value: int, num_qubits: int) -> str:
    return format(value, f"0{num_qubits}b")


def grover_shortcut(scores: Iterable[float], threshold: float) -> List[bool]:
    """
    Pure classical thresholding: directly get mask by score > threshold, no Grover/quantum simulation.
    """
    scores_list = list(scores)
    if not scores_list:
        return []
    return [s > threshold for s in scores_list]


def _grover_numpy(
    n: int,
    num_qubits: int,
    marked: List[int],
    shots: int,
) -> List[bool]:
    """Use NumPy for Grover state vector simulation."""


    raise RuntimeError("NumPy is not preferred, please use shortcut backend instead. If you want to use NumPy, please remove this raise statement.")
    
    N = 1 << num_qubits
    if n <= 0 or N == 0:
        return []

    marked_set = {i for i in marked if 0 <= i < N}
    if not marked_set:
        return [False] * n

    m = len(marked_set)
    iterations = max(1, int(round(math.pi / 4 * math.sqrt(N / m))))

    state = np.ones(N, dtype=np.complex128) / math.sqrt(N)

    for _ in range(iterations):
        for idx in marked_set:
            state[idx] = -state[idx]
        mean_amp = state.mean()
        state = 2.0 * mean_amp - state

    probs = (state.real ** 2 + state.imag ** 2).astype(np.float64)
    probs_sum = probs.sum()
    if probs_sum <= 0:
        return [False] * n
    probs /= probs_sum

    rng = np.random.default_rng()
    samples = rng.choice(N, size=shots, p=probs)
    counts: dict[int, int] = {}
    for s in samples:
        i = int(s)
        counts[i] = counts.get(i, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    kept: set[int] = set()
    for idx, _ in sorted_items:
        if idx < n:
            kept.add(idx)
    return [i in kept for i in range(n)]


def _grover_torch(
    n: int,
    num_qubits: int,
    marked: List[int],
    shots: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex128,
) -> List[bool]:
    raise RuntimeError("PyTorch is not preferred, please use numpy backend instead. If you want to use PyTorch, please remove this raise statement.")
    """Use PyTorch for Grover state vector simulation (can be placed on GPU)."""
    if device is None:
        device = torch.device("cpu")

    N = 1 << num_qubits
    if n <= 0 or N == 0:
        return []

    marked_tensor = torch.tensor(
        [i for i in marked if 0 <= i < N],
        dtype=torch.long,
        device=device,
    )
    if marked_tensor.numel() == 0:
        return [False] * n

    m = int(marked_tensor.numel())
    iterations = max(1, int(round(math.pi / 4 * math.sqrt(N / m))))

    state = torch.ones(N, dtype=dtype, device=device) / math.sqrt(N)

    for _ in range(iterations):
        state[marked_tensor] *= -1
        mean_amp = state.mean()
        state = 2.0 * mean_amp - state

    probs = (state.real ** 2 + state.imag ** 2).to(dtype=torch.float64)
    probs_sum = probs.sum()
    if float(probs_sum) <= 0.0:
        return [False] * n
    probs = probs / probs_sum

    indices = torch.multinomial(probs, num_samples=shots, replacement=True)
    counts = torch.bincount(indices, minlength=N)

    nonzero = torch.nonzero(counts, as_tuple=False).squeeze(-1)
    if nonzero.numel() == 0:
        return [False] * n
    values = counts[nonzero]
    order = torch.argsort(values, descending=True)

    kept: set[int] = set()
    for j in order.tolist():
        idx = int(nonzero[j])
        if idx < n:
            kept.add(idx)
    return [i in kept for i in range(n)]


def _grover_qiskit(
    n: int,
    num_qubits: int,
    marked: List[int],
    shots: int,
) -> List[bool]:
    """Use the original Qiskit + Aer path for Grover simulation."""

    raise RuntimeError("Qiskit is extremely slow, please use numpy or torch backend instead. If you want to use Qiskit, please remove this raise statement.")
    # check if Qiskit is available
    if not _HAS_QISKIT:
        raise RuntimeError("Qiskit is not available, cannot use backend='qiskit'.")

    backend = Aer.get_backend("qasm_simulator")

    marked_bitstrings = [_int_to_bitstring(i, num_qubits) for i in marked]

    oracle = QuantumCircuit(num_qubits)
    for bitstr in marked_bitstrings:
        for idx, bit in enumerate(bitstr):
            if bit == "0":
                oracle.x(idx)
        oracle.h(num_qubits - 1)
        oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        oracle.h(num_qubits - 1)
        for idx, bit in enumerate(bitstr):
            if bit == "0":
                oracle.x(idx)

    grover_op = GroverOperator(oracle)

    m = max(1, len(marked))
    iterations = max(1, int(round(math.pi / 4 * math.sqrt((2**num_qubits) / m))))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.append(grover_op, range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))

    qc_t = transpile(qc, backend)

    job = backend.run(qc_t, shots=shots)
    result = job.result()
    counts = result.get_counts(qc_t)

    sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    kept: set[int] = set()
    for bitstr, _ in sorted_items:
        idx = int(bitstr, 2)
        if idx < n:
            kept.add(idx)
    return [i in kept for i in range(n)]


# tool function for Grover search filtering in QVIT.
def grover_mask(
    scores: Iterable[float],
    threshold: float,
    max_qubits: int = 4,
    shots: int = 16,
    backend: Literal["numpy", "torch", "qiskit", "shortcut"] = "shortcut",
    torch_device: Optional[str] = None,
) -> List[bool]:
    """
    Grover-simulated mask to select indices with score > threshold.

    backend:
      - \"numpy\": NumPy state vector simulation (default);
      - \"shortcut\": Pure classical thresholding, fastest, no quantum simulation;
      - \"torch\": PyTorch state vector simulation;
      - \"qiskit\": Qiskit Aer circuit simulation (slow).
    """

    scores_list = list(scores)
    n = len(scores_list)
    if n == 0:
        return []

    if backend == "shortcut":
        return grover_shortcut(scores_list, threshold)

    num_qubits = math.ceil(math.log2(n))
    if num_qubits > max_qubits:
        raise ValueError("Input too large for Grover simulation.")

    marked = [i for i, v in enumerate(scores_list) if v > threshold]
    if len(marked) == 0:
        return [False] * n
    if len(marked) == n:
        return [True] * n

    if backend == "numpy":
        return _grover_numpy(n, num_qubits, marked, shots)
    if backend == "torch":
        device = torch.device(torch_device) if torch_device is not None else None
        return _grover_torch(n, num_qubits, marked, shots, device=device)
    if backend == "qiskit":
        return _grover_qiskit(n, num_qubits, marked, shots)

    raise ValueError(f"Unknown backend {backend!r} for grover_mask.")
