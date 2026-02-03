from __future__ import annotations

from typing import Iterable, List

import math


def _int_to_bitstring(value: int, num_qubits: int) -> str:
    return format(value, f"0{num_qubits}b")


def grover_mask(
    scores: Iterable[float],
    threshold: float,
    max_qubits: int = 4,
    shots: int = 16,
) -> List[bool]:
    """
    Quantum-simulated Grover search to select indices with score > threshold.

    This is a demonstration-level implementation intended for small vectors.
    If the problem size is too large, it raises ValueError.
    """

    scores = list(scores)
    n = len(scores)
    if n == 0:
        return []

    num_qubits = math.ceil(math.log2(n))
    if num_qubits > max_qubits:
        raise ValueError("Input too large for Grover simulation.")

    marked = [i for i, v in enumerate(scores) if v > threshold]
    if len(marked) == 0:
        return [False] * n
    if len(marked) == n:
        return [True] * n

    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import GroverOperator
        from qiskit import Aer
        backend = Aer.get_backend("qasm_simulator")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Qiskit is not available.") from exc

    marked_bitstrings = [_int_to_bitstring(i, num_qubits) for i in marked]

    oracle = QuantumCircuit(num_qubits)
    for bitstr in marked_bitstrings:
        # Apply X to qubits where bit is 0
        for idx, bit in enumerate(bitstr):
            if bit == "0":
                oracle.x(idx)
        # Multi-controlled Z
        oracle.h(num_qubits - 1)
        oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        oracle.h(num_qubits - 1)
        # Uncompute X
        for idx, bit in enumerate(bitstr):
            if bit == "0":
                oracle.x(idx)

    grover_op = GroverOperator(oracle)

    # Grover iterations
    m = len(marked)
    iterations = max(1, int(round(math.pi / 4 * math.sqrt((2**num_qubits) / m))))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.append(grover_op, range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))

    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)

    # Select the most frequent outcomes as indices to keep
    sorted_keys = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    kept = set()
    for key, _ in sorted_keys:
        idx = int(key, 2)
        if idx < n:
            kept.add(idx)
    return [i in kept for i in range(n)]
