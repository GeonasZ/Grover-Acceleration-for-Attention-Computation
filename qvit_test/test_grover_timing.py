"""
测试单次 Grover 搜索的运行时间。
使用 Qiskit 的 GroverOperator 和 qasm_simulator 测量：构建电路、transpile、执行各阶段耗时。
"""

from __future__ import annotations

import math
import time
from typing import List

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import GroverOperator
    try:
        from qiskit_aer import Aer
    except ImportError:
        from qiskit.providers.aer import Aer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def _int_to_bitstring(value: int, num_qubits: int) -> str:
    return format(value, f"0{num_qubits}b")


def build_grover_circuit(
    num_qubits: int,
    marked_indices: List[int],
    shots: int = 16,
) -> QuantumCircuit:
    """构建一次 Grover 搜索电路（与 qiskit_grover.grover_mask 逻辑一致）。"""
    marked_bitstrings = [_int_to_bitstring(i, num_qubits) for i in marked_indices]

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
    m = len(marked_indices)
    iterations = max(1, int(round(math.pi / 4 * math.sqrt((2**num_qubits) / m))))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.append(grover_op, range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def run_single_grover_timing(
    num_qubits: int,
    num_marked: int = 1,
    shots: int = 16,
    warmup: bool = True,
) -> dict:
    """
    跑一次 Grover，分别计时：构建电路、transpile、backend.run。
    若 warmup=True，先跑一次不计时的 warmup 再计时。
    """
    if not QISKIT_AVAILABLE:
        return {"error": "Qiskit not available"}

    n = 2**num_qubits
    marked = list(range(min(num_marked, n)))

    # 1) 构建电路
    t0 = time.perf_counter()
    qc = build_grover_circuit(num_qubits, marked, shots=shots)
    t_build = time.perf_counter() - t0

    backend = Aer.get_backend("qasm_simulator")

    if warmup:
        _ = backend.run(transpile(qc, backend), shots=shots).result()

    # 2) transpile
    t0 = time.perf_counter()
    qc_t = transpile(qc, backend)
    t_transpile = time.perf_counter() - t0

    # 3) 执行
    t0 = time.perf_counter()
    job = backend.run(qc_t, shots=shots)
    result = job.result()
    t_run = time.perf_counter() - t0

    total = t_build + t_transpile + t_run
    return {
        "num_qubits": num_qubits,
        "shots": shots,
        "t_build_ms": t_build * 1000,
        "t_transpile_ms": t_transpile * 1000,
        "t_run_ms": t_run * 1000,
        "t_total_ms": total * 1000,
        "counts": result.get_counts(qc_t),
    }


def main():
    if not QISKIT_AVAILABLE:
        print("Qiskit 未安装，无法运行测试。")
        return

    print("=" * 60)
    print("单次 Grover 运行时间测试 (Qiskit Aer qasm_simulator)")
    print("=" * 60)

    for num_qubits in [2, 3, 4]:
        r = run_single_grover_timing(num_qubits=num_qubits, shots=16, warmup=True)
        if "error" in r:
            print(r["error"])
            continue
        print(f"\n--- {num_qubits} qubits, 2^{num_qubits} = {2**num_qubits} 个态, shots=16 ---")
        print(f"  构建电路:   {r['t_build_ms']:.2f} ms")
        print(f"  transpile:  {r['t_transpile_ms']:.2f} ms")
        print(f"  执行 run:   {r['t_run_ms']:.2f} ms")
        print(f"  总耗时:     {r['t_total_ms']:.2f} ms")
        print(f"  测量结果 (前5): {dict(list(r['counts'].items())[:5])}")

    print("\n" + "=" * 60)
    print("单次 Grover 总耗时（可直接用于评估 QVIT 中一次 grover_mask 调用）")
    print("=" * 60)


if __name__ == "__main__":
    main()
