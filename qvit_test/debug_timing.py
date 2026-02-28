"""Optional debug timing for QVIT training. Enable with env DEBUG_QVIT_TIMING=1."""

from __future__ import annotations

import os
import time
from typing import Dict, Any

DEBUG_QVIT_TIMING = os.environ.get("DEBUG_QVIT_TIMING", "").lower() in ("1", "true", "yes")

# Per-batch accumulators (reset at start of each batch)
_current: Dict[str, float] = {}
# Per-batch records for epoch summary
_batch_records: list[Dict[str, Any]] = []


def reset_batch() -> None:
    _current.clear()
    _current["grover_mask_sec"] = 0.0
    _current["grover_filter_sec"] = 0.0


def add_grover_mask_sec(sec: float) -> None:
    _current["grover_mask_sec"] = _current.get("grover_mask_sec", 0.0) + sec


def add_grover_filter_sec(sec: float) -> None:
    _current["grover_filter_sec"] = _current.get("grover_filter_sec", 0.0) + sec


def get_batch_grover() -> Dict[str, float]:
    return dict(_current)


def record_batch(record: Dict[str, Any]) -> None:
    _batch_records.append(record)


def report_epoch(epoch: int, name: str = "QVIT") -> None:
    if not _batch_records:
        return
    n = len(_batch_records)
    total_batch = sum(r.get("batch_sec", 0) for r in _batch_records)
    total_forward = sum(r.get("forward_sec", 0) for r in _batch_records)
    total_tokenizer = sum(r.get("tokenizer_sec", 0) for r in _batch_records)
    total_backward = sum(r.get("backward_sec", 0) for r in _batch_records)
    total_step = sum(r.get("step_sec", 0) for r in _batch_records)
    total_grover_filter = sum(r.get("grover_filter_sec", 0) for r in _batch_records)
    total_grover_mask = sum(r.get("grover_mask_sec", 0) for r in _batch_records)

    denom = max(total_batch, 1e-9)
    print(f"\n[{name}] Epoch {epoch} timing ({n} batches):")
    print(f"  total batch:     {total_batch:.2f}s  (100%)")
    print(f"  tokenizer:       {total_tokenizer:.2f}s  ({100*total_tokenizer/denom:.0f}%)")
    print(f"  forward:         {total_forward:.2f}s  ({100*total_forward/denom:.0f}%)")
    print(f"    grover_filter: {total_grover_filter:.2f}s  ({100*total_grover_filter/denom:.0f}%)")
    print(f"    grover_mask:   {total_grover_mask:.2f}s  ({100*total_grover_mask/denom:.0f}%)  <- Aer simulation")
    print(f"  backward:       {total_backward:.2f}s  ({100*total_backward/denom:.0f}%)")
    print(f"  optimizer.step:  {total_step:.2f}s  ({100*total_step/denom:.0f}%)")
    print(f"  avg per batch:  {total_batch/n:.2f}s\n")
    _batch_records.clear()
