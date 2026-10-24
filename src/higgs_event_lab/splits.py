"""Explicit, reproducible split policy module for HIGGS benchmarks.

Re-exports the core splitting interfaces from the protocol contract.
"""

from __future__ import annotations

from higgs_event_lab.protocol import (
    SPLIT_PROTOCOL_VERSION,
    SplitConfig,
    SplitIndices,
    create_stratified_splits,
)

__all__ = [
    "SPLIT_PROTOCOL_VERSION",
    "SplitConfig",
    "SplitIndices",
    "create_stratified_splits",
]
