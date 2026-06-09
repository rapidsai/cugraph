# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Security modules for cuGraph.

Provides defensive detection for adversarial graph topologies.
"""

from .atgc_detection import (
    detect_clique_chain,
    validate_graph_input,
    ATGC_GUARD_ENABLED,
    ATGC_ACTION,
)

__all__ = [
    "detect_clique_chain",
    "validate_graph_input",
    "ATGC_GUARD_ENABLED",
    "ATGC_ACTION",
]
