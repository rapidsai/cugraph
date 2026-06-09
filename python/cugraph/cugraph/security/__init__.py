# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Security modules for cuGraph.

Provides production-grade defensive detection for adversarial graph topologies
that can cause GPU kernel hangs in graph analytics frameworks.

The main components are:
    ATGCConfig: Configuration management for ATGC detection
    ATGCDetector: Main detection engine for clique-chain topologies
    ATGCResult: Detection result container
    detect_clique_chain: Legacy detection function
    validate_graph_input: Legacy validation function

Example:
    >>> from cugraph.security import ATGCDetector, ATGCConfig
    >>> 
    >>> # Create detector with default configuration
    >>> detector = ATGCDetector()
    >>> 
    >>> # Or with custom configuration
    >>> config = ATGCConfig(max_clique_size=25, variance_threshold=150.0)
    >>> detector = ATGCDetector(config)
    >>> 
    >>> # Detect adversarial topology
    >>> result = detector.detect(edgelist_df)
    >>> if result.is_adversarial:
    ...     print(f"Adversarial graph detected: {result.message}")
"""

from .atgc_detection import (
    ATGCConfig,
    ATGCDetector,
    ATGCResult,
    detect_clique_chain,
    validate_graph_input,
)

__all__ = [
    "ATGCConfig",
    "ATGCDetector",
    "ATGCResult",
    "detect_clique_chain",
    "validate_graph_input",
]

__version__ = "1.0.0"
