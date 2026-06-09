# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
ATGC (Adversarial Topology Against GPU Compute) Detection Module

This module provides defensive detection for clique-chain graph topologies
that can cause GPU kernel hangs in graph analytics frameworks.

Reference: ATGC Security Research - CVE-2026-XXXX (pending)
"""

import os
import warnings
from typing import Dict, List, Set, Tuple, Union

import cudf
import numpy as np


# Configuration via environment variables
ATGC_GUARD_ENABLED = os.environ.get("CUGRAPH_ENABLE_ATGC_GUARD", "true").lower() == "true"
ATGC_LOG_LEVEL = os.environ.get("CUGRAPH_ATGC_LOG_LEVEL", "WARNING").upper()
ATGC_ACTION = os.environ.get("CUGRAPH_ATGC_ACTION", "reject").lower()

# Thresholds for detection
MAX_DEGREE_VARIANCE_THRESHOLD = 100.0  # Trigger deep inspection if exceeded
MAX_CLIQUE_SIZE_THRESHOLD = 20  # Cliques larger than this are suspicious
MIN_CLIQUE_COUNT = 2  # Minimum number of cliques to trigger detection


def _compute_degree_variance(edgelist_df: cudf.DataFrame, source_col: str, dest_col: str) -> float:
    """
    Compute degree variance of the graph.
    High variance is a signature of adversarial topologies.
    """
    # Count degrees using groupby
    src_counts = edgelist_df[source_col].value_counts()
    dst_counts = edgelist_df[dest_col].value_counts()
    
    # Combine degrees
    all_degrees = cudf.concat([src_counts, dst_counts])
    degree_counts = all_degrees.groupby(all_degrees.index).sum()
    
    if len(degree_counts) == 0:
        return 0.0
    
    mean_degree = degree_counts.mean()
    variance = ((degree_counts - mean_degree) ** 2).mean()
    
    return float(variance)


def _detect_cliques(edgelist_df: cudf.DataFrame, source_col: str, dest_col: str, 
                    min_clique_size: int = 3) -> List[Set[int]]:
    """
    Detect potential cliques in the graph using a greedy heuristic.
    
    Returns list of vertex sets that form potential cliques.
    """
    # Build adjacency list
    vertices = set(edgelist_df[source_col].to_pandas().tolist())
    vertices.update(edgelist_df[dest_col].to_pandas().tolist())
    
    neighbors = {v: set() for v in vertices}
    for _, row in edgelist_df.to_pandas().iterrows():
        u, v = row[source_col], row[dest_col]
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    # Find high-degree vertices (potential clique members)
    high_degree_vertices = [v for v in vertices if len(neighbors[v]) >= min_clique_size - 1]
    
    cliques = []
    visited = set()
    
    for start in high_degree_vertices:
        if start in visited:
            continue
        
        # Find candidates that are connected to start
        candidates = [v for v in neighbors[start] 
                     if v not in visited and len(neighbors[v]) >= min_clique_size - 1]
        candidates.append(start)
        
        if len(candidates) < min_clique_size:
            continue
        
        # Check if candidates form a clique (all connected to each other)
        # Use a greedy approach: start with start, add vertices that are connected to all current members
        clique = {start}
        for candidate in candidates:
            if candidate == start:
                continue
            # Check if candidate is connected to all members of current clique
            if all(candidate in neighbors[m] for m in clique):
                clique.add(candidate)
        
        if len(clique) >= min_clique_size:
            cliques.append(clique)
            visited.update(clique)
    
    return cliques


def _check_clique_chain(cliques: List[Set[int]], edgelist_df: cudf.DataFrame, 
                        source_col: str, dest_col: str) -> bool:
    """
    Check if detected cliques form a clique-chain topology.
    
    A clique-chain has cliques connected by single bridge edges.
    """
    if len(cliques) < MIN_CLIQUE_COUNT:
        return False
    
    # Check if cliques are connected by bridge edges
    # Build adjacency between cliques
    clique_adjacency = {i: set() for i in range(len(cliques))}
    
    for i, clique_i in enumerate(cliques):
        for j, clique_j in enumerate(cliques):
            if i >= j:
                continue
            
            # Check if there are edges between cliques
            bridge_edges = 0
            for _, row in edgelist_df.to_pandas().iterrows():
                u, v = row[source_col], row[dest_col]
                if (u in clique_i and v in clique_j) or (u in clique_j and v in clique_i):
                    bridge_edges += 1
            
            if bridge_edges > 0:
                clique_adjacency[i].add(j)
                clique_adjacency[j].add(i)
    
    # Check if cliques form a chain (each connected to at most 2 others)
    for i, adj in clique_adjacency.items():
        if len(adj) > 2:  # More than 2 neighbors means not a simple chain
            return False
    
    # Check if all cliques are connected (single component)
    visited = set()
    stack = [0]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in clique_adjacency[current]:
            if neighbor not in visited:
                stack.append(neighbor)
    
    return len(visited) == len(cliques)


def detect_clique_chain(edgelist_df: cudf.DataFrame, 
                        source_col: str = "source", 
                        dest_col: str = "destination") -> Dict[str, Union[bool, str, float]]:
    """
    Detect if a graph contains a clique-chain topology.
    
    Clique-chain graphs are adversarial inputs that can cause exponential
    per-thread work in GPU graph kernels, leading to TDR timeouts.
    
    Parameters
    ----------
    edgelist_df : cudf.DataFrame
        DataFrame containing edge list with source and destination columns
    source_col : str, optional (default='source')
        Name of the source column
    dest_col : str, optional (default='destination')
        Name of the destination column
    
    Returns
    -------
    dict
        Dictionary with detection results:
        - 'is_adversarial': bool, True if clique-chain detected
        - 'confidence': str, 'high', 'medium', or 'low'
        - 'degree_variance': float, graph degree variance
        - 'clique_count': int, number of detected cliques
        - 'max_clique_size': int, size of largest detected clique
        - 'message': str, human-readable detection result
    
    Examples
    --------
    >>> import cudf
    >>> from cugraph.security.atgc_detection import detect_clique_chain
    >>> 
    >>> # Safe graph (karate club)
    >>> df = cudf.DataFrame({'source': [0, 1, 2], 'destination': [1, 2, 3]})
    >>> result = detect_clique_chain(df)
    >>> print(result['is_adversarial'])  # False
    
    >>> # Adversarial graph (clique-chain)
    >>> # This would be constructed using ATGC methods
    """
    result = {
        'is_adversarial': False,
        'confidence': 'low',
        'degree_variance': 0.0,
        'clique_count': 0,
        'max_clique_size': 0,
        'message': 'No adversarial topology detected',
    }
    
    # Quick check: degree variance
    degree_variance = _compute_degree_variance(edgelist_df, source_col, dest_col)
    result['degree_variance'] = degree_variance
    
    # Fast path: low variance graphs are safe
    if degree_variance < MAX_DEGREE_VARIANCE_THRESHOLD:
        result['message'] = 'Graph degree variance is low, safe for GPU processing'
        return result
    
    # Deep inspection: detect cliques
    cliques = _detect_cliques(edgelist_df, source_col, dest_col, 
                              min_clique_size=MAX_CLIQUE_SIZE_THRESHOLD)
    
    result['clique_count'] = len(cliques)
    result['max_clique_size'] = max(len(c) for c in cliques) if cliques else 0
    
    if len(cliques) < MIN_CLIQUE_COUNT:
        result['message'] = f'Graph has high degree variance but only {len(cliques)} cliques detected (minimum {MIN_CLIQUE_COUNT})'
        return result
    
    # Check if cliques form a chain
    is_chain = _check_clique_chain(cliques, edgelist_df, source_col, dest_col)
    
    if is_chain:
        result['is_adversarial'] = True
        result['confidence'] = 'high'
        result['message'] = (
            f'Clique-chain topology detected: {len(cliques)} cliques '
            f'(max size: {result["max_clique_size"]}) connected in a chain. '
            f'This graph may cause GPU kernel hangs. '
            f'Reference: ATGC (Adversarial Topology Against GPU Compute)'
        )
    else:
        result['message'] = (
            f'High degree variance detected ({len(cliques)} cliques) but '
            f'cliques do not form a chain topology'
        )
    
    return result


def validate_graph_input(edgelist_df: cudf.DataFrame,
                         source_col: str = "source",
                         dest_col: str = "destination") -> None:
    """
    Validate graph input and raise error if adversarial topology detected.
    
    This function is called by the graph ingestion layer to prevent
    adversarial inputs from reaching GPU kernels.
    
    Parameters
    ----------
    edgelist_df : cudf.DataFrame
        DataFrame containing edge list
    source_col : str, optional (default='source')
        Name of the source column
    dest_col : str, optional (default='destination')
        Name of the destination column
    
    Raises
    ------
    ValueError
        If adversarial topology is detected and CUGRAPH_ATGC_ACTION is 'reject'
    
    Warnings
    --------
    UserWarning
        If adversarial topology is detected and CUGRAPH_ATGC_ACTION is 'warn'
    """
    if not ATGC_GUARD_ENABLED:
        return
    
    result = detect_clique_chain(edgelist_df, source_col, dest_col)
    
    if result['is_adversarial']:
        msg = (
            f"ATGC Detection: {result['message']}\n"
            f"To allow this graph, set CUGRAPH_ATGC_ACTION=allow\n"
            f"Reference: ATGC Security Research (CVE-2026-XXXX pending)"
        )
        
        if ATGC_ACTION == "reject":
            raise ValueError(msg)
        elif ATGC_ACTION == "warn":
            warnings.warn(msg, UserWarning)
        # elif ATGC_ACTION == "allow", do nothing
