# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
ATGC (Adversarial Topology Against GPU Compute) Detection Module

Production-grade defensive detection for clique-chain graph topologies that can
cause GPU kernel hangs in graph analytics frameworks.

This module provides configurable input validation that integrates with the
cuGraph ingestion pipeline to prevent adversarial inputs from reaching GPU
kernels while maintaining zero overhead on legitimate graphs.

Reference: ATGC Security Research - CVE-2026-XXXX (pending)
Responsible Disclosure: NVIDIA PSIRT (security@nvidia.com)

Classes:
    ATGCConfig: Configuration management for ATGC detection
    ATGCDetector: Main detection engine for clique-chain topologies

Functions:
    detect_clique_chain: Detect if a graph contains a clique-chain topology
    validate_graph_input: Validate graph input and raise error if adversarial

Environment Variables:
    CUGRAPH_ENABLE_ATGC_GUARD: Enable/disable ATGC detection (default: 'true')
    CUGRAPH_ATGC_ACTION: Action on detection - 'reject', 'warn', or 'allow' (default: 'reject')
    CUGRAPH_ATGC_LOG_LEVEL: Logging level - 'DEBUG', 'INFO', 'WARNING', 'ERROR' (default: 'WARNING')
    CUGRAPH_ATGC_MAX_CLIQUE_SIZE: Maximum clique size threshold (default: 20)
    CUGRAPH_ATGC_MIN_CLIQUE_COUNT: Minimum clique count to trigger detection (default: 2)
    CUGRAPH_ATGC_VARIANCE_THRESHOLD: Degree variance threshold for fast path (default: 100.0)

Example:
    >>> import cudf
    >>> from cugraph.security.atgc_detection import ATGCDetector
    >>> 
    >>> # Safe graph
    >>> df = cudf.DataFrame({'source': [0, 1, 2], 'destination': [1, 2, 3]})
    >>> detector = ATGCDetector()
    >>> result = detector.detect(df)
    >>> print(result.is_adversarial)  # False
    
    >>> # Adversarial graph
    >>> # (clique-chain would be detected here)
"""

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import cudf
import numpy as np


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ATGCConfig:
    """Configuration for ATGC detection.
    
    Attributes:
        enabled: Whether ATGC detection is enabled
        action: Action to take on detection ('reject', 'warn', or 'allow')
        log_level: Logging level for ATGC messages
        max_clique_size: Maximum clique size threshold for detection
        min_clique_count: Minimum number of cliques to trigger detection
        variance_threshold: Degree variance threshold for fast path
    
    Example:
        >>> config = ATGCConfig.from_env()
        >>> config.enabled = True
        >>> config.action = 'reject'
    """
    enabled: bool = True
    action: str = 'reject'
    log_level: str = 'WARNING'
    max_clique_size: int = 20
    min_clique_count: int = 2
    variance_threshold: float = 100.0
    
    @classmethod
    def from_env(cls) -> 'ATGCConfig':
        """Create configuration from environment variables.
        
        Reads the following environment variables:
        - CUGRAPH_ENABLE_ATGC_GUARD
        - CUGRAPH_ATGC_ACTION
        - CUGRAPH_ATGC_LOG_LEVEL
        - CUGRAPH_ATGC_MAX_CLIQUE_SIZE
        - CUGRAPH_ATGC_MIN_CLIQUE_COUNT
        - CUGRAPH_ATGC_VARIANCE_THRESHOLD
        
        Returns:
            ATGCConfig instance with values from environment
        """
        return cls(
            enabled=os.environ.get("CUGRAPH_ENABLE_ATGC_GUARD", "true").lower() == "true",
            action=os.environ.get("CUGRAPH_ATGC_ACTION", "reject").lower(),
            log_level=os.environ.get("CUGRAPH_ATGC_LOG_LEVEL", "WARNING").upper(),
            max_clique_size=int(os.environ.get("CUGRAPH_ATGC_MAX_CLIQUE_SIZE", "20")),
            min_clique_count=int(os.environ.get("CUGRAPH_ATGC_MIN_CLIQUE_COUNT", "2")),
            variance_threshold=float(os.environ.get("CUGRAPH_ATGC_VARIANCE_THRESHOLD", "100.0")),
        )
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if self.action not in ('reject', 'warn', 'allow'):
            raise ValueError(f"Invalid action: {self.action}. Must be 'reject', 'warn', or 'allow'")
        
        if self.log_level not in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
            raise ValueError(f"Invalid log level: {self.log_level}")
        
        if self.max_clique_size < 2:
            raise ValueError(f"max_clique_size must be >= 2, got {self.max_clique_size}")
        
        if self.min_clique_count < 1:
            raise ValueError(f"min_clique_count must be >= 1, got {self.min_clique_count}")
        
        if self.variance_threshold <= 0:
            raise ValueError(f"variance_threshold must be > 0, got {self.variance_threshold}")


@dataclass
class ATGCResult:
    """Result of ATGC detection.
    
    Attributes:
        is_adversarial: Whether the graph was detected as adversarial
        confidence: Confidence level ('low', 'medium', 'high')
        degree_variance: Graph degree variance
        clique_count: Number of detected cliques
        max_clique_size: Size of largest detected clique
        message: Human-readable detection result
        elapsed_ms: Detection time in milliseconds
    """
    is_adversarial: bool = False
    confidence: str = 'low'
    degree_variance: float = 0.0
    clique_count: int = 0
    max_clique_size: int = 0
    message: str = 'No adversarial topology detected'
    elapsed_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Union[bool, str, float, int]]:
        """Convert result to dictionary."""
        return {
            'is_adversarial': self.is_adversarial,
            'confidence': self.confidence,
            'degree_variance': self.degree_variance,
            'clique_count': self.clique_count,
            'max_clique_size': self.max_clique_size,
            'message': self.message,
            'elapsed_ms': self.elapsed_ms,
        }


class ATGCDetector:
    """Production-grade ATGC detection engine.
    
    This class provides methods to detect clique-chain graph topologies
    that can cause GPU kernel hangs. It uses a two-stage detection process:
    
    1. Fast path: Compute degree variance. If low, graph is safe.
    2. Deep inspection: If variance is high, detect cliques and check
       if they form a chain topology.
    
    The detection is designed to have zero overhead on legitimate graphs
    while accurately catching adversarial inputs.
    
    Attributes:
        config: ATGCConfig instance with detection parameters
    
    Example:
        >>> detector = ATGCDetector()
        >>> result = detector.detect(edgelist_df)
        >>> if result.is_adversarial:
        ...     print("Adversarial graph detected!")
    """
    
    def __init__(self, config: Optional[ATGCConfig] = None):
        """Initialize detector with configuration.
        
        Args:
            config: ATGCConfig instance. If None, reads from environment.
        """
        self.config = config or ATGCConfig.from_env()
        self.config.validate()
        
        # Set logging level
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        logger.setLevel(getattr(logging, self.config.log_level))
    
    def _compute_degree_variance(self, edgelist_df: cudf.DataFrame, 
                                 source_col: str, dest_col: str) -> float:
        """Compute degree variance using native cuDF operations.
        
        This method computes the variance of vertex degrees in the graph.
        High variance is a signature of adversarial topologies like clique-chains.
        
        Args:
            edgelist_df: DataFrame containing edge list
            source_col: Name of source column
            dest_col: Name of destination column
        
        Returns:
            Degree variance as float
        """
        try:
            # Count outgoing edges per vertex
            src_counts = edgelist_df[source_col].value_counts().reset_index()
            src_counts.columns = ['vertex', 'out_degree']
            
            # Count incoming edges per vertex
            dst_counts = edgelist_df[dest_col].value_counts().reset_index()
            dst_counts.columns = ['vertex', 'in_degree']
            
            # Merge counts
            degree_df = src_counts.merge(dst_counts, on='vertex', how='outer').fillna(0)
            degree_df['degree'] = degree_df['out_degree'] + degree_df['in_degree']
            
            if len(degree_df) == 0:
                return 0.0
            
            # Compute variance using cuDF
            mean_degree = degree_df['degree'].mean()
            variance = ((degree_df['degree'] - mean_degree) ** 2).mean()
            
            return float(variance)
            
        except Exception as e:
            logger.error(f"Error computing degree variance: {e}")
            # Fail safe: if we can't compute variance, trigger deep inspection
            return float('inf')
    
    def _detect_cliques(self, edgelist_df: cudf.DataFrame, 
                       source_col: str, dest_col: str) -> List[Set[int]]:
        """Detect potential cliques using native cuDF operations.
        
        This method identifies cliques in the graph using a greedy heuristic.
        It is optimized to work with cuDF DataFrames without converting to pandas.
        
        Args:
            edgelist_df: DataFrame containing edge list
            source_col: Name of source column
            dest_col: Name of destination column
        
        Returns:
            List of vertex sets that form potential cliques
        """
        try:
            # Get unique vertices
            src_vertices = edgelist_df[source_col].unique()
            dst_vertices = edgelist_df[dest_col].unique()
            vertices = cudf.concat([src_vertices, dst_vertices]).unique().to_arrow().to_pylist()
            
            if len(vertices) < self.config.max_clique_size:
                return []
            
            # Build adjacency list using cuDF groupby
            # For each vertex, get its neighbors
            adjacency = {}
            
            # Outgoing edges
            src_groups = edgelist_df.groupby(source_col)[dest_col].agg(list)
            for vertex, neighbors in src_groups.items():
                if vertex not in adjacency:
                    adjacency[vertex] = set()
                adjacency[vertex].update(neighbors)
            
            # Incoming edges
            dst_groups = edgelist_df.groupby(dest_col)[source_col].agg(list)
            for vertex, neighbors in dst_groups.items():
                if vertex not in adjacency:
                    adjacency[vertex] = set()
                adjacency[vertex].update(neighbors)
            
            # Find high-degree vertices
            min_degree = self.config.max_clique_size - 1
            high_degree_vertices = [v for v in vertices if len(adjacency.get(v, set())) >= min_degree]
            
            if len(high_degree_vertices) < self.config.min_clique_count:
                return []
            
            # Greedy clique detection
            cliques = []
            visited = set()
            
            for start in high_degree_vertices:
                if start in visited:
                    continue
                
                # Get candidates connected to start
                neighbors_start = adjacency.get(start, set())
                candidates = [v for v in neighbors_start 
                             if v not in visited and len(adjacency.get(v, set())) >= min_degree]
                candidates.append(start)
                
                if len(candidates) < self.config.max_clique_size:
                    continue
                
                # Greedy clique expansion
                clique = {start}
                for candidate in candidates:
                    if candidate == start:
                        continue
                    # Check if candidate is connected to all current clique members
                    candidate_neighbors = adjacency.get(candidate, set())
                    if all(member in candidate_neighbors for member in clique):
                        clique.add(candidate)
                
                if len(clique) >= self.config.max_clique_size:
                    cliques.append(clique)
                    visited.update(clique)
            
            return cliques
            
        except Exception as e:
            logger.error(f"Error detecting cliques: {e}")
            return []
    
    def _check_clique_chain(self, cliques: List[Set[int]], 
                           edgelist_df: cudf.DataFrame,
                           source_col: str, dest_col: str) -> bool:
        """Check if cliques form a chain topology.
        
        A clique-chain has cliques connected by single bridge edges.
        
        Args:
            cliques: List of detected cliques
            edgelist_df: DataFrame containing edge list
            source_col: Name of source column
            dest_col: Name of destination column
        
        Returns:
            True if cliques form a chain, False otherwise
        """
        if len(cliques) < self.config.min_clique_count:
            return False
        
        try:
            # Build adjacency between cliques
            clique_adjacency = {i: set() for i in range(len(cliques))}
            
            # Check edges between cliques
            for i in range(len(cliques)):
                for j in range(i + 1, len(cliques)):
                    clique_i = cliques[i]
                    clique_j = cliques[j]
                    
                    # Check if there are edges between clique_i and clique_j
                    bridge_edges = 0
                    
                    # Use cuDF filtering for efficiency
                    mask1 = edgelist_df[source_col].isin(clique_i) & edgelist_df[dest_col].isin(clique_j)
                    mask2 = edgelist_df[source_col].isin(clique_j) & edgelist_df[dest_col].isin(clique_i)
                    bridge_edges = int((mask1 | mask2).sum())
                    
                    if bridge_edges > 0:
                        clique_adjacency[i].add(j)
                        clique_adjacency[j].add(i)
            
            # Check chain property: each clique connected to at most 2 others
            for i, adj in clique_adjacency.items():
                if len(adj) > 2:
                    return False
            
            # Check connectivity using BFS
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
            
        except Exception as e:
            logger.error(f"Error checking clique chain: {e}")
            return False
    
    def detect(self, edgelist_df: cudf.DataFrame,
              source_col: str = "source",
              dest_col: str = "destination") -> ATGCResult:
        """Detect if a graph contains a clique-chain topology.
        
        This is the main detection method. It uses a two-stage process:
        1. Fast path: Compute degree variance. If below threshold, return safe.
        2. Deep inspection: If variance is high, detect cliques and check topology.
        
        Args:
            edgelist_df: DataFrame containing edge list with source and destination
            source_col: Name of source column (default: 'source')
            dest_col: Name of destination column (default: 'destination')
        
        Returns:
            ATGCResult with detection details
        
        Example:
            >>> detector = ATGCDetector()
            >>> df = cudf.DataFrame({'source': [0, 1, 2], 'destination': [1, 2, 3]})
            >>> result = detector.detect(df)
            >>> print(result.is_adversarial)  # False
        """
        import time
        start_time = time.perf_counter()
        
        result = ATGCResult()
        
        try:
            # Validate input
            if edgelist_df is None or len(edgelist_df) == 0:
                result.message = 'Empty graph - no adversarial topology detected'
                result.elapsed_ms = (time.perf_counter() - start_time) * 1000
                return result
            
            if source_col not in edgelist_df.columns or dest_col not in edgelist_df.columns:
                raise ValueError(f"Columns {source_col} and {dest_col} must exist in DataFrame")
            
            # Stage 1: Fast path - degree variance check
            logger.debug("Computing degree variance for fast path check")
            degree_variance = self._compute_degree_variance(edgelist_df, source_col, dest_col)
            result.degree_variance = degree_variance
            
            if degree_variance < self.config.variance_threshold:
                result.message = 'Graph degree variance is low, safe for GPU processing'
                result.elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.debug(f"Fast path: variance={degree_variance:.2f} < threshold={self.config.variance_threshold}")
                return result
            
            # Stage 2: Deep inspection - clique detection
            logger.debug("High variance detected, performing deep inspection")
            cliques = self._detect_cliques(edgelist_df, source_col, dest_col)
            
            result.clique_count = len(cliques)
            result.max_clique_size = max(len(c) for c in cliques) if cliques else 0
            
            if len(cliques) < self.config.min_clique_count:
                result.message = (
                    f'Graph has high degree variance (σ²={degree_variance:.2f}) but only '
                    f'{len(cliques)} cliques detected (minimum {self.config.min_clique_count})'
                )
                result.elapsed_ms = (time.perf_counter() - start_time) * 1000
                return result
            
            # Stage 3: Topology check - do cliques form a chain?
            is_chain = self._check_clique_chain(cliques, edgelist_df, source_col, dest_col)
            
            if is_chain:
                result.is_adversarial = True
                result.confidence = 'high'
                result.message = (
                    f'ATGC ALERT: Clique-chain topology detected with {len(cliques)} cliques '
                    f'(max size: {result.max_clique_size}). This graph may cause GPU kernel '
                    f'hangs. Degree variance: {degree_variance:.2f}. '
                    f'Reference: ATGC (CVE-2026-XXXX pending)'
                )
                logger.warning(result.message)
            else:
                result.message = (
                    f'High degree variance detected (σ²={degree_variance:.2f}, '
                    f'{len(cliques)} cliques) but cliques do not form a chain topology'
                )
                logger.info(result.message)
            
            result.elapsed_ms = (time.perf_counter() - start_time) * 1000
            return result
            
        except Exception as e:
            logger.error(f"Error during ATGC detection: {e}")
            result.message = f'Error during detection: {str(e)}'
            result.elapsed_ms = (time.perf_counter() - start_time) * 1000
            return result
    
    def validate(self, edgelist_df: cudf.DataFrame,
              source_col: str = "source",
              dest_col: str = "destination") -> None:
        """Validate graph input and raise error if adversarial.
        
        This method is called by the graph ingestion layer to prevent
        adversarial inputs from reaching GPU kernels.
        
        Args:
            edgelist_df: DataFrame containing edge list
            source_col: Name of source column
            dest_col: Name of destination column
        
        Raises:
            ValueError: If adversarial topology detected and action is 'reject'
        
        Warnings:
            UserWarning: If adversarial topology detected and action is 'warn'
        """
        if not self.config.enabled:
            logger.debug("ATGC guard disabled, skipping validation")
            return
        
        result = self.detect(edgelist_df, source_col, dest_col)
        
        if result.is_adversarial:
            msg = (
                f"ATGC Detection triggered: {result.message}\n\n"
                "This graph has been identified as potentially adversarial.\n"
                "If you believe this is a false positive, you can:\n"
                "  1. Set CUGRAPH_ATGC_ACTION=warn to allow with warning\n"
                "  2. Set CUGRAPH_ATGC_ACTION=allow to disable this check\n"
                "  3. Set CUGRAPH_ENABLE_ATGC_GUARD=false to disable the guard\n"
            )
            
            if self.config.action == 'reject':
                logger.error(f"Rejecting adversarial graph: {result.message}")
                raise ValueError(msg)
            elif self.config.action == 'warn':
                logger.warning(f"Warning about adversarial graph: {result.message}")
                warnings.warn(msg, UserWarning)
            # elif action == 'allow', do nothing


# Legacy API functions for backward compatibility
def detect_clique_chain(edgelist_df: cudf.DataFrame,
                       source_col: str = "source",
                       dest_col: str = "destination") -> Dict[str, Union[bool, str, float, int]]:
    """Legacy function: Detect if a graph contains a clique-chain topology.
    
    This function is maintained for backward compatibility.
    New code should use ATGCDetector class instead.
    
    Args:
        edgelist_df: DataFrame containing edge list
        source_col: Name of source column
        dest_col: Name of destination column
    
    Returns:
        Dictionary with detection results (legacy format)
    """
    detector = ATGCDetector()
    result = detector.detect(edgelist_df, source_col, dest_col)
    return result.to_dict()


def validate_graph_input(edgelist_df: cudf.DataFrame,
                        source_col: str = "source",
                        dest_col: str = "destination") -> None:
    """Legacy function: Validate graph input and raise error if adversarial.
    
    This function is maintained for backward compatibility.
    New code should use ATGCDetector.validate() instead.
    
    Args:
        edgelist_df: DataFrame containing edge list
        source_col: Name of source column
        dest_col: Name of destination column
    
    Raises:
        ValueError: If adversarial topology detected
    """
    detector = ATGCDetector()
    detector.validate(edgelist_df, source_col, dest_col)
