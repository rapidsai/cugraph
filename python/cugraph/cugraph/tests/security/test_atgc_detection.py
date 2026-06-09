# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ATGC (Adversarial Topology Against GPU Compute) detection module.
"""

import pytest
import cudf
import numpy as np

from cugraph.security.atgc_detection import (
    detect_clique_chain,
    validate_graph_input,
    _compute_degree_variance,
    _detect_cliques,
)


class TestATGCDetection:
    """Test suite for ATGC detection functionality."""
    
    def test_safe_graph_low_variance(self):
        """Test that a simple path graph is detected as safe."""
        # Simple path graph: low degree variance
        edges = cudf.DataFrame({
            'source': [0, 1, 2, 3, 4],
            'destination': [1, 2, 3, 4, 5]
        })
        
        result = detect_clique_chain(edges)
        
        assert result['is_adversarial'] is False
        assert result['confidence'] == 'low'
        assert result['degree_variance'] < 100.0
        assert result['clique_count'] == 0
    
    def test_safe_grid_graph(self):
        """Test that a grid graph (common in HPC) is detected as safe."""
        # Create a small grid graph
        edges = []
        n = 5  # 5x5 grid
        for i in range(n):
            for j in range(n):
                node = i * n + j
                # Connect to right neighbor
                if j < n - 1:
                    edges.append((node, node + 1))
                # Connect to bottom neighbor
                if i < n - 1:
                    edges.append((node, node + n))
        
        edges_df = cudf.DataFrame({
            'source': [e[0] for e in edges],
            'destination': [e[1] for e in edges]
        })
        
        result = detect_clique_chain(edges_df)
        
        assert result['is_adversarial'] is False
        assert result['degree_variance'] < 100.0
    
    def test_adversarial_clique_chain(self):
        """Test that a clique-chain is detected as adversarial."""
        # Create a clique-chain G(k=10, m=2)
        k = 10
        m = 2
        edges = []
        
        # First clique
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((i, j))
        
        # Bridge edge
        edges.append((k - 1, k))
        
        # Second clique
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((k + i, k + j))
        
        edges_df = cudf.DataFrame({
            'source': [e[0] for e in edges],
            'destination': [e[1] for e in edges]
        })
        
        result = detect_clique_chain(edges_df)
        
        assert result['is_adversarial'] is True
        assert result['confidence'] == 'high'
        assert result['degree_variance'] > 100.0
        assert result['clique_count'] >= 2
        assert result['max_clique_size'] >= 10
    
    def test_single_clique_not_chain(self):
        """Test that a single clique is not detected as chain."""
        # Single clique K_5
        edges = []
        for i in range(5):
            for j in range(i + 1, 5):
                edges.append((i, j))
        
        edges_df = cudf.DataFrame({
            'source': [e[0] for e in edges],
            'destination': [e[1] for e in edges]
        })
        
        result = detect_clique_chain(edges_df)
        
        # Single clique should not be detected as chain (need m >= 2)
        assert result['is_adversarial'] is False
    
    def test_degree_variance_computation(self):
        """Test degree variance computation."""
        # Star graph: high variance
        edges = cudf.DataFrame({
            'source': [0, 0, 0, 0, 0],
            'destination': [1, 2, 3, 4, 5]
        })
        
        variance = _compute_degree_variance(edges, 'source', 'destination')
        
        assert variance > 0
        # Star graph has center with degree 5, leaves with degree 1
        # Variance should be significant
    
    def test_clique_detection(self):
        """Test clique detection algorithm."""
        # Triangle (3-clique)
        edges = cudf.DataFrame({
            'source': [0, 1, 2],
            'destination': [1, 2, 0]
        })
        
        cliques = _detect_cliques(edges, 'source', 'destination', min_clique_size=3)
        
        assert len(cliques) >= 1
        assert any(len(c) >= 3 for c in cliques)
    
    def test_validate_graph_input_safe(self):
        """Test that validation passes for safe graphs."""
        edges = cudf.DataFrame({
            'source': [0, 1, 2, 3],
            'destination': [1, 2, 3, 4]
        })
        
        # Should not raise
        validate_graph_input(edges)
    
    def test_validate_graph_input_reject(self):
        """Test that validation rejects adversarial graphs when configured."""
        import os
        
        # Save original action
        original_action = os.environ.get('CUGRAPH_ATGC_ACTION', 'reject')
        os.environ['CUGRAPH_ATGC_ACTION'] = 'reject'
        
        try:
            # Create adversarial graph
            k = 10
            edges = []
            for i in range(k):
                for j in range(i + 1, k):
                    edges.append((i, j))
            edges.append((k - 1, k))
            for i in range(k):
                for j in range(i + 1, k):
                    edges.append((k + i, k + j))
            
            edges_df = cudf.DataFrame({
                'source': [e[0] for e in edges],
                'destination': [e[1] for e in edges]
            })
            
            # Should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                validate_graph_input(edges_df)
            
            assert 'ATGC' in str(exc_info.value)
        finally:
            # Restore original action
            os.environ['CUGRAPH_ATGC_ACTION'] = original_action
    
    def test_empty_graph(self):
        """Test detection on empty graph."""
        edges = cudf.DataFrame({
            'source': [],
            'destination': []
        })
        
        result = detect_clique_chain(edges)
        
        assert result['is_adversarial'] is False
        assert result['degree_variance'] == 0.0
    
    def test_large_safe_graph(self):
        """Test detection on large legitimate graph."""
        # Generate a random ER graph (safe)
        np.random.seed(42)
        n = 100
        p = 0.05
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < p:
                    edges.append((i, j))
        
        edges_df = cudf.DataFrame({
            'source': [e[0] for e in edges],
            'destination': [e[1] for e in edges]
        })
        
        result = detect_clique_chain(edges_df)
        
        # Random ER graphs should not have clique-chains
        assert result['is_adversarial'] is False
    
    def test_custom_column_names(self):
        """Test detection with custom column names."""
        edges = cudf.DataFrame({
            'src': [0, 1, 2],
            'dst': [1, 2, 0]
        })
        
        result = detect_clique_chain(edges, source_col='src', dest_col='dst')
        
        assert result['is_adversarial'] is False


class TestATGCConfiguration:
    """Test suite for ATGC configuration via environment variables."""
    
    def test_guard_enabled_by_default(self):
        """Test that guard is enabled by default."""
        from cugraph.security.atgc_detection import ATGC_GUARD_ENABLED
        assert ATGC_GUARD_ENABLED is True
    
    def test_default_action(self):
        """Test default action is 'reject'."""
        from cugraph.security.atgc_detection import ATGC_ACTION
        assert ATGC_ACTION == 'reject'
