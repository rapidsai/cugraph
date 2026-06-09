# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Production-grade test suite for ATGC (Adversarial Topology Against GPU Compute) detection.

This module provides comprehensive tests for the ATGC detection system including:
- Unit tests for individual detection components
- Integration tests for the full detection pipeline
- Performance tests for detection speed
- Edge case and error handling tests

Test Categories:
    TestATGCConfig: Configuration validation tests
    TestATGCDetectorUnit: Unit tests for detector components
    TestATGCDetectorIntegration: Integration tests for full detection
    TestATGCDetectorPerformance: Performance benchmarks
    TestATGCDetectorEdgeCases: Edge case and error handling tests
"""

import os
import time
import warnings

import pytest
import cudf
import numpy as np

from cugraph.security.atgc_detection import (
    ATGCConfig,
    ATGCDetector,
    ATGCResult,
    detect_clique_chain,
    validate_graph_input,
    _compute_degree_variance,
    _detect_cliques,
    _check_clique_chain,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def safe_path_graph():
    """Generate a simple path graph (safe, low variance)."""
    n = 100
    edges = [(i, i + 1) for i in range(n - 1)]
    return cudf.DataFrame({
        'source': [e[0] for e in edges],
        'destination': [e[1] for e in edges]
    })


@pytest.fixture
def safe_grid_graph():
    """Generate a grid graph (safe, moderate variance)."""
    n = 10
    edges = []
    for i in range(n):
        for j in range(n):
            node = i * n + j
            if j < n - 1:
                edges.append((node, node + 1))
            if i < n - 1:
                edges.append((node, node + n))
    return cudf.DataFrame({
        'source': [e[0] for e in edges],
        'destination': [e[1] for e in edges]
    })


@pytest.fixture
def safe_random_graph():
    """Generate an Erdos-Renyi random graph (safe)."""
    np.random.seed(42)
    n = 100
    p = 0.05
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < p:
                edges.append((i, j))
    return cudf.DataFrame({
        'source': [e[0] for e in edges],
        'destination': [e[1] for e in edges]
    })


@pytest.fixture
def adversarial_clique_chain():
    """Generate an adversarial clique-chain graph G(10, 2)."""
    k = 10
    m = 2
    edges = []
    
    for clique_idx in range(m):
        offset = clique_idx * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((offset + i, offset + j))
        
        if clique_idx < m - 1:
            bridge_u = offset + k - 1
            bridge_v = offset + k
            edges.append((bridge_u, bridge_v))
    
    return cudf.DataFrame({
        'source': [e[0] for e in edges],
        'destination': [e[1] for e in edges]
    })


@pytest.fixture
def adversarial_clique_chain_large():
    """Generate a larger adversarial clique-chain graph G(15, 3)."""
    k = 15
    m = 3
    edges = []
    
    for clique_idx in range(m):
        offset = clique_idx * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((offset + i, offset + j))
        
        if clique_idx < m - 1:
            bridge_u = offset + k - 1
            bridge_v = offset + k
            edges.append((bridge_u, bridge_v))
    
    return cudf.DataFrame({
        'source': [e[0] for e in edges],
        'destination': [e[1] for e in edges]
    })


# ============================================================================
# TestATGCConfig
# ============================================================================

class TestATGCConfig:
    """Test suite for ATGCConfig class."""
    
    def test_default_values(self):
        """Test that default configuration values are correct."""
        config = ATGCConfig()
        assert config.enabled is True
        assert config.action == 'reject'
        assert config.log_level == 'WARNING'
        assert config.max_clique_size == 20
        assert config.min_clique_count == 2
        assert config.variance_threshold == 100.0
    
    def test_from_env_defaults(self):
        """Test configuration from environment with default values."""
        # Clear any existing env vars
        for key in ['CUGRAPH_ENABLE_ATGC_GUARD', 'CUGRAPH_ATGC_ACTION', 
                     'CUGRAPH_ATGC_LOG_LEVEL', 'CUGRAPH_ATGC_MAX_CLIQUE_SIZE',
                     'CUGRAPH_ATGC_MIN_CLIQUE_COUNT', 'CUGRAPH_ATGC_VARIANCE_THRESHOLD']:
            if key in os.environ:
                del os.environ[key]
        
        config = ATGCConfig.from_env()
        assert config.enabled is True
        assert config.action == 'reject'
    
    def test_from_env_custom(self):
        """Test configuration from environment with custom values."""
        os.environ['CUGRAPH_ENABLE_ATGC_GUARD'] = 'false'
        os.environ['CUGRAPH_ATGC_ACTION'] = 'warn'
        os.environ['CUGRAPH_ATGC_LOG_LEVEL'] = 'DEBUG'
        os.environ['CUGRAPH_ATGC_MAX_CLIQUE_SIZE'] = '25'
        os.environ['CUGRAPH_ATGC_MIN_CLIQUE_COUNT'] = '3'
        os.environ['CUGRAPH_ATGC_VARIANCE_THRESHOLD'] = '200.0'
        
        try:
            config = ATGCConfig.from_env()
            assert config.enabled is False
            assert config.action == 'warn'
            assert config.log_level == 'DEBUG'
            assert config.max_clique_size == 25
            assert config.min_clique_count == 3
            assert config.variance_threshold == 200.0
        finally:
            # Clean up
            for key in ['CUGRAPH_ENABLE_ATGC_GUARD', 'CUGRAPH_ATGC_ACTION', 
                         'CUGRAPH_ATGC_LOG_LEVEL', 'CUGRAPH_ATGC_MAX_CLIQUE_SIZE',
                         'CUGRAPH_ATGC_MIN_CLIQUE_COUNT', 'CUGRAPH_ATGC_VARIANCE_THRESHOLD']:
                if key in os.environ:
                    del os.environ[key]
    
    def test_validate_valid(self):
        """Test validation with valid configuration."""
        config = ATGCConfig()
        config.validate()  # Should not raise
    
    def test_validate_invalid_action(self):
        """Test validation with invalid action."""
        config = ATGCConfig(action='invalid')
        with pytest.raises(ValueError, match="Invalid action"):
            config.validate()
    
    def test_validate_invalid_log_level(self):
        """Test validation with invalid log level."""
        config = ATGCConfig(log_level='INVALID')
        with pytest.raises(ValueError, match="Invalid log level"):
            config.validate()
    
    def test_validate_invalid_clique_size(self):
        """Test validation with invalid clique size."""
        config = ATGCConfig(max_clique_size=1)
        with pytest.raises(ValueError, match="max_clique_size must be >= 2"):
            config.validate()
    
    def test_validate_invalid_clique_count(self):
        """Test validation with invalid clique count."""
        config = ATGCConfig(min_clique_count=0)
        with pytest.raises(ValueError, match="min_clique_count must be >= 1"):
            config.validate()
    
    def test_validate_invalid_variance(self):
        """Test validation with invalid variance threshold."""
        config = ATGCConfig(variance_threshold=0)
        with pytest.raises(ValueError, match="variance_threshold must be > 0"):
            config.validate()


# ============================================================================
# TestATGCDetectorUnit
# ============================================================================

class TestATGCDetectorUnit:
    """Unit tests for ATGCDetector components."""
    
    def test_detector_init_default(self):
        """Test detector initialization with default config."""
        detector = ATGCDetector()
        assert detector.config.enabled is True
        assert detector.config.action == 'reject'
    
    def test_detector_init_custom(self):
        """Test detector initialization with custom config."""
        config = ATGCConfig(max_clique_size=25, action='warn')
        detector = ATGCDetector(config)
        assert detector.config.max_clique_size == 25
        assert detector.config.action == 'warn'
    
    def test_compute_degree_variance_path(self, safe_path_graph):
        """Test degree variance computation for path graph."""
        detector = ATGCDetector()
        variance = detector._compute_degree_variance(safe_path_graph, 'source', 'destination')
        
        # Path graph should have low variance
        assert variance >= 0
        assert variance < 100.0
    
    def test_compute_degree_variance_clique_chain(self, adversarial_clique_chain):
        """Test degree variance computation for clique-chain."""
        detector = ATGCDetector()
        variance = detector._compute_degree_variance(adversarial_clique_chain, 'source', 'destination')
        
        # Clique-chain should have high variance
        assert variance > 100.0
    
    def test_detect_cliques_safe(self, safe_path_graph):
        """Test clique detection on safe graph."""
        detector = ATGCDetector()
        cliques = detector._detect_cliques(safe_path_graph, 'source', 'destination')
        
        # Path graph should have no large cliques
        assert len(cliques) == 0
    
    def test_detect_cliques_adversarial(self, adversarial_clique_chain):
        """Test clique detection on adversarial graph."""
        detector = ATGCDetector()
        cliques = detector._detect_cliques(adversarial_clique_chain, 'source', 'destination')
        
        # Should detect cliques
        assert len(cliques) >= 2
        assert all(len(c) >= 10 for c in cliques)
    
    def test_check_clique_chain_true(self, adversarial_clique_chain):
        """Test chain check on true clique-chain."""
        detector = ATGCDetector()
        cliques = detector._detect_cliques(adversarial_clique_chain, 'source', 'destination')
        
        assert len(cliques) >= 2
        is_chain = detector._check_clique_chain(cliques, adversarial_clique_chain, 'source', 'destination')
        assert is_chain is True
    
    def test_check_clique_chain_false(self, safe_grid_graph):
        """Test chain check on non-chain graph."""
        detector = ATGCDetector()
        cliques = detector._detect_cliques(safe_grid_graph, 'source', 'destination')
        
        # Should either have no cliques or not form a chain
        if len(cliques) >= 2:
            is_chain = detector._check_clique_chain(cliques, safe_grid_graph, 'source', 'destination')
            assert is_chain is False


# ============================================================================
# TestATGCDetectorIntegration
# ============================================================================

class TestATGCDetectorIntegration:
    """Integration tests for full ATGCDetector pipeline."""
    
    def test_detect_safe_path(self, safe_path_graph):
        """Test detection on safe path graph."""
        detector = ATGCDetector()
        result = detector.detect(safe_path_graph)
        
        assert result.is_adversarial is False
        assert result.confidence == 'low'
        assert result.elapsed_ms > 0
    
    def test_detect_safe_grid(self, safe_grid_graph):
        """Test detection on safe grid graph."""
        detector = ATGCDetector()
        result = detector.detect(safe_grid_graph)
        
        assert result.is_adversarial is False
    
    def test_detect_safe_random(self, safe_random_graph):
        """Test detection on safe random graph."""
        detector = ATGCDetector()
        result = detector.detect(safe_random_graph)
        
        assert result.is_adversarial is False
    
    def test_detect_adversarial_clique_chain(self, adversarial_clique_chain):
        """Test detection on adversarial clique-chain."""
        detector = ATGCDetector()
        result = detector.detect(adversarial_clique_chain)
        
        assert result.is_adversarial is True
        assert result.confidence == 'high'
        assert result.clique_count >= 2
        assert result.max_clique_size >= 10
    
    def test_detect_adversarial_large(self, adversarial_clique_chain_large):
        """Test detection on larger adversarial clique-chain."""
        detector = ATGCDetector()
        result = detector.detect(adversarial_clique_chain_large)
        
        assert result.is_adversarial is True
        assert result.clique_count >= 3
        assert result.max_clique_size >= 15
    
    def test_validate_safe(self, safe_path_graph):
        """Test validation on safe graph."""
        detector = ATGCDetector()
        
        # Should not raise
        detector.validate(safe_path_graph)
    
    def test_validate_reject(self, adversarial_clique_chain):
        """Test validation rejects adversarial graph."""
        detector = ATGCDetector()
        
        with pytest.raises(ValueError, match="ATGC Detection triggered"):
            detector.validate(adversarial_clique_chain)
    
    def test_validate_warn(self, adversarial_clique_chain):
        """Test validation warns on adversarial graph."""
        config = ATGCConfig(action='warn')
        detector = ATGCDetector(config)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            detector.validate(adversarial_clique_chain)
            
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "ATGC Detection triggered" in str(w[0].message)
    
    def test_validate_allow(self, adversarial_clique_chain):
        """Test validation allows adversarial graph with allow action."""
        config = ATGCConfig(action='allow')
        detector = ATGCDetector(config)
        
        # Should not raise or warn
        detector.validate(adversarial_clique_chain)
    
    def test_validate_disabled(self, adversarial_clique_chain):
        """Test validation skipped when disabled."""
        config = ATGCConfig(enabled=False)
        detector = ATGCDetector(config)
        
        # Should not raise
        detector.validate(adversarial_clique_chain)
    
    def test_detect_empty_graph(self):
        """Test detection on empty graph."""
        detector = ATGCDetector()
        empty_df = cudf.DataFrame({'source': [], 'destination': []})
        
        result = detector.detect(empty_df)
        
        assert result.is_adversarial is False
        assert 'Empty graph' in result.message
    
    def test_detect_invalid_columns(self):
        """Test detection with invalid column names."""
        detector = ATGCDetector()
        df = cudf.DataFrame({'src': [0, 1], 'dst': [1, 2]})
        
        # Should raise ValueError for missing columns
        with pytest.raises(ValueError, match="must exist in DataFrame"):
            detector.detect(df, source_col='invalid', dest_col='destination')
    
    def test_detect_custom_columns(self):
        """Test detection with custom column names."""
        detector = ATGCDetector()
        df = cudf.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 3]})
        
        result = detector.detect(df, source_col='src', dest_col='dst')
        
        assert result.is_adversarial is False
    
    def test_result_to_dict(self):
        """Test ATGCResult conversion to dictionary."""
        result = ATGCResult(
            is_adversarial=True,
            confidence='high',
            degree_variance=150.0,
            clique_count=3,
            max_clique_size=20,
            message='Test message',
            elapsed_ms=5.0
        )
        
        d = result.to_dict()
        
        assert d['is_adversarial'] is True
        assert d['confidence'] == 'high'
        assert d['degree_variance'] == 150.0
        assert d['clique_count'] == 3
        assert d['max_clique_size'] == 20
        assert d['message'] == 'Test message'
        assert d['elapsed_ms'] == 5.0


# ============================================================================
# TestATGCDetectorPerformance
# ============================================================================

class TestATGCDetectorPerformance:
    """Performance tests for ATGCDetector."""
    
    def test_fast_path_performance(self, safe_path_graph):
        """Test that fast path completes quickly."""
        detector = ATGCDetector()
        
        # Warm up
        detector.detect(safe_path_graph)
        
        # Benchmark
        iterations = 10
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = detector.detect(safe_path_graph)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time_ms = np.mean(times) * 1000
        
        # Fast path should be < 100ms for 100-node graph
        assert avg_time_ms < 100.0, f"Fast path took {avg_time_ms:.2f}ms, expected < 100ms"
    
    def test_deep_inspection_performance(self, adversarial_clique_chain):
        """Test that deep inspection completes in reasonable time."""
        detector = ATGCDetector()
        
        # Warm up
        detector.detect(adversarial_clique_chain)
        
        # Benchmark
        iterations = 10
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = detector.detect(adversarial_clique_chain)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time_ms = np.mean(times) * 1000
        
        # Deep inspection should be < 500ms for clique-chain
        assert avg_time_ms < 500.0, f"Deep inspection took {avg_time_ms:.2f}ms, expected < 500ms"
    
    def test_large_graph_performance(self):
        """Test performance on large legitimate graph."""
        np.random.seed(42)
        n = 1000
        p = 0.01
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < p:
                    edges.append((i, j))
        
        df = cudf.DataFrame({
            'source': [e[0] for e in edges],
            'destination': [e[1] for e in edges]
        })
        
        detector = ATGCDetector()
        
        start = time.perf_counter()
        result = detector.detect(df)
        end = time.perf_counter()
        
        elapsed_ms = (end - start) * 1000
        
        # Should be < 1000ms for 1000-node graph
        assert elapsed_ms < 1000.0, f"Large graph took {elapsed_ms:.2f}ms, expected < 1000ms"
        assert result.is_adversarial is False


# ============================================================================
# TestLegacyAPI
# ============================================================================

class TestLegacyAPI:
    """Tests for legacy API functions (backward compatibility)."""
    
    def test_detect_clique_chain_safe(self, safe_path_graph):
        """Test legacy detect_clique_chain on safe graph."""
        result = detect_clique_chain(safe_path_graph)
        
        assert result['is_adversarial'] is False
        assert result['confidence'] == 'low'
    
    def test_detect_clique_chain_adversarial(self, adversarial_clique_chain):
        """Test legacy detect_clique_chain on adversarial graph."""
        result = detect_clique_chain(adversarial_clique_chain)
        
        assert result['is_adversarial'] is True
        assert result['confidence'] == 'high'
    
    def test_validate_graph_input_safe(self, safe_path_graph):
        """Test legacy validate_graph_input on safe graph."""
        # Should not raise
        validate_graph_input(safe_path_graph)
    
    def test_validate_graph_input_reject(self, adversarial_clique_chain):
        """Test legacy validate_graph_input rejects adversarial graph."""
        with pytest.raises(ValueError, match="ATGC Detection"):
            validate_graph_input(adversarial_clique_chain)
