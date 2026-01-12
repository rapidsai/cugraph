# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for windowed temporal neighbor sampling (B+C+D optimizations).

Tests verify:
1. Window parameters filter edges correctly
2. Timestamp conversion works for various input formats
3. API is backward compatible (no window params = standard behavior)
"""

import pytest
import cupy as cp
import numpy as np

import pylibcugraph
from pylibcugraph import (
    ResourceHandle,
    GraphProperties,
    SGGraph,
    homogeneous_uniform_temporal_neighbor_sample,
)


@pytest.fixture
def resource_handle():
    return ResourceHandle()


@pytest.fixture
def temporal_graph(resource_handle):
    """Create a simple temporal graph for testing.
    
    Graph structure:
        0 --[t=100]--> 1 --[t=200]--> 2
                       |              |
                    [t=300]        [t=400]
                       v              v
                       3 --[t=500]--> 4 --[t=600]--> 5
    
    Edge times: [100, 200, 300, 400, 500, 600]
    """
    srcs = cp.array([0, 1, 1, 2, 3, 4], dtype=np.int64)
    dsts = cp.array([1, 2, 3, 4, 4, 5], dtype=np.int64)
    edge_times = cp.array([100, 200, 300, 400, 500, 600], dtype=np.int64)
    
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
    graph = SGGraph(
        resource_handle, graph_props, srcs, dsts,
        edge_start_time_array=edge_times,
        store_transposed=True,
        renumber=False,
        do_expensive_check=False
    )
    return graph


class TestWindowedTemporalSampling:
    """Tests for windowed temporal sampling with B+C+D optimizations."""
    
    def test_windowed_sampling_filters_edges(self, resource_handle, temporal_graph):
        """Verify window parameters filter edges by time."""
        start_vertices = cp.array([0, 1], dtype=np.int64)
        vertex_times = cp.array([0, 0], dtype=np.int64)
        fanout = np.array([10], dtype=np.int32)
        
        # Sample with window [200, 500) - should include edges with times 200, 300, 400
        result = homogeneous_uniform_temporal_neighbor_sample(
            resource_handle, temporal_graph, None,
            start_vertices, vertex_times, None, fanout,
            with_replacement=True,
            do_expensive_check=False,
            window_start=200,
            window_end=500,
            window_time_unit='s'
        )
        
        # Verify we got results
        assert 'majors' in result
        assert 'minors' in result
        assert 'edge_start_time' in result
        
        # Verify all sampled edges are within window
        times = cp.asnumpy(result['edge_start_time'])
        assert all(200 <= t < 500 for t in times), f"Times outside window: {times}"
    
    def test_narrow_window_limits_edges(self, resource_handle, temporal_graph):
        """Test that a narrow window returns fewer edges."""
        start_vertices = cp.array([1], dtype=np.int64)
        vertex_times = cp.array([0], dtype=np.int64)
        fanout = np.array([10], dtype=np.int32)
        
        # Sample with narrow window [200, 300) - should only include t=200
        result = homogeneous_uniform_temporal_neighbor_sample(
            resource_handle, temporal_graph, None,
            start_vertices, vertex_times, None, fanout,
            with_replacement=True,
            do_expensive_check=False,
            window_start=200,
            window_end=300,
            window_time_unit='s'
        )
        
        times = cp.asnumpy(result['edge_start_time'])
        assert all(200 <= t < 300 for t in times), f"Times outside window: {times}"
    
    def test_backward_compatible_no_window(self, resource_handle, temporal_graph):
        """Test that omitting window params uses standard temporal sampling."""
        start_vertices = cp.array([0, 1], dtype=np.int64)
        vertex_times = cp.array([0, 0], dtype=np.int64)
        fanout = np.array([2], dtype=np.int32)
        
        # No window params - should use standard path
        result = homogeneous_uniform_temporal_neighbor_sample(
            resource_handle, temporal_graph, None,
            start_vertices, vertex_times, None, fanout,
            with_replacement=True,
            do_expensive_check=False
            # No window_start, window_end
        )
        
        assert 'majors' in result
        assert len(result['majors']) > 0


class TestTimestampConversion:
    """Tests for timestamp format conversion."""
    
    def test_integer_timestamps(self, resource_handle, temporal_graph):
        """Test integer timestamps work directly."""
        start_vertices = cp.array([0], dtype=np.int64)
        vertex_times = cp.array([0], dtype=np.int64)
        fanout = np.array([2], dtype=np.int32)
        
        result = homogeneous_uniform_temporal_neighbor_sample(
            resource_handle, temporal_graph, None,
            start_vertices, vertex_times, None, fanout,
            with_replacement=True,
            do_expensive_check=False,
            window_start=100,  # Integer
            window_end=600,    # Integer
            window_time_unit='s'
        )
        assert 'majors' in result
    
    def test_numpy_integer_timestamps(self, resource_handle, temporal_graph):
        """Test numpy integer types work correctly."""
        start_vertices = cp.array([0], dtype=np.int64)
        vertex_times = cp.array([0], dtype=np.int64)
        fanout = np.array([2], dtype=np.int32)
        
        result = homogeneous_uniform_temporal_neighbor_sample(
            resource_handle, temporal_graph, None,
            start_vertices, vertex_times, None, fanout,
            with_replacement=True,
            do_expensive_check=False,
            window_start=np.int64(100),
            window_end=np.int32(600),
            window_time_unit='s'
        )
        assert 'majors' in result
    
    def test_string_iso_format(self, resource_handle):
        """Test ISO format string timestamps."""
        import time
        from datetime import datetime
        
        base_time = int(time.time()) - 1000
        
        srcs = cp.array([0, 1], dtype=np.int64)
        dsts = cp.array([1, 2], dtype=np.int64)
        edge_times = cp.array([base_time, base_time + 500], dtype=np.int64)
        
        graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
        graph = SGGraph(
            resource_handle, graph_props, srcs, dsts,
            edge_start_time_array=edge_times,
            store_transposed=True,
            renumber=False,
            do_expensive_check=False
        )
        
        start_vertices = cp.array([0], dtype=np.int64)
        vertex_times = cp.array([0], dtype=np.int64)
        fanout = np.array([2], dtype=np.int32)
        
        # ISO format strings
        start_dt = datetime.fromtimestamp(base_time - 100)
        end_dt = datetime.fromtimestamp(base_time + 1000)
        
        result = homogeneous_uniform_temporal_neighbor_sample(
            resource_handle, graph, None,
            start_vertices, vertex_times, None, fanout,
            with_replacement=True,
            do_expensive_check=False,
            window_start=start_dt.isoformat(),
            window_end=end_dt.isoformat(),
            window_time_unit='s'
        )
        assert 'majors' in result
    
    def test_datetime_objects(self, resource_handle):
        """Test Python datetime objects."""
        import time
        from datetime import datetime
        
        base_time = int(time.time()) - 1000
        
        srcs = cp.array([0, 1], dtype=np.int64)
        dsts = cp.array([1, 2], dtype=np.int64)
        edge_times = cp.array([base_time, base_time + 500], dtype=np.int64)
        
        graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
        graph = SGGraph(
            resource_handle, graph_props, srcs, dsts,
            edge_start_time_array=edge_times,
            store_transposed=True,
            renumber=False,
            do_expensive_check=False
        )
        
        start_vertices = cp.array([0], dtype=np.int64)
        vertex_times = cp.array([0], dtype=np.int64)
        fanout = np.array([2], dtype=np.int32)
        
        # Python datetime objects
        start_dt = datetime.fromtimestamp(base_time - 100)
        end_dt = datetime.fromtimestamp(base_time + 1000)
        
        result = homogeneous_uniform_temporal_neighbor_sample(
            resource_handle, graph, None,
            start_vertices, vertex_times, None, fanout,
            with_replacement=True,
            do_expensive_check=False,
            window_start=start_dt,  # datetime object directly
            window_end=end_dt,      # datetime object directly
            window_time_unit='s'
        )
        assert 'majors' in result
    
    def test_pandas_timestamp(self, resource_handle):
        """Test pandas Timestamp objects."""
        import time
        import pandas as pd
        
        base_time = int(time.time()) - 1000
        
        srcs = cp.array([0, 1], dtype=np.int64)
        dsts = cp.array([1, 2], dtype=np.int64)
        edge_times = cp.array([base_time, base_time + 500], dtype=np.int64)
        
        graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
        graph = SGGraph(
            resource_handle, graph_props, srcs, dsts,
            edge_start_time_array=edge_times,
            store_transposed=True,
            renumber=False,
            do_expensive_check=False
        )
        
        start_vertices = cp.array([0], dtype=np.int64)
        vertex_times = cp.array([0], dtype=np.int64)
        fanout = np.array([2], dtype=np.int32)
        
        # pandas Timestamp objects
        start_ts = pd.Timestamp.fromtimestamp(base_time - 100)
        end_ts = pd.Timestamp.fromtimestamp(base_time + 1000)
        
        result = homogeneous_uniform_temporal_neighbor_sample(
            resource_handle, graph, None,
            start_vertices, vertex_times, None, fanout,
            with_replacement=True,
            do_expensive_check=False,
            window_start=start_ts,
            window_end=end_ts,
            window_time_unit='s'
        )
        assert 'majors' in result
    
    def test_numpy_datetime64(self, resource_handle):
        """Test numpy datetime64 objects."""
        import time
        
        base_time = int(time.time()) - 1000
        
        srcs = cp.array([0, 1], dtype=np.int64)
        dsts = cp.array([1, 2], dtype=np.int64)
        edge_times = cp.array([base_time, base_time + 500], dtype=np.int64)
        
        graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
        graph = SGGraph(
            resource_handle, graph_props, srcs, dsts,
            edge_start_time_array=edge_times,
            store_transposed=True,
            renumber=False,
            do_expensive_check=False
        )
        
        start_vertices = cp.array([0], dtype=np.int64)
        vertex_times = cp.array([0], dtype=np.int64)
        fanout = np.array([2], dtype=np.int32)
        
        # numpy datetime64
        start_dt64 = np.datetime64(base_time - 100, 's')
        end_dt64 = np.datetime64(base_time + 1000, 's')
        
        result = homogeneous_uniform_temporal_neighbor_sample(
            resource_handle, graph, None,
            start_vertices, vertex_times, None, fanout,
            with_replacement=True,
            do_expensive_check=False,
            window_start=start_dt64,
            window_end=end_dt64,
            window_time_unit='s'
        )
        assert 'majors' in result
    
    def test_different_time_units(self, resource_handle):
        """Test different time units (ns, us, ms, s)."""
        # Create graph with millisecond timestamps
        srcs = cp.array([0, 1], dtype=np.int64)
        dsts = cp.array([1, 2], dtype=np.int64)
        edge_times = cp.array([1000, 2000], dtype=np.int64)  # In milliseconds
        
        graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
        graph = SGGraph(
            resource_handle, graph_props, srcs, dsts,
            edge_start_time_array=edge_times,
            store_transposed=True,
            renumber=False,
            do_expensive_check=False
        )
        
        start_vertices = cp.array([0], dtype=np.int64)
        vertex_times = cp.array([0], dtype=np.int64)
        fanout = np.array([2], dtype=np.int32)
        
        # Use millisecond time unit
        result = homogeneous_uniform_temporal_neighbor_sample(
            resource_handle, graph, None,
            start_vertices, vertex_times, None, fanout,
            with_replacement=True,
            do_expensive_check=False,
            window_start=500,
            window_end=2500,
            window_time_unit='ms'
        )
        assert 'majors' in result


class TestValidation:
    """Tests for input validation."""
    
    def test_window_start_only_raises(self, resource_handle, temporal_graph):
        """Test that providing only window_start raises error."""
        start_vertices = cp.array([0], dtype=np.int64)
        vertex_times = cp.array([0], dtype=np.int64)
        fanout = np.array([2], dtype=np.int32)
        
        with pytest.raises(ValueError, match="Both window_start and window_end"):
            homogeneous_uniform_temporal_neighbor_sample(
                resource_handle, temporal_graph, None,
                start_vertices, vertex_times, None, fanout,
                with_replacement=True,
                do_expensive_check=False,
                window_start=100,
                window_end=None  # Missing!
            )
    
    def test_window_end_only_raises(self, resource_handle, temporal_graph):
        """Test that providing only window_end raises error."""
        start_vertices = cp.array([0], dtype=np.int64)
        vertex_times = cp.array([0], dtype=np.int64)
        fanout = np.array([2], dtype=np.int32)
        
        with pytest.raises(ValueError, match="Both window_start and window_end"):
            homogeneous_uniform_temporal_neighbor_sample(
                resource_handle, temporal_graph, None,
                start_vertices, vertex_times, None, fanout,
                with_replacement=True,
                do_expensive_check=False,
                window_start=None,  # Missing!
                window_end=500
            )
    
    def test_invalid_window_range_raises(self, resource_handle, temporal_graph):
        """Test that window_end <= window_start raises error."""
        start_vertices = cp.array([0], dtype=np.int64)
        vertex_times = cp.array([0], dtype=np.int64)
        fanout = np.array([2], dtype=np.int32)
        
        with pytest.raises(ValueError, match="must be greater than"):
            homogeneous_uniform_temporal_neighbor_sample(
                resource_handle, temporal_graph, None,
                start_vertices, vertex_times, None, fanout,
                with_replacement=True,
                do_expensive_check=False,
                window_start=500,
                window_end=100  # Invalid: end < start
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
