# Copyright (c) 2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.types cimport (
    bool_t,
)
from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_count_multi_edges,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)


def count_multi_edges(ResourceHandle resource_handle,
                      _GPUGraph graph,
                      bool_t do_expensive_check):
    """
    Count the number of multi-edges in the graph.  This returns
    the number of duplicates.  If the edge (u, v) appears k times
    in the graph, then that edge will contribute (k-1) toward the
    total number of duplicates.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments if True.

    Returns
    -------
    Total count of duplicate edges in the graph

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 0, 0], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 1, 1], dtype=numpy.int32)
    >>> weights = cupy.asarray([1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> count = pylibcugraph.count_multi_edges(resource_handle, G, False)

    """

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef size_t result
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    error_code = cugraph_count_multi_edges(c_resource_handle_ptr,
                                           c_graph_ptr,
                                           do_expensive_check,
                                           &result,
                                           &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_count_multi_edges")

    return result;
