# Copyright (c) 2022, NVIDIA CORPORATION.
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

from pylibcugraph._cugraph_c.cugraph_api cimport (
    bool_t,
    data_type_id_t,
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
# FIXME: update this with corresponding HITS
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_hits_result_t,
    cugraph_hits,
    cugraph_hits_result_get_vertices,
    cugraph_hits_result_get_hubs,
    cugraph_hits_result_get_authorities,
    cugraph_hits_result_free,
)

from pylibcugraph.resource_handle cimport (
    EXPERIMENTAL__ResourceHandle,
)
# FIXME: update with EXPERIMENTAL__MGGraph(_GPUGraph):
""""
from pylibcugraph.graphs cimport (
    _GPUGraph,
)
"""
from pylibcugraph.utils cimport (
    assert_success,
    assert_CAI_type,
    copy_to_cupy_array,
)

# FIXME: nstart is not supported in sg_hits, check if it is the case
# for mg_hits
def EXPERIMENTAL__hits(EXPERIMENTAL__ResourceHandle resource_handle,
                           _GPUGraph graph,
                           size_t max_iter,
                           double tol,
                           nstart,
                           bool_t normalized,
                           bool_t do_expensive_check):
    """
    Compute HITS hubs and authorities values for each vertex

    The HITS algorithm computes two numbers for a node.  Authorities
    estimates the node value based on the incoming links.  Hubs estimates
    the node value based on outgoing links.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : MGGraph
        The input graph.
    
    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned.
    
    tol : float, optional (default=1.0e-5)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.  This parameter is not currently supported.

    nstart : None
        This parameter is unsupported in this release and only None is
        accepted.

    normalized : bool, optional (default=True)
        This parameter is unsupported in this release and only True is
        accepted.
    
    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    A tuple of device arrays, where the third item in the tuple is a device
    array containing the vertex identifiers, the first and second items are device
    arrays containing respectively the hubs and authorities values for the corresponding
    vertices 

    Examples
    --------
    # FIXME: cugraph mg_hits will call pylibcugraph hits and pass the __cuda_array_interface__
    # of srcs, dsts and weights

    >>> srcs = ...
    >>> dsts = ...
    >>> weights = ...
    >>> resource_handle = pylibcugraph.experimental.ResourceHandle()
    >>> graph_props = pylibcugraph.experimental.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibcugraph.experimental.MGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weights,
    ...     store_transposed=True, renumber=False, do_expensive_check=False)
    >>> (hubs, authorities, vertex) = pylibcugraph.experimental.hits(
    ...     resource_handle, G, max_iter=100, tol=1.0e-5, None,
    ...     normalized=True)
    """

    # FIXME: import these modules here for now until a better pattern can be
    # used for optional imports (perhaps 'import_optional()' from cugraph), or
    # these are made hard dependencies.
    try:
        import cupy
    except ModuleNotFoundError:
        raise RuntimeError("hits requires the cupy package, which could "
                           "not be imported")
    try:
        import numpy
    except ModuleNotFoundError:
        raise RuntimeError("hits requires the numpy package, which could "
                           "not be imported")
    
    assert_CAI_type(nstart,
                    "nstart",
                    allow_None=True)

    # FIXME: Maybe add a function to ensure that a parameter is set to a default 
    # value because Not Implemented in utils.pyx

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef cugraph_type_erased_device_array_view_t* nstart_ptr = NULL
    if nstart:
        raise NotImplementedError("None is temporarily the only supported "
                                  "value for nstart")
    if normalized is not True:
        raise NotImplementedError("True is temporarily the only supported "
                                  "value for normalized")

    cdef cugraph_hits_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    error_code = cugraph_hits(c_resource_handle_ptr,
                                  c_graph_ptr,
                                  max_iter,
                                  tol,
                                  nstart_ptr,
                                  normalized,
                                  do_expensive_check,
                                  &result_ptr,
                                  &error_ptr)
    assert_success(error_code, error_ptr, "hits")

    # Extract individual device array pointers from result and copy to cupy
    # arrays for returning.
    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_hits_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* hubs_ptr = \
        cugraph_hits_result_get_hubs(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* authorities_ptr = \
        cugraph_hits_result_get_authorities(result_ptr)

    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
    cupy_hubs = copy_to_cupy_array(c_resource_handle_ptr, hubs_ptr)
    cupy_authorities = copy_to_cupy_array(c_resource_handle_ptr, authorities_ptr)

    cugraph_hits_result_free(result_ptr)

    return (cupy_vertices, cupy_hubs, cupy_authorities)