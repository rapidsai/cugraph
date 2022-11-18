# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr

from pylibcugraph.components._connectivity cimport *
from pylibraft.common.handle cimport *


def _ensure_arg_types(**kwargs):
    """
    Ensure all args have a __cuda_array_interface__ attr
    """
    for (arg_name, arg_val) in kwargs.items():
        # FIXME: remove this special case when weights are supported: weights
        # can only be None
        if arg_name is "weights":
            if arg_val is not None:
                raise TypeError("weights are currently not supported and must "
                                "be None")

        elif not(hasattr(arg_val, "__cuda_array_interface__")):
            raise TypeError(f"{arg_name} must support __cuda_array_interface__")

        if arg_name in ["offsets", "indices", "labels"]:
            typestr = arg_val.__cuda_array_interface__.get("typestr")
            if typestr != "<i4":
                raise TypeError(f"{arg_name} array must have a dtype of int32")


def strongly_connected_components(offsets, indices, weights, num_verts, num_edges, labels):
    """
    Generate the Strongly Connected Components and attach a component label to
    each vertex.

    Parameters
    ----------
    offsets : object supporting a __cuda_array_interface__ interface
        Array containing the offsets values of a Compressed Sparse Row matrix
        that represents the graph.

    indices : object supporting a __cuda_array_interface__ interface
        Array containing the indices values of a Compressed Sparse Row matrix
        that represents the graph.

    weights : object supporting a __cuda_array_interface__ interface
        Array containing the weights values of a Compressed Sparse Row matrix
        that represents the graph.

        NOTE: weighted graphs are currently unsupported, and because of this the
        weights parameter can only be set to None.

    num_verts : int
        The number of vertices present in the graph represented by the CSR
        arrays above.

    num_edges : int
        The number of edges present in the graph represented by the CSR arrays
        above.

    labels : object supporting a __cuda_array_interface__ interface
        Array of size num_verts that will be populated with component label
        values. The component lables in the array are ordered based on the
        sorted vertex ID values of the graph.  For example, labels [9, 9, 7]
        mean vertex 0 is labelled 9, vertex 1 is labelled 9, and vertex 2 is
        labelled 7.

    Returns
    -------
    None

    Examples
    --------
    >>> import cupy as cp
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>>
    >>> graph = [
    ... [0, 1, 1, 0, 0],
    ... [0, 0, 1, 0, 0],
    ... [0, 0, 0, 0, 0],
    ... [0, 0, 0, 0, 1],
    ... [0, 0, 0, 0, 0],
    ... ]
    >>> scipy_csr = csr_matrix(graph)
    >>> num_verts = scipy_csr.get_shape()[0]
    >>> num_edges = scipy_csr.nnz
    >>>
    >>> cp_offsets = cp.asarray(scipy_csr.indptr)
    >>> cp_indices = cp.asarray(scipy_csr.indices, dtype=np.int32)
    >>> cp_labels = cp.asarray(np.zeros(num_verts, dtype=np.int32))
    >>>
    >>> strongly_connected_components(offsets=cp_offsets,
    ...                               indices=cp_indices,
    ...                               weights=None,
    ...                               num_verts=num_verts,
    ...                               num_edges=num_edges,
    ...                               labels=cp_labels)
    >>> print(f"{len(set(cp_labels.tolist()))} - {cp_labels}")
    5 - [0 1 2 3 4]
    """
    _ensure_arg_types(offsets=offsets, indices=indices,
                      weights=weights, labels=labels)

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get()

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_edge_weights = <uintptr_t>NULL
    cdef uintptr_t c_labels = labels.__cuda_array_interface__['data'][0]

    cdef GraphCSRView[int,int,float] g

    g = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>NULL, num_verts, num_edges)

    cdef cugraph_cc_t connect_type=CUGRAPH_STRONG
    connected_components(g, <cugraph_cc_t>connect_type, <int *>c_labels)

