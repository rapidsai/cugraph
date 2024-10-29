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

from libc.stdint cimport uintptr_t

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
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_view_free,
    cugraph_type_erased_host_array_view_t,
    cugraph_type_erased_host_array_view_create,
    cugraph_type_erased_host_array_view_free,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.sampling_algorithms cimport (
    cugraph_negative_sampling,
)
from pylibcugraph._cugraph_c.coo cimport (
    cugraph_coo_t,
)
from pylibcugraph.internal_types.coo cimport (
    COO,
)
from pylibcugraph.utils cimport (
    assert_success,
    assert_CAI_type,
    create_cugraph_type_erased_device_array_view_from_py_obj,
)
from pylibcugraph._cugraph_c.random cimport (
    cugraph_rng_state_t
)
from pylibcugraph.random cimport (
    CuGraphRandomState
)

def negative_sampling(ResourceHandle resource_handle,
                      _GPUGraph graph,
                      size_t num_samples,
                      random_state=None,
                      vertices=None,
                      src_bias=None,
                      dst_bias=None,
                      remove_duplicates=False,
                      remove_false_negatives=False,
                      exact_number_of_samples=False,
                      do_expensive_check=False):
    """
    Performs negative sampling, which is essentially a form of graph generation.

    By setting vertices, src_bias, and dst_bias, this function can perform
    biased negative sampling.

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle to the underlying device and host resources needed for
        referencing data and running algorithms.
    input_graph: SGGraph or MGGraph
        The stored cuGraph graph to create negative samples for.
    num_samples: int
        The number of negative edges to generate for each positive edge.
    random_state: int (Optional)
        Random state to use when generating samples.  Optional argument,
        defaults to a hash of process id, time, and hostname.
        (See pylibcugraph.random.CuGraphRandomState)
    vertices: device array type (Optional)
        Vertex ids corresponding to the src/dst biases, if provided.
        Ignored if src/dst biases are not provided.
    src_bias: device array type (Optional)
        Probability per edge that a vertex is selected as a source vertex.
        Does not have to be normalized.  Uses a uniform distribution if
        not provided.
    dst_bias: device array type (Optional)
        Probability per edge that a vertex is selected as a destination vertex.
        Does not have to be normalized.  Uses a uniform distribution if
        not provided.
    remove_duplicates: bool (Optional)
        Whether to remove duplicate edges from the generated edgelist.
        Defaults to False (does not remove duplicates).
    remove_false_negatives: bool (Optional)
        Whether to remove false negatives from the generated edgelist.
        Defaults to False (does not check for and remove false negatives).
    exact_number_of_samples: bool (Optional)
        Whether to manually regenerate samples until the desired number
        as specified by num_samples has been generated.
        Defaults to False (does not regenerate if enough samples are not
        produced in the initial round).
    do_expensive_check: bool (Optional)
        Whether to perform an expensive error check at the C++ level.
        Defaults to False (no error check).

    Returns
    -------
    dict[str, cupy.ndarray]
        Generated edges in COO format.
    """

    assert_CAI_type(vertices, "vertices", True)
    assert_CAI_type(src_bias, "src_bias", True)
    assert_CAI_type(dst_bias, "dst_bias", True)

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = (
        resource_handle.c_resource_handle_ptr
    )

    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef bool_t c_remove_duplicates = remove_duplicates
    cdef bool_t c_remove_false_negatives = remove_false_negatives
    cdef bool_t c_exact_number_of_samples = exact_number_of_samples
    cdef bool_t c_do_expensive_check = do_expensive_check

    cg_rng_state = CuGraphRandomState(resource_handle, random_state)

    cdef cugraph_rng_state_t* rng_state_ptr = \
        cg_rng_state.rng_state_ptr

    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        create_cugraph_type_erased_device_array_view_from_py_obj(vertices)
    cdef cugraph_type_erased_device_array_view_t* src_bias_ptr = \
        create_cugraph_type_erased_device_array_view_from_py_obj(src_bias)
    cdef cugraph_type_erased_device_array_view_t* dst_bias_ptr = \
        create_cugraph_type_erased_device_array_view_from_py_obj(dst_bias)

    cdef cugraph_coo_t* result_ptr
    cdef cugraph_error_t* err_ptr
    cdef cugraph_error_code_t error_code

    error_code = cugraph_negative_sampling(
        c_resource_handle_ptr,
        rng_state_ptr,
        c_graph_ptr,
        vertices_ptr,
        src_bias_ptr,
        dst_bias_ptr,
        num_samples,
        c_remove_duplicates,
        c_remove_false_negatives,
        c_exact_number_of_samples,
        c_do_expensive_check,
        &result_ptr,
        &err_ptr,
    )
    assert_success(error_code, err_ptr, "cugraph_negative_sampling")

    coo = COO()
    coo.set_ptr(result_ptr)

    return {
        'sources': coo.get_sources(),
        'destinations': coo.get_destinations(),
        'edge_id': coo.get_edge_ids(),
        'edge_type': coo.get_edge_types(),
        'weight': coo.get_edge_weights(),
    }
