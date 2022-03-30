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

from libc.stdint cimport uintptr_t

from pylibcugraph._cugraph_c.resource_handle cimport (
    bool_t,
    cugraph_resource_handle_t,
    data_type_id_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_view_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
    cugraph_sg_graph_create,
    cugraph_mg_graph_create,
    cugraph_graph_properties_t,
    cugraph_sg_graph_free,
    cugraph_mg_graph_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
    cugraph_mg_graph_create,
    cugraph_graph_properties_t,
    cugraph_mg_graph_free,
)
from pylibcugraph.resource_handle cimport (
    EXPERIMENTAL__ResourceHandle,
)
from pylibcugraph.graph_properties cimport (
    EXPERIMENTAL__GraphProperties,
)
from pylibcugraph.utils cimport (
    assert_success,
    assert_CAI_type,
    get_c_type_from_numpy_type,
)


cdef class EXPERIMENTAL__SGGraph(_GPUGraph):
    """
    RAII-stye Graph class for use with single-GPU APIs that manages the
    individual create/free calls and the corresponding cugraph_graph_t pointer.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph_properties : GraphProperties
        Object defining intended properties for the graph.

    src_array : device array type
        Device array containing the vertex identifiers of the source of each
        directed edge. The order of the array corresponds to the ordering of the
        dst_array, where the ith item in src_array and the ith item in dst_array
        define the ith edge of the graph.

    dst_array : device array type
        Device array containing the vertex identifiers of the destination of
        each directed edge. The order of the array corresponds to the ordering
        of the src_array, where the ith item in src_array and the ith item in
        dst_array define the ith edge of the graph.

    weight_array : device array type
        Device array containing the weight values of each directed edge. The
        order of the array corresponds to the ordering of the src_array and
        dst_array arrays, where the ith item in weight_array is the weight value
        of the ith edge of the graph.

    store_transposed : bool
        Set to True if the graph should be transposed. This is required for some
        algorithms, such as pagerank.

    renumber : bool
        Set to True to indicate the vertices used in src_array and dst_array are
        not appropriate for use as internal array indices, and should be mapped
        to continuous integers starting from 0.

    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.
    """
    def __cinit__(self,
                  EXPERIMENTAL__ResourceHandle resource_handle,
                  EXPERIMENTAL__GraphProperties graph_properties,
                  src_array,
                  dst_array,
                  weight_array,
                  store_transposed,
                  renumber,
                  do_expensive_check):

        # FIXME: add tests for these
        if not(isinstance(store_transposed, (int, bool))):
            raise TypeError("expected int or bool for store_transposed, got "
                            f"{type(store_transposed)}")
        if not(isinstance(renumber, (int, bool))):
            raise TypeError("expected int or bool for renumber, got "
                            f"{type(renumber)}")
        if not(isinstance(do_expensive_check, (int, bool))):
            raise TypeError("expected int or bool for do_expensive_check, got "
                            f"{type(do_expensive_check)}")
        assert_CAI_type(src_array, "src_array")
        assert_CAI_type(dst_array, "dst_array")
        assert_CAI_type(weight_array, "weight_array")

        # FIXME: assert that src_array and dst_array have the same type

        cdef cugraph_error_t* error_ptr
        cdef cugraph_error_code_t error_code

        cdef uintptr_t cai_srcs_ptr = \
            src_array.__cuda_array_interface__["data"][0]
        cdef cugraph_type_erased_device_array_view_t* srcs_view_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_srcs_ptr,
                len(src_array),
                get_c_type_from_numpy_type(src_array.dtype))

        cdef uintptr_t cai_dsts_ptr = \
            dst_array.__cuda_array_interface__["data"][0]
        cdef cugraph_type_erased_device_array_view_t* dsts_view_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_dsts_ptr,
                len(dst_array),
                get_c_type_from_numpy_type(dst_array.dtype))

        cdef uintptr_t cai_weights_ptr = \
            weight_array.__cuda_array_interface__["data"][0]
        cdef cugraph_type_erased_device_array_view_t* weights_view_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_weights_ptr,
                len(weight_array),
                get_c_type_from_numpy_type(weight_array.dtype))

        error_code = cugraph_sg_graph_create(
            resource_handle.c_resource_handle_ptr,
            &(graph_properties.c_graph_properties),
            srcs_view_ptr,
            dsts_view_ptr,
            weights_view_ptr,
            store_transposed,
            renumber,
            do_expensive_check,
            &(self.c_graph_ptr),
            &error_ptr)

        assert_success(error_code, error_ptr,
                       "cugraph_sg_graph_create()")

        cugraph_type_erased_device_array_view_free(srcs_view_ptr)
        cugraph_type_erased_device_array_view_free(dsts_view_ptr)
        cugraph_type_erased_device_array_view_free(weights_view_ptr)

    def __dealloc__(self):
        if self.c_graph_ptr is not NULL:
            cugraph_sg_graph_free(self.c_graph_ptr)


cdef class EXPERIMENTAL__MGGraph(_GPUGraph):
    """
    RAII-stye Graph class for use with multi-GPU APIs that manages the
    individual create/free calls and the corresponding cugraph_graph_t pointer.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph_properties : GraphProperties
        Object defining intended properties for the graph.

    src_array : device array type
        Device array containing the vertex identifiers of the source of each
        directed edge. The order of the array corresponds to the ordering of the
        dst_array, where the ith item in src_array and the ith item in dst_array
        define the ith edge of the graph.

    dst_array : device array type
        Device array containing the vertex identifiers of the destination of
        each directed edge. The order of the array corresponds to the ordering
        of the src_array, where the ith item in src_array and the ith item in
        dst_array define the ith edge of the graph.

    weight_array : device array type
        Device array containing the weight values of each directed edge. The
        order of the array corresponds to the ordering of the src_array and
        dst_array arrays, where the ith item in weight_array is the weight value
        of the ith edge of the graph.

    store_transposed : bool
        Set to True if the graph should be transposed. This is required for some
        algorithms, such as pagerank.
    
    num_edges : int
        Number of edges
    
    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.
    """
    def __cinit__(self,
                  EXPERIMENTAL__ResourceHandle resource_handle,
                  EXPERIMENTAL__GraphProperties graph_properties,
                  src_array,
                  dst_array,
                  weight_array,
                  store_transposed,
                  num_edges,
                  do_expensive_check):

        # FIXME: add tests for these
        if not(isinstance(store_transposed, (int, bool))):
            raise TypeError("expected int or bool for store_transposed, got "
                            f"{type(store_transposed)}")
        if not(isinstance(num_edges, (int))):
            raise TypeError("expected int for num_edges, got "
                            f"{type(num_edges)}")
        if not(isinstance(do_expensive_check, (int, bool))):
            raise TypeError("expected int or bool for do_expensive_check, got "
                            f"{type(do_expensive_check)}")
        assert_CAI_type(src_array, "src_array")
        assert_CAI_type(dst_array, "dst_array")
        assert_CAI_type(weight_array, "weight_array")

        # FIXME: assert that src_array and dst_array have the same type

        cdef cugraph_error_t* error_ptr
        cdef cugraph_error_code_t error_code

        cdef uintptr_t cai_srcs_ptr = \
            src_array.__cuda_array_interface__["data"][0]
        cdef cugraph_type_erased_device_array_view_t* srcs_view_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_srcs_ptr,
                len(src_array),
                get_c_type_from_numpy_type(src_array.dtype))

        cdef uintptr_t cai_dsts_ptr = \
            dst_array.__cuda_array_interface__["data"][0]
        cdef cugraph_type_erased_device_array_view_t* dsts_view_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_dsts_ptr,
                len(dst_array),
                get_c_type_from_numpy_type(dst_array.dtype))

        cdef uintptr_t cai_weights_ptr = \
            weight_array.__cuda_array_interface__["data"][0]
        cdef cugraph_type_erased_device_array_view_t* weights_view_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_weights_ptr,
                len(weight_array),
                get_c_type_from_numpy_type(weight_array.dtype))

        error_code = cugraph_mg_graph_create(
            resource_handle.c_resource_handle_ptr,
            &(graph_properties.c_graph_properties),
            srcs_view_ptr,
            dsts_view_ptr,
            weights_view_ptr,
            store_transposed,
            num_edges,
            do_expensive_check,
            &(self.c_graph_ptr),
            &error_ptr)

        assert_success(error_code, error_ptr,
                       "cugraph_mg_graph_create()")

        cugraph_type_erased_device_array_view_free(srcs_view_ptr)
        cugraph_type_erased_device_array_view_free(dsts_view_ptr)
        cugraph_type_erased_device_array_view_free(weights_view_ptr)

    def __dealloc__(self):
        if self.c_graph_ptr is not NULL:
            cugraph_mg_graph_free(self.c_graph_ptr)
