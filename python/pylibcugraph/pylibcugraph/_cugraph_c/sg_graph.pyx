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

from libc.stdint cimport uintptr_t

from pylibcugraph._cugraph_c.cugraph_api cimport (
    bool_t,
    cugraph_resource_handle_t,
    data_type_id_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_t,
    cugraph_type_erased_device_array_create,
    cugraph_type_erased_device_array_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
    cugraph_sg_graph_create,
    cugraph_graph_properties_t,
    cugraph_sg_graph_free,
)
from pylibcugraph._cugraph_c.resource_handle cimport (
    EXPERIMENTAL__ResourceHandle,
)
from pylibcugraph._cugraph_c.graph_properties cimport (
    EXPERIMENTAL__GraphProperties,
)


# FIXME: add tests for this
cdef assert_success(cugraph_error_code_t code,
                    cugraph_error_t* err,
                    api_name):
    if code != cugraph_error_code_t.CUGRAPH_SUCCESS:
        # FIXME: extract message using cugraph_error_message()
        raise RuntimeError(f"non-success value returned from {api_name}")


cdef class EXPERIMENTAL__SGGraph:
    """
    RAII-stye Graph class for use with single-GPU APIs that manages the
    individual create/free calls and the corresponding cugraph_graph_t pointer.
    """
    def __cinit__(self,
                  EXPERIMENTAL__ResourceHandle resource_handle,
                  EXPERIMENTAL__GraphProperties graph_properties,
                  src_array,
                  dst_array,
                  weight_array,
                  store_transposed,
                  renumber,
                  expensive_check):

        if not(isinstance(store_transposed, (int, bool))):
            raise TypeError("expected int or bool for store_transposed, got "
                            f"{type(store_transposed)}")
        if not(isinstance(renumber, (int, bool))):
            raise TypeError("expected int or bool for renumber, got "
                            f"{type(renumber)}")
        if not(isinstance(expensive_check, (int, bool))):
            raise TypeError("expected int or bool for expensive_check, got "
                            f"{type(expensive_check)}")
        if not(hasattr(src_array, "__cuda_array_interface__")):
            raise TypeError("src_array does not have required "
                            "__cuda_array_interface__ attr")
        if not(hasattr(dst_array, "__cuda_array_interface__")):
            raise TypeError("dst_array does not have required "
                            "__cuda_array_interface__ attr")
        if not(hasattr(weight_array, "__cuda_array_interface__")):
            raise TypeError("weight_array does not have required "
                            "__cuda_array_interface__ attr")

        cdef cugraph_error_t* error_ptr
        cdef cugraph_error_code_t err_code

        cdef cugraph_type_erased_device_array_t* srcs_ptr
        cdef cugraph_type_erased_device_array_t* dsts_ptr
        cdef cugraph_type_erased_device_array_t* weights_ptr

        # FIXME: set dtype properly
        err_code = cugraph_type_erased_device_array_create(
            resource_handle.c_resource_handle_ptr,
            data_type_id_t.INT32,
            len(src_array),
            &srcs_ptr,
            &error_ptr)

        assert_success(err_code, error_ptr,
                       "cugraph_type_erased_device_array_create()")

        # FIXME: set dtype properly
        err_code = cugraph_type_erased_device_array_create(
            resource_handle.c_resource_handle_ptr,
            data_type_id_t.INT32,
            len(dst_array),
            &dsts_ptr,
            &error_ptr)

        assert_success(err_code, error_ptr,
                       "cugraph_type_erased_device_array_create()")

        # FIXME: set dtype properly
        err_code = cugraph_type_erased_device_array_create(
            resource_handle.c_resource_handle_ptr,
            data_type_id_t.INT32,
            len(weight_array),
            &weights_ptr,
            &error_ptr)

        assert_success(err_code, error_ptr,
                       "cugraph_type_erased_device_array_create()")

        err_code = cugraph_sg_graph_create(
            resource_handle.c_resource_handle_ptr,
            &(graph_properties.c_graph_properties),
            srcs_ptr,
            dsts_ptr,
            weights_ptr,
            int(store_transposed),
            int(renumber),
            int(expensive_check),
            &(self.c_sg_graph_ptr),
            &error_ptr)

        assert_success(err_code, error_ptr,
                       "cugraph_sg_graph_create()")

    def __dealloc__(self):
        if self.c_sg_graph_ptr is not NULL:
            cugraph_sg_graph_free(self.c_sg_graph_ptr)
