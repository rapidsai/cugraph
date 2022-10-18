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

from pylibcugraph._cugraph_c.resource_handle cimport (
    data_type_id_t,
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)


cdef assert_success(cugraph_error_code_t code,
                    cugraph_error_t* err,
                    api_name)

cdef assert_CAI_type(obj, var_name, allow_None=*)

cdef assert_AI_type(obj, var_name, allow_None=*)

cdef get_numpy_type_from_c_type(data_type_id_t c_type)

cdef get_c_type_from_numpy_type(numpy_type)

cdef get_c_weight_type_from_numpy_edge_ids_type(numpy_type)

cdef get_numpy_edge_ids_type_from_c_weight_type(data_type_id_t c_type)

cdef copy_to_cupy_array(
   cugraph_resource_handle_t* c_resource_handle_ptr,
   cugraph_type_erased_device_array_view_t* device_array_view_ptr)

cdef copy_to_cupy_array_ids(
   cugraph_resource_handle_t* c_resource_handle_ptr,
   cugraph_type_erased_device_array_view_t* device_array_view_ptr)

cdef cugraph_type_erased_device_array_view_t* \
    create_cugraph_type_erased_device_array_view_from_py_obj(python_obj)

cdef create_cupy_array_view_for_device_ptr(
    cugraph_type_erased_device_array_view_t* device_array_view_ptr,
    owning_py_object)
