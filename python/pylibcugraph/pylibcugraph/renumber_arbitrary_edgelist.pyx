# Copyright (c) 2025, NVIDIA CORPORATION.
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

from pylibcugraph.resource_handle cimport ResourceHandle

from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_host_array_view_t,
    cugraph_type_erased_device_array_view_free,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_host_array_view_free,
    cugraph_type_erased_host_array_view_create,
)

from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_renumber_arbitrary_edgelist,
)

from pylibcugraph.utils cimport (
    assert_success,
    assert_CAI_type,
    assert_AI_type,
    get_c_type_from_numpy_type,
)

def renumber_arbitrary_edgelist(
  ResourceHandle handle,
  renumber_map, # host array
  srcs, # device array
  dsts, # device array
):
    """
    Multi-GPU supporting function that accepts a local edgelist
    and global renumber map and renumbers the edgelist in place.

    Parameters
    ----------
    handle: ResourceHandle
        Resource handle to use.
    renumber_map: ndarray
        Host array type containing the renumber map.
    src: ndarray
        Device array type containing the source vertices.
    dst: ndarray
        Device array type containing the destination vertices.

    Returns
    -------
    Nothing.
    """

    assert_CAI_type(srcs, "srcs")
    assert_CAI_type(dsts, "dsts")

    assert_AI_type(renumber_map, "renumber_map")

    cdef uintptr_t cai_renumber_map_ptr = \
        renumber_map.__array_interface__['data'][0]
    cdef cugraph_type_erased_host_array_view_t* map_view = \
        cugraph_type_erased_host_array_view_create(
            <void*>cai_renumber_map_ptr,
            len(renumber_map),
            get_c_type_from_numpy_type(renumber_map.dtype)
        )

    cdef uintptr_t cai_srcs_ptr = \
        srcs.__cuda_array_interface__['data'][0]
    cdef cugraph_type_erased_device_array_view_t* srcs_view = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_srcs_ptr,
            len(srcs),
            get_c_type_from_numpy_type(srcs.dtype)
        )

    cdef uintptr_t cai_dsts_ptr = \
        dsts.__cuda_array_interface__['data'][0]
    cdef cugraph_type_erased_device_array_view_t* dsts_view = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_dsts_ptr,
            len(dsts),
            get_c_type_from_numpy_type(dsts.dtype)
        )

    cdef cugraph_resource_handle_t* handle_cptr = handle.c_resource_handle_ptr

    cdef cugraph_error_t* err_cptr

    cdef cugraph_error_code_t err_code = cugraph_renumber_arbitrary_edgelist(
        handle_cptr,
        map_view,
        srcs_view,
        dsts_view,
        &err_cptr,
    )

    # Verify that the C API call completed successfully and fail if it did not.
    assert_success(err_code, err_cptr, "cugraph_renumber_arbitrary_edgelist")

    # Free the views
    cugraph_type_erased_device_array_view_free(srcs_view)
    cugraph_type_erased_device_array_view_free(dsts_view)
    cugraph_type_erased_host_array_view_free(map_view)
