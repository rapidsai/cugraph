# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
    assert_device_accessible,
    assert_host_accessible,
    get_c_type_from_numpy_type,
    get_c_type_from_py_obj,
    get_size_from_py_obj,
    get_data_ptr_from_py_obj,
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

    assert_device_accessible(srcs, "srcs")
    assert_device_accessible(dsts, "dsts")

    assert_host_accessible(renumber_map, "renumber_map")

    cdef uintptr_t cai_renumber_map_ptr = \
        get_data_ptr_from_py_obj(renumber_map)
    cdef cugraph_type_erased_host_array_view_t* map_view = \
        cugraph_type_erased_host_array_view_create(
            <void*>cai_renumber_map_ptr,
            get_size_from_py_obj(renumber_map),
            get_c_type_from_py_obj(renumber_map)
        )

    cdef uintptr_t cai_srcs_ptr = \
        get_data_ptr_from_py_obj(srcs)
    cdef cugraph_type_erased_device_array_view_t* srcs_view = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_srcs_ptr,
            get_size_from_py_obj(srcs),
            get_c_type_from_py_obj(srcs)
        )

    cdef uintptr_t cai_dsts_ptr = \
        get_data_ptr_from_py_obj(dsts)
    cdef cugraph_type_erased_device_array_view_t* dsts_view = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_dsts_ptr,
            get_size_from_py_obj(dsts),
            get_c_type_from_py_obj(dsts)
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
