# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

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
    cugraph_type_erased_host_array_view_free,
)

from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_renumber_arbitrary_edgelist,
)

from pylibcugraph.utils cimport (
    assert_success,
    assert_device_accessible,
    assert_host_accessible,
    get_c_type_from_numpy_type,
    create_cugraph_type_erased_device_array_view_from_py_obj,
    create_cugraph_type_erased_host_array_view_from_py_obj,
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

    cdef cugraph_type_erased_host_array_view_t* map_view = \
        create_cugraph_type_erased_host_array_view_from_py_obj(renumber_map)

    cdef cugraph_type_erased_device_array_view_t* srcs_view = \
        create_cugraph_type_erased_device_array_view_from_py_obj(srcs)

    cdef cugraph_type_erased_device_array_view_t* dsts_view = \
        create_cugraph_type_erased_device_array_view_from_py_obj(dsts)

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
