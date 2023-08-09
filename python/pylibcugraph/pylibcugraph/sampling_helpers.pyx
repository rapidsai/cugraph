from libc.stdint cimport uintptr_t

from pylibcugraph._cugraph_c.resource_handle cimport (
    bool_t,
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

    cugraph_type_erased_device_array_t,
    cugraph_type_erased_device_array_view,
    cugraph_type_erased_device_array_free,
    cugraph_type_erased_device_array_view_size,
)

from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_get_num_vertices_per_hop,
)

from pylibcugraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    get_c_type_from_numpy_type,
)

from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)

def get_num_vertices_per_hop(ResourceHandle resource_handle,
                             srcs,
                             dsts,
                             hop,
                             size_t num_hops,):
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef uintptr_t cai_srcs_ptr = \
        srcs.__cuda_array_interface__['data'][0]
    
    cdef cugraph_type_erased_device_array_view_t* srcs_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_srcs_ptr,
            len(srcs),
            get_c_type_from_numpy_type(srcs.dtype))
    
    cdef uintptr_t cai_dsts_ptr = \
        dsts.__cuda_array_interface__['data'][0]
    
    cdef cugraph_type_erased_device_array_view_t* dsts_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_dsts_ptr,
            len(dsts),
            get_c_type_from_numpy_type(dsts.dtype))
    
    cdef uintptr_t cai_hop_ptr = \
        hop.__cuda_array_interface__['data'][0]
    
    cdef cugraph_type_erased_device_array_view_t* hop_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_hop_ptr,
            len(hop),
            get_c_type_from_numpy_type(hop.dtype))

    cdef cugraph_type_erased_device_array_t* result_ptr

    error_code = cugraph_get_num_vertices_per_hop(
        c_resource_handle_ptr,
        srcs_ptr,
        dsts_ptr,
        hop_ptr,
        num_hops,
        &result_ptr,
        &error_ptr,
    )

    assert_success(error_code, error_ptr, "cugraph_get_num_vertices_per_hop")

    cdef cugraph_type_erased_device_array_view_t* result_view = \
        cugraph_type_erased_device_array_view(result_ptr)
    
    size = cugraph_type_erased_device_array_view_size(result_view)

    cupy_result = copy_to_cupy_array(c_resource_handle_ptr, result_view)
    cugraph_type_erased_device_array_free(result_ptr)
    
    return cupy_result