use libcugraph_c_sys::cugraph_resource_handle_t;

/// Create a libcugraph resource handle.
///
/// Returns an error string if libcugraph returns a null pointer.
pub fn create_resource_handle() -> Result<*mut cugraph_resource_handle_t, String> {
    let handle = unsafe { libcugraph_c_sys::cugraph_create_resource_handle(std::ptr::null_mut()) };

    if handle.is_null() {
        return Err("failed to create resource handle".to_string());
    }

    Ok(handle)
}

/// Free a resource handle created by `create_resource_handle`.
///
/// Safe to call with null.
pub fn free_resource_handle(handle: *mut cugraph_resource_handle_t) {
    if handle.is_null() {
        return;
    }

    unsafe {
        libcugraph_c_sys::cugraph_free_resource_handle(handle);
    }
}
