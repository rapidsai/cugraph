#[cfg(feature = "runtime-link")]
#[test]
fn create_and_free_resource_handle() {
    unsafe {
        let handle = libcugraph_c_sys::cugraph_create_resource_handle(std::ptr::null_mut());
        assert!(!handle.is_null(), "expected a non-null resource handle");

        let rank = libcugraph_c_sys::cugraph_resource_handle_get_rank(handle);
        let size = libcugraph_c_sys::cugraph_resource_handle_get_comm_size(handle);
        assert!(rank >= 0);
        assert!(size >= 1);

        libcugraph_c_sys::cugraph_free_resource_handle(handle);
    }
}

#[cfg(not(feature = "runtime-link"))]
#[test]
fn runtime_smoke_requires_runtime_link_feature() {
    eprintln!("Skipping runtime smoke: run with --features runtime-link");
}
