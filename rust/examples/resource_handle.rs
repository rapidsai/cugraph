#[cfg(feature = "runtime-link")]
fn main() {
    unsafe {
        let handle = libcugraph_c_sys::cugraph_create_resource_handle(std::ptr::null_mut());
        if handle.is_null() {
            eprintln!("failed to create resource handle");
            std::process::exit(1);
        }

        let rank = libcugraph_c_sys::cugraph_resource_handle_get_rank(handle);
        let size = libcugraph_c_sys::cugraph_resource_handle_get_comm_size(handle);
        println!("created resource handle (rank={rank}, size={size})");

        libcugraph_c_sys::cugraph_free_resource_handle(handle);
    }
}

#[cfg(not(feature = "runtime-link"))]
fn main() {
    println!("Enable feature runtime-link to run this example against libcugraph_c");
}
