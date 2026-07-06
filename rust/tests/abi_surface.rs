use static_assertions::assert_type_eq_all;

#[test]
fn exposes_expected_core_types() {
    assert_type_eq_all!(libcugraph_c_sys::byte_t, i8);
    assert_eq!(
        std::mem::size_of::<libcugraph_c_sys::bool_t>(),
        std::mem::size_of::<i32>()
    );
}

#[test]
fn exposes_expected_core_constants() {
    assert_eq!(
        libcugraph_c_sys::cugraph_error_code__CUGRAPH_SUCCESS as u32,
        0
    );
}

#[test]
fn can_reference_key_symbols() {
    let _ = libcugraph_c_sys::cugraph_create_resource_handle as usize;
    let _ = libcugraph_c_sys::cugraph_free_resource_handle as usize;
    let _ = libcugraph_c_sys::cugraph_error_message as usize;
    let _ = libcugraph_c_sys::cugraph_error_free as usize;
}
