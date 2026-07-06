#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::ffi::CStr;

include!(concat!(env!("OUT_DIR"), "/bindgen.rs"));

/// Convert a cugraph error object into an owned Rust string.
///
/// # Safety
/// The pointer must either be null or point to a valid `cugraph_error_t` object.
pub unsafe fn error_message_owned(error: *const cugraph_error_t) -> Option<String> {
    if error.is_null() {
        return None;
    }

    let ptr = cugraph_error_message(error);
    if ptr.is_null() {
        return None;
    }

    Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
}
