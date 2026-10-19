/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::env;
use std::path::{Path, PathBuf};

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name).map(PathBuf::from)
}

fn default_include_dir(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join("..").join("cpp").join("include")
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let include_dir = env_path("CUGRAPH_INCLUDE_DIR").unwrap_or_else(|| default_include_dir(&manifest_dir));

    if !include_dir.exists() {
        panic!(
            "CUGRAPH include dir does not exist: {}. Set CUGRAPH_INCLUDE_DIR.",
            include_dir.display()
        );
    }

    let wrapper = manifest_dir.join("wrapper.h");
    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set")).join("bindgen.rs");

    println!("cargo:rerun-if-changed={}", wrapper.display());
    println!("cargo:rerun-if-env-changed=CUGRAPH_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CUGRAPH_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUGRAPH_EXPORT_BINDINGS");

    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_type("cugraph_.*")
        .allowlist_type("bool_t")
        .allowlist_type("byte_t")
        .allowlist_var("CUGRAPH_.*")
        .allowlist_var("FALSE|TRUE")
        .allowlist_var("INT8|INT16|INT32|INT64|UINT8|UINT16|UINT32|UINT64|FLOAT32|FLOAT64|SIZE_T|BOOL|NTYPES")
        .allowlist_function("cugraph_.*")
        .derive_default(true)
        .generate_comments(true)
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate libcugraph_c bindings");

    bindings
        .write_to_file(&out_path)
        .expect("Unable to write generated bindings");

    if let Some(export_path) = env_path("CUGRAPH_EXPORT_BINDINGS") {
        bindings
            .write_to_file(&export_path)
            .expect("Unable to export generated bindings");
        println!("cargo:warning=Exported bindings to {}", export_path.display());
    }

    if env::var_os("CARGO_FEATURE_RUNTIME_LINK").is_some() {
        if let Some(lib_dir) = env_path("CUGRAPH_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }

        println!("cargo:rustc-link-lib=dylib=cugraph_c");
    }
}
