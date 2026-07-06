# How To Use libcugraph C API from Rust

This guide is for generating and using low-level Rust FFI bindings to
`libcugraph_c` in this repository.

## 1) Generate bindings with default header path

From this directory:

```bash
cargo check
```

This generates bindings in Cargo `OUT_DIR` and compiles the crate.

## 2) Generate a single checked file named bindgen.rs

If you want one file in this folder (for sharing or reuse), run:

```bash
CUGRAPH_EXPORT_BINDINGS=$PWD/bindgen.rs cargo check
```

The generated file will be written to `rust/bindgen.rs`.

## 3) Use a custom include directory

```bash
CUGRAPH_INCLUDE_DIR=/custom/include/root cargo check
```

The include root must contain `cugraph_c/*.h`.

## 4) Run runtime example (links to libcugraph_c)

```bash
CUGRAPH_LIB_DIR=/path/to/lib cargo run --example resource_handle --features runtime-link
```

## 5) Run tests

Compile-time API tests:

```bash
cargo test --test abi_surface
```

Runtime smoke test (requires `libcugraph_c.so`):

```bash
CUGRAPH_LIB_DIR=/path/to/lib cargo test --test runtime_smoke --features runtime-link
```

## Notes

- This is C API only. No C++ wrappers.
- No Cython.
- `tree_algorithms.h` and `layout_algorithms.h` are currently omitted from the
  wrapper in this initial setup.
