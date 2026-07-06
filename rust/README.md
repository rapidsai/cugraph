# libcugraph C API Rust Bindings (bindgen)

This folder contains a minimal Rust setup to generate FFI bindings for
`libcugraph_c` directly from the C API headers in this repository.

It is intentionally low-level and does not provide a high-level Rust wrapper.

## What this provides

- `bindgen`-based generation from headers under `cpp/include/cugraph_c`
- A single Rust crate (`libcugraph-c-sys`) with generated raw FFI items
- Optional runtime linking support for smoke tests and examples
- Example + tests showing direct C API usage from Rust

## Layout

- `Cargo.toml` : crate manifest
- `build.rs` : runs bindgen and (optionally) emits linker directives
- `wrapper.h` : aggregation header including the full C API surface
- `src/lib.rs` : includes generated bindings and one small utility helper
- `examples/resource_handle.rs` : minimal runtime example
- `tests/abi_surface.rs` : compile-time shape checks
- `tests/runtime_smoke.rs` : runtime smoke test (feature-gated)

## Requirements

- Rust toolchain (`cargo`, `rustc`)
- `libclang` available to `bindgen` (for example, via conda or system install)
- C headers from this repository (defaults to `../cpp/include`)
- Optional runtime library for execution tests/examples: `libcugraph_c.so`

## Generate bindings

From this directory:

```bash
cargo check
```

This runs `build.rs`, which generates `bindgen.rs` into Cargo `OUT_DIR`.

## Environment variables

- `CUGRAPH_INCLUDE_DIR`
  - Optional include root for `cugraph_c/*.h`
  - Default: `../cpp/include`
- `CUGRAPH_LIB_DIR`
  - Optional directory containing `libcugraph_c.so`
  - Used when `runtime-link` feature is enabled
- `CUGRAPH_EXPORT_BINDINGS`
  - Optional output path for writing a single generated bindings file
  - Example: `CUGRAPH_EXPORT_BINDINGS=$PWD/bindgen.rs cargo check`

## Run example with runtime linking

```bash
CUGRAPH_LIB_DIR=/path/to/lib cargo run --example resource_handle --features runtime-link
```

## Run tests

Compile-only API checks:

```bash
cargo test --test abi_surface
```

Runtime smoke test (requires shared library):

```bash
CUGRAPH_LIB_DIR=/path/to/lib cargo test --test runtime_smoke --features runtime-link
```

## Notes

- This setup targets the C API only (`libcugraph_c`).
- No C++ wrappers and no Cython are used.
- Generated bindings are intentionally broad to expose the raw API surface.
- `tree_algorithms.h` and `layout_algorithms.h` are currently excluded from
  `wrapper.h` for this initial Rust binding scope.
