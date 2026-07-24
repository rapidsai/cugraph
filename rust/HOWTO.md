# How To Use libcugraph C API from Rust

This How-To is broken into two sections
  1) Generating the Rust bindgen binding
  2) Using the new binding to run PageRank from libcugraph C


## __Generating low-level Rust FFI bindings__

This guide is for generating and using low-level Rust FFI bindings to
`libcugraph_c` in this repository.

### 1) Generate bindings with default header path

From this directory:

```bash
cargo check
```

This generates bindings in Cargo `OUT_DIR` and compiles the crate.

### 2) Generate a single checked file named bindgen.rs

If you want one file in this folder (for sharing or reuse), run:

```bash
CUGRAPH_EXPORT_BINDINGS=$PWD/bindgen.rs cargo check
```

The generated file will be written to `rust/bindgen.rs`.

### 3) Use a custom include directory

```bash
CUGRAPH_INCLUDE_DIR=/custom/include/root cargo check
```

The include root must contain `cugraph_c/*.h`.

### 4) Run runtime example (links to libcugraph_c)

```bash
CUGRAPH_LIB_DIR=/path/to/lib cargo run --example resource_handle_example --features runtime-link
```

### 5) Run tests

Compile-time API tests:

```bash
cargo test --test abi_surface
```

Runtime smoke test (requires `libcugraph_c.so`):

```bash
CUGRAPH_LIB_DIR=/path/to/lib cargo test --test runtime_smoke --features runtime-link
```

### Binding Notes

- This is C API only. No C++ wrappers.
- No Cython.
- `tree_algorithms.h` and `layout_algorithms.h` are currently omitted from the
  wrapper in this initial setup.

<br>
<br>

## Creating a PageRank example

The code example is under _examples/pagerank_exmple.rs_

### Step 1: Create a CUDA Context Resource Handle
A standard step for all CUDA application is to create a resource handle.  
A simple resource creation utility function is available for use
```eamples/utils/resourcehandle.rx```
  * create_resource_handle() - returns the resource handle
  * free_resource_handle(handle) - frees up the resource

### Step 2: Data Format - Not always Needed
cuGraph expects column-based data in either COO, CSR, or CSC format. If your data is row-based, then it is up to you to first convert it. Likewise, it is up to you to get the data into COO, CSR, or CSC format.

Using a package like [__Polars__](https://docs.pola.rs/) can help with getting the data in columar-format. Additional Polars supports the ability to read data directly into the GPU.  Lastely, Polars also has the ability to support data organized by the as defined by the [Arrow](https://arrow.apache.org/) specifiction.

Assuming COO format, the needed GPU data variables would be
```
  let mut src_dev: *mut cugraph_type_erased_device_array_t = ptr::null_mut();
  let mut dst_dev: *mut cugraph_type_erased_device_array_t = ptr::null_mut();
  let mut wt_dev: *mut cugraph_type_erased_device_array_t = ptr::null_mut();
  let mut graph: *mut cugraph_graph_t = ptr::null_mut();
```

The next step is to get your data from host-base memory into the GPU.

Example using just the `src` data. First let get the data in the correct format.
```
let src = series_to_i32(
      src_df
          .column("src")
          .map_err(|e| format!("missing src column: {e}"))?,
          "src",
  )?;
```

Now move the data to the GPU:
```
check_status(
      cugraph_type_erased_device_array_create(
          handle,
          src.len(),
          data_type_id__INT32,
          &mut src_dev,
          &mut err,
      ),
      &mut err,
      "create src device array",
  )?;
```
Repeat the above for all the columns.
Once the data is ready, the next step is to get the data into the GPU and a cuGraph Graph created. 

### Step 3: Create a cuGraph Graph
Before any graph algorithm can be executed, a _Graph_ needs to be created.  Now that data is on the GPU it can be passed to the graph creation function.  

```
  check_status(
      cugraph_graph_create_sg(
          handle,
          &properties,
          ptr::null(),
          src_view,
          dst_view,
          wt_view,
          ptr::null(),
          ptr::null(),
          libcugraph_c_sys::bool__FALSE,
          libcugraph_c_sys::bool__TRUE,
          libcugraph_c_sys::bool__FALSE,
          libcugraph_c_sys::bool__FALSE,
          libcugraph_c_sys::bool__FALSE,
          libcugraph_c_sys::bool__FALSE,
          &mut graph,
          &mut err,
      ),
      &mut err,
      "create graph",
  )?;
```

### Step 4: Run PageRank
The next step is to run the PageRank algorithm.

```
  check_status(
      cugraph_pagerank(
          handle,
          graph,
          ptr::null(),
          ptr::null(),
          ptr::null(),
          ptr::null(),
          0.85,
          1e-6,
          100,
          libcugraph_c_sys::bool__FALSE,
          &mut result,
          &mut err,
      ),
      &mut err,
      "run pagerank",
  )?;
```


### Step 5: Cleanup

```
    {
        if !result.is_null() {
            unsafe {
                cugraph_centrality_result_free(result);
            }
        }
        if !graph.is_null() {
            unsafe {
                cugraph_graph_free(graph);
            }
        }
        if !src_dev.is_null() {
            unsafe {
                cugraph_type_erased_device_array_free(src_dev);
            }
        }
        if !dst_dev.is_null() {
            unsafe {
                cugraph_type_erased_device_array_free(dst_dev);
            }
        }
        if !wt_dev.is_null() {
            unsafe {
                cugraph_type_erased_device_array_free(wt_dev);
            }
        }
        free_error(&mut err);
    }
    utils::free_resource_handle(handle);
```