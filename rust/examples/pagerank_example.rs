mod utils;

#[cfg(feature = "runtime-link")]
fn main() {
    use libcugraph_c_sys::{
        cugraph_centrality_result_free, cugraph_centrality_result_get_values,
        cugraph_centrality_result_get_vertices, cugraph_centrality_result_t, cugraph_error_free,
        cugraph_error_t, cugraph_graph_create_sg, cugraph_graph_free, cugraph_graph_properties_t,
        cugraph_graph_t, cugraph_pagerank, cugraph_type_erased_device_array_create,
        cugraph_type_erased_device_array_free, cugraph_type_erased_device_array_t,
        cugraph_type_erased_device_array_view, cugraph_type_erased_device_array_view_copy_from_host,
        cugraph_type_erased_device_array_view_copy_to_host, cugraph_type_erased_device_array_view_size,
        cugraph_error_code__CUGRAPH_SUCCESS, data_type_id__FLOAT32, data_type_id__INT32,
    };
    use polars::prelude::DataType;
    use std::ptr;

    fn free_error(err: &mut *mut cugraph_error_t) {
        if !(*err).is_null() {
            unsafe {
                cugraph_error_free(*err);
            }
            *err = ptr::null_mut();
        }
    }

    fn check_status(
        code: libcugraph_c_sys::cugraph_error_code_t,
        err: &mut *mut cugraph_error_t,
        context: &str,
    ) -> Result<(), String> {
        if code == cugraph_error_code__CUGRAPH_SUCCESS {
            return Ok(());
        }

        let msg = unsafe { libcugraph_c_sys::error_message_owned(*err) }
            .unwrap_or_else(|| "no libcugraph error message available".to_string());
        free_error(err);
        Err(format!("{context} failed: {msg}"))
    }

    fn series_to_i32(series: &polars::prelude::Series, name: &str) -> Result<Vec<i32>, String> {
        let casted = series
            .cast(&DataType::Int32)
            .map_err(|e| format!("failed casting {name} to Int32: {e}"))?;
        Ok(casted.i32().map_err(|e| format!("failed reading {name}: {e}"))?.into_no_null_iter().collect())
    }

    fn series_to_f32(series: &polars::prelude::Series, name: &str) -> Result<Vec<f32>, String> {
        let casted = series
            .cast(&DataType::Float32)
            .map_err(|e| format!("failed casting {name} to Float32: {e}"))?;
        Ok(casted
            .f32()
            .map_err(|e| format!("failed reading {name}: {e}"))?
            .into_no_null_iter()
            .collect())
    }

    let mut err: *mut cugraph_error_t = ptr::null_mut();
    let mut src_dev: *mut cugraph_type_erased_device_array_t = ptr::null_mut();
    let mut dst_dev: *mut cugraph_type_erased_device_array_t = ptr::null_mut();
    let mut wt_dev: *mut cugraph_type_erased_device_array_t = ptr::null_mut();
    let mut graph: *mut cugraph_graph_t = ptr::null_mut();
    let mut result: *mut cugraph_centrality_result_t = ptr::null_mut();

    let handle = match utils::create_resource_handle() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let run_result = (|| -> Result<(), String> {
        let (src_df, dst_df, wt_df) =
            utils::read_triplet_dataframes("karate.csv", Some("../datasets")).map_err(|e| e.to_string())?;

        let src = series_to_i32(
            src_df
                .column("src")
                .map_err(|e| format!("missing src column: {e}"))?,
            "src",
        )?;
        let dst = series_to_i32(
            dst_df
                .column("dst")
                .map_err(|e| format!("missing dst column: {e}"))?,
            "dst",
        )?;
        let wt = series_to_f32(
            wt_df
                .column("wt")
                .map_err(|e| format!("missing wt column: {e}"))?,
            "wt",
        )?;

        if src.len() != dst.len() || src.len() != wt.len() {
            return Err("src, dst, wt column lengths must match".to_string());
        }

        let (vertices_host, pagerank_host, n) = unsafe {
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

            check_status(
                cugraph_type_erased_device_array_create(
                    handle,
                    dst.len(),
                    data_type_id__INT32,
                    &mut dst_dev,
                    &mut err,
                ),
                &mut err,
                "create dst device array",
            )?;

            check_status(
                cugraph_type_erased_device_array_create(
                    handle,
                    wt.len(),
                    data_type_id__FLOAT32,
                    &mut wt_dev,
                    &mut err,
                ),
                &mut err,
                "create wt device array",
            )?;

            let src_view = cugraph_type_erased_device_array_view(src_dev);
            let dst_view = cugraph_type_erased_device_array_view(dst_dev);
            let wt_view = cugraph_type_erased_device_array_view(wt_dev);

            check_status(
                cugraph_type_erased_device_array_view_copy_from_host(
                    handle,
                    src_view,
                    src.as_ptr() as *const libcugraph_c_sys::byte_t,
                    &mut err,
                ),
                &mut err,
                "copy src to device",
            )?;

            check_status(
                cugraph_type_erased_device_array_view_copy_from_host(
                    handle,
                    dst_view,
                    dst.as_ptr() as *const libcugraph_c_sys::byte_t,
                    &mut err,
                ),
                &mut err,
                "copy dst to device",
            )?;

            check_status(
                cugraph_type_erased_device_array_view_copy_from_host(
                    handle,
                    wt_view,
                    wt.as_ptr() as *const libcugraph_c_sys::byte_t,
                    &mut err,
                ),
                &mut err,
                "copy wt to device",
            )?;

            let properties = cugraph_graph_properties_t {
                is_symmetric: libcugraph_c_sys::bool__FALSE,
                is_multigraph: libcugraph_c_sys::bool__FALSE,
            };

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

            let vertices_view = cugraph_centrality_result_get_vertices(result);
            let pagerank_view = cugraph_centrality_result_get_values(result);
            let n = cugraph_type_erased_device_array_view_size(vertices_view);

            let mut vertices_host = vec![0i32; n];
            let mut pagerank_host = vec![0.0f32; n];

            check_status(
                cugraph_type_erased_device_array_view_copy_to_host(
                    handle,
                    vertices_host.as_mut_ptr() as *mut libcugraph_c_sys::byte_t,
                    vertices_view,
                    &mut err,
                ),
                &mut err,
                "copy pagerank vertices to host",
            )?;

            check_status(
                cugraph_type_erased_device_array_view_copy_to_host(
                    handle,
                    pagerank_host.as_mut_ptr() as *mut libcugraph_c_sys::byte_t,
                    pagerank_view,
                    &mut err,
                ),
                &mut err,
                "copy pagerank scores to host",
            )?;

            (vertices_host, pagerank_host, n)
        };

        println!("Computed PageRank for {} vertices", n);
        for (v, pr) in vertices_host.iter().zip(pagerank_host.iter()).take(10) {
            println!("vertex={} pagerank={:.6}", v, pr);
        }

        Ok(())
    })();

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

    if let Err(e) = run_result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "runtime-link"))]
fn main() {
    println!(
        "Enable feature runtime-link to run this example against libcugraph_c (pagerank example)"
    );
}
