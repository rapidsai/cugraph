/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fmt;
use std::ptr;

use crate::*;

#[derive(Debug)]
pub struct CugraphError {
    pub code: cugraph_error_code_t,
    pub message: Option<String>,
}

impl fmt::Display for CugraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.message {
            Some(message) => write!(f, "cuGraph error {:?}: {}", self.code, message),
            None => write!(f, "cuGraph error {:?}", self.code),
        }
    }
}

impl std::error::Error for CugraphError {}

type Result<T> = std::result::Result<T, CugraphError>;

#[allow(non_upper_case_globals)]
const cugraph_error_code_enum__CUGRAPH_SUCCESS: cugraph_error_code_t =
    cugraph_error_code__CUGRAPH_SUCCESS;
#[allow(non_upper_case_globals)]
const cugraph_error_code_enum__CUGRAPH_UNKNOWN_ERROR: cugraph_error_code_t =
    cugraph_error_code__CUGRAPH_UNKNOWN_ERROR;
#[allow(non_upper_case_globals)]
const cugraph_error_code_enum__CUGRAPH_INVALID_INPUT: cugraph_error_code_t =
    cugraph_error_code__CUGRAPH_INVALID_INPUT;

fn status_ok(status: cugraph_error_code_t) -> bool {
    status == cugraph_error_code_enum__CUGRAPH_SUCCESS
}

fn bool_t_from_bool(value: bool) -> bool_t {
    if value {
        bool__TRUE
    } else {
        bool__FALSE
    }
}

fn check_status(status: cugraph_error_code_t, error: *mut cugraph_error_t) -> Result<()> {
    if status_ok(status) {
        return Ok(());
    }

    let message = unsafe { error_message_owned(error as *const cugraph_error_t) };
    if !error.is_null() {
        unsafe { cugraph_error_free(error) };
    }
    Err(CugraphError { code: status, message })
}

pub struct ResourceHandle {
    raw: *mut cugraph_resource_handle_t,
}

impl ResourceHandle {
    pub fn new() -> Result<Self> {
        let raw = unsafe { cugraph_create_resource_handle(ptr::null_mut()) };
        if raw.is_null() {
            return Err(CugraphError {
                code: cugraph_error_code_enum__CUGRAPH_UNKNOWN_ERROR,
                message: Some("failed to create cugraph resource handle".to_string()),
            });
        }
        Ok(Self { raw })
    }

    pub fn as_ptr(&self) -> *const cugraph_resource_handle_t {
        self.raw as *const cugraph_resource_handle_t
    }
}

impl Drop for ResourceHandle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_free_resource_handle(self.raw) };
        }
    }
}

pub struct DeviceArray {
    raw: *mut cugraph_type_erased_device_array_t,
}

impl DeviceArray {
    pub fn from_host_slice<T: Copy>(
        handle: &ResourceHandle,
        data: &[T],
        dtype: cugraph_data_type_id_t,
    ) -> Result<Self> {
        let mut array = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            cugraph_type_erased_device_array_create(
                handle.as_ptr(),
                data.len(),
                dtype,
                &mut array,
                &mut error,
            )
        };
        check_status(status, error)?;

        let view = unsafe { cugraph_type_erased_device_array_view(array) };
        if view.is_null() {
            unsafe { cugraph_type_erased_device_array_free(array) };
            return Err(CugraphError {
                code: cugraph_error_code_enum__CUGRAPH_UNKNOWN_ERROR,
                message: Some("failed to create device array view".to_string()),
            });
        }

        let mut copy_error = ptr::null_mut();
        let copy_status = unsafe {
            cugraph_type_erased_device_array_view_copy_from_host(
                handle.as_ptr(),
                view,
                data.as_ptr() as *const byte_t,
                &mut copy_error,
            )
        };

        unsafe {
            cugraph_type_erased_device_array_view_free(view);
        }

        check_status(copy_status, copy_error)?;

        Ok(Self { raw: array })
    }

    pub fn view(&self) -> *mut cugraph_type_erased_device_array_view_t {
        unsafe { cugraph_type_erased_device_array_view(self.raw) }
    }
}

impl Drop for DeviceArray {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_type_erased_device_array_free(self.raw) };
        }
    }
}

pub struct Graph {
    raw: *mut cugraph_graph_t,
}

impl Graph {
    pub fn as_ptr_mut(&mut self) -> *mut cugraph_graph_t {
        self.raw
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { cugraph_graph_free(self.raw) };
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GraphBuildOptions {
    pub is_symmetric: bool,
    pub is_multigraph: bool,
    pub store_transposed: bool,
    pub renumber: bool,
    pub drop_self_loops: bool,
    pub drop_multi_edges: bool,
    pub symmetrize: bool,
    pub do_expensive_check: bool,
}

impl Default for GraphBuildOptions {
    fn default() -> Self {
        Self {
            is_symmetric: false,
            is_multigraph: false,
            store_transposed: false,
            renumber: true,
            drop_self_loops: false,
            drop_multi_edges: false,
            symmetrize: false,
            do_expensive_check: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CscInput<'a, V, W> {
    pub row: &'a [V],
    pub colptr: &'a [V],
    pub num_dst_vertices: usize,
    pub weights: Option<&'a [W]>,
}

impl<'a, V, W> CscInput<'a, V, W>
where
    V: Copy + TryInto<usize>,
{
    pub fn validate(&self) -> Result<()> {
        if self.colptr.len() != self.num_dst_vertices + 1 {
            return Err(CugraphError {
                code: cugraph_error_code_enum__CUGRAPH_INVALID_INPUT,
                message: Some("colptr length must be num_dst_vertices + 1".to_string()),
            });
        }

        let nnz = self
            .colptr
            .get(self.num_dst_vertices)
            .and_then(|v| (*v).try_into().ok())
            .ok_or_else(|| CugraphError {
                code: cugraph_error_code_enum__CUGRAPH_INVALID_INPUT,
                message: Some("invalid terminal colptr value".to_string()),
            })?;

        if self.row.len() != nnz {
            return Err(CugraphError {
                code: cugraph_error_code_enum__CUGRAPH_INVALID_INPUT,
                message: Some("row length must equal colptr[num_dst_vertices]".to_string()),
            });
        }

        if let Some(weights) = self.weights {
            if weights.len() != nnz {
                return Err(CugraphError {
                    code: cugraph_error_code_enum__CUGRAPH_INVALID_INPUT,
                    message: Some("weights length must equal nnz".to_string()),
                });
            }
        }

        Ok(())
    }
}

pub fn expand_csc_to_coo_dst<V>(colptr: &[V], num_dst_vertices: usize) -> Result<Vec<V>>
where
    V: Copy + TryInto<usize> + TryFrom<usize>,
{
    if colptr.len() != num_dst_vertices + 1 {
        return Err(CugraphError {
            code: cugraph_error_code_enum__CUGRAPH_INVALID_INPUT,
            message: Some("colptr length must be num_dst_vertices + 1".to_string()),
        });
    }

    let nnz = colptr[num_dst_vertices]
        .try_into()
        .map_err(|_| CugraphError {
            code: cugraph_error_code_enum__CUGRAPH_INVALID_INPUT,
            message: Some("invalid terminal colptr value".to_string()),
        })?;

    let mut dst = Vec::with_capacity(nnz);
    for j in 0..num_dst_vertices {
        let start = colptr[j].try_into().map_err(|_| CugraphError {
            code: cugraph_error_code_enum__CUGRAPH_INVALID_INPUT,
            message: Some("invalid colptr start value".to_string()),
        })?;
        let end = colptr[j + 1].try_into().map_err(|_| CugraphError {
            code: cugraph_error_code_enum__CUGRAPH_INVALID_INPUT,
            message: Some("invalid colptr end value".to_string()),
        })?;
        if end < start {
            return Err(CugraphError {
                code: cugraph_error_code_enum__CUGRAPH_INVALID_INPUT,
                message: Some("colptr must be non-decreasing".to_string()),
            });
        }

        let dst_id = V::try_from(j).map_err(|_| CugraphError {
            code: cugraph_error_code_enum__CUGRAPH_INVALID_INPUT,
            message: Some("failed to cast destination id to vertex dtype".to_string()),
        })?;

        for _ in start..end {
            dst.push(dst_id);
        }
    }

    Ok(dst)
}

pub struct CscGraphArtifacts {
    pub graph: Graph,
    pub src: DeviceArray,
    pub dst: DeviceArray,
    pub weights: Option<DeviceArray>,
}

pub fn build_graph_from_csc_as_edgelist<V, W>(
    handle: &ResourceHandle,
    input: &CscInput<'_, V, W>,
    vertex_dtype: cugraph_data_type_id_t,
    weight_dtype: cugraph_data_type_id_t,
    options: GraphBuildOptions,
) -> Result<CscGraphArtifacts>
where
    V: Copy + TryInto<usize> + TryFrom<usize>,
    W: Copy,
{
    input.validate()?;

    let dst_host = expand_csc_to_coo_dst(input.colptr, input.num_dst_vertices)?;
    let src = DeviceArray::from_host_slice(handle, input.row, vertex_dtype)?;
    let dst = DeviceArray::from_host_slice(handle, &dst_host, vertex_dtype)?;
    let weights = match input.weights {
        Some(w) => Some(DeviceArray::from_host_slice(handle, w, weight_dtype)?),
        None => None,
    };

    let src_view = src.view();
    let dst_view = dst.view();
    let weights_view = match &weights {
        Some(w) => w.view(),
        None => ptr::null_mut(),
    };

    if src_view.is_null() || dst_view.is_null() || (weights.is_some() && weights_view.is_null()) {
        unsafe {
            if !src_view.is_null() {
                cugraph_type_erased_device_array_view_free(src_view);
            }
            if !dst_view.is_null() {
                cugraph_type_erased_device_array_view_free(dst_view);
            }
            if !weights_view.is_null() {
                cugraph_type_erased_device_array_view_free(weights_view);
            }
        }
        return Err(CugraphError {
            code: cugraph_error_code_enum__CUGRAPH_UNKNOWN_ERROR,
            message: Some("failed to create one or more device views".to_string()),
        });
    }

    let properties = cugraph_graph_properties_t {
        is_symmetric: bool_t_from_bool(options.is_symmetric),
        is_multigraph: bool_t_from_bool(options.is_multigraph),
    };

    let mut graph_ptr = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        cugraph_graph_create_sg(
            handle.as_ptr(),
            &properties,
            ptr::null(),
            src_view,
            dst_view,
            weights_view,
            ptr::null(),
            ptr::null(),
            bool_t_from_bool(options.store_transposed),
            bool_t_from_bool(options.renumber),
            bool_t_from_bool(options.drop_self_loops),
            bool_t_from_bool(options.drop_multi_edges),
            bool_t_from_bool(options.symmetrize),
            bool_t_from_bool(options.do_expensive_check),
            &mut graph_ptr,
            &mut error,
        )
    };

    unsafe {
        cugraph_type_erased_device_array_view_free(src_view);
        cugraph_type_erased_device_array_view_free(dst_view);
        if !weights_view.is_null() {
            cugraph_type_erased_device_array_view_free(weights_view);
        }
    }

    check_status(status, error)?;

    Ok(CscGraphArtifacts {
        graph: Graph { raw: graph_ptr },
        src,
        dst,
        weights,
    })
}

pub struct CsrGraphArtifacts {
    pub graph: Graph,
    pub offsets: DeviceArray,
    pub indices: DeviceArray,
    pub weights: Option<DeviceArray>,
}

pub fn build_graph_from_csc_as_csr<V, W>(
    handle: &ResourceHandle,
    input: &CscInput<'_, V, W>,
    vertex_dtype: cugraph_data_type_id_t,
    weight_dtype: cugraph_data_type_id_t,
    options: GraphBuildOptions,
) -> Result<CsrGraphArtifacts>
where
    V: Copy + TryInto<usize>,
    W: Copy,
{
    input.validate()?;

    let offsets = DeviceArray::from_host_slice(handle, input.colptr, vertex_dtype)?;
    let indices = DeviceArray::from_host_slice(handle, input.row, vertex_dtype)?;
    let weights = match input.weights {
        Some(w) => Some(DeviceArray::from_host_slice(handle, w, weight_dtype)?),
        None => None,
    };

    let offsets_view = offsets.view();
    let indices_view = indices.view();
    let weights_view = match &weights {
        Some(w) => w.view(),
        None => ptr::null_mut(),
    };
    if offsets_view.is_null()
        || indices_view.is_null()
        || (weights.is_some() && weights_view.is_null())
    {
        unsafe {
            if !offsets_view.is_null() {
                cugraph_type_erased_device_array_view_free(offsets_view);
            }
            if !indices_view.is_null() {
                cugraph_type_erased_device_array_view_free(indices_view);
            }
            if !weights_view.is_null() {
                cugraph_type_erased_device_array_view_free(weights_view);
            }
        }
        return Err(CugraphError {
            code: cugraph_error_code_enum__CUGRAPH_UNKNOWN_ERROR,
            message: Some("failed to create one or more device views".to_string()),
        });
    }

    let properties = cugraph_graph_properties_t {
        is_symmetric: bool_t_from_bool(options.is_symmetric),
        is_multigraph: bool_t_from_bool(options.is_multigraph),
    };

    let mut graph_ptr = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        cugraph_graph_create_sg_from_csr(
            handle.as_ptr(),
            &properties,
            offsets_view,
            indices_view,
            weights_view,
            ptr::null(),
            ptr::null(),
            bool_t_from_bool(options.store_transposed),
            bool_t_from_bool(options.renumber),
            bool_t_from_bool(options.symmetrize),
            bool_t_from_bool(options.do_expensive_check),
            &mut graph_ptr,
            &mut error,
        )
    };

    unsafe {
        cugraph_type_erased_device_array_view_free(offsets_view);
        cugraph_type_erased_device_array_view_free(indices_view);
        if !weights_view.is_null() {
            cugraph_type_erased_device_array_view_free(weights_view);
        }
    }

    check_status(status, error)?;

    Ok(CsrGraphArtifacts {
        graph: Graph { raw: graph_ptr },
        offsets,
        indices,
        weights,
    })
}

pub fn default_vertex_dtype_i64() -> cugraph_data_type_id_t {
    data_type_id__INT64
}

pub fn default_weight_dtype_f32() -> cugraph_data_type_id_t {
    data_type_id__FLOAT32
}
