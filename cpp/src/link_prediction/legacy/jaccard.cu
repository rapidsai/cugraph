/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>
#include <utilities/graph_utils.cuh>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

namespace cugraph {
namespace detail {

// Volume of neighboors (*weight_s)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_row_sum(
  vertex_t n, edge_t const* csrPtr, vertex_t const* csrInd, weight_t const* v, weight_t* work)
{
  vertex_t row;
  edge_t start, end, length;
  weight_t sum;

  for (row = threadIdx.y + blockIdx.y * blockDim.y; row < n; row += gridDim.y * blockDim.y) {
    start  = csrPtr[row];
    end    = csrPtr[row + 1];
    length = end - start;

    // compute row sums
    if (weighted) {
      sum = parallel_prefix_sum(length, csrInd + start, v);
      if (threadIdx.x == 0) work[row] = sum;
    } else {
      work[row] = static_cast<weight_t>(length);
    }
  }
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_is(vertex_t n,
                           edge_t const* csrPtr,
                           vertex_t const* csrInd,
                           weight_t const* v,
                           weight_t* work,
                           weight_t* weight_i,
                           weight_t* weight_s)
{
  edge_t i, j, Ni, Nj;
  vertex_t row, col;
  vertex_t ref, cur, ref_col, cur_col, match;
  weight_t ref_val;

  for (row = threadIdx.z + blockIdx.z * blockDim.z; row < n; row += gridDim.z * blockDim.z) {
    for (j = csrPtr[row] + threadIdx.y + blockIdx.y * blockDim.y; j < csrPtr[row + 1];
         j += gridDim.y * blockDim.y) {
      col = csrInd[j];
      // find which row has least elements (and call it reference row)
      Ni  = csrPtr[row + 1] - csrPtr[row];
      Nj  = csrPtr[col + 1] - csrPtr[col];
      ref = (Ni < Nj) ? row : col;
      cur = (Ni < Nj) ? col : row;

      // compute new sum weights
      weight_s[j] = work[row] + work[col];

      // compute new intersection weights
      // search for the element with the same column index in the reference row
      for (i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x; i < csrPtr[ref + 1];
           i += gridDim.x * blockDim.x) {
        match   = -1;
        ref_col = csrInd[i];
        if (weighted) {
          ref_val = v[ref_col];
        } else {
          ref_val = 1.0;
        }

        // binary search (column indices are sorted within each row)
        edge_t left  = csrPtr[cur];
        edge_t right = csrPtr[cur + 1] - 1;
        while (left <= right) {
          edge_t middle = (left + right) >> 1;
          cur_col       = csrInd[middle];
          if (cur_col > ref_col) {
            right = middle - 1;
          } else if (cur_col < ref_col) {
            left = middle + 1;
          } else {
            match = middle;
            break;
          }
        }

        // if the element with the same column index in the reference row has been found
        if (match != -1) { atomicAdd(&weight_i[j], ref_val); }
      }
    }
  }
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
// Using list of node pairs
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_is_pairs(edge_t num_pairs,
                                 edge_t const* csrPtr,
                                 vertex_t const* csrInd,
                                 vertex_t const* first_pair,
                                 vertex_t const* second_pair,
                                 weight_t const* v,
                                 weight_t* work,
                                 weight_t* weight_i,
                                 weight_t* weight_s)
{
  edge_t i, idx, Ni, Nj, match;
  vertex_t row, col, ref, cur, ref_col, cur_col;
  weight_t ref_val;

  for (idx = threadIdx.z + blockIdx.z * blockDim.z; idx < num_pairs;
       idx += gridDim.z * blockDim.z) {
    row = first_pair[idx];
    col = second_pair[idx];

    // find which row has least elements (and call it reference row)
    Ni  = csrPtr[row + 1] - csrPtr[row];
    Nj  = csrPtr[col + 1] - csrPtr[col];
    ref = (Ni < Nj) ? row : col;
    cur = (Ni < Nj) ? col : row;

    // compute new sum weights
    weight_s[idx] = work[row] + work[col];

    // compute new intersection weights
    // search for the element with the same column index in the reference row
    for (i = csrPtr[ref] + threadIdx.x + blockIdx.x * blockDim.x; i < csrPtr[ref + 1];
         i += gridDim.x * blockDim.x) {
      match   = -1;
      ref_col = csrInd[i];
      if (weighted) {
        ref_val = v[ref_col];
      } else {
        ref_val = 1.0;
      }

      // binary search (column indices are sorted within each row)
      edge_t left  = csrPtr[cur];
      edge_t right = csrPtr[cur + 1] - 1;
      while (left <= right) {
        edge_t middle = (left + right) >> 1;
        cur_col       = csrInd[middle];
        if (cur_col > ref_col) {
          right = middle - 1;
        } else if (cur_col < ref_col) {
          left = middle + 1;
        } else {
          match = middle;
          break;
        }
      }

      // if the element with the same column index in the reference row has been found
      if (match != -1) { atomicAdd(&weight_i[idx], ref_val); }
    }
  }
}

// Jaccard  weights (*weight)
template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_jw(edge_t e,
                           weight_t const* weight_i,
                           weight_t const* weight_s,
                           weight_t* weight_j)
{
  edge_t j;
  weight_t Wi, Ws, Wu;

  for (j = threadIdx.x + blockIdx.x * blockDim.x; j < e; j += gridDim.x * blockDim.x) {
    Wi          = weight_i[j];
    Ws          = weight_s[j];
    Wu          = Ws - Wi;
    weight_j[j] = (Wi / Wu);
  }
}

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
int jaccard(vertex_t n,
            edge_t e,
            edge_t const* csrPtr,
            vertex_t const* csrInd,
            weight_t const* weight_in,
            weight_t* work,
            weight_t* weight_i,
            weight_t* weight_s,
            weight_t* weight_j)
{
  rmm::cuda_stream_view stream_view;
  dim3 nthreads, nblocks;
  int y = 4;

  // setup launch configuration
  nthreads.x = 32;
  nthreads.y = y;
  nthreads.z = 1;
  nblocks.x  = 1;
  nblocks.y  = min((n + nthreads.y - 1) / nthreads.y, vertex_t{CUDA_MAX_BLOCKS});
  nblocks.z  = 1;

  // launch kernel
  jaccard_row_sum<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads, 0, stream_view.value()>>>(n, csrPtr, csrInd, weight_in, work);

  thrust::fill(rmm::exec_policy(stream_view), weight_i, weight_i + e, weight_t{0.0});

  // setup launch configuration
  nthreads.x = 32 / y;
  nthreads.y = y;
  nthreads.z = 8;
  nblocks.x  = 1;
  nblocks.y  = 1;
  nblocks.z  = min((n + nthreads.z - 1) / nthreads.z, vertex_t{CUDA_MAX_BLOCKS});  // 1;

  // launch kernel
  jaccard_is<weighted, vertex_t, edge_t, weight_t><<<nblocks, nthreads, 0, stream_view.value()>>>(
    n, csrPtr, csrInd, weight_in, work, weight_i, weight_s);

  // setup launch configuration
  nthreads.x = min(e, edge_t{CUDA_MAX_KERNEL_THREADS});
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((e + nthreads.x - 1) / nthreads.x, edge_t{CUDA_MAX_BLOCKS});
  nblocks.y  = 1;
  nblocks.z  = 1;

  // launch kernel
  jaccard_jw<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads, 0, stream_view.value()>>>(e, weight_i, weight_s, weight_j);

  return 0;
}

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
int jaccard_pairs(vertex_t n,
                  edge_t num_pairs,
                  edge_t const* csrPtr,
                  vertex_t const* csrInd,
                  vertex_t const* first_pair,
                  vertex_t const* second_pair,
                  weight_t const* weight_in,
                  weight_t* work,
                  weight_t* weight_i,
                  weight_t* weight_s,
                  weight_t* weight_j)
{
  dim3 nthreads, nblocks;
  int y = 4;

  // setup launch configuration
  nthreads.x = 32;
  nthreads.y = y;
  nthreads.z = 1;
  nblocks.x  = 1;
  nblocks.y  = min((n + nthreads.y - 1) / nthreads.y, vertex_t{CUDA_MAX_BLOCKS});
  nblocks.z  = 1;

  // launch kernel
  jaccard_row_sum<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(n, csrPtr, csrInd, weight_in, work);
  cudaDeviceSynchronize();

  // NOTE: initilized weight_i vector with 0.0
  // fill(num_pairs, weight_i, weight_t{0.0});

  // setup launch configuration
  nthreads.x = 32;
  nthreads.y = 1;
  nthreads.z = 8;
  nblocks.x  = 1;
  nblocks.y  = 1;
  nblocks.z  = min((n + nthreads.z - 1) / nthreads.z, vertex_t{CUDA_MAX_BLOCKS});  // 1;

  // launch kernel
  jaccard_is_pairs<weighted, vertex_t, edge_t, weight_t><<<nblocks, nthreads>>>(
    num_pairs, csrPtr, csrInd, first_pair, second_pair, weight_in, work, weight_i, weight_s);

  // setup launch configuration
  nthreads.x = min(num_pairs, edge_t{CUDA_MAX_KERNEL_THREADS});
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((num_pairs + nthreads.x - 1) / nthreads.x, (edge_t)CUDA_MAX_BLOCKS);
  nblocks.y  = 1;
  nblocks.z  = 1;

  // launch kernel
  jaccard_jw<weighted, vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads>>>(num_pairs, weight_i, weight_s, weight_j);

  return 0;
}
}  // namespace detail

template <typename VT, typename ET, typename WT>
void jaccard(legacy::GraphCSRView<VT, ET, WT> const& graph, WT const* weights, WT* result)
{
  CUGRAPH_EXPECTS(result != nullptr, "Invalid input argument: result pointer is NULL");

  rmm::device_vector<WT> weight_i(graph.number_of_edges);
  rmm::device_vector<WT> weight_s(graph.number_of_edges);
  rmm::device_vector<WT> work(graph.number_of_vertices);

  if (weights == nullptr) {
    cugraph::detail::jaccard<false, VT, ET, WT>(graph.number_of_vertices,
                                                graph.number_of_edges,
                                                graph.offsets,
                                                graph.indices,
                                                weights,
                                                work.data().get(),
                                                weight_i.data().get(),
                                                weight_s.data().get(),
                                                result);
  } else {
    cugraph::detail::jaccard<true, VT, ET, WT>(graph.number_of_vertices,
                                               graph.number_of_edges,
                                               graph.offsets,
                                               graph.indices,
                                               weights,
                                               work.data().get(),
                                               weight_i.data().get(),
                                               weight_s.data().get(),
                                               result);
  }
}

template <typename VT, typename ET, typename WT>
void jaccard_list(legacy::GraphCSRView<VT, ET, WT> const& graph,
                  WT const* weights,
                  ET num_pairs,
                  VT const* first,
                  VT const* second,
                  WT* result)
{
  CUGRAPH_EXPECTS(result != nullptr, "Invalid input argument: result pointer is NULL");
  CUGRAPH_EXPECTS(first != nullptr, "Invalid input argument: first is NULL");
  CUGRAPH_EXPECTS(second != nullptr, "Invalid input argument: second in NULL");

  rmm::device_vector<WT> weight_i(num_pairs, WT{0.0});
  rmm::device_vector<WT> weight_s(num_pairs);
  rmm::device_vector<WT> work(graph.number_of_vertices);

  if (weights == nullptr) {
    cugraph::detail::jaccard_pairs<false, VT, ET, WT>(graph.number_of_vertices,
                                                      num_pairs,
                                                      graph.offsets,
                                                      graph.indices,
                                                      first,
                                                      second,
                                                      weights,
                                                      work.data().get(),
                                                      weight_i.data().get(),
                                                      weight_s.data().get(),
                                                      result);
  } else {
    cugraph::detail::jaccard_pairs<true, VT, ET, WT>(graph.number_of_vertices,
                                                     num_pairs,
                                                     graph.offsets,
                                                     graph.indices,
                                                     first,
                                                     second,
                                                     weights,
                                                     work.data().get(),
                                                     weight_i.data().get(),
                                                     weight_s.data().get(),
                                                     result);
  }
}

template void jaccard<int32_t, int32_t, float>(legacy::GraphCSRView<int32_t, int32_t, float> const&,
                                               float const*,
                                               float*);
template void jaccard<int32_t, int32_t, double>(
  legacy::GraphCSRView<int32_t, int32_t, double> const&, double const*, double*);
template void jaccard<int64_t, int64_t, float>(legacy::GraphCSRView<int64_t, int64_t, float> const&,
                                               float const*,
                                               float*);
template void jaccard<int64_t, int64_t, double>(
  legacy::GraphCSRView<int64_t, int64_t, double> const&, double const*, double*);
template void jaccard_list<int32_t, int32_t, float>(
  legacy::GraphCSRView<int32_t, int32_t, float> const&,
  float const*,
  int32_t,
  int32_t const*,
  int32_t const*,
  float*);
template void jaccard_list<int32_t, int32_t, double>(
  legacy::GraphCSRView<int32_t, int32_t, double> const&,
  double const*,
  int32_t,
  int32_t const*,
  int32_t const*,
  double*);
template void jaccard_list<int64_t, int64_t, float>(
  legacy::GraphCSRView<int64_t, int64_t, float> const&,
  float const*,
  int64_t,
  int64_t const*,
  int64_t const*,
  float*);
template void jaccard_list<int64_t, int64_t, double>(
  legacy::GraphCSRView<int64_t, int64_t, double> const&,
  double const*,
  int64_t,
  int64_t const*,
  int64_t const*,
  double*);

}  // namespace cugraph
