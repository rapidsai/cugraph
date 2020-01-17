/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "degree.cuh"
namespace cugraph { 
namespace snmg {
/**
 * Single node multi-GPU method for degree calculation on a partitioned graph.
 * @param x Indicates whether to compute in degree, out degree, or the sum of both.
 *    0 = in + out degree
 *    1 = in-degree
 *    2 = out-degree
 * @param part_off The vertex partitioning of the global graph
 * @param off The offsets array of the local partition
 * @param ind The indices array of the local partition
 * @param degree Pointer to pointers to memory on each GPU for the result
 * @return Error code
 */
template<typename idx_t>
void snmg_degree(int x, size_t* part_off, idx_t* off, idx_t* ind, idx_t** degree) {
  sync_all();
  SNMGinfo env;
  auto i = env.get_thread_num();
  auto p = env.get_num_threads();

  // Getting the global and local vertices and edges
  size_t glob_v = part_off[p];
  size_t loc_v = part_off[i + 1] - part_off[i];
  idx_t tmp;
  CUDA_TRY(cudaMemcpy(&tmp, &off[loc_v], sizeof(idx_t), cudaMemcpyDeviceToHost));
  size_t loc_e = tmp;

  // Allocating the local result array, and setting all entries to zero.
  idx_t* local_result;
  ALLOC_TRY((void** )&local_result, glob_v * sizeof(idx_t), nullptr);
  thrust::fill(rmm::exec_policy(nullptr)->on(nullptr), local_result, local_result + glob_v, 0);

  // In-degree
  if (x == 1 || x == 0) {
    dim3 nthreads, nblocks;
    nthreads.x = min(static_cast<idx_t>(loc_e), static_cast<idx_t>(CUDA_MAX_KERNEL_THREADS));
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min(static_cast<idx_t>((loc_e + nthreads.x - 1) / nthreads.x),
                    static_cast<idx_t>(env.get_num_sm() * 32));
    nblocks.y = 1;
    nblocks.z = 1;
    cugraph::detail::degree_coo<idx_t, idx_t> <<<nblocks, nthreads>>>(static_cast<idx_t>(loc_e),
                                                     static_cast<idx_t>(loc_e),
                                                     ind,
                                                     local_result);
    CUDA_CHECK_LAST();
  }

  // Out-degree
  if (x == 2 || x == 0) {
    dim3 nthreads, nblocks;
    nthreads.x = min(static_cast<idx_t>(loc_v), static_cast<idx_t>(CUDA_MAX_KERNEL_THREADS));
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min(static_cast<idx_t>((loc_v + nthreads.x - 1) / nthreads.x),
                    static_cast<idx_t>(env.get_num_sm() * 32));
    nblocks.y = 1;
    nblocks.z = 1;
    cugraph::detail::degree_offsets<idx_t, idx_t> <<<nblocks, nthreads>>>(static_cast<idx_t>(loc_v),
                                                         static_cast<idx_t>(loc_e),
                                                         off,
                                                         local_result + part_off[i]);
    CUDA_CHECK_LAST();
  }

  // Combining the local results into global results
  sync_all();
  treeReduce<idx_t, thrust::plus<idx_t> >(env, glob_v, local_result, degree);

  // Broadcasting the global result to all GPUs
  treeBroadcast(env, glob_v, local_result, degree);

  
}

template void snmg_degree<int>(int x, size_t* part_off, int* off, int* ind, int** degree);

template<>
void snmg_degree<int64_t>(int x,
                               size_t* part_off,
                               int64_t* off,
                               int64_t* ind,
                               int64_t** degree) {
  sync_all();
  SNMGinfo env;
  auto i = env.get_thread_num();
  auto p = env.get_num_threads();

  // Getting the global and local vertices and edges
  size_t glob_v = part_off[p];
  size_t loc_v = part_off[i + 1] - part_off[i];
  int64_t tmp;
  CUDA_TRY(cudaMemcpy(&tmp, &off[loc_v], sizeof(int64_t), cudaMemcpyDeviceToHost));
  size_t loc_e = tmp;

  // Allocating the local result array, and setting all entries to zero.
  int64_t* local_result;
  ALLOC_TRY((void** )&local_result, glob_v * sizeof(int64_t), nullptr);
  thrust::fill(rmm::exec_policy(nullptr)->on(nullptr), local_result, local_result + glob_v, 0);

  // In-degree
  if (x == 1 || x == 0) {
    dim3 nthreads, nblocks;
    nthreads.x = min(static_cast<int64_t>(loc_e), static_cast<int64_t>(CUDA_MAX_KERNEL_THREADS));
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min(static_cast<int64_t>((loc_e + nthreads.x - 1) / nthreads.x),
                    static_cast<int64_t>(env.get_num_sm() * 32));
    nblocks.y = 1;
    nblocks.z = 1;
    cugraph::detail::degree_coo<int64_t, double> <<<nblocks, nthreads>>>(static_cast<int64_t>(loc_e),
                                                        static_cast<int64_t>(loc_e),
                                                        ind,
                                                        reinterpret_cast<double*>(local_result));
    CUDA_CHECK_LAST();
  }

  // Out-degree
  if (x == 2 || x == 0) {
    dim3 nthreads, nblocks;
    nthreads.x = min(static_cast<int64_t>(loc_v), static_cast<int64_t>(CUDA_MAX_KERNEL_THREADS));
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min(static_cast<int64_t>((loc_v + nthreads.x - 1) / nthreads.x),
                    static_cast<int64_t>(env.get_num_sm() * 32));
    nblocks.y = 1;
    nblocks.z = 1;
    cugraph::detail::degree_offsets<int64_t, double> <<<nblocks, nthreads>>>(static_cast<int64_t>(loc_v),
                                                            static_cast<int64_t>(loc_e),
                                                            off,
                                                            reinterpret_cast<double*>(local_result
                                                                + part_off[i]));
    CUDA_CHECK_LAST();
  }

  // Convert the values written as doubles back to int64:
  dim3 nthreads, nblocks;
  nthreads.x = min(static_cast<int64_t>(glob_v), static_cast<int64_t>(CUDA_MAX_KERNEL_THREADS));
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x = min(static_cast<int64_t>((glob_v + nthreads.x - 1) / nthreads.x),
                  static_cast<int64_t>(env.get_num_sm() * 32));
  nblocks.y = 1;
  nblocks.z = 1;
  cugraph::detail::type_convert<double, int64_t> <<<nblocks, nthreads>>>(reinterpret_cast<double*>(local_result), glob_v);
  CUDA_CHECK_LAST();

  // Combining the local results into global results
  treeReduce<int64_t, thrust::plus<int64_t> >(env, glob_v, local_result, degree);

  // Broadcasting the global result to all GPUs
  treeBroadcast(env, glob_v, local_result, degree);

  
}

template<typename idx_t>
void snmg_degree_impl(int x,
                               size_t* part_offsets,
                               gdf_column* off,
                               gdf_column* ind,
                               gdf_column** x_cols) {
  CUGRAPH_EXPECTS(off->size > 0, "Invalid API parameter");
  CUGRAPH_EXPECTS(ind->size > 0, "Invalid API parameter");
  CUGRAPH_EXPECTS(off->dtype == ind->dtype, "Unsupported data type");
  CUGRAPH_EXPECTS(off->null_count + ind->null_count == 0, "Column must be valid");

  auto p = omp_get_num_threads();

  idx_t* degree[p];
  for (auto i = 0; i < p; ++i) {
    CUGRAPH_EXPECTS(x_cols[i] != nullptr, "Invalid API parameter");
    CUGRAPH_EXPECTS(x_cols[i]->size > 0, "Invalid API parameter");
    degree[i] = static_cast<idx_t*>(x_cols[i]->data);
  }

  snmg_degree(x,
              part_offsets,
              static_cast<idx_t*>(off->data),
              static_cast<idx_t*>(ind->data),
              degree);
}

} //namespace snmg

void snmg_degree(int x,
                 size_t* part_offsets,
                 gdf_column* off,
                 gdf_column* ind,
                 gdf_column** x_cols) {
  CUGRAPH_EXPECTS(part_offsets != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(off != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(ind != nullptr, "Invalid API parameter");
  CUGRAPH_EXPECTS(x_cols != nullptr, "Invalid API parameter");
  switch (off->dtype) {
    case GDF_INT32:
      return snmg::snmg_degree_impl<int32_t>(x, part_offsets, off, ind, x_cols);
    case GDF_INT64:
      return snmg::snmg_degree_impl<int64_t>(x, part_offsets, off, ind, x_cols);
    default:
      CUGRAPH_FAIL("Unsupported data type");
  }
}

} // namespace cugraph
