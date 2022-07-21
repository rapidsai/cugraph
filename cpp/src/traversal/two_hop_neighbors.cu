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

#include "two_hop_neighbors.cuh"
#include <cugraph/algorithms.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

namespace cugraph {

template <typename VT, typename ET, typename WT>
std::unique_ptr<legacy::GraphCOO<VT, ET, WT>> get_two_hop_neighbors(
  legacy::GraphCSRView<VT, ET, WT> const& graph)
{
  cudaStream_t stream{nullptr};

  rmm::device_vector<ET> exsum_degree(graph.number_of_edges + 1);
  ET* d_exsum_degree = exsum_degree.data().get();

  // Find the degree of the out vertex of each edge
  degree_iterator<ET> deg_it(graph.offsets);
  deref_functor<degree_iterator<ET>, ET> deref(deg_it);
  exsum_degree[0] = ET{0};
  thrust::transform(rmm::exec_policy(stream),
                    graph.indices,
                    graph.indices + graph.number_of_edges,
                    d_exsum_degree + 1,
                    deref);

  // Take the inclusive sum of the degrees
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         d_exsum_degree + 1,
                         d_exsum_degree + graph.number_of_edges + 1,
                         d_exsum_degree + 1);

  // Copy out the last value to get the size of scattered output
  ET output_size = exsum_degree[graph.number_of_edges];

  // Allocate memory for the scattered output
  rmm::device_vector<VT> first_pair(output_size);
  rmm::device_vector<VT> second_pair(output_size);

  VT* d_first_pair  = first_pair.data().get();
  VT* d_second_pair = second_pair.data().get();

  // Figure out number of blocks and allocate memory for block bucket offsets
  ET num_blocks = (output_size + TWO_HOP_BLOCK_SIZE - 1) / TWO_HOP_BLOCK_SIZE;
  rmm::device_vector<ET> block_bucket_offsets(num_blocks + 1);

  ET* d_block_bucket_offsets = block_bucket_offsets.data().get();

  // Compute the block bucket offsets
  dim3 grid, block;
  block.x = 512;
  grid.x  = min((ET)MAXBLOCKS, (num_blocks / 512) + 1);
  compute_bucket_offsets_kernel<<<grid, block, 0, nullptr>>>(
    d_exsum_degree, d_block_bucket_offsets, graph.number_of_edges, output_size);

  block_bucket_offsets[num_blocks] = graph.number_of_edges;

  // Scatter the expanded edge lists into temp space
  grid.x = min((ET)MAXBLOCKS, num_blocks);
  scatter_expand_kernel<<<grid, block, 0, nullptr>>>(d_exsum_degree,
                                                     graph.indices,
                                                     graph.offsets,
                                                     d_block_bucket_offsets,
                                                     graph.number_of_vertices,
                                                     output_size,
                                                     num_blocks,
                                                     d_first_pair,
                                                     d_second_pair);

  // TODO:  This would be faster in a hash table (no sorting), unless there's
  //        some reason that the result has to be sorted
  // Remove duplicates and self pairings
  auto tuple_start = thrust::make_zip_iterator(thrust::make_tuple(d_first_pair, d_second_pair));
  auto tuple_end   = tuple_start + output_size;
  thrust::sort(rmm::exec_policy(stream), tuple_start, tuple_end);
  tuple_end = thrust::copy_if(
    rmm::exec_policy(stream), tuple_start, tuple_end, tuple_start, self_loop_flagger<VT>());
  tuple_end = thrust::unique(rmm::exec_policy(stream), tuple_start, tuple_end);

  // Get things ready to return
  ET outputSize = tuple_end - tuple_start;

  auto result =
    std::make_unique<legacy::GraphCOO<VT, ET, WT>>(graph.number_of_vertices, outputSize, false);

  cudaMemcpy(result->src_indices(), d_first_pair, sizeof(VT) * outputSize, cudaMemcpyDefault);
  cudaMemcpy(result->dst_indices(), d_second_pair, sizeof(VT) * outputSize, cudaMemcpyDefault);

  return result;
}

template std::unique_ptr<legacy::GraphCOO<int, int, float>> get_two_hop_neighbors(
  legacy::GraphCSRView<int, int, float> const&);

template std::unique_ptr<legacy::GraphCOO<int, int, double>> get_two_hop_neighbors(
  legacy::GraphCSRView<int, int, double> const&);

}  // namespace cugraph
