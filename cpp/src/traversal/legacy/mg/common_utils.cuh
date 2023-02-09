/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#pragma once

#include "../traversal_common.cuh"

#include <cub/cub.cuh>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

namespace cugraph {

namespace mg {

namespace detail {

template <typename degree_t>
constexpr int BitsPWrd = sizeof(degree_t) * 8;

template <typename degree_t>
constexpr int NumberBins = sizeof(degree_t) * 8 + 1;

template <typename return_t>
constexpr inline return_t number_of_words(return_t number_of_bits)
{
  return raft::div_rounding_up_safe(number_of_bits, static_cast<return_t>(BitsPWrd<uint32_t>));
}

template <typename edge_t>
struct isDegreeZero {
  edge_t const* offset_;
  isDegreeZero(edge_t const* offset) : offset_(offset) {}

  __device__ bool operator()(const edge_t& id) const { return (offset_[id + 1] == offset_[id]); }
};

struct set_nth_bit {
  uint32_t* bmap_;
  set_nth_bit(uint32_t* bmap) : bmap_(bmap) {}

  template <typename return_t>
  __device__ void operator()(const return_t& id)
  {
    atomicOr(bmap_ + (id / BitsPWrd<uint32_t>), (uint32_t{1} << (id % BitsPWrd<uint32_t>)));
  }
};

template <typename vertex_t>
bool is_vertex_isolated(rmm::device_vector<uint32_t>& bmap, vertex_t id)
{
  uint32_t word       = bmap[id / BitsPWrd<uint32_t>];
  uint32_t active_bit = static_cast<uint32_t>(1) << (id % BitsPWrd<uint32_t>);
  // If idth bit of bmap is set to 1 then return true
  return ((active_bit & word) != 0);
}

template <typename vertex_t, typename edge_t>
struct BFSStepNoDist {
  uint32_t* output_frontier_;
  uint32_t* visited_;
  vertex_t* predecessors_;

  BFSStepNoDist(uint32_t* output_frontier, uint32_t* visited, vertex_t* predecessors)
    : output_frontier_(output_frontier), visited_(visited), predecessors_(predecessors)
  {
  }

  __device__ bool operator()(vertex_t src, vertex_t dst)
  {
    uint32_t active_bit = static_cast<uint32_t>(1) << (dst % BitsPWrd<uint32_t>);
    uint32_t prev_word  = atomicOr(output_frontier_ + (dst / BitsPWrd<uint32_t>), active_bit);
    bool dst_not_visited_earlier = !(active_bit & visited_[dst / BitsPWrd<uint32_t>]);
    bool dst_not_visited_current = !(prev_word & active_bit);
    // If this thread activates the frontier bitmap for a destination
    // then the source is the predecessor of that destination
    if (dst_not_visited_earlier && dst_not_visited_current) {
      predecessors_[dst] = src;
      return true;
    } else {
      return false;
    }
  }

  // No-op
  void increment_level(void) {}
};

template <typename vertex_t, typename edge_t>
struct BFSStep {
  uint32_t* output_frontier_;
  uint32_t* visited_;
  vertex_t* predecessors_;
  vertex_t* distances_;
  vertex_t level_;

  BFSStep(uint32_t* output_frontier, uint32_t* visited, vertex_t* predecessors, vertex_t* distances)
    : output_frontier_(output_frontier),
      visited_(visited),
      predecessors_(predecessors),
      distances_(distances),
      level_(0)
  {
  }

  __device__ bool operator()(vertex_t src, vertex_t dst)
  {
    uint32_t active_bit = static_cast<uint32_t>(1) << (dst % BitsPWrd<uint32_t>);
    uint32_t prev_word  = atomicOr(output_frontier_ + (dst / BitsPWrd<uint32_t>), active_bit);
    bool dst_not_visited_earlier = !(active_bit & visited_[dst / BitsPWrd<uint32_t>]);
    bool dst_not_visited_current = !(prev_word & active_bit);
    // If this thread activates the frontier bitmap for a destination
    // then the source is the predecessor of that destination
    if (dst_not_visited_earlier && dst_not_visited_current) {
      distances_[dst]    = level_;
      predecessors_[dst] = src;
      return true;
    } else {
      return false;
    }
  }

  void increment_level(void) { ++level_; }
};

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t populate_isolated_vertices(
  raft::handle_t const& handle,
  cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
  rmm::device_vector<vertex_t>& isolated_vertex_ids)
{
  bool is_mg = (handle.comms_initialized() && (graph.local_vertices != nullptr) &&
                (graph.local_offsets != nullptr));

  edge_t vertex_begin_, vertex_end_;
  if (is_mg) {
    vertex_begin_ = graph.local_offsets[handle.get_comms().get_rank()];
    vertex_end_   = graph.local_offsets[handle.get_comms().get_rank()] +
                  graph.local_vertices[handle.get_comms().get_rank()];
  } else {
    vertex_begin_ = 0;
    vertex_end_   = graph.number_of_vertices;
  }
  auto count = thrust::copy_if(handle.get_thrust_policy(),
                               thrust::make_counting_iterator<vertex_t>(vertex_begin_),
                               thrust::make_counting_iterator<vertex_t>(vertex_end_),
                               thrust::make_counting_iterator<edge_t>(0),
                               isolated_vertex_ids.begin(),
                               isDegreeZero<edge_t>(graph.offsets)) -
               isolated_vertex_ids.begin();
  return static_cast<vertex_t>(count);
}

template <typename return_t>
return_t collect_vectors(raft::handle_t const& handle,
                         rmm::device_vector<size_t>& buffer_len,
                         rmm::device_vector<return_t>& local,
                         return_t local_count,
                         rmm::device_vector<return_t>& global)
{
  RAFT_CHECK_CUDA(handle.get_stream());
  buffer_len.resize(handle.get_comms().get_size());
  auto my_rank        = handle.get_comms().get_rank();
  buffer_len[my_rank] = static_cast<size_t>(local_count);
  handle.get_comms().allgather(
    buffer_len.data().get() + my_rank, buffer_len.data().get(), 1, handle.get_stream());
  RAFT_CHECK_CUDA(handle.get_stream());
  // buffer_len now contains the lengths of all local buffers
  // for all ranks

  thrust::host_vector<size_t> h_buffer_len = buffer_len;
  // h_buffer_offsets has to be int because raft allgatherv expects
  // int array for displacement vector. This should be changed in
  // raft so that the displacement is templated
  thrust::host_vector<size_t> h_buffer_offsets(h_buffer_len.size());

  thrust::exclusive_scan(
    thrust::host, h_buffer_len.begin(), h_buffer_len.end(), h_buffer_offsets.begin());
  return_t global_buffer_len = h_buffer_len.back() + h_buffer_offsets.back();

  handle.get_comms().allgatherv(local.data().get(),
                                global.data().get(),
                                h_buffer_len.data(),
                                h_buffer_offsets.data(),
                                handle.get_stream());
  RAFT_CHECK_CUDA(handle.get_stream());
  return global_buffer_len;
}

template <typename return_t>
void add_to_bitmap(raft::handle_t const& handle,
                   rmm::device_vector<uint32_t>& bmap,
                   rmm::device_vector<return_t>& id,
                   return_t count)
{
  cudaStream_t stream = handle.get_stream();
  thrust::for_each(
    handle.get_thrust_policy(), id.begin(), id.begin() + count, set_nth_bit(bmap.data().get()));
  RAFT_CHECK_CUDA(stream);
}

// For all vertex ids i which are isolated (out degree is 0), set
// ith bit of isolated_bmap to 1
template <typename vertex_t, typename edge_t, typename weight_t>
void create_isolated_bitmap(raft::handle_t const& handle,
                            cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                            rmm::device_vector<vertex_t>& local_isolated_ids,
                            rmm::device_vector<vertex_t>& global_isolated_ids,
                            rmm::device_vector<size_t>& temp_buffer_len,
                            rmm::device_vector<uint32_t>& isolated_bmap)
{
  size_t word_count = detail::number_of_words(graph.number_of_vertices);
  local_isolated_ids.resize(graph.number_of_vertices);
  global_isolated_ids.resize(graph.number_of_vertices);
  temp_buffer_len.resize(handle.get_comms().get_size());
  isolated_bmap.resize(word_count);

  vertex_t local_isolated_count  = populate_isolated_vertices(handle, graph, local_isolated_ids);
  vertex_t global_isolated_count = collect_vectors(
    handle, temp_buffer_len, local_isolated_ids, local_isolated_count, global_isolated_ids);
  add_to_bitmap(handle, isolated_bmap, global_isolated_ids, global_isolated_count);
}

template <typename return_t>
return_t remove_duplicates(raft::handle_t const& handle,
                           rmm::device_vector<return_t>& data,
                           return_t data_len)
{
  cudaStream_t stream = handle.get_stream();
  thrust::sort(handle.get_thrust_policy(), data.begin(), data.begin() + data_len);
  auto unique_count =
    thrust::unique(handle.get_thrust_policy(), data.begin(), data.begin() + data_len) -
    data.begin();
  return static_cast<return_t>(unique_count);
}

// Use the fact that any value in id array can only be in
// the range [id_begin, id_end) to create a unique set of
// ids. bmap is expected to be of the length
// id_end/BitsPWrd<uint32_t> and is set to 0 initially
template <uint32_t BLOCK_SIZE, typename return_t>
__global__ void remove_duplicates_kernel(uint32_t* bmap,
                                         return_t* in_id,
                                         return_t id_begin,
                                         return_t id_end,
                                         return_t count,
                                         return_t* out_id,
                                         return_t* out_count)
{
  return_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  return_t id;
  if (tid < count) {
    id = in_id[tid];
  } else {
    // Invalid vertex id to avoid partial thread block execution
    id = id_end;
  }

  int acceptable_vertex = 0;
  // If id is not in the acceptable range then set it to
  // an invalid vertex id
  if ((id >= id_begin) && (id < id_end)) {
    uint32_t active_bit = static_cast<uint32_t>(1) << (id % BitsPWrd<uint32_t>);
    uint32_t prev_word  = atomicOr(bmap + (id / BitsPWrd<uint32_t>), active_bit);
    // If bit was set by this thread then the id is unique
    if (!(prev_word & active_bit)) { acceptable_vertex = 1; }
  }

  __shared__ return_t block_offset;
  typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int thread_write_offset;
  int block_acceptable_vertex_count;
  BlockScan(temp_storage)
    .ExclusiveSum(acceptable_vertex, thread_write_offset, block_acceptable_vertex_count);

  // If the block is not going to write unique ids then return
  if (block_acceptable_vertex_count == 0) { return; }

  if (threadIdx.x == 0) {
    block_offset = cugraph::detail::traversal::atomicAdd(
      out_count, static_cast<return_t>(block_acceptable_vertex_count));
  }
  __syncthreads();

  if (acceptable_vertex) { out_id[block_offset + thread_write_offset] = id; }
}

template <uint32_t BLOCK_SIZE, typename return_t>
__global__ void remove_duplicates_kernel(uint32_t* bmap,
                                         uint32_t* isolated_bmap,
                                         return_t* in_id,
                                         return_t id_begin,
                                         return_t id_end,
                                         return_t count,
                                         return_t* out_id,
                                         return_t* out_count)
{
  return_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  return_t id;
  if (tid < count) {
    id = in_id[tid];
  } else {
    // Invalid vertex id to avoid partial thread block execution
    id = id_end;
  }

  int acceptable_vertex = 0;
  // If id is not in the acceptable range then set it to
  // an invalid vertex id
  if ((id >= id_begin) && (id < id_end)) {
    uint32_t active_bit = static_cast<uint32_t>(1) << (id % BitsPWrd<uint32_t>);
    uint32_t prev_word  = atomicOr(bmap + (id / BitsPWrd<uint32_t>), active_bit);
    // If bit was set by this thread then the id is unique
    if (!(prev_word & active_bit)) {
      // If id is isolated (out-degree == 0) then mark it as unacceptable
      bool is_dst_isolated = active_bit & isolated_bmap[id / BitsPWrd<uint32_t>];
      acceptable_vertex    = !is_dst_isolated;
    }
  }

  __shared__ return_t block_offset;
  typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int thread_write_offset;
  int block_acceptable_vertex_count;
  BlockScan(temp_storage)
    .ExclusiveSum(acceptable_vertex, thread_write_offset, block_acceptable_vertex_count);

  // If the block is not going to write unique ids then return
  if (block_acceptable_vertex_count == 0) { return; }

  if (threadIdx.x == 0) {
    block_offset = cugraph::detail::traversal::atomicAdd(
      out_count, static_cast<return_t>(block_acceptable_vertex_count));
  }
  __syncthreads();

  if (acceptable_vertex) { out_id[block_offset + thread_write_offset] = id; }
}

template <typename return_t>
return_t remove_duplicates(raft::handle_t const& handle,
                           rmm::device_vector<uint32_t>& bmap,
                           rmm::device_vector<return_t>& data,
                           return_t data_len,
                           return_t data_begin,
                           return_t data_end,
                           rmm::device_vector<return_t>& out_data)
{
  cudaStream_t stream = handle.get_stream();

  rmm::device_vector<return_t> unique_count(1, 0);

  thrust::fill(handle.get_thrust_policy(), bmap.begin(), bmap.end(), static_cast<uint32_t>(0));
  constexpr return_t threads = 256;
  return_t blocks            = raft::div_rounding_up_safe(data_len, threads);
  remove_duplicates_kernel<threads><<<blocks, threads, 0, stream>>>(bmap.data().get(),
                                                                    data.data().get(),
                                                                    data_begin,
                                                                    data_end,
                                                                    data_len,
                                                                    out_data.data().get(),
                                                                    unique_count.data().get());
  RAFT_CHECK_CUDA(stream);
  return static_cast<return_t>(unique_count[0]);
}

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t preprocess_input_frontier(
  raft::handle_t const& handle,
  cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
  rmm::device_vector<uint32_t>& bmap,
  rmm::device_vector<uint32_t>& isolated_bmap,
  rmm::device_vector<vertex_t>& input_frontier,
  vertex_t input_frontier_len,
  rmm::device_vector<vertex_t>& output_frontier)
{
  cudaStream_t stream = handle.get_stream();

  vertex_t vertex_begin = graph.local_offsets[handle.get_comms().get_rank()];
  vertex_t vertex_end   = graph.local_offsets[handle.get_comms().get_rank()] +
                        graph.local_vertices[handle.get_comms().get_rank()];
  rmm::device_vector<vertex_t> unique_count(1, 0);

  thrust::fill(handle.get_thrust_policy(), bmap.begin(), bmap.end(), static_cast<uint32_t>(0));
  constexpr vertex_t threads = 256;
  vertex_t blocks            = raft::div_rounding_up_safe(input_frontier_len, threads);
  remove_duplicates_kernel<threads><<<blocks, threads, 0, stream>>>(bmap.data().get(),
                                                                    isolated_bmap.data().get(),
                                                                    input_frontier.data().get(),
                                                                    vertex_begin,
                                                                    vertex_end,
                                                                    input_frontier_len,
                                                                    output_frontier.data().get(),
                                                                    unique_count.data().get());
  RAFT_CHECK_CUDA(stream);
  return static_cast<vertex_t>(unique_count[0]);
}

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t preprocess_input_frontier(
  raft::handle_t const& handle,
  cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
  rmm::device_vector<uint32_t>& bmap,
  rmm::device_vector<vertex_t>& input_frontier,
  vertex_t input_frontier_len,
  rmm::device_vector<vertex_t>& output_frontier)
{
  cudaStream_t stream = handle.get_stream();

  vertex_t vertex_begin = graph.local_offsets[handle.get_comms().get_rank()];
  vertex_t vertex_end   = graph.local_offsets[handle.get_comms().get_rank()] +
                        graph.local_vertices[handle.get_comms().get_rank()];
  rmm::device_vector<vertex_t> unique_count(1, 0);

  thrust::fill(handle.get_thrust_policy(), bmap.begin(), bmap.end(), static_cast<uint32_t>(0));
  constexpr vertex_t threads = 256;
  vertex_t blocks            = raft::div_rounding_up_safe(input_frontier_len, threads);
  remove_duplicates_kernel<threads><<<blocks, threads, 0, stream>>>(bmap.data().get(),
                                                                    input_frontier.data().get(),
                                                                    vertex_begin,
                                                                    vertex_end,
                                                                    input_frontier_len,
                                                                    output_frontier.data().get(),
                                                                    unique_count.data().get());
  RAFT_CHECK_CUDA(stream);
  return static_cast<vertex_t>(unique_count[0]);
}

template <typename vertex_t>
__global__ void fill_kernel(vertex_t* distances, vertex_t count, vertex_t start_vertex)
{
  vertex_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) { return; }
  if (tid == start_vertex) {
    distances[tid] = vertex_t{0};
  } else {
    distances[tid] = cugraph::detail::traversal::vec_t<vertex_t>::max;
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void fill_max_dist(raft::handle_t const& handle,
                   cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                   vertex_t start_vertex,
                   vertex_t global_number_of_vertices,
                   vertex_t* distances)
{
  if (distances == nullptr) { return; }
  vertex_t array_size        = global_number_of_vertices;
  constexpr vertex_t threads = 256;
  vertex_t blocks            = raft::div_rounding_up_safe(array_size, threads);
  fill_kernel<<<blocks, threads, 0, handle.get_stream()>>>(distances, array_size, start_vertex);
}

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t get_global_vertex_count(
  raft::handle_t const& handle,
  cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph)
{
  rmm::device_vector<vertex_t> id(1);
  id[0] = *thrust::max_element(
    handle.get_thrust_policy(), graph.indices, graph.indices + graph.number_of_edges);
  handle.get_comms().allreduce(
    id.data().get(), id.data().get(), 1, raft::comms::op_t::MAX, handle.get_stream());
  vertex_t max_vertex_id = id[0];

  if ((graph.number_of_vertices - 1) > max_vertex_id) {
    max_vertex_id = graph.number_of_vertices - 1;
  }

  return max_vertex_id + 1;
}

}  // namespace detail

}  // namespace mg

}  // namespace cugraph
