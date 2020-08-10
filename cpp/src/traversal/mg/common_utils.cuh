/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <raft/integer_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <cub/cub.cuh>
#include "../traversal_common.cuh"

namespace cugraph {

namespace mg {

namespace detail {

template <typename degree_t>
constexpr int BitsPWrd = sizeof(degree_t) * 8;

template <typename degree_t>
constexpr int NumberBins = sizeof(degree_t) * 8 + 1;

template <typename T>
constexpr inline T number_of_words(T number_of_bits)
{
  return raft::div_rounding_up_safe(number_of_bits, static_cast<T>(BitsPWrd<unsigned>));
}

template <typename T>
void print(T * ptr, T count, std::string prefix = "", std::string delim = " ") {
  thrust::device_ptr<T> p(ptr);
  std::string out = prefix;
  for(T i = 0; i < count; ++i) {
    T val = p[i];
    out += std::to_string(val) + delim;
  }
  out += "\n";
  std::cout<<out;
}

template <typename T>
void print(rmm::device_vector<T> &data, T count, std::string prefix = "", std::string delim = " ") {
  print(data.data().get(), count, prefix, delim);
}

template <typename edge_t>
struct isDegreeZero {
edge_t * offset_;
isDegreeZero(edge_t * offset) : offset_(offset) {}

__device__
bool operator()(const edge_t& id) {
  return (offset_[id+1] == offset_[id]);
}
};

struct set_nth_bit {
unsigned * bmap_;
set_nth_bit(unsigned * bmap) : bmap_(bmap) {}

template <typename T>
__device__
void operator()(const T& id) {
  atomicOr(
      bmap_ + (id / BitsPWrd<unsigned>),
      (unsigned{1} << (id % BitsPWrd<unsigned>)));
}
};

template <typename VT, typename ET>
struct BFSPred {
  unsigned* output_frontier_;
  unsigned* visited_;
  VT* predecessors_;

  BFSPred(
    unsigned* output_frontier, unsigned* visited, VT* predecessors)
    : output_frontier_(output_frontier),
      visited_(visited),
      predecessors_(predecessors)
  {
  }

  __device__ bool operator()(VT src, VT dst)
  {
    unsigned active_bit = static_cast<unsigned>(1) << (dst % BitsPWrd<unsigned>);
    unsigned prev_word  = atomicOr(output_frontier_ + (dst / BitsPWrd<unsigned>), active_bit);
    bool dst_not_visited_earlier = !(active_bit & visited_[dst / BitsPWrd<unsigned>]);
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
};

template <typename VT, typename ET>
struct BFSPredDist {
  unsigned* output_frontier_;
  unsigned* visited_;
  VT* predecessors_;
  VT* distances_;
  VT level_;

  BFSPredDist(
    unsigned* output_frontier, unsigned* visited, VT* predecessors, VT* distances, VT level)
    : output_frontier_(output_frontier),
      visited_(visited),
      predecessors_(predecessors),
      distances_(distances),
      level_(level)
  {
  }

  __device__ bool operator()(VT src, VT dst)
  {
    //printf("e : %d %d\n", (int)src, (int)dst);
    unsigned active_bit = static_cast<unsigned>(1) << (dst % BitsPWrd<unsigned>);
    unsigned prev_word  = atomicOr(output_frontier_ + (dst / BitsPWrd<unsigned>), active_bit);
    bool dst_not_visited_earlier = !(active_bit & visited_[dst / BitsPWrd<unsigned>]);
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
};

template <typename VT, typename ET>
struct bfs_pred {
  unsigned* output_frontier_;
  unsigned* visited_;
  VT* predecessors_;

  bfs_pred(
    unsigned* output_frontier, unsigned* visited, VT* predecessors)
    : output_frontier_(output_frontier),
      visited_(visited),
      predecessors_(predecessors)
  {
  }

  __device__ void operator()(VT src, VT dst, VT * frontier, ET * frontier_count)
  {
    unsigned active_bit = static_cast<unsigned>(1) << (dst % BitsPWrd<unsigned>);
    unsigned prev_word  = atomicOr(output_frontier_ + (dst / BitsPWrd<unsigned>), active_bit);
    bool dst_not_visited_earlier = !(active_bit & visited_[dst / BitsPWrd<unsigned>]);
    bool dst_not_visited_current = !(prev_word & active_bit);
    // If this thread activates the frontier bitmap for a destination
    // then the source is the predecessor of that destination
    if (dst_not_visited_earlier && dst_not_visited_current) {
      predecessors_[dst] = src;
      auto count = *frontier_count;
      frontier[count] = dst;
      *frontier_count = count+1;
    }
  }
};

template <typename VT, typename ET>
struct bfs_pred_dist {
  unsigned* output_frontier_;
  unsigned* visited_;
  VT* predecessors_;
  VT* distances_;
  VT level_;

  bfs_pred_dist(
    unsigned* output_frontier, unsigned* visited, VT* predecessors, VT* distances, VT level)
    : output_frontier_(output_frontier),
      visited_(visited),
      predecessors_(predecessors),
      distances_(distances),
      level_(level)
  {
  }

  __device__ void operator()(VT src, VT dst, VT * frontier, ET * frontier_count)
  {
    //printf("e : %d %d\n", (int)src, (int)dst);
    unsigned active_bit = static_cast<unsigned>(1) << (dst % BitsPWrd<unsigned>);
    unsigned prev_word  = atomicOr(output_frontier_ + (dst / BitsPWrd<unsigned>), active_bit);
    bool dst_not_visited_earlier = !(active_bit & visited_[dst / BitsPWrd<unsigned>]);
    bool dst_not_visited_current = !(prev_word & active_bit);
    // If this thread activates the frontier bitmap for a destination
    // then the source is the predecessor of that destination
    if (dst_not_visited_earlier && dst_not_visited_current) {
      distances_[dst]    = level_;
      predecessors_[dst] = src;
      auto count = *frontier_count;
      frontier[count] = dst;
      *frontier_count = count+1;
    }
  }
};

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t populate_isolated_vertices(raft::handle_t const &handle,
    cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
    rmm::device_vector<vertex_t> &isolated_vertex_ids) {

  bool is_mg = (handle.comms_initialized() && (graph.local_vertices != nullptr) &&
                  (graph.local_offsets != nullptr));
  cudaStream_t stream = handle.get_stream();

  edge_t vertex_begin_, vertex_end_;
  if (is_mg) {
    //isolated_vertex_ids.resize(graph.local_vertices[handle.get_comms().get_rank()]);
    vertex_begin_ = graph.local_offsets[handle.get_comms().get_rank()];
    vertex_end_   = graph.local_offsets[handle.get_comms().get_rank()] +
                  graph.local_vertices[handle.get_comms().get_rank()];
  } else {
    //isolated_vertex_ids.resize(graph.number_of_vertices);
    vertex_begin_ = 0;
    vertex_end_   = graph.number_of_vertices;
  }
  auto count = thrust::copy_if(rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator<vertex_t>(vertex_begin_),
      thrust::make_counting_iterator<vertex_t>(vertex_end_),
      thrust::make_counting_iterator<edge_t>(0),
      isolated_vertex_ids.begin(),
      isDegreeZero<edge_t>(graph.offsets)) - isolated_vertex_ids.begin();
  return static_cast<vertex_t>(count);
}

template <typename T>
T collect_vectors(raft::handle_t const &handle,
    rmm::device_vector<size_t> &buffer_len,
    rmm::device_vector<T> &local,
    T local_count,
    rmm::device_vector<T> &global) {
  CHECK_CUDA(handle.get_stream());
  buffer_len.resize(handle.get_comms().get_size());
  auto my_rank = handle.get_comms().get_rank();
  buffer_len[my_rank] = static_cast<size_t>(local_count);
  handle.get_comms().allgather(
      buffer_len.data().get() + my_rank, buffer_len.data().get(),
      1, handle.get_stream());
  CHECK_CUDA(handle.get_stream());
  //buffer_len now contains the lengths of all local buffers
  //for all ranks

  thrust::host_vector<size_t> h_buffer_len = buffer_len;
  //h_buffer_offsets has to be int because raft allgatherv expects
  //int array for displacement vector. This should be changed in
  //raft so that the displacement is templated
  thrust::host_vector<int> h_buffer_offsets(h_buffer_len.size());
  int global_buffer_len = 0;
  for (size_t i = 0; i < h_buffer_len.size(); ++i) {
    h_buffer_offsets[i] = global_buffer_len;
    global_buffer_len += h_buffer_len[i];
  }
  //global.resize(global_buffer_len);

  handle.get_comms().allgatherv(
      local.data().get(),
      global.data().get(),
      h_buffer_len.data(),
      h_buffer_offsets.data(),
      handle.get_stream());
  CHECK_CUDA(handle.get_stream());
  return static_cast<T>(global_buffer_len);
}

template <typename T>
void add_to_bitmap(raft::handle_t const &handle,
    rmm::device_vector<unsigned> &bmap,
    rmm::device_vector<T> &id,
    T count) {
  cudaStream_t stream = handle.get_stream();
  thrust::for_each(rmm::exec_policy(stream)->on(stream),
      id.begin(), id.begin() + count,
      set_nth_bit(bmap.data().get()));
  CHECK_CUDA(stream);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void create_isolated_bitmap(raft::handle_t const &handle,
    cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
    rmm::device_vector<vertex_t> &local_isolated_ids,
    rmm::device_vector<vertex_t> &global_isolated_ids,
    rmm::device_vector<size_t> &temp_buffer_len,
    rmm::device_vector<unsigned> &isolated_bmap) {

  size_t word_count = detail::number_of_words(graph.number_of_vertices);
  local_isolated_ids.resize(graph.number_of_vertices);
  global_isolated_ids.resize(graph.number_of_vertices);
  temp_buffer_len.resize(handle.get_comms().get_size());
  isolated_bmap.resize(word_count);

  vertex_t local_isolated_count =
    populate_isolated_vertices(handle, graph, local_isolated_ids);
  vertex_t global_isolated_count =
    collect_vectors(
      handle,
      temp_buffer_len,
      local_isolated_ids,
      local_isolated_count,
      global_isolated_ids);
  add_to_bitmap(handle,
      isolated_bmap,
      global_isolated_ids,
      global_isolated_count);
}

template <typename T>
T remove_duplicates(raft::handle_t const &handle,
    rmm::device_vector<T> &data,
    T data_len)
{
  cudaStream_t stream = handle.get_stream();
  thrust::sort(
      rmm::exec_policy(stream)->on(stream),
      data.begin(), data.begin() + data_len);
  auto unique_count = thrust::unique(
      rmm::exec_policy(stream)->on(stream),
      data.begin(), data.begin() + data_len) - data.begin();
  return static_cast<T>(unique_count);
}

//Use the fact that any value in id array can only be in
//the range [id_begin, id_end) to create a unique set of
//ids. bmap is expected to be of the length
//id_end/BitsPWrd<unsigned> and is set to 0 initially
template <unsigned BLOCK_SIZE, typename T>
__global__
void remove_duplicates_kernel(
    unsigned * bmap,
    T * in_id,
    T id_begin,
    T id_end,
    T count,
    T * out_id,
    T * out_count) {
  T tid = blockIdx.x*blockDim.x + threadIdx.x;
  T id;
  if (tid < count) {
    id = in_id[tid];
  } else {
    //Invalid vertex id to avoid partial thread block execution
    id = id_end;
  }

  int acceptable_vertex = 0;
  //If id is not in the acceptable range then set it to
  //an invalid vertex id
  if ((id >= id_begin) && (id < id_end)) {
    unsigned active_bit = static_cast<unsigned>(1) << (id % BitsPWrd<unsigned>);
    unsigned prev_word  = atomicOr(bmap + (id / BitsPWrd<unsigned>), active_bit);
    //If bit was set by this thread then the id is unique
    if (!(prev_word & active_bit)) {
      acceptable_vertex = 1;
    }
  }

  __shared__ T block_offset;
  typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int thread_write_offset;
  int block_acceptable_vertex_count;
  BlockScan(temp_storage).ExclusiveSum(
      acceptable_vertex,
      thread_write_offset,
      block_acceptable_vertex_count);

  //If the block is not going to write unique ids then return
  if (block_acceptable_vertex_count == 0) {
    return;
  }

  if (threadIdx.x == 0) {
    block_offset = cugraph::detail::traversal::atomicAdd(out_count,
        static_cast<T>(block_acceptable_vertex_count));
  }
  __syncthreads();

  if (acceptable_vertex) {
    out_id[block_offset + thread_write_offset] = id;
  }

}

template <unsigned BLOCK_SIZE, typename T>
__global__
void remove_duplicates_kernel(
    unsigned * bmap,
    unsigned * isolated_bmap,
    T * in_id,
    T id_begin,
    T id_end,
    T count,
    T * out_id,
    T * out_count) {
  T tid = blockIdx.x*blockDim.x + threadIdx.x;
  T id;
  if (tid < count) {
    id = in_id[tid];
  } else {
    //Invalid vertex id to avoid partial thread block execution
    id = id_end;
  }

  int acceptable_vertex = 0;
  //If id is not in the acceptable range then set it to
  //an invalid vertex id
  if ((id >= id_begin) && (id < id_end)) {
    unsigned active_bit = static_cast<unsigned>(1) << (id % BitsPWrd<unsigned>);
    unsigned prev_word  = atomicOr(bmap + (id / BitsPWrd<unsigned>), active_bit);
    //If bit was set by this thread then the id is unique
    if (!(prev_word & active_bit)) {
      //If id is isolated (out-degree == 0) then mark it as unacceptable
      bool is_dst_isolated =
        active_bit & isolated_bmap[id / BitsPWrd<unsigned>];
      acceptable_vertex = !is_dst_isolated;
    }
  }

  __shared__ T block_offset;
  typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  int thread_write_offset;
  int block_acceptable_vertex_count;
  BlockScan(temp_storage).ExclusiveSum(
      acceptable_vertex,
      thread_write_offset,
      block_acceptable_vertex_count);

  //If the block is not going to write unique ids then return
  if (block_acceptable_vertex_count == 0) {
    return;
  }

  if (threadIdx.x == 0) {
    block_offset = cugraph::detail::traversal::atomicAdd(out_count,
        static_cast<T>(block_acceptable_vertex_count));
  }
  __syncthreads();

  if (acceptable_vertex) {
    out_id[block_offset + thread_write_offset] = id;
  }

}

template <typename T>
T remove_duplicates(raft::handle_t const &handle,
    rmm::device_vector<unsigned> &bmap,
    rmm::device_vector<T> &data,
    T data_len,
    T data_begin, T data_end,
    rmm::device_vector<T> &out_data)
{
  cudaStream_t stream = handle.get_stream();

  rmm::device_vector<T> unique_count(1,0);

  thrust::fill(rmm::exec_policy(stream)->on(stream),
                bmap.begin(),
                bmap.end(),
                static_cast<unsigned>(0));
  T threads = 256;
  T blocks = raft::div_rounding_up_safe(data_len, threads);
  remove_duplicates_kernel<256><<<blocks, threads, 0, stream>>>(
      bmap.data().get(),
      data.data().get(),
      data_begin,
      data_end,
      data_len,
      out_data.data().get(),
      unique_count.data().get()
      );
  CHECK_CUDA(stream);
  return static_cast<T>(unique_count[0]);
}


template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t
preprocess_input_frontier(raft::handle_t const &handle,
    cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
    rmm::device_vector<unsigned> &bmap,
    rmm::device_vector<unsigned> &isolated_bmap,
    rmm::device_vector<vertex_t> &input_frontier,
    vertex_t input_frontier_len,
    rmm::device_vector<vertex_t> &output_frontier)
{
  cudaStream_t stream = handle.get_stream();

  vertex_t vertex_begin = graph.local_offsets[handle.get_comms().get_rank()];
  vertex_t vertex_end   = graph.local_offsets[handle.get_comms().get_rank()] +
      graph.local_vertices[handle.get_comms().get_rank()];
  rmm::device_vector<vertex_t> unique_count(1,0);

  thrust::fill(rmm::exec_policy(stream)->on(stream),
                bmap.begin(),
                bmap.end(),
                static_cast<unsigned>(0));
  vertex_t threads = 256;
  vertex_t blocks = raft::div_rounding_up_safe(input_frontier_len, threads);
  remove_duplicates_kernel<256><<<blocks, threads, 0, stream>>>(
      bmap.data().get(),
      isolated_bmap.data().get(),
      input_frontier.data().get(),
      vertex_begin,
      vertex_end,
      input_frontier_len,
      output_frontier.data().get(),
      unique_count.data().get()
      );
  CHECK_CUDA(stream);
  return static_cast<vertex_t>(unique_count[0]);
}

template <typename vertex_t, typename edge_t, typename weight_t>
vertex_t
preprocess_input_frontier(raft::handle_t const &handle,
    cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
    rmm::device_vector<unsigned> &bmap,
    rmm::device_vector<vertex_t> &input_frontier,
    vertex_t input_frontier_len,
    rmm::device_vector<vertex_t> &output_frontier)
{
  cudaStream_t stream = handle.get_stream();

  vertex_t vertex_begin = graph.local_offsets[handle.get_comms().get_rank()];
  vertex_t vertex_end   = graph.local_offsets[handle.get_comms().get_rank()] +
      graph.local_vertices[handle.get_comms().get_rank()];
  rmm::device_vector<vertex_t> unique_count(1,0);

  thrust::fill(rmm::exec_policy(stream)->on(stream),
                bmap.begin(),
                bmap.end(),
                static_cast<unsigned>(0));
  vertex_t threads = 256;
  vertex_t blocks = raft::div_rounding_up_safe(input_frontier_len, threads);
  remove_duplicates_kernel<256><<<blocks, threads, 0, stream>>>(
      bmap.data().get(),
      input_frontier.data().get(),
      vertex_begin,
      vertex_end,
      input_frontier_len,
      output_frontier.data().get(),
      unique_count.data().get()
      );
  CHECK_CUDA(stream);
  return static_cast<vertex_t>(unique_count[0]);
}

template <typename vertex_t>
__global__
void fill_kernel(
         vertex_t *distances,
         vertex_t count,
         vertex_t start_vertex) {
  vertex_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid >= count) {
    return;
  }
  if (tid == start_vertex) {
    distances[tid] = vertex_t{0};
  } else {
    distances[tid] = cugraph::detail::traversal::vec_t<vertex_t>::max;
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void fill_max_dist(raft::handle_t const &handle,
         cugraph::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
         vertex_t start_vertex,
         vertex_t *distances) {
  vertex_t array_size = graph.number_of_vertices;
  vertex_t threads = 256;
  vertex_t blocks = raft::div_rounding_up_safe(array_size, threads);
  fill_kernel<<<blocks, threads, 0, handle.get_stream()>>>(
      distances, array_size, start_vertex);
}

}  // namespace detail

}  // namespace mg

}  // namespace cugraph
