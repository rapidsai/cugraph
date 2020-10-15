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

#include <raft/comms/comms.hpp>
#include <raft/device_atomics.cuh>

namespace cugraph {
namespace experimental {

namespace detail {

//
// FIXME:   This implementation of variable_shuffle stages the data for transfer
//          in host memory.  It would be more efficient, I believe, to stage the
//          data in device memory, but it would require actually instantiating
//          the data in device memory which is already precious in the Louvain
//          implementation.  We should explore if it's actually more efficient
//          through device memory and whether the improvement is worth the extra
//          memory required.
//
template <typename data_t, typename iterator_t, typename partition_iter_t>
rmm::device_vector<data_t> variable_shuffle(raft::handle_t const &handle,
                                            std::size_t n_elements,
                                            iterator_t data_iter,
                                            partition_iter_t partition_iter)
{
  //
  // We need to compute the size of data movement
  //
  raft::comms::comms_t const &comms = handle.get_comms();

  cudaStream_t stream = handle.get_stream();
  int num_gpus        = comms.get_size();
  int my_gpu          = comms.get_rank();

  rmm::device_vector<size_t> local_sizes_v(num_gpus, size_t{0});

  thrust::for_each(rmm::exec_policy(stream)->on(stream),
                   partition_iter,
                   partition_iter + n_elements,
                   [num_gpus, d_local_sizes = local_sizes_v.data().get()] __device__(auto p) {
                     atomicAdd(d_local_sizes + p, size_t{1});
                   });

  std::vector<size_t> h_local_sizes_v(num_gpus);
  std::vector<size_t> h_global_sizes_v(num_gpus);
  std::vector<data_t> h_input_v(n_elements);
  std::vector<int32_t> h_partitions_v(n_elements);

  thrust::copy(local_sizes_v.begin(), local_sizes_v.end(), h_local_sizes_v.begin());
  thrust::copy(partition_iter, partition_iter + n_elements, h_partitions_v.begin());

  std::vector<raft::comms::request_t> requests(2 * num_gpus);

  int request_pos = 0;

  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    if (gpu != my_gpu) {
      comms.irecv(&h_global_sizes_v[gpu], 1, gpu, 0, &requests[request_pos]);
      ++request_pos;
      comms.isend(&h_local_sizes_v[gpu], 1, gpu, 0, &requests[request_pos]);
      ++request_pos;
    } else {
      h_global_sizes_v[gpu] = h_local_sizes_v[gpu];
    }
  }

  if (request_pos > 0) { comms.waitall(request_pos, requests.data()); }

  comms.barrier();

  //
  //  Now global_sizes contains all of the counts, we need to
  //  allocate an array of the appropriate size
  //
  int64_t receive_size =
    thrust::reduce(thrust::host, h_global_sizes_v.begin(), h_global_sizes_v.end());

  std::vector<data_t> temp_data;

  if (receive_size > 0) temp_data.resize(receive_size);

  rmm::device_vector<data_t> input_v(n_elements);

  auto input_start = input_v.begin();

  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    input_start = thrust::copy_if(rmm::exec_policy(stream)->on(stream),
                                  data_iter,
                                  data_iter + n_elements,
                                  partition_iter,
                                  input_start,
                                  [gpu] __device__(int32_t p) { return p == gpu; });
  }

  thrust::copy(input_v.begin(), input_v.end(), h_input_v.begin());

  std::vector<size_t> temp_v(num_gpus + 1);

  thrust::exclusive_scan(
    thrust::host, h_global_sizes_v.begin(), h_global_sizes_v.end(), temp_v.begin());

  temp_v[num_gpus] = temp_v[num_gpus - 1] + h_global_sizes_v[num_gpus - 1];
  h_global_sizes_v = temp_v;

  thrust::exclusive_scan(
    thrust::host, h_local_sizes_v.begin(), h_local_sizes_v.end(), temp_v.begin());

  temp_v[num_gpus] = temp_v[num_gpus - 1] + h_local_sizes_v[num_gpus - 1];
  h_local_sizes_v  = temp_v;

  CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));
  comms.barrier();

  request_pos = 0;

  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    size_t to_receive = h_global_sizes_v[gpu + 1] - h_global_sizes_v[gpu];
    size_t to_send    = h_local_sizes_v[gpu + 1] - h_local_sizes_v[gpu];

    if (gpu != my_gpu) {
      if (to_receive > 0) {
        comms.irecv(
          temp_data.data() + h_global_sizes_v[gpu], to_receive, gpu, 0, &requests[request_pos]);
        ++request_pos;
      }

      if (to_send > 0) {
        comms.isend(
          h_input_v.data() + h_local_sizes_v[gpu], to_send, gpu, 0, &requests[request_pos]);
        ++request_pos;
      }
    } else if (to_receive > 0) {
      std::copy(h_input_v.begin() + h_local_sizes_v[gpu],
                h_input_v.begin() + h_local_sizes_v[gpu + 1],
                temp_data.begin() + h_global_sizes_v[gpu]);
    }
  }

  comms.barrier();

  if (request_pos > 0) { comms.waitall(request_pos, requests.data()); }

  comms.barrier();

  return rmm::device_vector<data_t>(temp_data);
}

}  // namespace detail

/**
 * @brief shuffle data to the desired partition
 *
 * MNMG algorithms require shuffling data between partitions
 * to get the data to the right location for computation.
 *
 * This function operates dynamically, there is no
 * a priori knowledge about where the data will need
 * to be transferred.
 *
 * This function will be executed on each GPU.  Each gpu
 * has a portion of the data (specified by begin_data and
 * end_data iterators) and an iterator that identifies
 * (for each corresponding element) which GPU the data
 * should be shuffled to.
 *
 * The return value will be a device vector containing
 * the data received by this GPU.
 *
 * Note that this function accepts iterators as input.
 * `partition_iterator` will be traversed multiple times.
 *
 * @tparam is_multi_gpu     If true, multi-gpu - shuffle will occur
 *                          If false, single GPU - simple copy will occur
 * @tparam data_t           Type of the data being shuffled
 * @tparam iterator_t       Iterator referencing data to be shuffled
 * @tparam partition_iter_t Iterator identifying the destination partition
 *
 * @param  handle         Library handle (RAFT)
 * @param  n_elements     Number of elements to transfer
 * @param  data_iter      Iterator that returns the elements to be transfered
 * @param  partition_iter Iterator that returns the partition where elements
 *                        should be transfered.
 */
template <bool is_multi_gpu,
          typename data_t,
          typename iterator_t,
          typename partition_iter_t,
          typename std::enable_if_t<is_multi_gpu> * = nullptr>
rmm::device_vector<data_t> variable_shuffle(raft::handle_t const &handle,
                                            std::size_t n_elements,
                                            iterator_t data_iter,
                                            partition_iter_t partition_iter)
{
  return detail::variable_shuffle<data_t>(handle, n_elements, data_iter, partition_iter);
}

template <bool is_multi_gpu,
          typename data_t,
          typename iterator_t,
          typename partition_iter_t,
          typename std::enable_if_t<!is_multi_gpu> * = nullptr>
rmm::device_vector<data_t> variable_shuffle(raft::handle_t const &handle,
                                            std::size_t n_elements,
                                            iterator_t data_iter,
                                            partition_iter_t partition_iter)
{
  return rmm::device_vector<data_t>(data_iter, data_iter + n_elements);
}

}  // namespace experimental
}  // namespace cugraph
