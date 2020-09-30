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

  rmm::device_vector<uint64_t> local_sizes_v(num_gpus);
  rmm::device_vector<uint64_t> global_sizes_v(num_gpus);

  // TODO:
  rmm::device_vector<data_t> input_v(data_iter, data_iter + n_elements);
  rmm::device_vector<int32_t> partitions_v(partition_iter, partition_iter + n_elements);

  auto d_local_sizes  = local_sizes_v.data().get();
  auto d_global_sizes = global_sizes_v.data().get();

  thrust::for_each(
    rmm::exec_policy(stream)->on(stream),
    partitions_v.begin(),
    partitions_v.end(),
    [d_local_sizes] __device__(auto p) { atomicAdd(d_local_sizes + p, uint64_t{1}); });

  std::vector<raft::comms::request_t> requests(2 * num_gpus);

  std::for_each(thrust::make_counting_iterator<int>(0),
                thrust::make_counting_iterator<int>(num_gpus),
                [my_gpu, &global_sizes_v, local_sizes_v, &comms, &requests, stream](int gpu) {
                  if (gpu != my_gpu) {
                    comms.irecv(global_sizes_v.data().get() + gpu, 1, gpu, 0, &requests[2 * gpu]);
                    comms.isend(
                      local_sizes_v.data().get() + gpu, 1, gpu, 0, &requests[2 * gpu + 1]);
                  } else {
                    CUDA_CHECK(cudaMemcpyAsync(global_sizes_v.data().get() + gpu,
                                               local_sizes_v.data().get() + gpu,
                                               sizeof(int),
                                               cudaMemcpyDeviceToDevice,
                                               stream));

                    requests[2 * gpu]     = std::numeric_limits<raft::comms::request_t>::max();
                    requests[2 * gpu + 1] = std::numeric_limits<raft::comms::request_t>::max();
                  }
                });

  std::vector<raft::comms::request_t> requests_wait(2 * (num_gpus - 1));

  std::copy_if(
    requests.begin(), requests.end(), requests_wait.begin(), [](raft::comms::request_t r) {
      return r != std::numeric_limits<raft::comms::request_t>::max();
    });

  comms.waitall(requests_wait.size(), requests_wait.data());
  comms.barrier();

  //
  //  Now global_sizes contains all of the counts, we need to
  //  allocate an array of the appropriate size
  //
  int64_t receive_size = thrust::reduce(
    rmm::exec_policy(stream)->on(stream), global_sizes_v.begin(), global_sizes_v.end());

  rmm::device_vector<data_t> temp_data(receive_size);

  // FIXME:  Don't really need sort_by_key here.  We have a host
  //         loop that iterates over all of the gpus.  We could
  //         do num_gpu copy_if calls rather than this sort_by_key
  //         which might be faster.  If isend/irecv do buffering
  //         then we could also potentially save memory by never
  //         realizing partitions_v and only realizing input_v
  //         one partition at a time.
  thrust::sort_by_key(rmm::exec_policy(stream)->on(stream),
                      partitions_v.begin(),
                      partitions_v.end(),
                      input_v.begin());

  std::vector<raft::comms::request_t> requests2(2 * (num_gpus - 1));
  rmm::device_vector<int64_t> temp_v(num_gpus + 1);

  thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                         global_sizes_v.begin(),
                         global_sizes_v.end(),
                         temp_v.begin());

  thrust::host_vector<int64_t> h_global_sizes_v(temp_v);
  h_global_sizes_v[num_gpus] = h_global_sizes_v[num_gpus - 1] + h_global_sizes_v[num_gpus - 1];

  thrust::exclusive_scan(rmm::exec_policy(stream)->on(stream),
                         local_sizes_v.begin(),
                         local_sizes_v.end(),
                         temp_v.begin());

  thrust::host_vector<int64_t> h_local_sizes_v(temp_v);
  h_local_sizes_v[num_gpus] = h_local_sizes_v[num_gpus - 1] + local_sizes_v[num_gpus - 1];

  std::for_each(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_gpus),
    [my_gpu, input_v, &temp_data, h_global_sizes_v, h_local_sizes_v, &comms, &requests2, stream](
      int gpu) {
      if (gpu != my_gpu) {
        comms.irecv(temp_data.data().get() + h_global_sizes_v[gpu],
                    h_global_sizes_v[gpu + 1] - h_global_sizes_v[gpu],
                    gpu,
                    0,
                    &requests2[2 * gpu]);
        comms.isend(input_v.data().get() + h_local_sizes_v[gpu],
                    h_local_sizes_v[gpu + 1] - h_local_sizes_v[gpu],
                    gpu,
                    0,
                    &requests2[2 * gpu - 1]);
      } else {
        CUDA_CHECK(
          cudaMemcpyAsync(temp_data.data().get() + h_local_sizes_v[gpu],
                          input_v.data().get() + h_global_sizes_v[gpu],
                          sizeof(data_t) * (h_local_sizes_v[gpu + 1] - h_local_sizes_v[gpu]),
                          cudaMemcpyDeviceToDevice,
                          stream));
        requests2[2 * gpu]     = std::numeric_limits<raft::comms::request_t>::max();
        requests2[2 * gpu + 1] = std::numeric_limits<raft::comms::request_t>::max();
      }
    });

  std::vector<raft::comms::request_t> requests_wait2(2 * (num_gpus - 1));

  std::copy_if(
    requests2.begin(), requests2.end(), requests_wait2.begin(), [](raft::comms::request_t r) {
      return r != std::numeric_limits<raft::comms::request_t>::max();
    });

  comms.waitall(requests_wait2.size(), requests_wait2.data());
  comms.barrier();

  return temp_data;
}
}  // namespace detail

/**
 * @brief shuffle data to the desired partition
 *
 * MNMG algorithms require shuffling data between partitions
 * to get the data to the right location for computation.
 *
 * This function will be executed on each GPU.  Each gpu
 * has a portion of the data (specified by begin_data and
 * end_data iterators) and an iterator that identifies
 * (for each corresponding element) which GPU the data
 * should be shuffled to.
 *
 * This function operates dynamically, there is no
 * a priori knowledge about where the data will need
 * to be transferred.
 *
 * The return value will be a unique pointer to a newly
 * allocated device pointer containing the data sent to
 * this GPU.
 *
 * @tparam data_t           Type of the data being shuffled
 * @tparam iterator_t       Iterator referencing data to be shuffled
 * @tparam partition_iter_t Iterator identifying the destination partition
 *
 * @param  handle         Library handle (RAFT)
 * @param  data           Device vector containing the data.  Note the
 *                        contents of this vector will be reordered
 * @param  partition_iter Random access iterator for the partition
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
