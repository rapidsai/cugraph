/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>

namespace cugraph {
namespace mtmg {

class handle_t {
 public:
  /**
   * @brief Get the raft handle
   *
   * @return const reference to a raft handle
   */
  raft::handle_t const& raft_handle() const { return *raft_handle_; }

  /**
   * @brief Get thread rank
   *
   * @return thread rank
   */
  int get_thread_rank() const { return thread_rank_; }

  /**
   * @brief Get number of gpus
   *
   * @return number of gpus
   */
  int get_size() const { return raft_handle_->get_comms().get_size(); }

  /**
   * @brief Get gpu rank
   *
   * @return gpu rank
   */
  int get_rank() const { return raft_handle_->get_comms().get_rank(); }

 private:
  std::shared_ptr<raft::handle_t> raft_handle_;
  int thread_id;
};

}  // namespace mtmg
}  // namespace cugraph

void sample_code()
{
  /*
    The initial model assumes the following sequence of events:
      1. Create a resource_manager_t instance for each process
      2. Add each GPU to its local resource_manager as a local resource and each remote GPU to all
    other resource managers as a remote resource
      3. When we run a GSQL query:
        a. Determine how many GPUs will be required
        b. Allocate an instance manager with that many GPUs
        c. Each thread can request a handle from the instance manager.  The handle contains the
    necessary information to properly coordinate sending data to GPU memory and launching any
    required GPU kernels.  Threads are assigned to GPUs in a round-robin fashion.  We will need to
    have at least one thread for each local GPU in order for the cuGraph code to execute properly.
        d. Create an edge list for assembling edges for graph creation
        e. The threads can execute CPU code to do whatever is required.  As edges are extracted they
    can be sent to GPU memory. f. As threads complete they can flush their local buffers over to the
    GPU. g. Once all of the threads are complete, the call can be made to create the graph. h. Once
    graph creation is complete, the call can be made to execute the graph algorithm(s). i. CPU
    threads can request subsets of the results to be returned to host memory.

     Need to make sure the API handles this entire work flow.
  */

  // SNMG example
  using vertex_t    = int32_t;
  using weight_t    = float;
  using edge_t      = int32_t;
  using edge_type_t = int32_t;

  size_t const device_buffer_size{32 * 1024 * 1024};
  size_t const thread_buffer_size{1024};
  bool const use_weight{true};
  bool const use_edge_id{true};
  bool const use_edge_type{false};

  //  In main thread...

  resource_manager_t resource_manager;

  // Register 4 GPUs with the manager
  for (int i = 0; i < 4; ++i)
    resource_manager.register_local_gpu(i, i);

  auto instance_manager = manager.create_instance_manager({0, 1, 2, 3});

  //
  // When a query begins, in one thread create this edgelist.
  //
  // Could be a pointer and the pointer/reference could be passed
  // to the threads... not worrying about that mechanism at this point.
  //
  device_shared_wrapper_t<per_device_edgelist_t<vertex_t, weight_t, edge_t, edge_type_t>> edgelist;

  //
  //  Within each thread, do something like this
  //
  auto handle = instance_manager.get_handle();

  if (handle.get_thread_rank() == 0) {
    edgelist.initialize_pointer(handle, device_buffer_size, use_weight, use_edge_id, use_edge_type);
  }

  // Do some thread synchronization.  Initialization of the edgelist is now complete

  thread_edgelist_t<vertex_t, weight_t, edge_t, edge_type_t> thread_edgelist(
    edgelist.get_pointer(handle).value(), thread_buffer_size);

  while (true) {
    // Do database work to extract edges, break out of loop at appropriate time

    // Single edge insertion syntax
    thread_edgelist.append(handle,
                           src,
                           dst,
                           std::make_optional<weight_t>(1),
                           std::make_optional<edge_t>(1),
                           std::nullopt);

    // Multiple edge insertion syntax
    thread_edgelist.append(handle,
                           src_list,
                           dst_list,
                           std::make_optional(wgt_list),
                           std::make_optional(edge_id_list),
                           std::nullopt);
  }

  thread_edgelist.flush(handle);

  // Now we can use the edgelist to construct a graph
}

// I should create a unit test from this.  We can test even with a single GPU system.
//
//   1) Get the number of gpus (let's call this P)
//   2) Create a set of edges (use standard data set creation things in C++)
//   3) Pull the edges back to CPU memory
//   4) Clean up GPU memory
//   5) Launch n > P threads, each operating on a subset of the edge list
//   6) Do a thread join to synchronize
//   7) Launch n > P threads to create the graph, only thread_id == 0 calls the create graph
//   function 8) Synchronization 9) Launch n > P threads to call pagerank, only thread_id == 0 calls
//   the create graph function
//  10) Synchronization
//  11) Launch n > P threads to bring pagerank results back to host memory
//  12) Synchronization
//  13) Validate results
//
