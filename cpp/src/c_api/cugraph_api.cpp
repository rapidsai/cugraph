/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cugraph_c/cugraph_api.h>
#include <cugraph/api_helpers.hpp>

#include <cugraph/visitors/rw_visitor.hpp>

#include <cugraph/visitors/erased_api.hpp>

#include <raft/handle.hpp>

#include <rmm/device_uvector.hpp>

extern "C" cugraph_error_t cugraph_random_walks(const cugraph_raft_handle_t* ptr_handle,
                                                cugraph_graph_envelope_t* ptr_graph_envelope,
                                                cugraph_device_array_t* ptr_d_start,
                                                size_t num_paths,
                                                size_t max_depth,
                                                bool_t flag_use_padding,
                                                cugraph_unique_ptr_t* ptr_sampling_strategy,
                                                cugraph_rw_ret_t* ret)
{
  using namespace cugraph::visitors;

  using ptr_sampling_t = std::unique_ptr<cugraph::sampling_params_t>;

  cugraph_error_t status = CUGRAPH_SUCCESS;

  try {
    // unpack C stub arguments:
    //
    graph_envelope_t* p_g_env = reinterpret_cast<graph_envelope_t*>(ptr_graph_envelope);

    raft::handle_t const* p_raft_handle = reinterpret_cast<raft::handle_t const*>(ptr_handle);
    void* p_d_start                     = reinterpret_cast<void*>(ptr_d_start);

    bool use_padding = static_cast<bool>(flag_use_padding);

    ptr_sampling_t* p_uniq_sampling = reinterpret_cast<ptr_sampling_t*>(ptr_sampling_strategy);

    // pack type-erased algorithm arguments:
    //
    erased_pack_t ep{const_cast<raft::handle_t*>(p_raft_handle),
                     p_d_start,
                     &num_paths,
                     &max_depth,
                     &use_padding,
                     p_uniq_sampling};

    // call algorithm:
    //
    return_t ret_erased = cugraph::api::random_walks(*p_g_env, ep);

    // caller deffered type-reconstruction: caller has the knoweldge
    // and means to reconstruct the result:
    // (CAVEAT: must allocate, because `ret_erased` is local to this function!)
    //
    ret->p_erased_ret = new return_t(std::move(ret_erased));

  } catch (...) {
    status = CUGRAPH_ERROR_UNKNOWN;
  }

  return status;
}
