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

#include <cugraph/cugraph_api.h>
#include <cugraph/api_helpers.hpp>

#include <cugraph/visitors/rw_visitor.hpp>

#include <cugraph/visitors/erased_api.hpp>

#include <raft/handle.hpp>

#include <rmm/device_uvector.hpp>

extern "C" cugraph_error_t c_random_walks(const c_raft_handle_t* ptr_handle,
                                          c_graph_envelope_t* ptr_graph_envelope,
                                          c_device_array_t* ptr_d_start,
                                          size_t num_paths,
                                          size_t max_depth,
                                          bool_t flag_use_padding,
                                          c_unique_ptr_t* ptr_sampling_strategy,
                                          c_rw_ret_t* ret)
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

#ifdef _USE_CALLEE_TYPE_RECONSTR
    // callee type-reconstruction: this function has the knoweldge
    // and means to reconstruct the result:
    //
    using algo_ret_t = std::tuple<rmm::device_uvector<vertex_t>,  // requires dispatching
                                  rmm::device_uvector<weight_t>,  // requires dispatching
                                  rmm::device_uvector<index_t>>;  // requires dispatching

    // unpack type-erased result:
    // requires type reconstruction for vertex_t, edge_t, weight_t;
    //
    auto&& ret_tuple = ret_erased.get<algo_ret_t>();

    //(CAVEAT: must allocate, because returns are local to this function!)
    //
    {
      rmm::device_buffer* p_d_vs = new rmm::device_buffer(std::get<0>(ret_tuple).release());
      ret->vertex_paths_         = {p_d_vs, p_d_vs->size()};
    }
    {
      rmm::device_buffer* p_d_ws = new rmm::device_buffer(std::get<1>(ret_tuple).release());
      ret->weight_paths_         = {p_d_ws, p_d_ws->size()};
    }
    {
      rmm::device_buffer* p_d_sz = new rmm::device_buffer(std::get<2>(ret_tuple).release());
      ret->sizes_                = {p_d_sz, p_d_sz->size()};
    }
#else
    // caller deffered type-reconstruction: caller has the knoweldge
    // and means to reconstruct the result:
    // (CAVEAT: must allocate, because `ret_erased` is local to this function!)
    //
    ret->p_erased_ret = new return_t(std::move(ret_erased));
#endif

  } catch (...) {
    status = CUGRAPH_ERROR_UNKNOWN;
  }

  return status;
}
