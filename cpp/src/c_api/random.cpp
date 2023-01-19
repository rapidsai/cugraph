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

#include <c_api/error.hpp>
#include <c_api/random.hpp>
#include <c_api/resource_handle.hpp>

#include <cugraph/utilities/host_scalar_comm.hpp>

extern "C" cugraph_error_code_t cugraph_rng_state_create(const cugraph_resource_handle_t* handle,
                                                         uint64_t seed,
                                                         cugraph_rng_state_t** state,
                                                         cugraph_error_t** error)
{
  *state = nullptr;
  *error = nullptr;

  try {
    if (!handle) {
      *error = reinterpret_cast<cugraph_error_t*>(
        new cugraph::c_api::cugraph_error_t{"invalid resource handle"});
      return CUGRAPH_INVALID_HANDLE;
    }

    auto p_handle = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);

    if (p_handle->handle_->comms_initialized()) {
      // need to verify that every seed is different
      auto seed_v = cugraph::host_scalar_allgather(
        p_handle->handle_->get_comms(), seed, p_handle->handle_->get_stream());
      std::sort(seed_v.begin(), seed_v.end());
      if (std::unique(seed_v.begin(), seed_v.end()) != seed_v.end()) {
        *error = reinterpret_cast<cugraph_error_t*>(
          new cugraph::c_api::cugraph_error_t{"seed must be different on each GPU"});
        return CUGRAPH_INVALID_INPUT;
      }
    }

    *state = reinterpret_cast<cugraph_rng_state_t*>(
      new cugraph::c_api::cugraph_rng_state_t{raft::random::RngState{seed}});
    return CUGRAPH_SUCCESS;
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }
}

extern "C" void cugraph_rng_state_free(cugraph_rng_state_t* p)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(p);
  delete internal_pointer;
}
