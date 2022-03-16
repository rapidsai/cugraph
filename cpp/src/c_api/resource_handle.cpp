/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cugraph_c/resource_handle.h>

#include <c_api/resource_handle.hpp>

extern "C" cugraph_resource_handle_t* cugraph_create_resource_handle(void* raft_handle)
{
  try {
    return reinterpret_cast<cugraph_resource_handle_t*>(
      new cugraph::c_api::cugraph_resource_handle_t(raft_handle));
  } catch (...) {
    return nullptr;
  }
}

extern "C" void cugraph_free_resource_handle(cugraph_resource_handle_t* handle)
{
  auto internal = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t*>(handle);
  if (internal->allocated_) delete internal->handle_;
  delete internal;
}

extern "C" int cugraph_resource_handle_get_rank(const cugraph_resource_handle_t* handle)
{
  auto internal = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
  auto& comm    = internal->handle_->get_comms();
  return static_cast<int>(comm.get_rank());
}
