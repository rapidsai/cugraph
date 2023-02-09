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
#pragma once

#include <raft/core/handle.hpp>

namespace cugraph {
namespace c_api {

struct cugraph_resource_handle_t {
  raft::handle_t* handle_{nullptr};
  bool allocated_{false};

  cugraph_resource_handle_t(void* raft_handle)
  {
    if (raft_handle == nullptr) {
      handle_    = new raft::handle_t{};
      allocated_ = true;
    } else {
      handle_ = reinterpret_cast<raft::handle_t*>(raft_handle);
    }
  }
};

}  // namespace c_api
}  // namespace cugraph
