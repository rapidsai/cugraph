/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
