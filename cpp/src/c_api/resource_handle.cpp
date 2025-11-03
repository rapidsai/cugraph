/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/resource_handle.hpp"

#include <cugraph_c/resource_handle.h>

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

extern "C" int cugraph_resource_handle_get_comm_size(const cugraph_resource_handle_t* handle)
{
  auto internal = reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle);
  auto& comm    = internal->handle_->get_comms();
  return static_cast<int>(comm.get_size());
}
