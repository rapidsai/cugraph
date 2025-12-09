/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph_c/random.h>

#include <raft/random/rng_state.hpp>

namespace cugraph {
namespace c_api {

struct cugraph_rng_state_t {
  raft::random::RngState rng_state_;
};

}  // namespace c_api
}  // namespace cugraph
