/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

// Andrei Schaffer, aschaffer@nvidia.com
//
// This is a collection of aggregates used by (parts of) the API defined in algorithms.hpp;
// These aggregates get propagated to the C-only API (which is why they're non-template aggregates)

#pragma once

namespace cugraph {

enum class sampling_strategy_t : int { UNIFORM = 0, BIASED, NODE2VEC };

struct sampling_params_t {
  sampling_params_t(void) {}

  sampling_params_t(sampling_strategy_t sampling_type,
                    double p             = 1.0,
                    double q             = 1.0,
                    bool use_alpha_cache = false)
    : sampling_type_(sampling_type), p_(p), q_(q), use_alpha_cache_(use_alpha_cache)
  {
  }

  // FIXME: The new C API uses the above constructor, this constructor
  //        is only used by the legacy python/cython calls.  It should be
  //        removed once it is no longer called.
  sampling_params_t(int sampling_type, double p = 1.0, double q = 1.0, bool use_alpha_cache = false)
    : sampling_type_(static_cast<sampling_strategy_t>(sampling_type)),
      p_(p),
      q_(q),
      use_alpha_cache_(use_alpha_cache)
  {
  }

  sampling_strategy_t sampling_type_{sampling_strategy_t::UNIFORM};

  // node2vec specific:
  //
  double p_;
  double q_;
  bool use_alpha_cache_{false};
};
}  // namespace cugraph
