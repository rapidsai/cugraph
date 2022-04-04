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

#pragma once

#ifdef _DEBUG_
#include <iostream>
#endif

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace cugraph {
namespace visitors {

struct erased_pack_t {
  erased_pack_t(void** p_args, size_t n)
    : args_{[](void** p, size_t n) {
        std::vector<void*> v;
        v.insert(v.begin(), p, p + n);
        return v;
      }(p_args, n)}
  {
    // args_.insert(args_.begin(), p_args, p_args + n);
  }

  erased_pack_t(std::initializer_list<void*> args) : args_{args} {}

  std::vector<void*> const& get_args(void) const { return args_; }

  erased_pack_t(erased_pack_t const&) = delete;
  erased_pack_t& operator=(erased_pack_t const&) = delete;

  erased_pack_t(erased_pack_t&& other) : args_(std::move(other.args_)) {}

  erased_pack_t& operator=(erased_pack_t&& other)
  {
    args_ = std::move(other.args_);
    return *this;
  }

#ifdef _DEBUG_
  void print(void) const
  {
    std::cout << "list args addresses:\n";
    for (auto&& elem : args_)
      std::cout << elem << ", ";
    std::cout << '\n';
  }
#endif

 private:
  std::vector<void*> args_;
};

}  // namespace visitors
}  // namespace cugraph
