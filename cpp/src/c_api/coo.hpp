/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "c_api/array.hpp"

#include <vector>

namespace cugraph {
namespace c_api {

struct cugraph_coo_t {
  std::unique_ptr<cugraph_type_erased_device_array_t> src_{};
  std::unique_ptr<cugraph_type_erased_device_array_t> dst_{};
  std::unique_ptr<cugraph_type_erased_device_array_t> wgt_{};
  std::unique_ptr<cugraph_type_erased_device_array_t> id_{};
  std::unique_ptr<cugraph_type_erased_device_array_t> type_{};
};

struct cugraph_coo_list_t {
  std::vector<std::unique_ptr<cugraph_coo_t>> list_;
};

}  // namespace c_api
}  // namespace cugraph
