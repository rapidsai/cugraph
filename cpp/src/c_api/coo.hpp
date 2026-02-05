/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
