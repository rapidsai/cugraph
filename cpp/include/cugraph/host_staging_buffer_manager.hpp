/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

#include <cstddef>
#include <memory>
#include <optional>

namespace cugraph {

class host_staging_buffer_manager {
 public:
  using buffer_element_t =
    uint64_t;  // use uint64_t to ensure 8B alignment (this might be redundant as the underlying rmm
               // pinned_memory_resource already guarantees this but to be more future proof)
  static constexpr size_t minimum_alignment =
    sizeof(uint64_t);  // guarantees alignment by at least minimum_alginment (more likely to be at
                       // least 256B alignmened, but this is not guaranteed by the API)
  static constexpr size_t staging_buffer_size =
    size_t{16384} /* # GPUs */ * 10 /* tuple size */ * sizeof(int64_t);
  static_assert((staging_buffer_size % sizeof(buffer_element_t)) == 0);

  static void init(raft::handle_t const& handle,
                   std::vector<rmm::cuda_stream_view> const& streams,
                   std::shared_ptr<rmm::mr::pinned_memory_resource> pinned_mr)
  {
    auto& s = state();
    CUGRAPH_EXPECTS(s.initialized == false, "staging buffer is already initialized.");
    s.initialized = true;
    for (size_t i = 0; i < streams.size(); ++i) {
      s.stream_to_idx_map.insert({streams[i], i});
    }
    s.pinned_mr = std::move(pinned_mr);
    s.bufs.reserve(streams.size());
    for (size_t i = 0; i < streams.size(); ++i) {
      s.bufs.emplace_back(
        staging_buffer_size / sizeof(buffer_element_t), handle.get_stream(), s.pinned_mr.get());
    }
    RAFT_CUDA_TRY(cudaDeviceSynchronize());  // now ready for use by every registered stream
  }

  static std::optional<void*> buffer_ptr(size_t req_size, rmm::cuda_stream_view stream)
  {
    if (req_size >= staging_buffer_size) { return std::nullopt; }
    auto& s = state();
    if (s.initialized == false) { return std::nullopt; }
    auto it = s.stream_to_idx_map.find(stream);
    if (it == s.stream_to_idx_map.end()) { return std::nullopt; }
    return std::make_optional(static_cast<void*>(s.bufs[it->second].data()));
  }

 private:
  struct stream_hash_t {
    std::size_t operator()(rmm::cuda_stream_view stream) const noexcept
    {
      return std::hash<cudaStream_t>{}(stream.value());
    }
  };

  struct stream_eqaul_t {
    bool operator()(rmm::cuda_stream_view lhs, rmm::cuda_stream_view rhs) const noexcept
    {
      return lhs.value() == rhs.value();
    }
  };

  struct state_t {
    bool initialized = false;
    std::unordered_map<rmm::cuda_stream_view, size_t, stream_hash_t, stream_eqaul_t>
      stream_to_idx_map{};
    std::shared_ptr<rmm::mr::pinned_memory_resource> pinned_mr{};
    std::vector<rmm::device_uvector<buffer_element_t>> bufs{};
  };

  static state_t& state()
  {
    static state_t s{};
    return s;
  };
};

}  // namespace cugraph
