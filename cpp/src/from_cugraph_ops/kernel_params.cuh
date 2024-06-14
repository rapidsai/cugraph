/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 */

#pragma once

#include "device_core.hpp"
#include "format.hpp"

#include <cstdint>
#include <type_traits>

namespace cugraph::ops::utils {

// We ignore clang-tidy warnings about [[nodiscard]] here, because this
// part of the code should stay compatible with C++ 11 if possible.
// Additionally, we ignore implicit constructors since the following structs
// are explicitly made to handle parameters more easily and should not add
// unnecessary overhead when creating higher-level structs using them.
// NOLINTBEGIN(modernize-use-nodiscard, google-explicit-constructor)

template <typename DataT, bool HAS_DATA>
class IntegralVariable;

template <typename DataT>
class IntegralVariable<DataT, true> {
 private:
  DataT data_{};

 public:
  IntegralVariable() = default;
  IntegralVariable(DataT data) : data_(data) {}
  /* [[nodiscard]] */ inline DataT __host__ __device__ get() const { return data_; }
  inline void __host__ __device__ set(DataT data) { data_ = data; }
};

template <typename DataT>
class IntegralVariable<DataT, false> {
 public:
  IntegralVariable() = default;
  IntegralVariable(DataT /* data */) {}
  /* [[nodiscard]] */ inline DataT __host__ __device__ get() const { return DataT{}; }
  inline void __host__ __device__ set(DataT data) {}  // do nothing
};

template <typename DataT, bool HAS_DATA>
class PointerVariable;

template <typename DataT>
class PointerVariable<DataT, true> {
 private:
  DataT* data_{nullptr};

 public:
  PointerVariable() = default;
  PointerVariable(DataT* data) : data_(data) {}
  PointerVariable(const DataT* data) : data_(const_cast<DataT*>(data)) {}
  /* [[nodiscard]] */ inline DataT* __host__ __device__ get() { return data_; }
  /* [[nodiscard]] */ inline DataT* __host__ __device__ get() const { return data_; }
  inline void __host__ __device__ set(DataT* data) { data_ = data; }
  inline void __host__ __device__ set(const DataT* data) { data_ = const_cast<DataT*>(data); }
};

template <typename DataT>
class PointerVariable<DataT, false> {
 public:
  PointerVariable() = default;
  PointerVariable(DataT* /* data */) {}
  PointerVariable(const DataT* /* data */) {}
  /* [[nodiscard]] */ inline DataT* __host__ __device__ get() { return nullptr; }
  /* [[nodiscard]] */ inline DataT* __host__ __device__ get() const { return nullptr; }
  inline void __host__ __device__ set(DataT* data) {}        // do nothing
  inline void __host__ __device__ set(const DataT* data) {}  // do nothing
};
// NOLINTEND(modernize-use-nodiscard, google-explicit-constructor)
template <typename EdgeIdxT, typename NodeIdxT, bool IS_HG>
struct __attribute__((aligned(16))) csc_params {
  using nidx_t             = NodeIdxT;
  using eidx_t             = EdgeIdxT;
  static constexpr bool HG = IS_HG;

  template <bool HG_ALIAS = IS_HG, typename std::enable_if<!HG_ALIAS, bool>::type = true>
  explicit csc_params(const graph::csc<EdgeIdxT, NodeIdxT>& graph)
    : offsets(graph.offsets),
      indices(graph.indices),
      rev_offsets(graph.rev_offsets),
      rev_indices(graph.rev_indices),
      map_csc_to_coo(graph.map_csc_to_coo),
      map_rev_to_coo(graph.map_rev_to_coo),
      map_dst_to_src(graph.map_dst_to_src),
      edge_types(),
      n_edge_types(),
      n_src_nodes(graph.n_src_nodes),
      n_dst_nodes(graph.n_dst_nodes),
      n_indices(graph.n_indices)
  {
  }

  template <bool HG_ALIAS = IS_HG, typename std::enable_if<!HG_ALIAS, bool>::type = true>
  explicit csc_params(const graph::bipartite_csc<EdgeIdxT, NodeIdxT>& graph)
    : offsets(graph.offsets),
      indices(graph.indices),
      rev_offsets(graph.rev_offsets),
      rev_indices(graph.rev_indices),
      map_csc_to_coo(graph.map_csc_to_coo),
      map_rev_to_coo(graph.map_rev_to_coo),
      map_dst_to_src(),
      edge_types(),
      n_edge_types(),
      n_src_nodes(graph.n_src_nodes),
      n_dst_nodes(graph.n_dst_nodes),
      n_indices(graph.n_indices)
  {
  }

  template <bool HG_ALIAS = IS_HG, typename std::enable_if<HG_ALIAS, bool>::type = true>
  explicit csc_params(const graph::csc_hg<EdgeIdxT, NodeIdxT>& graph)
    : offsets(graph.offsets),
      indices(graph.indices),
      rev_offsets(graph.rev_offsets),
      rev_indices(graph.rev_indices),
      map_csc_to_coo(graph.map_csc_to_coo),
      map_rev_to_coo(graph.map_rev_to_coo),
      map_dst_to_src(graph.map_dst_to_src),
      edge_types(graph.edge_types),
      n_edge_types(graph.n_edge_types),
      n_src_nodes(graph.n_src_nodes),
      n_dst_nodes(graph.n_dst_nodes),
      n_indices(graph.n_indices)
  {
  }

  template <bool HG_ALIAS = IS_HG, typename std::enable_if<HG_ALIAS, bool>::type = true>
  explicit csc_params(const graph::bipartite_csc_hg<EdgeIdxT, NodeIdxT>& graph)
    : offsets(graph.offsets),
      indices(graph.indices),
      rev_offsets(graph.rev_offsets),
      rev_indices(graph.rev_indices),
      map_csc_to_coo(graph.map_csc_to_coo),
      map_rev_to_coo(graph.map_rev_to_coo),
      map_dst_to_src(),
      edge_types(graph.edge_types),
      n_edge_types(graph.n_edge_types),
      n_src_nodes(graph.n_src_nodes),
      n_dst_nodes(graph.n_dst_nodes),
      n_indices(graph.n_indices)
  {
  }

  const EdgeIdxT* offsets{nullptr};
  const NodeIdxT* indices{nullptr};
  const NodeIdxT* map_dst_to_src{nullptr};

  const EdgeIdxT* rev_offsets{nullptr};
  const NodeIdxT* rev_indices{nullptr};

  const EdgeIdxT* map_csc_to_coo{nullptr};
  const EdgeIdxT* map_rev_to_coo{nullptr};

  const PointerVariable<int32_t, IS_HG> edge_types;

  NodeIdxT n_src_nodes;
  NodeIdxT n_dst_nodes;
  EdgeIdxT n_indices;

  const IntegralVariable<int32_t, IS_HG> n_edge_types;
};

}  // namespace cugraph::ops::utils
