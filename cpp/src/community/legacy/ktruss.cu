/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

/**
 * ---------------------------------------------------------------------------*
 * @brief KTruss implementation
 *
 * @file ktruss.cu
 * --------------------------------------------------------------------------*/

#include <cugraph/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <Hornet.hpp>
#include <StandardAPI.hpp>
#include <Static/KTruss/KTruss.cuh>

using namespace hornets_nest;

namespace cugraph {

namespace detail {

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> ktruss_subgraph_impl(
  raft::handle_t const& handle,
  raft::device_span<vertex_t> src,
  raft::device_span<vertex_t> dst,
  size_t number_of_vertices,
  int k)
{
  using HornetGraph = hornet::gpu::Hornet<vertex_t>;
  using UpdatePtr   = hornet::BatchUpdatePtr<vertex_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
  using Update      = hornet::gpu::BatchUpdate<vertex_t>;

  HornetGraph hnt(number_of_vertices + 1);

  // NOTE: Should a constant pointer be passed for @src and @dst
  UpdatePtr ptr(static_cast<int>(src.size()), src.data(), dst.data());
  Update batch(ptr);

  hnt.insert(batch);
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to initialize graph");

  KTruss kt(hnt);

  kt.init();
  kt.reset();
  kt.createOffSetArray();
  // NOTE : These parameters will become obsolete once we move to the updated
  // algorithm (https://ieeexplore.ieee.org/document/8547581)
  kt.setInitParameters(4,      // Number of threads per block per list intersection
                       8,      // Number of intersections per block
                       2,      // log2(Number of threads)
                       64000,  // Total number of blocks launched
                       32);    // Thread block dimension
  kt.reset();
  kt.sortHornet();

  kt.runForK(k);
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to run");

  rmm::device_uvector<vertex_t> result_src(kt.getGraphEdgeCount(), handle.get_stream());
  rmm::device_uvector<vertex_t> result_dst(kt.getGraphEdgeCount(), handle.get_stream());

  kt.copyGraph(result_src.data(), result_dst.data());

  kt.release();
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to release");

  return std::make_tuple(std::move(result_src), std::move(result_dst));
}

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
weighted_ktruss_subgraph_impl(raft::handle_t const& handle,
                              raft::device_span<vertex_t> src,
                              raft::device_span<vertex_t> dst,
                              std::optional<raft::device_span<weight_t>> wgt,
                              size_t number_of_vertices,
                              int k)
{
  using HornetGraph = hornet::gpu::Hornet<vertex_t, hornet::EMPTY, hornet::TypeList<weight_t>>;
  using UpdatePtr =
    hornet::BatchUpdatePtr<vertex_t, hornet::TypeList<weight_t>, hornet::DeviceType::DEVICE>;
  using Update = hornet::gpu::BatchUpdate<vertex_t, hornet::TypeList<weight_t>>;

  HornetGraph hnt(number_of_vertices + 1);

  UpdatePtr ptr(static_cast<int>(src.size()), src.data(), dst.data(), wgt->data());
  Update batch(ptr);

  hnt.insert(batch);
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to initialize graph");

  KTrussWeighted<weight_t> kt(hnt);

  kt.init();
  kt.reset();
  kt.createOffSetArray();
  // NOTE : These parameters will become obsolete once we move to the updated
  // algorithm (https://ieeexplore.ieee.org/document/8547581)
  kt.setInitParameters(4,      // Number of threads per block per list intersection
                       8,      // Number of intersections per block
                       2,      // log2(Number of threads)
                       64000,  // Total number of blocks launched
                       32);    // Thread block dimension
  kt.reset();
  kt.sortHornet();

  kt.runForK(k);
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to run");

  rmm::device_uvector<vertex_t> result_src(kt.getGraphEdgeCount(), handle.get_stream());
  rmm::device_uvector<vertex_t> result_dst(kt.getGraphEdgeCount(), handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> result_wgt{std::nullopt};

  result_wgt = rmm::device_uvector<weight_t>(kt.getGraphEdgeCount(), handle.get_stream());
  kt.copyGraph(result_src.data(), result_dst.data(), result_wgt->data());

  kt.release();
  CUGRAPH_EXPECTS(cudaPeekAtLastError() == cudaSuccess, "KTruss : Failed to release");

  return std::make_tuple(std::move(result_src), std::move(result_dst), std::move(result_wgt));
}

}  // namespace detail

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
k_truss_subgraph(raft::handle_t const& handle,
                 raft::device_span<vertex_t> src,
                 raft::device_span<vertex_t> dst,
                 std::optional<raft::device_span<weight_t>> wgt,
                 size_t number_of_vertices,
                 int k)
{
  if (wgt.has_value()) {
    return detail::weighted_ktruss_subgraph_impl(handle, src, dst, wgt, number_of_vertices, k);
  } else {
    auto [result_src, result_dst] =
      detail::ktruss_subgraph_impl(handle, src, dst, number_of_vertices, k);
    std::optional<rmm::device_uvector<weight_t>> result_wgt{std::nullopt};
    return std::make_tuple(std::move(result_src), std::move(result_dst), std::move(result_wgt));
  }
}

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
k_truss_subgraph(raft::handle_t const& handle,
                 raft::device_span<int32_t> src,
                 raft::device_span<int32_t> dst,
                 std::optional<raft::device_span<float>> wgt,
                 size_t number_of_vertices,
                 int k);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
k_truss_subgraph(raft::handle_t const& handle,
                 raft::device_span<int32_t> src,
                 raft::device_span<int32_t> dst,
                 std::optional<raft::device_span<double>> wgt,
                 size_t number_of_vertices,
                 int k);

}  // namespace cugraph
