/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>

#include <community/egonet_validate.hpp>
#include <utilities/test_utilities.hpp>

#include <gtest/gtest.h>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace cugraph {
namespace test {

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<size_t>>
egonet_reference(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<vertex_t const> ego_sources,
  int radius)
{
#if 1
  auto [d_coo_src, d_coo_dst, d_coo_wgt] =
    cugraph::decompress_to_edgelist(handle,
                                    graph_view,
                                    edge_weight_view,
                                    std::optional<raft::device_span<vertex_t const>>{std::nullopt});
#else
  // FIXME: This should be faster (smaller list of edges to operate on), but uniform_nbr_sample
  // doesn't preserve multi-edges (which is probably a bug)
  std::vector<int> fan_out(radius + 1, -1);

  rmm::device_uvector<vertex_t> local_ego_sources(ego_sources.size(), handle.get_stream());
  raft::copy(local_ego_sources.data(), ego_sources.data(), ego_sources.size(), handle.get_stream());

  auto [d_coo_src, d_coo_dst, d_coo_wgt, d_coo_counts] = cugraph::uniform_nbr_sample(
    handle,
    graph_view,
    edge_weight_view,
    raft::device_span<vertex_t>{local_ego_sources.data(), local_ego_sources.size()},
    raft::host_span<const int>{fan_out.data(), fan_out.size()},
    false);

  d_coo_counts.resize(0, handle.get_stream());
  d_coo_counts.shrink_to_fit(handle.get_stream());
#endif

  rmm::device_uvector<vertex_t> d_reference_src(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_reference_dst(0, handle.get_stream());

  auto d_reference_wgt =
    edge_weight_view ? std::make_optional(rmm::device_uvector<weight_t>(0, handle.get_stream()))
                     : std::nullopt;

  std::vector<size_t> h_reference_offsets;

  size_t offset{0};

  for (size_t idx = 0; idx < ego_sources.size(); ++idx) {
    h_reference_offsets.push_back(offset);

    rmm::device_uvector<vertex_t> frontier(1, handle.get_stream());
    rmm::device_uvector<vertex_t> visited(1, handle.get_stream());
    raft::copy(frontier.data(), ego_sources.data() + idx, 1, handle.get_stream());
    raft::copy(visited.data(), ego_sources.data() + idx, 1, handle.get_stream());

    for (int hop = 0; hop < radius; ++hop) {
      size_t new_entries = thrust::count_if(
        handle.get_thrust_policy(),
        d_coo_src.begin(),
        d_coo_src.end(),
        [frontier_begin = frontier.begin(),
         frontier_end   = frontier.end()] __device__(vertex_t src) {
          return thrust::binary_search(thrust::seq, frontier_begin, frontier_end, src);
        });

      size_t old_size = d_reference_src.size();

      d_reference_src.resize(old_size + new_entries, handle.get_stream());
      d_reference_dst.resize(old_size + new_entries, handle.get_stream());

      if (edge_weight_view) {
        d_reference_wgt->resize(old_size + new_entries, handle.get_stream());

        thrust::copy_if(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(d_coo_src.begin(), d_coo_dst.begin(), d_coo_wgt->begin()),
          thrust::make_zip_iterator(d_coo_src.end(), d_coo_dst.end(), d_coo_wgt->end()),
          thrust::make_zip_iterator(d_reference_src.begin() + old_size,
                                    d_reference_dst.begin() + old_size,
                                    d_reference_wgt->begin() + old_size),
          [frontier_begin = frontier.begin(),
           frontier_end   = frontier.end()] __device__(auto tuple) {
            vertex_t src = thrust::get<0>(tuple);
            return thrust::binary_search(thrust::seq, frontier_begin, frontier_end, src);
          });
      } else {
        thrust::copy_if(handle.get_thrust_policy(),
                        thrust::make_zip_iterator(d_coo_src.begin(), d_coo_dst.begin()),
                        thrust::make_zip_iterator(d_coo_src.end(), d_coo_dst.end()),
                        thrust::make_zip_iterator(d_reference_src.begin() + old_size,
                                                  d_reference_dst.begin() + old_size),
                        [frontier_begin = frontier.begin(),
                         frontier_end   = frontier.end()] __device__(auto tuple) {
                          vertex_t src = thrust::get<0>(tuple);
                          return thrust::binary_search(
                            thrust::seq, frontier_begin, frontier_end, src);
                        });
      }

      frontier.resize(new_entries, handle.get_stream());
      auto new_end = thrust::copy_if(
        handle.get_thrust_policy(),
        d_reference_dst.begin() + old_size,
        d_reference_dst.end(),
        frontier.begin(),
        [visited_begin = visited.begin(), visited_end = visited.end()] __device__(auto v) {
          return !thrust::binary_search(thrust::seq, visited_begin, visited_end, v);
        });
      frontier.resize(thrust::distance(frontier.begin(), new_end), handle.get_stream());
      thrust::sort(handle.get_thrust_policy(), frontier.begin(), frontier.end());
      new_end = thrust::unique(handle.get_thrust_policy(), frontier.begin(), frontier.end());
      frontier.resize(thrust::distance(frontier.begin(), new_end), handle.get_stream());

      size_t old_visited_size = visited.size();
      visited.resize(old_visited_size + frontier.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   frontier.begin(),
                   frontier.end(),
                   visited.begin() + old_visited_size);
      thrust::sort(handle.get_thrust_policy(), visited.begin(), visited.end());

      offset += new_entries;
    }

    if (frontier.size() > 0) {
      size_t new_entries = thrust::count_if(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(d_coo_src.begin(), d_coo_dst.begin()),
        thrust::make_zip_iterator(d_coo_src.end(), d_coo_dst.end()),
        [frontier_begin = frontier.begin(),
         frontier_end   = frontier.end(),
         visited_begin  = visited.begin(),
         visited_end    = visited.end()] __device__(auto tuple) {
          vertex_t src = thrust::get<0>(tuple);
          vertex_t dst = thrust::get<1>(tuple);
          return thrust::binary_search(thrust::seq, frontier_begin, frontier_end, src) &&
                 thrust::binary_search(thrust::seq, visited_begin, visited_end, dst);
        });

      size_t old_size = d_reference_src.size();

      d_reference_src.resize(old_size + new_entries, handle.get_stream());
      d_reference_dst.resize(old_size + new_entries, handle.get_stream());

      if (edge_weight_view) {
        d_reference_wgt->resize(old_size + new_entries, handle.get_stream());

        thrust::copy_if(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(d_coo_src.begin(), d_coo_dst.begin(), d_coo_wgt->begin()),
          thrust::make_zip_iterator(d_coo_src.end(), d_coo_dst.end(), d_coo_wgt->end()),
          thrust::make_zip_iterator(d_reference_src.begin() + old_size,
                                    d_reference_dst.begin() + old_size,
                                    d_reference_wgt->begin() + old_size),
          [frontier_begin = frontier.begin(),
           frontier_end   = frontier.end(),
           visited_begin  = visited.begin(),
           visited_end    = visited.end()] __device__(auto tuple) {
            vertex_t src = thrust::get<0>(tuple);
            vertex_t dst = thrust::get<1>(tuple);
            return thrust::binary_search(thrust::seq, frontier_begin, frontier_end, src) &&
                   thrust::binary_search(thrust::seq, visited_begin, visited_end, dst);
          });
      } else {
        thrust::copy_if(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(d_coo_src.begin(), d_coo_dst.begin()),
          thrust::make_zip_iterator(d_coo_src.end(), d_coo_dst.end()),
          thrust::make_zip_iterator(d_reference_src.begin() + old_size,
                                    d_reference_dst.begin() + old_size),
          [frontier_begin = frontier.begin(),
           frontier_end   = frontier.end(),
           visited_begin  = visited.begin(),
           visited_end    = visited.end()] __device__(auto tuple) {
            vertex_t src = thrust::get<0>(tuple);
            vertex_t dst = thrust::get<1>(tuple);
            return thrust::binary_search(thrust::seq, frontier_begin, frontier_end, src) &&
                   thrust::binary_search(thrust::seq, visited_begin, visited_end, dst);
          });
      }

      offset += new_entries;
    }
  }

  h_reference_offsets.push_back(offset);

  auto d_reference_offsets = cugraph::test::to_device(handle, h_reference_offsets);

  return std::make_tuple(std::move(d_reference_src),
                         std::move(d_reference_dst),
                         std::move(d_reference_wgt),
                         std::move(d_reference_offsets));
}

template <typename vertex_t, typename weight_t>
void egonet_validate(raft::handle_t const& handle,
                     rmm::device_uvector<vertex_t>& d_cugraph_egonet_src,
                     rmm::device_uvector<vertex_t>& d_cugraph_egonet_dst,
                     std::optional<rmm::device_uvector<weight_t>>& d_cugraph_egonet_wgt,
                     rmm::device_uvector<size_t>& d_cugraph_egonet_offsets,
                     rmm::device_uvector<vertex_t>& d_reference_egonet_src,
                     rmm::device_uvector<vertex_t>& d_reference_egonet_dst,
                     std::optional<rmm::device_uvector<weight_t>>& d_reference_egonet_wgt,
                     rmm::device_uvector<size_t>& d_reference_egonet_offsets)
{
  ASSERT_EQ(d_reference_egonet_offsets.size(), d_cugraph_egonet_offsets.size())
    << "Returned edge offset vector has an invalid size.";

  ASSERT_TRUE(thrust::equal(handle.get_thrust_policy(),
                            d_reference_egonet_offsets.begin(),
                            d_reference_egonet_offsets.end(),
                            d_cugraph_egonet_offsets.begin()))
    << "Returned egonet edge offset values do not match with the reference values.";

  ASSERT_EQ(d_cugraph_egonet_wgt.has_value(), d_reference_egonet_wgt.has_value());

  auto h_offsets =
    cugraph::test::to_host(handle,
                           raft::device_span<size_t const>{d_cugraph_egonet_offsets.data(),
                                                           d_cugraph_egonet_offsets.size()});

  for (size_t i = 0; i < (h_offsets.size() - 1); ++i) {
    if (d_reference_egonet_wgt) {
      thrust::sort(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(d_reference_egonet_src.begin(),
                                             d_reference_egonet_dst.begin(),
                                             d_reference_egonet_wgt->begin()) +
                     h_offsets[i],
                   thrust::make_zip_iterator(d_reference_egonet_src.begin(),
                                             d_reference_egonet_dst.begin(),
                                             d_reference_egonet_wgt->begin()) +
                     h_offsets[i + 1]);
      thrust::sort(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(d_cugraph_egonet_src.begin(),
                                             d_cugraph_egonet_dst.begin(),
                                             d_cugraph_egonet_wgt->begin()) +
                     h_offsets[i],
                   thrust::make_zip_iterator(d_cugraph_egonet_src.begin(),
                                             d_cugraph_egonet_dst.begin(),
                                             d_cugraph_egonet_wgt->begin()) +
                     h_offsets[i + 1]);

      ASSERT_TRUE(thrust::equal(handle.get_thrust_policy(),
                                thrust::make_zip_iterator(d_reference_egonet_src.begin(),
                                                          d_reference_egonet_dst.begin(),
                                                          d_reference_egonet_wgt->begin()) +
                                  h_offsets[i],
                                thrust::make_zip_iterator(d_reference_egonet_src.begin(),
                                                          d_reference_egonet_dst.begin(),
                                                          d_reference_egonet_wgt->begin()) +
                                  h_offsets[i + 1],
                                thrust::make_zip_iterator(d_cugraph_egonet_src.begin(),
                                                          d_cugraph_egonet_dst.begin(),
                                                          d_cugraph_egonet_wgt->begin()) +
                                  h_offsets[i],
                                [] __device__(auto left, auto right) {
                                  auto l0 = thrust::get<0>(left);
                                  auto l1 = thrust::get<1>(left);
                                  auto l2 = thrust::get<2>(left);
                                  auto r0 = thrust::get<0>(right);
                                  auto r1 = thrust::get<1>(right);
                                  auto r2 = thrust::get<2>(right);
                                  if (!((l0 == r0) && (l1 == r1) && (l2 == r2)))
                                    printf("edge mismatch: (%d,%d,%g) != (%d,%d,%g)\n",
                                           (int)l0,
                                           (int)l1,
                                           (float)l2,
                                           (int)r0,
                                           (int)r1,
                                           (float)r2);
                                  return (l0 == r0) && (l1 == r1) && (l2 == r2);
                                }))
        << "Extracted egonet edges do not match with the edges extracted by the reference "
           "implementation.";
    } else {
      thrust::sort(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(d_reference_egonet_src.begin(), d_reference_egonet_dst.begin()) +
          h_offsets[i],
        thrust::make_zip_iterator(d_reference_egonet_src.begin(), d_reference_egonet_dst.begin()) +
          h_offsets[i + 1]);
      thrust::sort(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(d_cugraph_egonet_src.begin(), d_cugraph_egonet_dst.begin()) +
          h_offsets[i],
        thrust::make_zip_iterator(d_cugraph_egonet_src.begin(), d_cugraph_egonet_dst.begin()) +
          h_offsets[i + 1]);

      ASSERT_TRUE(thrust::equal(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(d_reference_egonet_src.begin(), d_reference_egonet_dst.begin()) +
          h_offsets[i],
        thrust::make_zip_iterator(d_reference_egonet_src.begin(), d_reference_egonet_dst.begin()) +
          h_offsets[i + 1],
        thrust::make_zip_iterator(d_cugraph_egonet_src.begin(), d_cugraph_egonet_dst.begin()) +
          h_offsets[i],
        [] __device__(auto left, auto right) {
          auto l0 = thrust::get<0>(left);
          auto l1 = thrust::get<1>(left);
          auto r0 = thrust::get<0>(right);
          auto r1 = thrust::get<1>(right);
          if (!((l0 == r0) && (l1 == r1)))
            printf("edge mismatch: (%d,%d) != (%d,%d)\n", (int)l0, (int)l1, (int)r0, (int)r1);
          return (l0 == r0) && (l1 == r1);
        }))
        << "Extracted egonet edges do not match with the edges extracted by the reference "
           "implementation.";
    }
  }
}

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    rmm::device_uvector<size_t>>
egonet_reference(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  raft::device_span<int32_t const> ego_sources,
  int radius);

template void egonet_validate(raft::handle_t const& handle,
                              rmm::device_uvector<int32_t>& d_cugraph_egonet_src,
                              rmm::device_uvector<int32_t>& d_cugraph_egonet_dst,
                              std::optional<rmm::device_uvector<float>>& d_cugraph_egonet_wgt,
                              rmm::device_uvector<size_t>& d_cugraph_egonet_offsets,
                              rmm::device_uvector<int32_t>& d_reference_egonet_src,
                              rmm::device_uvector<int32_t>& d_reference_egonet_dst,
                              std::optional<rmm::device_uvector<float>>& d_reference_egonet_wgt,
                              rmm::device_uvector<size_t>& d_reference_egonet_offsets);

template void egonet_validate(raft::handle_t const& handle,
                              rmm::device_uvector<int64_t>& d_cugraph_egonet_src,
                              rmm::device_uvector<int64_t>& d_cugraph_egonet_dst,
                              std::optional<rmm::device_uvector<float>>& d_cugraph_egonet_wgt,
                              rmm::device_uvector<size_t>& d_cugraph_egonet_offsets,
                              rmm::device_uvector<int64_t>& d_reference_egonet_src,
                              rmm::device_uvector<int64_t>& d_reference_egonet_dst,
                              std::optional<rmm::device_uvector<float>>& d_reference_egonet_wgt,
                              rmm::device_uvector<size_t>& d_reference_egonet_offsets);

#if 0
// NOT CURRENTLY USED
template
std::tuple<rmm::device_uvector<int32_t>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<float>>,
           rmm::device_uvector<size_t>>
egonet_reference(raft::handle_t const& handle,
                 cugraph::graph_view_t<int32_t, int64_t, float, false, false> const& graph_view,
                 std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
                 raft::device_span<int32_t const> ego_sources,
                 int radius);

template
std::tuple<rmm::device_uvector<int64_t>,
           rmm::device_uvector<int64_t>,
           std::optional<rmm::device_uvector<float>>,
           rmm::device_uvector<size_t>>
egonet_reference(raft::handle_t const& handle,
                 cugraph::graph_view_t<int64_t, int64_t, float, false, false> const& graph_view,
                 std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
                 raft::device_span<int64_t const> ego_sources,
                 int radius);

template
std::tuple<rmm::device_uvector<int32_t>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<double>>,
           rmm::device_uvector<size_t>>
egonet_reference(raft::handle_t const& handle,
                 cugraph::graph_view_t<int32_t, int32_t, double, false, false> const& graph_view,
                 std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
                 raft::device_span<int32_t const> ego_sources,
                 int radius);

template
std::tuple<rmm::device_uvector<int32_t>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<double>>,
           rmm::device_uvector<size_t>>
egonet_reference(raft::handle_t const& handle,
                 cugraph::graph_view_t<int32_t, int64_t, double, false, false> const& graph_view,
                 std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
                 raft::device_span<int32_t const> ego_sources,
                 int radius);

template
std::tuple<rmm::device_uvector<int64_t>,
           rmm::device_uvector<int64_t>,
           std::optional<rmm::device_uvector<double>>,
           rmm::device_uvector<size_t>>
egonet_reference(raft::handle_t const& handle,
                 cugraph::graph_view_t<int64_t, int64_t, double, false, false> const& graph_view,
                 std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
                 raft::device_span<int64_t const> ego_sources,
                 int radius);

template
void egonet_validate(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>& d_cugraph_egonet_src,
  rmm::device_uvector<int32_t>& d_cugraph_egonet_dst,
  std::optional<rmm::device_uvector<double>>& d_cugraph_egonet_wgt,
  rmm::device_uvector<size_t>& d_cugraph_egonet_offsets,
  rmm::device_uvector<int32_t>& d_reference_egonet_src,
  rmm::device_uvector<int32_t>& d_reference_egonet_dst,
  std::optional<rmm::device_uvector<double>>& d_reference_egonet_wgt,
  rmm::device_uvector<size_t>& d_reference_egonet_offsets);

template
void egonet_validate(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>& d_cugraph_egonet_src,
  rmm::device_uvector<int64_t>& d_cugraph_egonet_dst,
  std::optional<rmm::device_uvector<double>>& d_cugraph_egonet_wgt,
  rmm::device_uvector<size_t>& d_cugraph_egonet_offsets,
  rmm::device_uvector<int64_t>& d_reference_egonet_src,
  rmm::device_uvector<int64_t>& d_reference_egonet_dst,
  std::optional<rmm::device_uvector<double>>& d_reference_egonet_wgt,
  rmm::device_uvector<size_t>& d_reference_egonet_offsets);

#endif

}  // namespace test
}  // namespace cugraph
