/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strongly_connected_components_reference.hpp"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

// Canonicalize a cycle by rotating it to the vertex with the smallest index.
template <typename vertex_t>
std::vector<vertex_t> canonicalize_cycle(std::vector<vertex_t>&& cycle_vertices)
{
  if (cycle_vertices.size() <= 1) { return cycle_vertices; }
  auto min_it    = std::min_element(cycle_vertices.begin(), cycle_vertices.end());
  auto min_index = static_cast<size_t>(std::distance(cycle_vertices.begin(), min_it));
  std::rotate(cycle_vertices.begin(), cycle_vertices.begin() + min_index, cycle_vertices.end());
  return cycle_vertices;
}

template <typename vertex_t>
std::vector<std::vector<vertex_t>> canonicalize_cycles(std::vector<std::vector<vertex_t>>&& cycles)
{
  for (auto& cycle : cycles) {
    cycle = canonicalize_cycle(std::move(cycle));
  }
  // lexicographic sort of the rotated vertex sequences (e.g. [0, 2] < [0, 2, 1] < [1, 2])
  std::sort(cycles.begin(), cycles.end());
  return cycles;
}

// SCCs of the subgraph induced by @p vertices.
template <typename vertex_t, typename edge_t>
std::vector<std::vector<vertex_t>> strongly_connected_components_from_induced_subgraph(
  std::span<edge_t const> offsets,
  std::span<vertex_t const> indices,
  std::span<vertex_t const> vertices)
{
  auto const n = static_cast<vertex_t>(offsets.size() - 1);
  std::vector<bool> include(n, false);
  for (auto v : vertices) {
    include[v] = true;
  }

  std::vector<edge_t> induced_offsets(n + 1);
  std::vector<vertex_t> induced_indices{};
  for (vertex_t v = 0; v < n; ++v) {
    induced_offsets[v] = static_cast<edge_t>(induced_indices.size());
    if (!include[v]) { continue; }
    for (edge_t e = offsets[v]; e < offsets[v + 1]; ++e) {
      auto w = indices[e];
      if (include[w]) { induced_indices.push_back(w); }
    }
  }
  induced_offsets[n] = static_cast<edge_t>(induced_indices.size());

  auto labels = strongly_connected_components_reference<vertex_t, edge_t>(
    std::span<edge_t const>{induced_offsets.data(), induced_offsets.size()},
    std::span<vertex_t const>{induced_indices.data(), induced_indices.size()});

  vertex_t max_label{0};
  for (auto v : vertices) {
    max_label = std::max(max_label, labels[v]);
  }
  std::vector<std::vector<vertex_t>> components(
    vertices.empty() ? size_t{0} : static_cast<size_t>(max_label) + size_t{1});
  for (auto v : vertices) {
    components[labels[v]].push_back(v);
  }
  components.erase(std::remove_if(components.begin(),
                                  components.end(),
                                  [](auto const& component) { return component.empty(); }),
                   components.end());
  return components;
}

// Non-recursive Johnson search for cycles through @p start in the subgraph given by CSR
// @p offsets / @p indices (NetworkX _johnson_cycle_search).
template <typename vertex_t, typename edge_t>
std::vector<std::vector<vertex_t>> johnson_cycle_search(std::span<edge_t const> offsets,
                                                        std::span<vertex_t const> indices,
                                                        vertex_t start)
{
  auto const num_vertices = static_cast<vertex_t>(offsets.size() - 1);
  std::vector<bool> blocked(num_vertices, false);
  std::vector<std::unordered_set<vertex_t>> B(num_vertices);
  blocked[start] = true;

  std::vector<vertex_t> path  = {start};
  std::vector<size_t> nbr_pos = {0};
  std::vector<bool> closed    = {false};
  std::vector<std::vector<vertex_t>> cycles{};

  while (!path.empty()) {
    auto v                = path.back();
    auto nbr_start_offset = offsets[v];
    auto nbr_end_offset   = offsets[v + 1];
    bool pushed           = false;
    while (nbr_pos.back() < static_cast<size_t>(nbr_end_offset - nbr_start_offset)) {
      auto w = indices[nbr_start_offset + nbr_pos.back()];
      ++nbr_pos.back();
      if (w == start) {
        cycles.push_back(path);
        closed.back() = true;
      } else if (!blocked[w]) {
        path.push_back(w);
        nbr_pos.push_back(0);
        closed.push_back(false);
        blocked[w] = true;
        pushed     = true;
        break;
      }
    }
    if (pushed) { continue; }

    nbr_pos.pop_back();
    v = path.back();
    path.pop_back();
    bool v_closed = closed.back();
    closed.pop_back();
    if (v_closed) {
      if (!closed.empty()) { closed.back() = true; }
      std::vector<vertex_t> unblock_stack{v};
      while (!unblock_stack.empty()) {
        auto u = unblock_stack.back();
        unblock_stack.pop_back();
        if (blocked[u]) {
          blocked[u] = false;
          unblock_stack.insert(unblock_stack.end(), B[u].begin(), B[u].end());
          B[u].clear();
        }
      }
    } else {
      for (edge_t e = offsets[v]; e < offsets[v + 1]; ++e) {
        B[indices[e]].insert(v);
      }
    }
  }

  return cycles;
}

// Host reference for directed simple_cycles (NetworkX simple_cycles / Johnson, unbounded).
template <typename vertex_t, typename edge_t>
std::vector<std::vector<vertex_t>> simple_cycles_reference(std::span<edge_t const> offsets,
                                                           std::span<vertex_t const> indices)
{
  auto num_vertices = static_cast<vertex_t>(offsets.size() - 1);

  std::vector<std::vector<vertex_t>> cycles{};

  std::vector<edge_t> simple_offsets(
    num_vertices + 1);  // simple_offsets & simple_indices: exclude self-loops and multi-edges
  std::vector<vertex_t> simple_indices{};
  simple_indices.reserve(indices.size());
  for (vertex_t v = 0; v < num_vertices; ++v) {
    simple_offsets[v] = static_cast<edge_t>(simple_indices.size());
    bool self_loop    = false;
    for (edge_t e = offsets[v]; e < offsets[v + 1]; ++e) {
      auto w = indices[e];
      if (w == v) {
        self_loop = true;
      } else {
        simple_indices.push_back(w);
      }
    }
    if (self_loop) { cycles.push_back(std::vector<vertex_t>{v}); }
    std::sort(simple_indices.begin() + simple_offsets[v], simple_indices.end());
    simple_indices.erase(
      std::unique(simple_indices.begin() + simple_offsets[v], simple_indices.end()),
      simple_indices.end());
  }
  simple_offsets[num_vertices] = static_cast<edge_t>(simple_indices.size());

  std::vector<vertex_t> all_vertices(num_vertices);
  std::iota(all_vertices.begin(), all_vertices.end(), vertex_t{0});

  auto components = strongly_connected_components_from_induced_subgraph<vertex_t, edge_t>(
    simple_offsets, simple_indices, all_vertices);
  std::vector<std::vector<vertex_t>> work{};
  work.reserve(components.size());
  std::copy_if(
    components.begin(), components.end(), std::back_inserter(work), [](auto const& component) {
      return component.size() >= 2;
    });

  while (!work.empty()) {
    auto c = std::move(work.back());
    work.pop_back();
    std::sort(c.begin(), c.end());
    if (c.size() < 2) {
      continue;
    } else if (c.size() == 2) {  // the only simple cycle is c[0] -> c[1] -> c[0]
      cycles.push_back(std::move(c));
      continue;
    }

    auto v = c.front();

    std::vector<edge_t> component_offsets(num_vertices + 1);
    std::vector<vertex_t> component_indices{};
    vertex_t last_v{0};
    for (auto u : c) {
      while (last_v <= u) {
        component_offsets[last_v] = static_cast<edge_t>(component_indices.size());
        ++last_v;
      }
      std::set_intersection(simple_indices.begin() + simple_offsets[u],
                            simple_indices.begin() + simple_offsets[u + 1],
                            c.begin(),
                            c.end(),
                            std::back_inserter(component_indices));
    }
    while (last_v <= num_vertices) {
      component_offsets[last_v] = static_cast<edge_t>(component_indices.size());
      ++last_v;
    }

    auto new_cycles =
      johnson_cycle_search<vertex_t, edge_t>(component_offsets, component_indices, v);
    cycles.insert(cycles.end(),
                  std::make_move_iterator(new_cycles.begin()),
                  std::make_move_iterator(new_cycles.end()));

    auto remaining = strongly_connected_components_from_induced_subgraph<vertex_t, edge_t>(
      simple_offsets, simple_indices, std::span<vertex_t const>{c.data() + 1, c.size() - 1});
    for (auto& next : remaining) {
      if (next.size() >= 2) { work.push_back(std::move(next)); }
    }
  }

  return cycles;
}

struct SimpleCycles_Usecase {
  size_t k{10};
  double seed_ratio{1.0};
  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_SimpleCycles
  : public ::testing::TestWithParam<std::tuple<SimpleCycles_Usecase, input_usecase_t>> {
 public:
  Tests_SimpleCycles() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(SimpleCycles_Usecase const& simple_cycles_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    using weight_t = float;  // dummy

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);
    std::optional<rmm::device_uvector<vertex_t>> d_renumber_map_labels{std::nullopt};
    std::tie(graph, std::ignore, d_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, false, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    ASSERT_FALSE(graph_view.is_symmetric())
      << "simple_cycles currently supports directed (asymmetric) graphs only.";

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (simple_cycles_usecase.edge_masking) {
      edge_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(handle, graph_view, 2);
      graph_view.attach_edge_mask(edge_mask->view());
    }

    ASSERT_TRUE(simple_cycles_usecase.seed_ratio > 0.0 && simple_cycles_usecase.seed_ratio <= 1.0)
      << "seed_ratio must be greater than 0.0 and less than or equal to 1.0.";
    std::optional<rmm::device_uvector<vertex_t>> d_seed_vertices{std::nullopt};
    if (simple_cycles_usecase.seed_ratio < 1.0) {
      auto num_seeds = static_cast<size_t>(static_cast<double>(graph_view.number_of_vertices()) *
                                           simple_cycles_usecase.seed_ratio);
      num_seeds      = std::max(num_seeds, size_t{1});
      num_seeds      = std::min(num_seeds, static_cast<size_t>(graph_view.number_of_vertices()));
      raft::random::RngState rng_state(0);
      d_seed_vertices = cugraph::select_random_vertices(
        handle,
        graph_view,
        std::optional<raft::device_span<vertex_t const>>{std::nullopt},
        rng_state,
        num_seeds,
        false /* with_replacement */,
        true /* sort_vertices */);
    }

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("simple_cycles");
    }

    auto [d_cugraph_cycle_vertices, d_cugraph_cycle_offsets] = cugraph::simple_cycles(
      handle,
      graph_view,
      d_seed_vertices ? std::make_optional(raft::device_span<vertex_t const>{
                          d_seed_vertices->data(), d_seed_vertices->size()})
                      : std::optional<raft::device_span<vertex_t const>>{std::nullopt},
      static_cast<vertex_t>(simple_cycles_usecase.k));

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (simple_cycles_usecase.check_correctness) {
      std::vector<edge_t> h_offsets{};
      std::vector<vertex_t> h_indices{};
      std::tie(h_offsets, h_indices, std::ignore) =
        cugraph::test::graph_to_host_csr<vertex_t, edge_t, weight_t, false, false>(
          handle,
          graph_view,
          std::nullopt,
          d_renumber_map_labels
            ? std::make_optional<raft::device_span<vertex_t const>>((*d_renumber_map_labels).data(),
                                                                    (*d_renumber_map_labels).size())
            : std::nullopt);

      auto h_reference_cycles =
        simple_cycles_reference(std::span<edge_t const>(h_offsets.data(), h_offsets.size()),
                                std::span<vertex_t const>(h_indices.data(), h_indices.size()));

      std::optional<std::unordered_set<vertex_t>> h_seed_set{std::nullopt};
      if (d_seed_vertices) {
        if (renumber) {
          cugraph::unrenumber_local_int_vertices(handle,
                                                 d_seed_vertices->data(),
                                                 d_seed_vertices->size(),
                                                 d_renumber_map_labels->data(),
                                                 vertex_t{0},
                                                 graph_view.number_of_vertices());
        }
        auto h_seeds = cugraph::test::to_host(handle, *d_seed_vertices);
        h_seed_set.emplace(h_seeds.begin(), h_seeds.end());
      }
      h_reference_cycles.erase(
        std::remove_if(h_reference_cycles.begin(),
                       h_reference_cycles.end(),
                       [k = simple_cycles_usecase.k, &h_seed_set](auto const& cycle) {
                         if (cycle.size() > k) { return true; }
                         if (!h_seed_set) { return false; }
                         return std::none_of(cycle.begin(), cycle.end(), [&h_seed_set](auto v) {
                           return h_seed_set->count(v) > 0;
                         });
                       }),
        h_reference_cycles.end());

      if (renumber) {
        cugraph::unrenumber_local_int_vertices(handle,
                                               d_cugraph_cycle_vertices.data(),
                                               d_cugraph_cycle_vertices.size(),
                                               (*d_renumber_map_labels).data(),
                                               vertex_t{0},
                                               graph_view.number_of_vertices());
      }
      auto h_cugraph_cycle_vertices = cugraph::test::to_host(handle, d_cugraph_cycle_vertices);
      auto h_cugraph_cycle_offsets  = cugraph::test::to_host(handle, d_cugraph_cycle_offsets);

      std::vector<std::vector<vertex_t>> h_cugraph_cycles{};
      h_cugraph_cycles.reserve(h_cugraph_cycle_offsets.size() - 1);
      for (size_t i = 0; i + 1 < h_cugraph_cycle_offsets.size(); ++i) {
        h_cugraph_cycles.emplace_back(
          h_cugraph_cycle_vertices.begin() + h_cugraph_cycle_offsets[i],
          h_cugraph_cycle_vertices.begin() + h_cugraph_cycle_offsets[i + 1]);
      }

      h_reference_cycles = canonicalize_cycles(std::move(h_reference_cycles));
      h_cugraph_cycles   = canonicalize_cycles(std::move(h_cugraph_cycles));

      ASSERT_EQ(h_reference_cycles.size(), h_cugraph_cycles.size())
        << "number of simple cycles does not match the reference.";
      ASSERT_TRUE(
        std::equal(h_reference_cycles.begin(), h_reference_cycles.end(), h_cugraph_cycles.begin()))
        << "simple cycles do not match the reference values.";
    }
  }
};

using Tests_SimpleCycles_File = Tests_SimpleCycles<cugraph::test::File_Usecase>;
using Tests_SimpleCycles_Rmat = Tests_SimpleCycles<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_SimpleCycles_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_SimpleCycles_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_SimpleCycles_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_SimpleCycles_File,
  ::testing::Combine(::testing::Values(SimpleCycles_Usecase{size_t{3}, 0.5, false},
                                       SimpleCycles_Usecase{size_t{3}, 0.5, true},
                                       SimpleCycles_Usecase{size_t{3}, 1.0, false},
                                       SimpleCycles_Usecase{size_t{3}, 1.0, true},
                                       SimpleCycles_Usecase{size_t{6}, 0.5, false},
                                       SimpleCycles_Usecase{size_t{6}, 0.5, true},
                                       SimpleCycles_Usecase{size_t{6}, 1.0, false},
                                       SimpleCycles_Usecase{size_t{6}, 1.0, true},
                                       SimpleCycles_Usecase{size_t{10}, 0.5, false},
                                       SimpleCycles_Usecase{size_t{10}, 0.5, true},
                                       SimpleCycles_Usecase{size_t{10}, 1.0, false},
                                       SimpleCycles_Usecase{size_t{10}, 1.0, true}),
                     ::testing::Values(cugraph::test::File_Usecase("karate-asymmetric.csv"),
                                       cugraph::test::File_Usecase("test/datasets/cage6.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_SimpleCycles_Rmat,
  ::testing::Combine(
    ::testing::Values(SimpleCycles_Usecase{size_t{3}, 0.5, false},
                      SimpleCycles_Usecase{size_t{3}, 0.5, true},
                      SimpleCycles_Usecase{size_t{3}, 1.0, false},
                      SimpleCycles_Usecase{size_t{3}, 1.0, true},
                      SimpleCycles_Usecase{size_t{6}, 0.5, false},
                      SimpleCycles_Usecase{size_t{6}, 0.5, true},
                      SimpleCycles_Usecase{size_t{6}, 1.0, false},
                      SimpleCycles_Usecase{size_t{6}, 1.0, true},
                      SimpleCycles_Usecase{size_t{10}, 0.5, false},
                      SimpleCycles_Usecase{size_t{10}, 0.5, true},
                      SimpleCycles_Usecase{size_t{10}, 1.0, false},
                      SimpleCycles_Usecase{size_t{10}, 1.0, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_SimpleCycles_Rmat,
  ::testing::Values(
    std::make_tuple(SimpleCycles_Usecase{size_t{10}, 1.0, false, false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false)),
    std::make_tuple(SimpleCycles_Usecase{size_t{10}, 1.0, true, false},
                    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
