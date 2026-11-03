/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "community/detail/leiden/leiden_csr.hpp"
#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

// Host copy of the CSR cugraph::leiden reads (single-GPU graph: one edge partition).
template <typename vertex_t, typename edge_t, typename weight_t>
struct host_csr_t {
  std::vector<edge_t> offsets{};
  std::vector<vertex_t> indices{};
  std::optional<std::vector<weight_t>> weights{};  // std::nullopt: every edge weighs 1
};

template <typename vertex_t, typename edge_t, typename weight_t>
host_csr_t<vertex_t, edge_t, weight_t> csr_to_host(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  auto partition = graph_view.local_edge_partition_view();
  host_csr_t<vertex_t, edge_t, weight_t> csr{};
  csr.offsets = cugraph::test::to_host(handle, partition.offsets());
  csr.indices = cugraph::test::to_host(handle, partition.indices());
  if (edge_weight_view) {
    csr.weights = cugraph::test::to_host(
      handle,
      raft::device_span<weight_t const>(edge_weight_view->value_firsts()[0],
                                        static_cast<size_t>(edge_weight_view->edge_counts()[0])));
  }
  return csr;
}

// Weight of stored entry j with the edge semantics of cugraph::leiden: self-loops and entries of
// weight <= 0 are not counted (parallel entries are counted individually, i.e. summed).
template <typename vertex_t, typename edge_t, typename weight_t>
double counted_weight(host_csr_t<vertex_t, edge_t, weight_t> const& csr, vertex_t v, edge_t j)
{
  if (csr.indices[j] == v) return 0.0;
  double const w = csr.weights ? static_cast<double>((*csr.weights)[j]) : 1.0;
  return w > 0.0 ? w : 0.0;
}

// Q(gamma) = sum_c [L_c / 2m - gamma (K_c / 2m)^2] in fp64, recomputed on the host.
template <typename vertex_t, typename edge_t, typename weight_t, typename label_t>
double host_modularity(host_csr_t<vertex_t, edge_t, weight_t> const& csr,
                       std::vector<label_t> const& labels,
                       double gamma)
{
  auto const n = static_cast<vertex_t>(csr.offsets.size() - 1);
  auto const k = static_cast<size_t>(*std::max_element(labels.begin(), labels.end())) + 1;
  std::vector<double> K(k, 0.0);
  double two_m{0.0}, intra{0.0};
  for (vertex_t v = 0; v < n; ++v) {
    for (edge_t j = csr.offsets[v]; j < csr.offsets[v + 1]; ++j) {
      double const w = counted_weight(csr, v, j);
      two_m += w;
      K[labels[v]] += w;
      if (labels[csr.indices[j]] == labels[v]) intra += w;
    }
  }
  if (two_m == 0.0) return 0.0;
  double sum_k2{0.0};
  for (auto x : K)
    sum_k2 += (x / two_m) * (x / two_m);
  return intra / two_m - gamma * sum_k2;
}

// Number of communities whose members are not connected by counted intra-community edges.
template <typename vertex_t, typename edge_t, typename weight_t, typename label_t>
size_t count_disconnected_communities(host_csr_t<vertex_t, edge_t, weight_t> const& csr,
                                      std::vector<label_t> const& labels)
{
  auto const n = static_cast<vertex_t>(csr.offsets.size() - 1);
  std::vector<vertex_t> parent(n);
  std::iota(parent.begin(), parent.end(), vertex_t{0});
  auto find = [&](vertex_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x         = parent[x];
    }
    return x;
  };
  for (vertex_t v = 0; v < n; ++v) {
    for (edge_t j = csr.offsets[v]; j < csr.offsets[v + 1]; ++j) {
      auto const u = csr.indices[j];
      if (counted_weight(csr, v, j) > 0.0 && labels[u] == labels[v]) {
        auto a = find(u), b = find(v);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
      }
    }
  }
  auto const k = static_cast<size_t>(*std::max_element(labels.begin(), labels.end())) + 1;
  std::vector<size_t> pieces(k, 0);
  for (vertex_t v = 0; v < n; ++v)
    if (find(v) == v) ++pieces[labels[v]];
  return static_cast<size_t>(
    std::count_if(pieces.begin(), pieces.end(), [](size_t p) { return p > 1; }));
}

// API contract of the returned clustering: labels in [0, num_clusters), every label used, label 0
// the largest community, ties ordered by the smallest vertex id.
template <typename label_t>
void check_label_contract(std::vector<label_t> const& labels, size_t num_clusters)
{
  ASSERT_GT(num_clusters, size_t{0});
  std::vector<size_t> size(num_clusters, 0);
  std::vector<size_t> min_v(num_clusters, std::numeric_limits<size_t>::max());
  for (size_t v = 0; v < labels.size(); ++v) {
    ASSERT_GE(labels[v], label_t{0});
    ASSERT_LT(static_cast<size_t>(labels[v]), num_clusters);
    ++size[labels[v]];
    min_v[labels[v]] = std::min(min_v[labels[v]], v);
  }
  for (size_t c = 0; c < num_clusters; ++c) {
    ASSERT_GT(size[c], size_t{0}) << "label " << c << " is unused";
    if (c > 0) {
      ASSERT_TRUE(size[c - 1] > size[c] || (size[c - 1] == size[c] && min_v[c - 1] < min_v[c]))
        << "labels are not ordered by decreasing size (ties by smallest vertex id)";
    }
  }
}

uint64_t bits_of(double x)
{
  uint64_t b{};
  std::memcpy(&b, &x, sizeof(b));
  return b;
}

// FNV-1a of the labels (pins the clustering of the larger golden graphs compactly).
template <typename label_t>
uint64_t fnv1a(std::vector<label_t> const& labels)
{
  uint64_t h = 14695981039346656037ull;
  for (auto x : labels) {
    h ^= static_cast<uint64_t>(static_cast<int64_t>(x));
    h *= 1099511628211ull;
  }
  return h;
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<std::vector<vertex_t>, cugraph::leiden_result_t> run_leiden(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  cugraph::leiden_params_t const& params,
  uint64_t seed)
{
  rmm::device_uvector<vertex_t> clustering(graph_view.number_of_vertices(), handle.get_stream());
  raft::random::RngState rng_state(seed);
  auto result = cugraph::leiden(handle,
                                rng_state,
                                graph_view,
                                edge_weight_view,
                                raft::device_span<vertex_t>(clustering.data(), clustering.size()),
                                params);
  return std::make_tuple(cugraph::test::to_host(handle, clustering), result);
}

// Edge list of an undirected graph with both directions of every edge stored (add()), or of
// single directed entries (add_entry()).
struct edge_list_t {
  int64_t num_vertices{0};
  std::vector<int64_t> srcs{};
  std::vector<int64_t> dsts{};
  std::vector<double> weights{};

  void add_entry(int64_t u, int64_t v, double w)
  {
    srcs.push_back(u);
    dsts.push_back(v);
    weights.push_back(w);
  }
  void add(int64_t u, int64_t v, double w)
  {
    add_entry(u, v, w);
    if (u != v) add_entry(v, u, w);
  }
};

// A ring of `num_cliques` cliques of `clique_size` vertices; consecutive cliques are joined by one
// edge. Weights are powers of two (exact in fp32 and fp64), chosen by a hash of the edge.
edge_list_t ring_of_cliques(int64_t num_cliques, int64_t clique_size)
{
  edge_list_t el{};
  el.num_vertices = num_cliques * clique_size;
  auto weight     = [](int64_t u, int64_t v) {
    uint64_t h = static_cast<uint64_t>(u) * 0x9E3779B97F4A7C15ull ^ static_cast<uint64_t>(v);
    h ^= h >> 29;
    return std::ldexp(1.0, static_cast<int>(h % 3));  // 1, 2 or 4
  };
  for (int64_t c = 0; c < num_cliques; ++c) {
    int64_t const b = c * clique_size;
    for (int64_t i = 0; i < clique_size; ++i)
      for (int64_t j = i + 1; j < clique_size; ++j)
        el.add(b + i, b + j, weight(b + i, b + j));
    el.add(b + clique_size - 1, (b + clique_size) % el.num_vertices, 1.0);
  }
  return el;
}

// The same edge list with its entries in a (seeded) random order.
edge_list_t shuffled(edge_list_t const& el, uint64_t seed)
{
  std::vector<size_t> perm(el.srcs.size());
  std::iota(perm.begin(), perm.end(), size_t{0});
  std::mt19937_64 gen(seed);
  std::shuffle(perm.begin(), perm.end(), gen);
  edge_list_t out{};
  out.num_vertices = el.num_vertices;
  for (auto i : perm)
    out.add_entry(el.srcs[i], el.dsts[i], el.weights[i]);
  return out;
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<cugraph::graph_t<vertex_t, edge_t, false, false>,
           std::optional<cugraph::edge_property_t<edge_t, weight_t>>>
graph_from_edge_list(raft::handle_t const& handle,
                     edge_list_t const& el,
                     bool weighted,
                     bool is_symmetric  = true,
                     bool is_multigraph = false)
{
  std::vector<vertex_t> h_srcs(el.srcs.begin(), el.srcs.end());
  std::vector<vertex_t> h_dsts(el.dsts.begin(), el.dsts.end());
  std::vector<vertex_t> h_vertices(el.num_vertices);
  std::iota(h_vertices.begin(), h_vertices.end(), vertex_t{0});
  std::vector<cugraph::arithmetic_device_uvector_t> props{};
  if (weighted) {
    std::vector<weight_t> h_w(el.weights.begin(), el.weights.end());
    props.push_back(cugraph::test::to_device(handle, h_w));
  }
  auto [graph, edge_props, renumber_map] =
    cugraph::create_graph_from_edgelist<vertex_t, edge_t, false, false>(
      handle,
      std::make_optional(cugraph::test::to_device(handle, h_vertices)),
      cugraph::test::to_device(handle, h_srcs),
      cugraph::test::to_device(handle, h_dsts),
      std::move(props),
      cugraph::graph_properties_t{is_symmetric, is_multigraph},
      false /* renumber: keep the vertex ids (they enter the hashes) */);
  std::optional<cugraph::edge_property_t<edge_t, weight_t>> weights{std::nullopt};
  if (weighted) {
    weights = std::move(std::get<cugraph::edge_property_t<edge_t, weight_t>>(edge_props[0]));
  }
  return std::make_tuple(std::move(graph), std::move(weights));
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<std::vector<vertex_t>, cugraph::leiden_result_t> run_leiden_on_edge_list(
  raft::handle_t const& handle,
  edge_list_t const& el,
  bool weighted,
  cugraph::leiden_params_t const& params,
  uint64_t seed,
  bool is_multigraph = false)
{
  auto [g, w] =
    graph_from_edge_list<vertex_t, edge_t, weight_t>(handle, el, weighted, true, is_multigraph);
  return run_leiden<vertex_t, edge_t, weight_t>(
    handle, g.view(), w ? std::make_optional(w->view()) : std::nullopt, params, seed);
}

}  // namespace

struct Leiden_Usecase {
  double resolution_{1.0};
  int32_t n_iterations_{2};
  bool test_weighted_{true};
  bool check_correctness_{true};
};

template <typename input_usecase_t>
class Tests_Leiden : public ::testing::TestWithParam<std::tuple<Leiden_Usecase, input_usecase_t>> {
 public:
  Tests_Leiden() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(std::tuple<Leiden_Usecase const&, input_usecase_t const&> const& param)
  {
    auto [usecase, input_usecase] = param;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    // renumber = false: the algorithm hashes vertex ids, so keeping the input ids makes results
    // comparable across vertex / edge / weight types.
    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, usecase.test_weighted_, false, true, true);
    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    cugraph::leiden_params_t params{};
    params.resolution   = usecase.resolution_;
    params.n_iterations = usecase.n_iterations_;

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Leiden");
    }

    auto [labels, result] = run_leiden<vertex_t, edge_t, weight_t>(
      handle, graph_view, edge_weight_view, params, uint64_t{42});

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (!usecase.check_correctness_) return;

    // API contract: one label per vertex, consecutive and size-ordered.
    ASSERT_EQ(labels.size(), static_cast<size_t>(graph_view.number_of_vertices()));
    ASSERT_NO_FATAL_FAILURE(check_label_contract(labels, result.num_clusters));
    ASSERT_EQ(result.num_clusters,
              static_cast<size_t>(*std::max_element(labels.begin(), labels.end())) + 1);
    ASSERT_GE(result.num_iterations, size_t{1});
    ASSERT_GE(result.num_levels, size_t{1});

    // Determinism: the same seed gives the identical clustering and the bitwise identical
    // modularity, also with every layout of the engine: each kernel launch mode (0: plain stream
    // launches, 1: CUDA graph replays, the path of CUDA < 12.4; the default 2 runs the
    // local-moving sweeps in a conditional WHILE node where supported), and a level arena too small
    // for the first level (the engine reruns on a larger one).
    {
      auto [labels2, result2] = run_leiden<vertex_t, edge_t, weight_t>(
        handle, graph_view, edge_weight_view, params, uint64_t{42});
      ASSERT_EQ(labels, labels2) << "same seed, different clustering";
      ASSERT_EQ(bits_of(result.modularity), bits_of(result2.modularity));
      ASSERT_EQ(result.num_levels, result2.num_levels);

      // A fresh RngState{42} gives cugraph::leiden the 32-bit seed 42.
      auto partition = graph_view.local_edge_partition_view();
      std::optional<raft::device_span<weight_t const>> weights{std::nullopt};
      if (edge_weight_view) {
        weights = raft::device_span<weight_t const>(edge_weight_view->value_firsts()[0],
                                                    partition.indices().size());
      }
      for (auto const& options : {cugraph::detail::leiden_layout_options_t{0, 1.9},
                                  cugraph::detail::leiden_layout_options_t{1, 1.9},
                                  cugraph::detail::leiden_layout_options_t{2, 0.01}}) {
        SCOPED_TRACE("graph_mode " + std::to_string(options.graph_mode) + ", arena_factor " +
                     std::to_string(options.arena_factor));
        rmm::device_uvector<vertex_t> d_labels3(graph_view.number_of_vertices(),
                                                handle.get_stream());
        auto const result3 = cugraph::detail::leiden_csr<vertex_t, edge_t, weight_t>(
          handle,
          uint32_t{42},
          raft::device_span<edge_t const>(partition.offsets().data(), partition.offsets().size()),
          raft::device_span<vertex_t const>(partition.indices().data(), partition.indices().size()),
          weights,
          raft::device_span<vertex_t>(d_labels3.data(), d_labels3.size()),
          params,
          options);
        ASSERT_EQ(labels, cugraph::test::to_host(handle, d_labels3))
          << "the engine layout changed the clustering";
        ASSERT_EQ(bits_of(result.modularity), bits_of(result3.modularity));
        ASSERT_EQ(result.num_levels, result3.num_levels);
      }
    }

    // Every community is connected, and the returned Q is the modularity of the returned
    // clustering (fp64 recomputation on the host).
    auto csr = csr_to_host(handle, graph_view, edge_weight_view);
    ASSERT_EQ(count_disconnected_communities(csr, labels), size_t{0});
    double const q_host = host_modularity(csr, labels, usecase.resolution_);
    EXPECT_NEAR(result.modularity, q_host, 1e-9) << "reported modularity is not that of the result";

    // A different seed gives a different but equally valid clustering.
    {
      auto [labels4, result4] = run_leiden<vertex_t, edge_t, weight_t>(
        handle, graph_view, edge_weight_view, params, uint64_t{7});
      ASSERT_NO_FATAL_FAILURE(check_label_contract(labels4, result4.num_clusters));
      ASSERT_EQ(count_disconnected_communities(csr, labels4), size_t{0});
      EXPECT_NEAR(result4.modularity, host_modularity(csr, labels4, usecase.resolution_), 1e-9);
    }
  }
};

using Tests_Leiden_File = Tests_Leiden<cugraph::test::File_Usecase>;
using Tests_Leiden_Rmat = Tests_Leiden<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_Leiden_File, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Leiden_File, CheckInt64Int64Float)
{
  run_current_test<int64_t, int64_t, float>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Leiden_Rmat, CheckInt32Int32Float)
{
  run_current_test<int32_t, int32_t, float>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_Leiden_Rmat, CheckInt64Int64Double)
{
  run_current_test<int64_t, int64_t, double>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_Leiden_File,
  ::testing::Combine(
    ::testing::Values(Leiden_Usecase{1.0, 2, true},
                      Leiden_Usecase{0.5, 2, true},
                      Leiden_Usecase{2.0, -1, true},
                      Leiden_Usecase{1.0, 1, false}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"),
                      cugraph::test::File_Usecase("test/datasets/dolphins.mtx"),
                      cugraph::test::File_Usecase("test/datasets/polbooks.mtx"),
                      cugraph::test::File_Usecase("test/datasets/netscience.mtx"))));

INSTANTIATE_TEST_SUITE_P(rmat_small_test,
                         Tests_Leiden_Rmat,
                         ::testing::Combine(::testing::Values(Leiden_Usecase{1.0, 2, true},
                                                              Leiden_Usecase{1.0, 2, false}),
                                            ::testing::Values(cugraph::test::Rmat_Usecase(
                                              10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_Leiden_Rmat,
  ::testing::Combine(
    ::testing::Values(Leiden_Usecase{1.0, 2, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(12, 32, 0.57, 0.19, 0.19, 0, true, false))));

// Pinned results of cugraph::leiden with the default parameters (resolution 1, two iterations), a
// fresh raft::random::RngState{0} (32-bit seed 0) and float weights, on the graphs read with
// renumber = false (vertex ids enter the hashes). The algorithm makes every decision in exact
// integer arithmetic and sums Q in a fixed order, so these values are exact and the same on every
// GPU: the labels, the bits of Q and the FNV-1a hash of the int32 labels are what this call
// returns. Rerunning it with the same inputs reproduces them; a mismatch means that the algorithm
// (or the graph reader) changed. The int64 / double variants below must give the same result.
TEST(Leiden, GoldenValues)
{
  struct golden_t {
    char const* file;
    size_t num_clusters;
    uint64_t modularity_bits;
    uint64_t labels_fnv1a;
    std::vector<int32_t> labels;  // empty: compare the hash only
  };
  std::vector<golden_t> const goldens{
    {"test/datasets/karate.mtx",
     4,
     0x3fdaddd53fca2404ull,  // 0x1.addd53fca2404p-2 = 0.41978961209730437
     0x4c1c3962cc809ab5ull,
     {1, 1, 1, 1, 3, 3, 3, 1, 0, 0, 3, 1, 1, 1, 0, 0, 3,
      1, 0, 1, 0, 1, 0, 2, 2, 2, 0, 2, 2, 0, 0, 2, 0, 0}},
    {"test/datasets/dolphins.mtx",
     5,
     0x3fe0e9a19a8e5039ull,  // 0.5285194414777897
     0x410d3f45fbbb6eb0ull,
     {3, 0, 3, 4, 2, 0, 0, 0, 4, 0, 3, 2, 1, 0, 1, 2, 1, 0, 2, 0, 3, 2, 0, 2, 2, 0, 0, 0, 3, 2, 3,
      0, 0, 1, 1, 2, 4, 1, 1, 4, 1, 0, 3, 1, 3, 2, 1, 3, 0, 1, 1, 2, 1, 1, 0, 2, 0, 0, 1, 4, 0, 1}},
    {"test/datasets/polbooks.mtx",
     5,
     0x3fe0df1f46f4d973ull,  // 0.5272365938060787
     0x07a6cd250a693bddull,
     {3, 3, 3, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 3, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 2, 2, 2, 0,
      0, 0, 0, 4, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2}},
    {"test/datasets/netscience.mtx",
     408,
     0x3fee8f4378bea679ull,  // 0.9549882276816809
     0xade061749613a608ull,
     {}},
  };

  raft::handle_t handle{};
  cugraph::leiden_params_t const params{};
  for (auto const& g : goldens) {
    SCOPED_TRACE(g.file);
    auto check = [&](auto const& labels, cugraph::leiden_result_t const& result) {
      EXPECT_EQ(result.num_clusters, g.num_clusters);
      EXPECT_EQ(bits_of(result.modularity), g.modularity_bits) << "Q = " << result.modularity;
      EXPECT_EQ(fnv1a(labels), g.labels_fnv1a);
      if (!g.labels.empty()) {
        EXPECT_TRUE(std::equal(labels.begin(), labels.end(), g.labels.begin(), g.labels.end()));
      }
    };
    auto [graph, weights, renumber_map] =
      cugraph::test::construct_graph<int32_t, int32_t, float, false, false>(
        handle, cugraph::test::File_Usecase(g.file), true, false);
    auto [labels, result] = run_leiden<int32_t, int32_t, float>(
      handle, graph.view(), std::make_optional(weights->view()), params, uint64_t{0});
    check(labels, result);

    auto [graph64, weights64, renumber_map64] =
      cugraph::test::construct_graph<int64_t, int64_t, float, false, false>(
        handle, cugraph::test::File_Usecase(g.file), true, false);
    auto [labels64, result64] = run_leiden<int64_t, int64_t, float>(
      handle, graph64.view(), std::make_optional(weights64->view()), params, uint64_t{0});
    check(labels64, result64);

    // the same weight values as double give the identical result
    auto csr = csr_to_host(handle, graph.view(), std::make_optional(weights->view()));
    edge_list_t el{};
    el.num_vertices = static_cast<int64_t>(csr.offsets.size()) - 1;
    for (int64_t v = 0; v < el.num_vertices; ++v)
      for (auto j = csr.offsets[v]; j < csr.offsets[v + 1]; ++j)
        el.add_entry(v, csr.indices[j], static_cast<double>((*csr.weights)[j]));
    auto [labels_d, result_d] =
      run_leiden_on_edge_list<int32_t, int32_t, double>(handle, el, true, params, uint64_t{0});
    check(labels_d, result_d);
  }
}

// The result depends on the vertex ids and the weight values only: int32 / int64 vertex and edge
// ids and float / double weights (with equal values) give the identical clustering and the
// bitwise identical modularity. Self-loops are ignored, an unweighted graph equals the graph with
// all weights 1, and a ring of cliques is split along the cliques.
TEST(Leiden, TypesSelfLoopsAndUnitWeights)
{
  raft::handle_t handle{};
  auto const el = ring_of_cliques(24, 6);
  cugraph::leiden_params_t params{};

  for (uint64_t seed : {0, 1, 2}) {
    auto [l32f, r32f] =
      run_leiden_on_edge_list<int32_t, int32_t, float>(handle, el, true, params, seed);
    auto [l32d, r32d] =
      run_leiden_on_edge_list<int32_t, int32_t, double>(handle, el, true, params, seed);
    auto [l64f, r64f] =
      run_leiden_on_edge_list<int64_t, int64_t, float>(handle, el, true, params, seed);
    auto [l64d, r64d] =
      run_leiden_on_edge_list<int64_t, int64_t, double>(handle, el, true, params, seed);

    ASSERT_EQ(l32f, l32d);
    ASSERT_TRUE(std::equal(l32f.begin(), l32f.end(), l64f.begin(), l64f.end()));
    ASSERT_EQ(l64f, l64d);
    ASSERT_EQ(bits_of(r32f.modularity), bits_of(r32d.modularity));
    ASSERT_EQ(bits_of(r32f.modularity), bits_of(r64f.modularity));
    ASSERT_EQ(bits_of(r32f.modularity), bits_of(r64d.modularity));

    // every clique lies inside one community
    for (int64_t v = 0; v < el.num_vertices; ++v)
      ASSERT_EQ(l32f[v], l32f[(v / 6) * 6]) << "clique " << v / 6 << " was split";

    // self-loops (any weight) are ignored
    auto with_loops = el;
    for (int64_t v = 0; v < el.num_vertices; v += 3)
      with_loops.add(v, v, 8.0);
    auto [ll, rl] =
      run_leiden_on_edge_list<int32_t, int32_t, float>(handle, with_loops, true, params, seed);
    ASSERT_EQ(l32f, ll);
    ASSERT_EQ(bits_of(r32f.modularity), bits_of(rl.modularity));

    // unweighted == all weights 1
    auto unit = el;
    std::fill(unit.weights.begin(), unit.weights.end(), 1.0);
    auto [lu, ru] =
      run_leiden_on_edge_list<int32_t, int32_t, float>(handle, unit, true, params, seed);
    auto [ln, rn] =
      run_leiden_on_edge_list<int32_t, int32_t, float>(handle, el, false, params, seed);
    ASSERT_EQ(lu, ln);
    ASSERT_EQ(bits_of(ru.modularity), bits_of(rn.modularity));
  }
}

// Parallel edges are summed: a multigraph gives the result of the graph with the summed weights,
// and the summation order is canonical (ascending weights), so the stored order of the parallel
// edges does not matter even when their fp64 sum depends on the order.
TEST(Leiden, ParallelEdgesAreSummed)
{
  raft::handle_t handle{};
  auto const el = ring_of_cliques(16, 5);
  cugraph::leiden_params_t params{};

  // split every undirected edge of weight w into parallel edges of weights w/4, w/4, w/2 (exact)
  edge_list_t multi{};
  multi.num_vertices = el.num_vertices;
  for (size_t i = 0; i < el.srcs.size(); ++i) {
    for (double part : {0.25, 0.25, 0.5})
      multi.add_entry(el.srcs[i], el.dsts[i], part * el.weights[i]);
  }
  // non-dyadic parallel edges: the fp64 sum depends on the summation order
  edge_list_t inexact = el;
  for (int64_t v = 0; v + 1 < el.num_vertices; v += 2) {
    for (double part : {0.1, 0.7, 0.2, 1.0 / 3.0})
      inexact.add(v, v + 1, part);
  }

  for (uint64_t seed : {0, 3}) {
    auto [ls, rs] =
      run_leiden_on_edge_list<int32_t, int32_t, float>(handle, el, true, params, seed);
    auto [lm, rm] =
      run_leiden_on_edge_list<int32_t, int32_t, float>(handle, multi, true, params, seed, true);
    ASSERT_EQ(ls, lm) << "multigraph != graph with summed weights";
    ASSERT_EQ(bits_of(rs.modularity), bits_of(rm.modularity));
    auto [lm64, rm64] =
      run_leiden_on_edge_list<int64_t, int64_t, double>(handle, multi, true, params, seed, true);
    ASSERT_TRUE(std::equal(ls.begin(), ls.end(), lm64.begin(), lm64.end()));
    ASSERT_EQ(bits_of(rs.modularity), bits_of(rm64.modularity));

    // unweighted multigraph: the multiplicity is the weight
    edge_list_t unit_multi{};
    unit_multi.num_vertices = el.num_vertices;
    for (size_t i = 0; i < el.srcs.size(); ++i) {
      auto const copies = static_cast<int>(el.weights[i]);  // 1, 2 or 4
      for (int c = 0; c < copies; ++c)
        unit_multi.add_entry(el.srcs[i], el.dsts[i], 1.0);
    }
    auto [lu, ru] = run_leiden_on_edge_list<int32_t, int32_t, float>(
      handle, unit_multi, false, params, seed, true);
    ASSERT_EQ(ls, lu) << "unweighted multigraph != weighted graph of the multiplicities";
    ASSERT_EQ(bits_of(rs.modularity), bits_of(ru.modularity));

    // any stored order of the parallel edges gives the identical result
    auto [li, ri] =
      run_leiden_on_edge_list<int32_t, int32_t, double>(handle, inexact, true, params, seed, true);
    for (uint64_t shuffle_seed : {1, 2, 3}) {
      auto [lp, rp] = run_leiden_on_edge_list<int32_t, int32_t, double>(
        handle, shuffled(inexact, shuffle_seed), true, params, seed, true);
      ASSERT_EQ(li, lp) << "the order of parallel edges changed the clustering";
      ASSERT_EQ(bits_of(ri.modularity), bits_of(rp.modularity));
    }
  }
}

// The CSR assembly of the multi-GPU path (detail::leiden_coo on the all-gathered edge list) does
// not depend on how the edges are split over the ranks: every split of the edge list into P
// chunks, concatenated in rank order, gives the result of the single-GPU path on the same vertex
// ids, bitwise. Together with the identical seed this is what makes multi-GPU Leiden equal to
// single-GPU Leiden for any number of GPUs.
TEST(Leiden, EdgeListSplitInvariance)
{
  raft::handle_t handle{};
  cugraph::leiden_params_t params{};
  auto el = ring_of_cliques(20, 6);
  for (int64_t v = 0; v + 2 < el.num_vertices; v += 3)
    el.add(v, v + 2, 0.1 * static_cast<double>(v % 7));  // parallel and zero-weight edges

  for (uint64_t seed : {0, 5}) {
    auto [ls, rs] =
      run_leiden_on_edge_list<int32_t, int32_t, float>(handle, el, true, params, seed, true);
    for (int P : {1, 2, 3, 4}) {
      SCOPED_TRACE("P = " + std::to_string(P));
      // rank r holds the entries i with hash(i) % P == r, in a shuffled order
      auto const mixed = shuffled(el, static_cast<uint64_t>(P));
      edge_list_t gathered{};
      gathered.num_vertices = el.num_vertices;
      for (int r = 0; r < P; ++r)
        for (size_t i = 0; i < mixed.srcs.size(); ++i)
          if (static_cast<int>((i * 2654435761ull) % P) == r)
            gathered.add_entry(mixed.srcs[i], mixed.dsts[i], mixed.weights[i]);

      std::vector<int32_t> h_srcs(gathered.srcs.begin(), gathered.srcs.end());
      std::vector<int32_t> h_dsts(gathered.dsts.begin(), gathered.dsts.end());
      std::vector<float> h_w(gathered.weights.begin(), gathered.weights.end());
      rmm::device_uvector<int32_t> labels(el.num_vertices, handle.get_stream());
      auto const r = cugraph::detail::leiden_coo<int32_t, int32_t, float>(
        handle,
        static_cast<uint32_t>(seed),
        static_cast<int32_t>(el.num_vertices),
        cugraph::test::to_device(handle, h_srcs),
        cugraph::test::to_device(handle, h_dsts),
        std::make_optional(cugraph::test::to_device(handle, h_w)),
        raft::device_span<int32_t>(labels.data(), labels.size()),
        params);
      ASSERT_EQ(ls, cugraph::test::to_host(handle, labels));
      ASSERT_EQ(bits_of(rs.modularity), bits_of(r.modularity));
      ASSERT_EQ(rs.num_clusters, r.num_clusters);
      ASSERT_EQ(rs.num_iterations, r.num_iterations);
      ASSERT_EQ(rs.num_levels, r.num_levels);
    }
  }
}

// The data is checked, not the is_symmetric() flag: a symmetric graph that is not flagged as such
// is accepted and gives the same result.
TEST(Leiden, SymmetricDataWithoutSymmetricFlag)
{
  raft::handle_t handle{};
  auto const el = ring_of_cliques(8, 4);
  cugraph::leiden_params_t params{};
  auto [g1, w1] = graph_from_edge_list<int32_t, int32_t, float>(handle, el, true, true);
  auto [g2, w2] = graph_from_edge_list<int32_t, int32_t, float>(handle, el, true, false);
  auto [l1, r1] = run_leiden<int32_t, int32_t, float>(
    handle, g1.view(), std::make_optional(w1->view()), params, 0);
  auto [l2, r2] = run_leiden<int32_t, int32_t, float>(
    handle, g2.view(), std::make_optional(w2->view()), params, 0);
  ASSERT_EQ(l1, l2);
  ASSERT_EQ(bits_of(r1.modularity), bits_of(r2.modularity));
}

// Successive calls with one RngState use different seeds; equal states give equal results.
TEST(Leiden, RngStateAdvances)
{
  raft::handle_t handle{};
  auto const el = ring_of_cliques(64, 5);
  auto [g, w]   = graph_from_edge_list<int32_t, int32_t, float>(handle, el, true);
  auto view     = g.view();
  auto wview    = std::make_optional(w->view());
  rmm::device_uvector<int32_t> c1(view.number_of_vertices(), handle.get_stream());
  rmm::device_uvector<int32_t> c2(view.number_of_vertices(), handle.get_stream());

  raft::random::RngState a(uint64_t{5}), b(uint64_t{5});
  auto r1 =
    cugraph::leiden(handle, a, view, wview, raft::device_span<int32_t>(c1.data(), c1.size()));
  auto r2 =
    cugraph::leiden(handle, b, view, wview, raft::device_span<int32_t>(c2.data(), c2.size()));
  ASSERT_EQ(cugraph::test::to_host(handle, c1), cugraph::test::to_host(handle, c2));
  ASSERT_EQ(bits_of(r1.modularity), bits_of(r2.modularity));
  ASSERT_EQ(a.base_subsequence, b.base_subsequence);
  ASSERT_GT(a.base_subsequence, uint64_t{0});  // advanced: the next call uses another seed
}

TEST(Leiden, GraphWithoutCountedEdges)
{
  raft::handle_t handle{};
  edge_list_t el{};
  el.num_vertices = 5;
  el.add(1, 1, 3.0);  // a self-loop only: no counted edge
  el.add(2, 3, 0.0);  // weight 0: not counted
  auto [labels, r] = run_leiden_on_edge_list<int32_t, int32_t, float>(
    handle, el, true, cugraph::leiden_params_t{}, 0);
  ASSERT_EQ(r.num_clusters, size_t{5});
  ASSERT_EQ(r.modularity, 0.0);
  ASSERT_EQ(labels, (std::vector<int32_t>{0, 1, 2, 3, 4}));
}

TEST(Leiden, RejectsInvalidInput)
{
  raft::handle_t handle{};
  // The graph is built first (that must succeed); only cugraph::leiden must reject it, with a
  // message naming the problem.
  auto expect_rejected = [&](edge_list_t const& el,
                             std::string const& fragment,
                             cugraph::leiden_params_t const& params = {}) {
    using graph_and_weights_t = std::tuple<cugraph::graph_t<int32_t, int32_t, false, false>,
                                           std::optional<cugraph::edge_property_t<int32_t, float>>>;
    std::optional<graph_and_weights_t> built{};
    auto build = [&] { return graph_from_edge_list<int32_t, int32_t, float>(handle, el, true); };
    ASSERT_NO_THROW(built.emplace(build()));
    auto& [g, w] = *built;
    try {
      run_leiden<int32_t, int32_t, float>(
        handle, g.view(), std::make_optional(w->view()), params, 0);
      ADD_FAILURE() << "not rejected (expected: " << fragment << ")";
    } catch (cugraph::logic_error const& e) {
      EXPECT_NE(std::string(e.what()).find(fragment), std::string::npos) << e.what();
    }
  };
  auto const good = ring_of_cliques(4, 4);

  // an edge without its reverse
  auto asymmetric = good;
  asymmetric.add_entry(0, 9, 1.0);
  expect_rejected(asymmetric, "not symmetric");
  // a reverse edge with a different weight
  auto asymmetric_weight       = good;
  asymmetric_weight.weights[0] = 2.0 * asymmetric_weight.weights[0] + 1.0;
  expect_rejected(asymmetric_weight, "not symmetric");
  // parallel edges whose sums differ from their reverse
  auto asymmetric_multi = good;
  asymmetric_multi.add_entry(0, 1, 1.0);
  expect_rejected(asymmetric_multi, "not symmetric");
  // negative and non-finite weights
  auto negative = good;
  negative.add(0, 15, -1.0);
  expect_rejected(negative, "finite and non-negative");
  auto not_finite = good;
  not_finite.add(0, 15, std::numeric_limits<double>::quiet_NaN());
  expect_rejected(not_finite, "finite and non-negative");

  // parameters
  auto with = [](auto&& set) {
    cugraph::leiden_params_t p{};
    set(p);
    return p;
  };
  expect_rejected(good, "resolution", with([](auto& p) { p.resolution = -1.0; }));
  expect_rejected(good, "resolution", with([](auto& p) { p.resolution = std::nan(""); }));
  expect_rejected(good, "resolution", with([](auto& p) { p.resolution = 2e6; }));
  expect_rejected(good, "n_iterations", with([](auto& p) { p.n_iterations = 0; }));
  expect_rejected(good, "n_iterations", with([](auto& p) { p.n_iterations = -2; }));
  expect_rejected(good, "beta", with([](auto& p) { p.beta = 0.01; }));
  expect_rejected(good, "beta", with([](auto& p) { p.beta = -1.0; }));
  expect_rejected(good, "max_level", with([](auto& p) { p.max_level = 0; }));
  expect_rejected(good, "max_level", with([](auto& p) { p.max_level = 65; }));

  auto [g, w] = graph_from_edge_list<int32_t, int32_t, float>(handle, good, true);
  auto view   = g.view();
  auto wview  = std::make_optional(w->view());
  rmm::device_uvector<int32_t> too_small(view.number_of_vertices() - 1, handle.get_stream());
  raft::random::RngState rng_state(uint64_t{0});
  EXPECT_THROW(cugraph::leiden(handle,
                               rng_state,
                               view,
                               wview,
                               raft::device_span<int32_t>(too_small.data(), too_small.size())),
               cugraph::logic_error);
  ASSERT_NO_THROW(
    (run_leiden<int32_t, int32_t, float>(handle, view, wview, cugraph::leiden_params_t{}, 0)));
}

CUGRAPH_TEST_PROGRAM_MAIN()
