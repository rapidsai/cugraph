/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */

#include <prims/property_generator.cuh>

#include <prims/fill_edge_src_dst_property.cuh>
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <utilities/base_fixture.hpp>
#include <utilities/test_graphs.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

template <typename vertex_t, typename result_t>
struct e_op_t {
  __device__ result_t operator()(vertex_t src,
                                 vertex_t dst,
                                 result_t src_property,
                                 result_t dst_property,
                                 thrust::nullopt_t) const
  {
    if (src_property < dst_property) {
      return src_property;
    } else {
      return dst_property;
    }
  }
};

struct MaximalIndependentSet_Usecase {
  size_t select_count{std::numeric_limits<size_t>::max()};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_MGMaximalIndependentSet
  : public ::testing::TestWithParam<std::tuple<MaximalIndependentSet_Usecase, input_usecase_t>> {
 public:
  Tests_MGMaximalIndependentSet() {}

  static void SetUpTestCase() { handle_ = cugraph::test::initialize_mg_handle(); }
  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
  void run_current_test(std::tuple<MaximalIndependentSet_Usecase, input_usecase_t> const& param)
  {
    auto [mis_usecase, input_usecase] = param;

    auto const comm_rank = handle_->get_comms().get_rank();
    auto const comm_size = handle_->get_comms().get_size();

    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.start("MG Construct graph");
    }

    auto [mg_graph, mg_edge_weights, mg_renumber_map] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, true>(
        *handle_, input_usecase, false, true);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto mg_graph_view = mg_graph.view();
    auto mg_edge_weight_view =
      mg_edge_weights ? std::make_optional((*mg_edge_weights).view()) : std::nullopt;

    // Test MIS

    auto d_mis =
      cugraph::compute_mis<vertex_t, edge_t, weight_t, true>(*handle_, mg_graph_view, std::nullopt);

    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    std::vector<vertex_t> h_mis(d_mis.size());
    raft::update_host(h_mis.data(), d_mis.data(), d_mis.size(), handle_->get_stream());

    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    for (int i = 0; i < comm_size; ++i) {
      if (comm_rank == i) {
        if (h_mis.size() <= 50 && mg_graph_view.number_of_vertices() < 35) {
          std::cout << "MIS (rank:" << comm_rank << "): ";
          std::copy(h_mis.begin(), h_mis.end(), std::ostream_iterator<int>(std::cout, " "));
          std::cout << std::endl;
        }
        auto vertex_first = mg_graph_view.local_vertex_partition_range_first();
        auto vertex_last  = mg_graph_view.local_vertex_partition_range_last();

        std::for_each(h_mis.begin(), h_mis.end(), [vertex_first, vertex_last](vertex_t v) {
          ASSERT_TRUE((v >= vertex_first) && (v < vertex_last));
        });
      }

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
    }

    // Test MIS correctness

    auto multi_gpu = true;

    vertex_t local_vtx_partitoin_size = mg_graph_view.local_vertex_partition_range_size();
    rmm::device_uvector<vertex_t> d_total_outgoing_nbrs_included_mis(local_vtx_partitoin_size,
                                                                     handle_->get_stream());

    rmm::device_uvector<vertex_t> inclusiong_flags(local_vtx_partitoin_size, handle_->get_stream());

    thrust::uninitialized_fill(
      handle_->get_thrust_policy(), inclusiong_flags.begin(), inclusiong_flags.end(), vertex_t{0});

    auto vertex_begin =
      thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_first());
    auto vertex_end =
      thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_last());

    thrust::for_each(
      handle_->get_thrust_policy(),
      d_mis.begin(),
      d_mis.end(),
      [inclusiong_flags =
         raft::device_span<vertex_t>(inclusiong_flags.data(), inclusiong_flags.size()),
       v_first = mg_graph_view.local_vertex_partition_range_first()] __device__(auto v) {
        auto v_offset              = v - v_first;
        inclusiong_flags[v_offset] = vertex_t{1};
      });

    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    for (int i = 0; i < comm_size; ++i) {
      if (comm_rank == i) {
        if (inclusiong_flags.size() <= 50 && mg_graph_view.number_of_vertices() < 35) {
          std::cout << "rank:(" << comm_rank << "): ";
          raft::print_device_vector(
            "inclusiong_flags:", inclusiong_flags.data(), inclusiong_flags.size(), std::cout);
        }
      }

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
    }

    using GraphViewType = cugraph::graph_view_t<vertex_t, edge_t, false, true>;
    cugraph::edge_src_property_t<GraphViewType, vertex_t> src_inclusion_cache(*handle_);
    cugraph::edge_dst_property_t<GraphViewType, vertex_t> dst_inclusion_cache(*handle_);

    // Update rank caches with temporary inclusiong_flags
    if (multi_gpu) {
      src_inclusion_cache =
        cugraph::edge_src_property_t<GraphViewType, vertex_t>(*handle_, mg_graph_view);
      dst_inclusion_cache =
        cugraph::edge_dst_property_t<GraphViewType, vertex_t>(*handle_, mg_graph_view);
      update_edge_src_property(
        *handle_, mg_graph_view, inclusiong_flags.begin(), src_inclusion_cache);
      update_edge_dst_property(
        *handle_, mg_graph_view, inclusiong_flags.begin(), dst_inclusion_cache);
    }

    per_v_transform_reduce_outgoing_e(
      *handle_,
      mg_graph_view,
      multi_gpu ? src_inclusion_cache.view()
                : cugraph::detail::edge_major_property_view_t<vertex_t, vertex_t const*>(
                    inclusiong_flags.data()),
      multi_gpu ? dst_inclusion_cache.view()
                : cugraph::detail::edge_minor_property_view_t<vertex_t, vertex_t const*>(
                    inclusiong_flags.data(), vertex_t{0}),
      cugraph::edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto src_included, auto dst_included, auto wt) {
        return (src == dst) ? 0 : dst_included;
      },
      vertex_t{0},
      cugraph::reduce_op::plus<vertex_t>{},
      d_total_outgoing_nbrs_included_mis.begin());

    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    std::vector<vertex_t> h_total_outgoing_nbrs_included_mis(
      d_total_outgoing_nbrs_included_mis.size());
    raft::update_host(h_total_outgoing_nbrs_included_mis.data(),
                      d_total_outgoing_nbrs_included_mis.data(),
                      d_total_outgoing_nbrs_included_mis.size(),
                      handle_->get_stream());

    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    {
      for (int i = 0; i < comm_size; ++i) {
        if (comm_rank == i) {
          if (h_total_outgoing_nbrs_included_mis.size() <= 50 &&
              mg_graph_view.number_of_vertices() < 35) {
            std::cout << "Total neighbors included in mis (rank:" << comm_rank << "): ";
            std::copy(h_total_outgoing_nbrs_included_mis.begin(),
                      h_total_outgoing_nbrs_included_mis.end(),
                      std::ostream_iterator<int>(std::cout, " "));
            std::cout << std::endl;
          }

          auto vertex_first = mg_graph_view.local_vertex_partition_range_first();
          auto vertex_last  = mg_graph_view.local_vertex_partition_range_last();

          std::for_each(
            h_mis.begin(),
            h_mis.end(),
            [vertex_first, vertex_last, &h_total_outgoing_nbrs_included_mis](vertex_t v) {
              ASSERT_TRUE((v >= vertex_first) && (v < vertex_last))
                << v << " is not within vertex parition range" << std::endl;

              ASSERT_TRUE(h_total_outgoing_nbrs_included_mis[v - vertex_first] == 0)
                << v << "'s neighbor is included in MIS" << std::endl;
            });

          RAFT_CUDA_TRY(cudaDeviceSynchronize());
          handle_->get_comms().barrier();
        }
      }
    }
    //

    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    for (int r = 0; r < comm_size; ++r) {
      if (comm_rank == r) {
        std::cout << "------------comm_rank = " << r << std::endl;
        std::cout << "#V = " << mg_graph_view.number_of_vertices()
                  << " #E = " << mg_graph_view.number_of_edges() << std::endl;

        auto vertex_partition_range_offsets = mg_graph_view.vertex_partition_range_offsets();

        std::cout << "vertex_partition_range_offsets: ";
        std::copy(vertex_partition_range_offsets.begin(),
                  vertex_partition_range_offsets.end(),
                  std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;

        auto vertex_partition_range_lasts = mg_graph_view.vertex_partition_range_lasts();

        std::cout << "vertex_partition_range_lasts: ";
        std::copy(vertex_partition_range_lasts.begin(),
                  vertex_partition_range_lasts.end(),
                  std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;

        std::cout << "number_of_local_edge_partitions: "
                  << mg_graph_view.number_of_local_edge_partitions() << std::endl;

        for (int lpidx = 0; lpidx < mg_graph_view.number_of_local_edge_partitions(); lpidx++) {
          std::cout << "rank " << comm_rank << ": edge partition-" << lpidx
                    << ", src_range: " << mg_graph_view.local_edge_partition_src_range_first(lpidx)
                    << " --" << mg_graph_view.local_edge_partition_src_range_last(lpidx)
                    << " dst_range: " << mg_graph_view.local_edge_partition_dst_range_first()
                    << " --" << mg_graph_view.local_edge_partition_dst_range_last()
                    << " #edges: " << mg_graph_view.number_of_local_edge_partition_edges(lpidx)
                    << std::endl;

          std::cout << "rank " << comm_rank << ": local_edge_partition_segment_offsets: ";
          auto local_edge_partition_segment_offsets =
            *(mg_graph_view.local_edge_partition_segment_offsets(lpidx));

          std::copy(local_edge_partition_segment_offsets.begin(),
                    local_edge_partition_segment_offsets.end(),
                    std::ostream_iterator<int>(std::cout, " "));

          std::cout << std::endl;

          if (mg_graph_view.number_of_vertices() < 35) {
            std::cout << "####--edge partition ---- " << lpidx << std::endl;
            auto local_edge_partition_view = mg_graph_view.local_edge_partition_view(lpidx);

            std::cout << "DCS" << (mg_graph_view.use_dcs() ? " yes" : " no") << std::endl;

            auto offsets = local_edge_partition_view.offsets();
            auto indices = local_edge_partition_view.indices();

            raft::print_device_vector("offsets: ", offsets.data(), offsets.size(), std::cout);

            raft::print_device_vector("indices: ", indices.data(), indices.size(), std::cout);

            RAFT_CUDA_TRY(cudaDeviceSynchronize());

            std::vector<edge_t> h_offsets(offsets.size());
            raft::update_host(
              h_offsets.data(), offsets.data(), offsets.size(), handle_->get_stream());

            std::vector<vertex_t> h_indices(indices.size());
            raft::update_host(
              h_indices.data(), indices.data(), indices.size(), handle_->get_stream());
            RAFT_CUDA_TRY(cudaDeviceSynchronize());

            auto major_hypersparse_first = *(local_edge_partition_view.major_hypersparse_first());

            auto dcs_nzd_vertices = *(local_edge_partition_view.dcs_nzd_vertices());

            std::vector<vertex_t> h_dcs_nzd_vertices(dcs_nzd_vertices.size());

            std::cout << "major_hypersparse_first: " << major_hypersparse_first << std::endl;
            std::cout << "h_dcs_nzd_vertices:";
            raft::update_host(h_dcs_nzd_vertices.data(),
                              dcs_nzd_vertices.data(),
                              dcs_nzd_vertices.size(),
                              handle_->get_stream());
            RAFT_CUDA_TRY(cudaDeviceSynchronize());

            std::copy(h_dcs_nzd_vertices.begin(),
                      h_dcs_nzd_vertices.end(),
                      std::ostream_iterator<int>(std::cout, " "));

            std::cout << std::endl;

            for (int idx = 0; idx < h_offsets.size() - 1; idx++) {
              std::cout << (idx + mg_graph_view.local_edge_partition_src_range_first(lpidx))
                        << ": ";

              if (idx < (major_hypersparse_first -
                         mg_graph_view.local_edge_partition_src_range_first(lpidx))) {
                std::copy(h_indices.begin() + h_offsets[idx],
                          h_indices.begin() + h_offsets[idx + 1],
                          std::ostream_iterator<int>(std::cout, " "));
                std::cout << std::endl;
              } else {
                auto hs_idx = idx - major_hypersparse_first;

                auto src = h_dcs_nzd_vertices[hs_idx];

                std::cout << std::endl << src << ":";
                std::copy(h_indices.begin() + h_offsets[idx],
                          h_indices.begin() + h_offsets[idx + 1],
                          std::ostream_iterator<int>(std::cout, " "));
                std::cout << std::endl;
              }
            }
          }
        }
      }

      RAFT_CUDA_TRY(cudaDeviceSynchronize());
      handle_->get_comms().barrier();
    }

    //

    cugraph::graph_t<vertex_t, edge_t, false, false> sg_graph(*handle_);

    std::tie(sg_graph, std::ignore, std::ignore) = cugraph::test::mg_graph_to_sg_graph(
      *handle_,
      mg_graph_view,
      std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>>{std::nullopt},

      std::optional<raft::device_span<vertex_t const>>{std::nullopt},
      false);

    if (handle_->get_comms().get_rank() == 0 && sg_graph.view().number_of_vertices() < 16) {
      ASSERT_TRUE(mg_graph_view.number_of_vertices() == sg_graph.view().number_of_vertices());

      auto offsets = sg_graph.view().local_edge_partition_view().offsets();
      auto indices = sg_graph.view().local_edge_partition_view().indices();

      raft::print_device_vector("offsets: ", offsets.data(), offsets.size(), std::cout);

      raft::print_device_vector("indices: ", indices.data(), indices.size(), std::cout);

      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      std::vector<edge_t> h_offsets(offsets.size());
      raft::update_host(h_offsets.data(), offsets.data(), offsets.size(), handle_->get_stream());

      std::vector<vertex_t> h_indices(indices.size());
      raft::update_host(h_indices.data(), indices.data(), indices.size(), handle_->get_stream());
      RAFT_CUDA_TRY(cudaDeviceSynchronize());

      for (int idx = 0; idx < h_offsets.size() - 1; idx++) {
        std::cout << idx << ": ";
        std::copy(h_indices.begin() + h_offsets[idx],
                  h_indices.begin() + h_offsets[idx + 1],
                  std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
      }
    }

    //
    const int initial_value = 0;
    auto property_initial_value =
      cugraph::test::generate<vertex_t, result_t>::initial_value(initial_value);

    std::cout << "result type, is arithmetic? " << (std::is_arithmetic_v<result_t> ? "yes" : "no")
              << std::endl;

    if (!std::is_arithmetic_v<result_t>) {
      // std::cout << "property_initial_value: " << thrust::get<0>(property_initial_value) << " "
      //           << thrust::get<1>(property_initial_value) << std::endl;
    } else {
      std::cout << "property_initial_value: " << property_initial_value << std::endl;
    }
    // auto mg_vertex_prop = cugraph::test::generate<vertex_t, result_t>::vertex_property(
    //   *handle_, *mg_renumber_map, hash_bin_count);
    // auto mg_src_prop = cugraph::test::generate<vertex_t, result_t>::src_property(
    //   *handle_, mg_graph_view, mg_vertex_prop);
    // auto mg_dst_prop = cugraph::test::generate<vertex_t, result_t>::dst_property(
    //   *handle_, mg_graph_view, mg_vertex_prop);

    rmm::device_uvector<vertex_t> mg_vertex_prop = std::move(inclusiong_flags);
    cugraph::edge_src_property_t<GraphViewType, vertex_t> mg_src_prop =
      std::move(src_inclusion_cache);
    cugraph::edge_dst_property_t<GraphViewType, vertex_t> mg_dst_prop =
      std::move(dst_inclusion_cache);

    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    handle_->get_comms().barrier();

    for (int r = 0; r < comm_size; ++r) {
      if (comm_rank == r) {
        std::cout << "---- Graph -----" << comm_rank << std::endl;

        if (mg_graph_view.local_vertex_partition_range_size() < 35) {
          if (!std::is_arithmetic_v<result_t>) {
            //   raft::print_device_vector("mg_vertex_prop 0",
            //                             std::get<0>(mg_vertex_prop).data(),
            //                             std::get<0>(mg_vertex_prop).size(),
            //                             std::cout);

            //   raft::print_device_vector("mg_vertex_prop 1",
            //                             std::get<1>(mg_vertex_prop).data(),
            //                             std::get<1>(mg_vertex_prop).size(),
            //                             std::cout);
          } else {
            raft::print_device_vector(
              "mg_vertex_prop", inclusiong_flags.data(), inclusiong_flags.size(), std::cout);
          }

          auto key_chuk_size      = mg_graph_view.local_sorted_unique_edge_src_chunk_size();
          auto key_chuk_size_prop = mg_src_prop.view().key_chunk_size();

          if (key_chuk_size) {
            std::cout << "key_chuk_size: " << *key_chuk_size
                      << " key_chuk_size_prop:" << *key_chuk_size_prop << std::endl;
          }

          for (int local_partition_idx = 0;
               local_partition_idx < mg_graph_view.number_of_local_edge_partitions();
               local_partition_idx++) {
            std::cout << local_partition_idx << " : src range size "
                      << mg_graph_view.local_edge_partition_src_range_size(local_partition_idx)
                      << " dst range size " << mg_graph_view.local_edge_partition_dst_range_size()
                      << std::endl;

            if (key_chuk_size) {
              auto key_chunk_start_offsets =
                *(mg_graph_view.local_sorted_unique_edge_src_chunk_start_offsets(
                  local_partition_idx));

              auto keys = *(mg_graph_view.local_sorted_unique_edge_srcs(local_partition_idx));

              for (int chunk_idx = 0; chunk_idx < key_chunk_start_offsets.size() - 1; chunk_idx++) {
                auto chunk_start = keys.begin() + key_chunk_start_offsets[chunk_idx];
                auto num_elements_in_chunk =
                  key_chunk_start_offsets[chunk_idx + 1] - key_chunk_start_offsets[chunk_idx];
                raft::print_device_vector(
                  "key chunk:", chunk_start, num_elements_in_chunk, std::cout);
              }

            } else {
              raft::print_device_vector(
                "mg_src_prop  : ",
                mg_src_prop.view().value_firsts()[local_partition_idx],
                mg_graph_view.local_edge_partition_src_range_size(local_partition_idx),
                std::cout);

              raft::print_device_vector("mg_dst_prop  : ",
                                        mg_dst_prop.view().value_first(),
                                        mg_graph_view.local_edge_partition_dst_range_size(),
                                        std::cout);
            }
          }
        }
        // raft::print_device_vector("mg_src_prop: 0 ",
        //                           std::get<0>(mg_src_prop).data(),
        //                           std::get<0>(mg_src_prop).size(),
        //                           std::cout);

        // raft::print_device_vector("mg_dst_prop: 0 ",
        //                           std::get<0>(mg_dst_prop).data(),
        //                           std::get<0>(mg_dst_prop).size(),
        //                           std::cout);

        //                           raft::print_device_vector("edges  : ",
        //                           (*current_edge_weight_view).value_firsts()[0],
        //                           (*current_edge_weight_view).edge_counts()[0],
        //                           std::cout);
      }

      handle_->get_comms().barrier();
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
    }

    enum class reduction_type_t { PLUS };
    std::array<reduction_type_t, 1> reduction_types = {reduction_type_t::PLUS};

    std::vector<decltype(cugraph::allocate_dataframe_buffer<result_t>(0, rmm::cuda_stream_view{}))>
      mg_out_results{};
    mg_out_results.reserve(reduction_types.size());

    mg_out_results.push_back(cugraph::allocate_dataframe_buffer<result_t>(
      mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream()));

    per_v_transform_reduce_outgoing_e(*handle_,
                                      mg_graph_view,
                                      mg_src_prop.view(),
                                      mg_dst_prop.view(),
                                      cugraph::edge_dummy_property_t{}.view(),
                                      e_op_t<vertex_t, result_t>{},
                                      vertex_t{0},
                                      cugraph::reduce_op::plus<result_t>{},
                                      cugraph::get_dataframe_buffer_begin(mg_out_results[0]));
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_MGMaximalIndependentSet<input_usecase_t>::handle_ = nullptr;

using Tests_MGMaximalIndependentSet_File =
  Tests_MGMaximalIndependentSet<cugraph::test::File_Usecase>;
using Tests_MGMaximalIndependentSet_Rmat =
  Tests_MGMaximalIndependentSet<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_MGMaximalIndependentSet_File, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_File_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt32Int32FloatFloat)
{
  run_current_test<int32_t, int32_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt32Int64FloatFloat)
{
  run_current_test<int32_t, int64_t, float, int>(
    override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
}

// TEST_P(Tests_MGMaximalIndependentSet_Rmat, CheckInt64Int64FloatFloat)
// {
//   run_current_test<int64_t, int64_t, float, int>(
//     override_Rmat_Usecase_with_cmd_line_arguments(GetParam()));
// }

INSTANTIATE_TEST_SUITE_P(
  file_test_pass,
  Tests_MGMaximalIndependentSet_File,
  ::testing::Combine(::testing::Values(MaximalIndependentSet_Usecase{20, false},
                                       MaximalIndependentSet_Usecase{20, false}),
                     ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_MGMaximalIndependentSet_Rmat,
  ::testing::Combine(
    ::testing::Values(MaximalIndependentSet_Usecase{50, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(3, 4, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_MGMaximalIndependentSet_Rmat,
  ::testing::Combine(
    ::testing::Values(MaximalIndependentSet_Usecase{500, false},
                      MaximalIndependentSet_Usecase{500, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
