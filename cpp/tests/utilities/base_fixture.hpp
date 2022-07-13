/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <utilities/cxxopts.hpp>
#include <utilities/test_graphs.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <optional>

namespace cugraph {
namespace test {

// FIXME: The BaseFixture class is not used in any tests. This file is only needed for the
// CUGRAPH_TEST_PROGRAM_MAIN macro and the code that it calls, so consider removing the BaseFixture
// class and renaming this file, or moving CUGRAPH_TEST_PROGRAM_MAIN to the test_utilities.hpp file
// and removing this file completely.

/**
 * @brief Base test fixture class from which all libcugraph tests should inherit.
 *
 * Example:
 * ```
 * class MyTestFixture : public cugraph::test::BaseFixture {};
 * ```
 **/
class BaseFixture : public ::testing::Test {
  rmm::mr::device_memory_resource* _mr{rmm::mr::get_current_device_resource()};

 public:
  /**
   * @brief Returns pointer to `device_memory_resource` that should be used for all tests inheriting
   *from this fixture
   **/
  rmm::mr::device_memory_resource* mr() { return _mr; }
};

/// MR factory functions
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }

inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda());
}

inline auto make_binning()
{
  auto pool = make_pool();
  // Add a fixed_size_memory_resource for bins of size 256, 512, 1024, 2048 and 4096KiB
  // Larger allocations will use the pool resource
  auto mr = rmm::mr::make_owning_wrapper<rmm::mr::binning_memory_resource>(pool, 18, 22);
  return mr;
}

/**
 * @brief Creates a memory resource for the unit test environment given the name of the allocation
 * mode.
 *
 * The returned resource instance must be kept alive for the duration of the tests. Attaching the
 * resource to a TestEnvironment causes issues since the environment objects are not destroyed until
 * after the runtime is shutdown.
 *
 * @throw cugraph::logic_error if the `allocation_mode` is unsupported.
 *
 * @param allocation_mode String identifies which resource type.
 *        Accepted types are "pool", "cuda", and "managed" only.
 * @return Memory resource instance
 */
inline std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(
  std::string const& allocation_mode)
{
  if (allocation_mode == "binning") return make_binning();
  if (allocation_mode == "cuda") return make_cuda();
  if (allocation_mode == "pool") return make_pool();
  if (allocation_mode == "managed") return make_managed();
  CUGRAPH_FAIL("Invalid RMM allocation mode");
}

// these variables are updated by command line arguments
static bool g_perf{false};
static std::optional<size_t> g_rmat_scale{std::nullopt};
static std::optional<size_t> g_rmat_edge_factor{std::nullopt};
static std::optional<std::string> g_test_file_name{std::nullopt};

inline Rmat_Usecase override_Rmat_Usecase_with_cmd_line_arguments(Rmat_Usecase usecase)
{
  if (g_rmat_scale) { usecase.set_scale(*g_rmat_scale); }
  if (g_rmat_edge_factor) { usecase.set_edge_factor(*g_rmat_edge_factor); }
  return usecase;
}

inline File_Usecase override_File_Usecase_with_cmd_line_arguments(File_Usecase usecase)
{
  if (g_test_file_name) { usecase.set_filename(*g_test_file_name); }
  return usecase;
}

template <typename T>
inline std::tuple<T, Rmat_Usecase> override_Rmat_Usecase_with_cmd_line_arguments(
  std::tuple<T, Rmat_Usecase> const& param)
{
  return std::make_tuple(std::get<0>(param),
                         override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

template <typename T>
inline std::tuple<T, File_Usecase> override_File_Usecase_with_cmd_line_arguments(
  std::tuple<T, File_Usecase> const& param)
{
  return std::make_tuple(std::get<0>(param),
                         override_File_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

}  // namespace test
}  // namespace cugraph

/**
 * @brief Parses the cuGraph test command line options.
 *
 * Currently supports 'rmm_mode', 'perf', 'rmat_scale', and 'rmat_edge_factor'.
 * 'rmm_mode` string paramater sets the rmm allocation mode. The default value of the parameter is
 * 'pool'.
 * `perf` boolean parameter enables performance measurements. The default value of the
 * parameter is 'false' (if this option is not provided).
 * 'rmat_scale' integer parameter overrides the hardcoded R-mat scale if provided.
 * 'rmat_edge_factor' integer parameter overrides the hardcoded R-mat edge factor if provided.
 *
 * 'perf', 'rmat_scale', 'rmat_edge_factor' are added mainly to support C++ benchmarking (to
 * override the input graph size through command line interface).
 *
 * Example:
 * ```
 * ./tests/PAGERANK_TEST --gtest_filter=rmat_benchmark_test/Tests_PageRank_Rmat.CheckInt32Int32*
 * --rmm_mode=pool --perf --rmat_scale=25 --rmat_edge_factor=16
 * ```
 *
 * @return Parsing results in the form of cxxopts::ParseResult
 */
inline auto parse_test_options(int argc, char** argv)
{
  try {
    cxxopts::Options options(argv[0], " - cuGraph tests command line options");
    options.allow_unrecognised_options().add_options()(
      "rmm_mode", "RMM allocation mode", cxxopts::value<std::string>()->default_value("pool"))(
      "perf", "enalbe performance measurements", cxxopts::value<bool>()->default_value("false"))(
      "rmat_scale", "override the hardcoded R-mat scale", cxxopts::value<size_t>())(
      "rmat_edge_factor", "override the hardcoded R-mat edge factor", cxxopts::value<size_t>())(
      "test_file_name", "override the hardcoded test filename", cxxopts::value<std::string>());

    return options.parse(argc, argv);
  } catch (const cxxopts::OptionException& e) {
    CUGRAPH_FAIL("Error parsing command line options");
  }
}

/**
 * @brief Macro that defines main function for gtest programs that use rmm
 *
 * Should be included in every test program that uses rmm allocators since it maintains the lifespan
 * of the rmm default memory resource. This `main` function is a wrapper around the google test
 * generated `main`, maintaining the original functionality. In addition, this custom `main`
 * function parses the command line to customize test behavior, like the allocation mode used for
 * creating the default memory resource.
 */
#define CUGRAPH_TEST_PROGRAM_MAIN()                                                     \
  int main(int argc, char** argv)                                                       \
  {                                                                                     \
    ::testing::InitGoogleTest(&argc, argv);                                             \
    auto const cmd_opts = parse_test_options(argc, argv);                               \
    auto const rmm_mode = cmd_opts["rmm_mode"].as<std::string>();                       \
    auto resource       = cugraph::test::create_memory_resource(rmm_mode);              \
    rmm::mr::set_current_device_resource(resource.get());                               \
    cugraph::test::g_perf = cmd_opts["perf"].as<bool>();                                \
    cugraph::test::g_rmat_scale =                                                       \
      (cmd_opts.count("rmat_scale") > 0)                                                \
        ? std::make_optional<size_t>(cmd_opts["rmat_scale"].as<size_t>())               \
        : std::nullopt;                                                                 \
    cugraph::test::g_rmat_edge_factor =                                                 \
      (cmd_opts.count("rmat_edge_factor") > 0)                                          \
        ? std::make_optional<size_t>(cmd_opts["rmat_edge_factor"].as<size_t>())         \
        : std::nullopt;                                                                 \
    cugraph::test::g_test_file_name =                                                   \
      (cmd_opts.count("test_file_name") > 0)                                            \
        ? std::make_optional<std::string>(cmd_opts["test_file_name"].as<std::string>()) \
        : std::nullopt;                                                                 \
    return RUN_ALL_TESTS();                                                             \
  }

#define CUGRAPH_MG_TEST_PROGRAM_MAIN()                                                  \
  int main(int argc, char** argv)                                                       \
  {                                                                                     \
    MPI_TRY(MPI_Init(&argc, &argv));                                                    \
    int comm_rank{};                                                                    \
    int comm_size{};                                                                    \
    MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));                                 \
    MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));                                 \
    int num_gpus_per_node{};                                                            \
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));                              \
    RAFT_CUDA_TRY(cudaSetDevice(comm_rank % num_gpus_per_node));                        \
    ::testing::InitGoogleTest(&argc, argv);                                             \
    auto const cmd_opts = parse_test_options(argc, argv);                               \
    auto const rmm_mode = cmd_opts["rmm_mode"].as<std::string>();                       \
    auto resource       = cugraph::test::create_memory_resource(rmm_mode);              \
    rmm::mr::set_current_device_resource(resource.get());                               \
    cugraph::test::g_perf = cmd_opts["perf"].as<bool>();                                \
    cugraph::test::g_rmat_scale =                                                       \
      (cmd_opts.count("rmat_scale") > 0)                                                \
        ? std::make_optional<size_t>(cmd_opts["rmat_scale"].as<size_t>())               \
        : std::nullopt;                                                                 \
    cugraph::test::g_rmat_edge_factor =                                                 \
      (cmd_opts.count("rmat_edge_factor") > 0)                                          \
        ? std::make_optional<size_t>(cmd_opts["rmat_edge_factor"].as<size_t>())         \
        : std::nullopt;                                                                 \
    cugraph::test::g_test_file_name =                                                   \
      (cmd_opts.count("test_file_name") > 0)                                            \
        ? std::make_optional<std::string>(cmd_opts["test_file_name"].as<std::string>()) \
        : std::nullopt;                                                                 \
    auto ret = RUN_ALL_TESTS();                                                         \
    MPI_TRY(MPI_Finalize());                                                            \
    return ret;                                                                         \
  }
