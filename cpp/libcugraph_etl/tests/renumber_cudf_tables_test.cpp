/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cugraph_etl/functions.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

struct vertex_name {
  std::string first;
  std::string second;
};

// Each row is one vertex, identified by the pair of strings. Source frequencies differ so the
// renumber order is not an input order, and one destination vertex never appears as a source.
std::vector<vertex_name> const src_vertices = {
  {"alice", "person"},
  {"alice", "person"},
  {"bob", "person"},
  {"dave", "person"},
};

std::vector<vertex_name> const dst_vertices = {
  {"bob", "person"},
  {"carol", "person"},
  {"alice", "person"},
  {"carol", "place"},
};

std::unique_ptr<cudf::column> make_strings_column(std::vector<std::string> const& strings,
                                                  cuda::stream_ref stream)
{
  std::vector<cudf::size_type> offsets;
  offsets.reserve(strings.size() + 1);
  offsets.push_back(0);
  std::string chars;
  for (auto const& value : strings) {
    chars.append(value);
    offsets.push_back(static_cast<cudf::size_type>(chars.size()));
  }

  rmm::device_uvector<cudf::size_type> offsets_device(offsets.size(), stream);
  if (!offsets.empty()) {
    RAFT_CUDA_TRY(cudaMemcpyAsync(offsets_device.data(),
                                  offsets.data(),
                                  offsets.size() * sizeof(cudf::size_type),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
  }
  auto const offsets_size = static_cast<cudf::size_type>(offsets_device.size());
  auto offsets_column =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                   offsets_size,
                                   offsets_device.release(),
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream),
                                   0);

  rmm::device_buffer chars_buffer(chars.data(), chars.size(), stream);
  // Both copies are asynchronous. The host sources must stay alive until they finish.
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  return cudf::make_strings_column(
    static_cast<cudf::size_type>(strings.size()),
    std::move(offsets_column),
    std::move(chars_buffer),
    0,
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream));
}

std::unique_ptr<cudf::table> make_vertex_table(std::vector<vertex_name> const& vertices,
                                               cuda::stream_ref stream)
{
  std::vector<std::string> first;
  std::vector<std::string> second;
  first.reserve(vertices.size());
  second.reserve(vertices.size());
  for (auto const& vertex : vertices) {
    first.push_back(vertex.first);
    second.push_back(vertex.second);
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(make_strings_column(first, stream));
  columns.push_back(make_strings_column(second, stream));
  return std::make_unique<cudf::table>(std::move(columns));
}

template <typename vertex_t>
std::vector<vertex_t> copy_vertices(cudf::column_view const& column, cudaStream_t stream)
{
  std::vector<vertex_t> host_vertices(static_cast<size_t>(column.size()));
  if (!host_vertices.empty()) {
    RAFT_CUDA_TRY(cudaMemcpyAsync(host_vertices.data(),
                                  column.data<vertex_t>(),
                                  host_vertices.size() * sizeof(vertex_t),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }
  return host_vertices;
}

template <typename vertex_t>
void record_endpoint_ids(std::vector<vertex_name> const& vertices,
                         std::vector<vertex_t> const& ids,
                         std::map<std::pair<std::string, std::string>, vertex_t>& id_by_name,
                         std::set<vertex_t>& used_ids)
{
  ASSERT_EQ(ids.size(), vertices.size());
  for (size_t i = 0; i < vertices.size(); ++i) {
    std::pair<std::string, std::string> const name{vertices[i].first, vertices[i].second};
    auto const found = id_by_name.find(name);
    if (found == id_by_name.end()) {
      id_by_name.emplace(name, ids[i]);
    } else {
      // The same vertex must keep the id assigned on its first occurrence.
      EXPECT_EQ(ids[i], found->second);
    }
    used_ids.insert(ids[i]);
  }
}

template <typename vertex_t>
void test_renumber_cudf_tables(cudf::type_id dtype)
{
  raft::handle_t handle{};
  auto const stream = handle.get_stream();

  auto src_table = make_vertex_table(src_vertices, stream);
  auto dst_table = make_vertex_table(dst_vertices, stream);

  auto [src_ids, dst_ids, renumber_map] =
    cugraph::etl::renumber_cudf_tables(handle, src_table->view(), dst_table->view(), dtype);
  handle.sync_stream();

  ASSERT_NE(src_ids, nullptr);
  ASSERT_NE(dst_ids, nullptr);
  ASSERT_NE(renumber_map, nullptr);
  ASSERT_EQ(src_ids->type().id(), dtype);
  ASSERT_EQ(dst_ids->type().id(), dtype);
  ASSERT_EQ(src_ids->size(), static_cast<cudf::size_type>(src_vertices.size()));
  ASSERT_EQ(dst_ids->size(), static_cast<cudf::size_type>(dst_vertices.size()));
  ASSERT_EQ(src_ids->null_count(), 0);
  ASSERT_EQ(dst_ids->null_count(), 0);
  ASSERT_EQ(renumber_map->num_columns(), 2);

  std::set<std::pair<std::string, std::string>> expected_names;
  for (auto const& vertex : src_vertices) {
    expected_names.emplace(vertex.first, vertex.second);
  }
  for (auto const& vertex : dst_vertices) {
    expected_names.emplace(vertex.first, vertex.second);
  }
  ASSERT_EQ(renumber_map->num_rows(), static_cast<cudf::size_type>(expected_names.size()));

  auto const cuda_stream = stream.get();
  // Read the ids through the column's declared dtype. Packed INT32 values in an INT64 column
  // do not come back as the dense ids 0 .. num_vertices-1.
  std::map<std::pair<std::string, std::string>, vertex_t> id_by_name;
  std::set<vertex_t> used_ids;
  record_endpoint_ids<vertex_t>(
    src_vertices, copy_vertices<vertex_t>(src_ids->view(), cuda_stream), id_by_name, used_ids);
  record_endpoint_ids<vertex_t>(
    dst_vertices, copy_vertices<vertex_t>(dst_ids->view(), cuda_stream), id_by_name, used_ids);

  ASSERT_EQ(id_by_name.size(), expected_names.size());
  ASSERT_EQ(used_ids.size(), expected_names.size());
  EXPECT_EQ(*used_ids.begin(), vertex_t{0});
  EXPECT_EQ(*used_ids.rbegin(), static_cast<vertex_t>(expected_names.size() - 1));
}

}  // namespace

TEST(RenumberCudfTables, Int32) { test_renumber_cudf_tables<int32_t>(cudf::type_id::INT32); }

TEST(RenumberCudfTables, Int64) { test_renumber_cudf_tables<int64_t>(cudf::type_id::INT64); }
