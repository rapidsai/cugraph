/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/array.h>
#include <cugraph_c/error.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/resource_handle.h>
#include <cugraph_c/traversal_algorithms.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const int32_t INF = INT32_MAX;
static const int64_t INF64 = INT64_MAX;

static int fail_with_error(const char* what, cugraph_error_t* error)
{
  fprintf(stderr, "%s failed", what);
  if (error != NULL) { fprintf(stderr, ": %s", cugraph_error_message(error)); }
  fprintf(stderr, "\n");
  return 1;
}

static int expect_success(cugraph_error_code_t code, cugraph_error_t* error, const char* what)
{
  if (code != CUGRAPH_SUCCESS) { return fail_with_error(what, error); }
  return 0;
}

static int expect_error(cugraph_error_code_t code,
                        cugraph_error_code_t expected,
                        cugraph_error_t* error,
                        const char* what)
{
  if (code != expected) {
    fprintf(stderr, "%s returned %d, expected %d", what, (int)code, (int)expected);
    if (error != NULL) { fprintf(stderr, ": %s", cugraph_error_message(error)); }
    fprintf(stderr, "\n");
    return 1;
  }
  cugraph_error_free(error);
  return 0;
}

static cugraph_type_erased_device_array_t* make_device_i32(const cugraph_resource_handle_t* handle,
                                                           const int32_t* host,
                                                           size_t count,
                                                           const char* name)
{
  cugraph_error_t* error = NULL;
  cugraph_type_erased_device_array_t* array = NULL;
  if (expect_success(cugraph_type_erased_device_array_create(handle, count, INT32, &array, &error),
                     error,
                     name)) {
    exit(1);
  }
  cugraph_type_erased_device_array_view_t* view = cugraph_type_erased_device_array_view(array);
  if (expect_success(cugraph_type_erased_device_array_view_copy_from_host(
                       handle, view, (const byte_t*)host, &error),
                     error,
                     name)) {
    exit(1);
  }
  cugraph_type_erased_device_array_view_free(view);
  return array;
}

static cugraph_type_erased_device_array_t* make_device_i64(const cugraph_resource_handle_t* handle,
                                                           const int64_t* host,
                                                           size_t count,
                                                           const char* name)
{
  cugraph_error_t* error = NULL;
  cugraph_type_erased_device_array_t* array = NULL;
  if (expect_success(cugraph_type_erased_device_array_create(handle, count, INT64, &array, &error),
                     error,
                     name)) {
    exit(1);
  }
  cugraph_type_erased_device_array_view_t* view = cugraph_type_erased_device_array_view(array);
  if (expect_success(cugraph_type_erased_device_array_view_copy_from_host(
                       handle, view, (const byte_t*)host, &error),
                     error,
                     name)) {
    exit(1);
  }
  cugraph_type_erased_device_array_view_free(view);
  return array;
}

static int fetch_distances_by_vertex(const cugraph_resource_handle_t* handle,
                                     cugraph_paths_result_t* result,
                                     int32_t* out,
                                     size_t n,
                                     const char* name)
{
  cugraph_error_t* error                         = NULL;
  cugraph_type_erased_device_array_view_t* verts = cugraph_paths_result_get_vertices(result);
  cugraph_type_erased_device_array_view_t* dists = cugraph_paths_result_get_distances(result);
  if (cugraph_type_erased_device_array_view_size(dists) != n) {
    fprintf(stderr, "%s returned wrong distance size\n", name);
    return 1;
  }
  size_t vertex_count = cugraph_type_erased_device_array_view_size(verts);
  if (vertex_count == 0) {
    if (expect_success(cugraph_type_erased_device_array_view_copy_to_host(
                         handle, (byte_t*)out, dists, &error),
                       error,
                       name)) {
      return 1;
    }
    return 0;
  }
  if (vertex_count != n) {
    fprintf(stderr, "%s returned wrong vertex size\n", name);
    return 1;
  }

  int32_t* tmp_vertices  = (int32_t*)malloc(n * sizeof(int32_t));
  int32_t* tmp_distances = (int32_t*)malloc(n * sizeof(int32_t));
  if (tmp_vertices == NULL || tmp_distances == NULL) {
    fprintf(stderr, "%s allocation failed\n", name);
    free(tmp_vertices);
    free(tmp_distances);
    return 1;
  }

  if (expect_success(cugraph_type_erased_device_array_view_copy_to_host(
                       handle, (byte_t*)tmp_vertices, verts, &error),
                     error,
                     name)) {
    free(tmp_vertices);
    free(tmp_distances);
    return 1;
  }
  if (expect_success(cugraph_type_erased_device_array_view_copy_to_host(
                       handle, (byte_t*)tmp_distances, dists, &error),
                     error,
                     name)) {
    free(tmp_vertices);
    free(tmp_distances);
    return 1;
  }

  for (size_t i = 0; i < n; ++i) { out[i] = INT32_MIN; }
  for (size_t i = 0; i < n; ++i) {
    if (tmp_vertices[i] < 0 || tmp_vertices[i] >= (int32_t)n) {
      fprintf(stderr, "%s returned out-of-range vertex id %d\n", name, tmp_vertices[i]);
      free(tmp_vertices);
      free(tmp_distances);
      return 1;
    }
    out[tmp_vertices[i]] = tmp_distances[i];
  }

  free(tmp_vertices);
  free(tmp_distances);
  return 0;
}

static int fetch_distances_by_vertex64(const cugraph_resource_handle_t* handle,
                                       cugraph_paths_result_t* result,
                                       int64_t* out,
                                       size_t n,
                                       const char* name)
{
  cugraph_error_t* error                         = NULL;
  cugraph_type_erased_device_array_view_t* verts = cugraph_paths_result_get_vertices(result);
  cugraph_type_erased_device_array_view_t* dists = cugraph_paths_result_get_distances(result);
  if (cugraph_type_erased_device_array_view_size(dists) != n) {
    fprintf(stderr, "%s returned wrong distance size\n", name);
    return 1;
  }
  size_t vertex_count = cugraph_type_erased_device_array_view_size(verts);
  if (vertex_count == 0) {
    if (expect_success(cugraph_type_erased_device_array_view_copy_to_host(
                         handle, (byte_t*)out, dists, &error),
                       error,
                       name)) {
      return 1;
    }
    return 0;
  }
  if (vertex_count != n) {
    fprintf(stderr, "%s returned wrong vertex size\n", name);
    return 1;
  }

  int64_t* tmp_vertices  = (int64_t*)malloc(n * sizeof(int64_t));
  int64_t* tmp_distances = (int64_t*)malloc(n * sizeof(int64_t));
  if (tmp_vertices == NULL || tmp_distances == NULL) {
    fprintf(stderr, "%s allocation failed\n", name);
    free(tmp_vertices);
    free(tmp_distances);
    return 1;
  }

  if (expect_success(cugraph_type_erased_device_array_view_copy_to_host(
                       handle, (byte_t*)tmp_vertices, verts, &error),
                     error,
                     name)) {
    free(tmp_vertices);
    free(tmp_distances);
    return 1;
  }
  if (expect_success(cugraph_type_erased_device_array_view_copy_to_host(
                       handle, (byte_t*)tmp_distances, dists, &error),
                     error,
                     name)) {
    free(tmp_vertices);
    free(tmp_distances);
    return 1;
  }

  for (size_t i = 0; i < n; ++i) { out[i] = INT64_MIN; }
  for (size_t i = 0; i < n; ++i) {
    if (tmp_vertices[i] < 0 || tmp_vertices[i] >= (int64_t)n) {
      fprintf(stderr, "%s returned out-of-range vertex id %ld\n", name, (long)tmp_vertices[i]);
      free(tmp_vertices);
      free(tmp_distances);
      return 1;
    }
    out[tmp_vertices[i]] = tmp_distances[i];
  }

  free(tmp_vertices);
  free(tmp_distances);
  return 0;
}

static int fetch_distances_view(const cugraph_resource_handle_t* handle,
                                cugraph_type_erased_device_array_view_t* dists,
                                int32_t* out,
                                size_t n,
                                const char* name)
{
  cugraph_error_t* error = NULL;
  if (cugraph_type_erased_device_array_view_size(dists) != n) {
    fprintf(stderr, "%s returned wrong distance size\n", name);
    return 1;
  }
  return expect_success(
    cugraph_type_erased_device_array_view_copy_to_host(handle, (byte_t*)out, dists, &error),
    error,
    name);
}

static int check_distances(const int32_t* got,
                           const int32_t* expected,
                           size_t n,
                           const char* label)
{
  for (size_t i = 0; i < n; ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr,
              "%s mismatch at %zu: got %d expected %d\n",
              label,
              i,
              got[i],
              expected[i]);
      return 1;
    }
  }
  return 0;
}

static int check_distances64(const int64_t* got,
                             const int64_t* expected,
                             size_t n,
                             const char* label)
{
  for (size_t i = 0; i < n; ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr,
              "%s mismatch at %zu: got %ld expected %ld\n",
              label,
              i,
              (long)got[i],
              (long)expected[i]);
      return 1;
    }
  }
  return 0;
}

static int create_i32_graph(const cugraph_resource_handle_t* handle,
                            bool_t store_transposed,
                            bool_t renumber,
                            cugraph_graph_t** graph)
{
  const size_t num_vertices = 6;
  const size_t num_edges    = 8;
  int32_t vertices[]        = {0, 1, 2, 3, 4, 5};
  int32_t src[]             = {0, 1, 1, 2, 2, 2, 3, 4};
  int32_t dst[]             = {1, 3, 4, 0, 1, 3, 5, 5};

  cugraph_error_t* error = NULL;
  cugraph_type_erased_device_array_t* vertices_array =
    make_device_i32(handle, vertices, num_vertices, "vertices");
  cugraph_type_erased_device_array_t* src_array = make_device_i32(handle, src, num_edges, "src");
  cugraph_type_erased_device_array_t* dst_array = make_device_i32(handle, dst, num_edges, "dst");
  cugraph_type_erased_device_array_view_t* vertices_view =
    cugraph_type_erased_device_array_view(vertices_array);
  cugraph_type_erased_device_array_view_t* src_view =
    cugraph_type_erased_device_array_view(src_array);
  cugraph_type_erased_device_array_view_t* dst_view =
    cugraph_type_erased_device_array_view(dst_array);

  cugraph_graph_properties_t properties;
  properties.is_symmetric  = FALSE;
  properties.is_multigraph = FALSE;

  int result = expect_success(cugraph_graph_create_sg(handle,
                                                      &properties,
                                                      vertices_view,
                                                      src_view,
                                                      dst_view,
                                                      NULL,
                                                      NULL,
                                                      NULL,
                                                      store_transposed,
                                                      renumber,
                                                      FALSE,
                                                      FALSE,
                                                      FALSE,
                                                      FALSE,
                                                      graph,
                                                      &error),
                              error,
                              "graph create");

  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_view_free(vertices_view);
  cugraph_type_erased_device_array_free(dst_array);
  cugraph_type_erased_device_array_free(src_array);
  cugraph_type_erased_device_array_free(vertices_array);
  cugraph_error_free(error);
  return result;
}

static int run_dawn_bfs_case(bool_t renumber, size_t depth_limit, const int32_t* expected)
{
  const size_t num_vertices = 6;
  int32_t source[]          = {0};
  int32_t distances[6];
  int result = 0;

  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(NULL);
  if (handle == NULL) { return 1; }

  cugraph_graph_t* graph = NULL;
  result |= create_i32_graph(handle, FALSE, renumber, &graph);

  cugraph_type_erased_device_array_t* source_array = make_device_i32(handle, source, 1, "source");
  cugraph_type_erased_device_array_view_t* source_view =
    cugraph_type_erased_device_array_view(source_array);

  cugraph_error_t* error       = NULL;
  cugraph_paths_result_t* path = NULL;
  result |= expect_success(
    cugraph_dawn_bfs(handle, graph, source_view, depth_limit, FALSE, &path, &error),
    error,
    "cugraph_dawn_bfs");
  if (result == 0) {
    result |= fetch_distances_by_vertex(handle, path, distances, num_vertices, "dawn distances");
    result |= check_distances(distances, expected, num_vertices, "dawn distances");
  }

  if (path != NULL) { cugraph_paths_result_free(path); }
  cugraph_type_erased_device_array_view_free(source_view);
  cugraph_type_erased_device_array_free(source_array);
  cugraph_graph_free(graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(error);
  return result;
}

static int test_dawn_bfs_non_renumbered(void)
{
  int32_t expected[] = {0, 1, INF, 2, 2, 3};
  return run_dawn_bfs_case(FALSE, SIZE_MAX, expected);
}

static int test_dawn_bfs_depth_limit(void)
{
  int32_t expected[] = {0, 1, INF, INF, INF, INF};
  return run_dawn_bfs_case(FALSE, 1, expected);
}

static int test_dawn_bfs_distances_api(void)
{
  const size_t num_vertices = 6;
  int32_t sources[]         = {0, 1};
  int32_t distances[6];
  int32_t expected[] = {0, 0, INF, 1, 1, 2};
  int result = 0;

  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(NULL);
  if (handle == NULL) { return 1; }

  cugraph_graph_t* graph = NULL;
  result |= create_i32_graph(handle, FALSE, FALSE, &graph);

  cugraph_type_erased_device_array_t* source_array = make_device_i32(handle, sources, 2, "sources");
  cugraph_type_erased_device_array_view_t* source_view =
    cugraph_type_erased_device_array_view(source_array);

  cugraph_error_t* error = NULL;
  cugraph_type_erased_device_array_t* distance_array = NULL;
  result |= expect_success(cugraph_type_erased_device_array_create(
                             handle, num_vertices, INT32, &distance_array, &error),
                           error,
                           "distance array");
  cugraph_type_erased_device_array_view_t* distance_view =
    cugraph_type_erased_device_array_view(distance_array);

  result |= expect_success(cugraph_dawn_bfs_distances(
                             handle, graph, source_view, SIZE_MAX, FALSE, distance_view, &error),
                           error,
                           "cugraph_dawn_bfs_distances");
  if (result == 0) {
    result |= fetch_distances_view(handle, distance_view, distances, num_vertices, "distances api");
    result |= check_distances(distances, expected, num_vertices, "distances api");
  }

  cugraph_type_erased_device_array_view_free(distance_view);
  cugraph_type_erased_device_array_free(distance_array);
  cugraph_type_erased_device_array_view_free(source_view);
  cugraph_type_erased_device_array_free(source_array);
  cugraph_graph_free(graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(error);
  return result;
}

static int test_dawn_bfs_renumbered_rejected(void)
{
  const size_t num_vertices = 6;
  int32_t source[]          = {0};
  int result = 0;

  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(NULL);
  if (handle == NULL) { return 1; }

  cugraph_graph_t* graph = NULL;
  result |= create_i32_graph(handle, FALSE, TRUE, &graph);

  cugraph_type_erased_device_array_t* source_array = make_device_i32(handle, source, 1, "source");
  cugraph_type_erased_device_array_view_t* source_view =
    cugraph_type_erased_device_array_view(source_array);

  cugraph_error_t* error       = NULL;
  cugraph_paths_result_t* path = NULL;
  result |= expect_error(
    cugraph_dawn_bfs(handle, graph, source_view, SIZE_MAX, FALSE, &path, &error),
    CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
    error,
    "renumbered cugraph_dawn_bfs");

  cugraph_type_erased_device_array_t* distance_array = NULL;
  result |= expect_success(cugraph_type_erased_device_array_create(
                             handle, num_vertices, INT32, &distance_array, &error),
                           error,
                           "distance array");
  cugraph_type_erased_device_array_view_t* distance_view =
    cugraph_type_erased_device_array_view(distance_array);
  result |= expect_error(cugraph_dawn_bfs_distances(
                           handle, graph, source_view, SIZE_MAX, FALSE, distance_view, &error),
                         CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                         error,
                         "renumbered cugraph_dawn_bfs_distances");

  if (path != NULL) { cugraph_paths_result_free(path); }
  cugraph_type_erased_device_array_view_free(distance_view);
  cugraph_type_erased_device_array_free(distance_array);
  cugraph_type_erased_device_array_view_free(source_view);
  cugraph_type_erased_device_array_free(source_array);
  cugraph_graph_free(graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(error);
  return result;
}

static int test_dawn_bfs_store_transposed_rejected(void)
{
  int32_t source[] = {0};
  int result       = 0;

  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(NULL);
  if (handle == NULL) { return 1; }

  cugraph_graph_t* graph = NULL;
  result |= create_i32_graph(handle, TRUE, FALSE, &graph);

  cugraph_type_erased_device_array_t* source_array = make_device_i32(handle, source, 1, "source");
  cugraph_type_erased_device_array_view_t* source_view =
    cugraph_type_erased_device_array_view(source_array);

  cugraph_error_t* error       = NULL;
  cugraph_paths_result_t* path = NULL;
  result |= expect_error(
    cugraph_dawn_bfs(handle, graph, source_view, SIZE_MAX, FALSE, &path, &error),
    CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
    error,
    "store_transposed cugraph_dawn_bfs");

  if (path != NULL) { cugraph_paths_result_free(path); }
  cugraph_type_erased_device_array_view_free(source_view);
  cugraph_type_erased_device_array_free(source_array);
  cugraph_graph_free(graph);
  cugraph_free_resource_handle(handle);
  return result;
}

static int test_dawn_bfs_multi_source(void)
{
  int32_t sources[] = {0, 1};
  int32_t distances[6];
  int result = 0;

  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(NULL);
  if (handle == NULL) { return 1; }

  cugraph_graph_t* graph = NULL;
  result |= create_i32_graph(handle, FALSE, FALSE, &graph);

  cugraph_type_erased_device_array_t* source_array = make_device_i32(handle, sources, 2, "sources");
  cugraph_type_erased_device_array_view_t* source_view =
    cugraph_type_erased_device_array_view(source_array);

  cugraph_error_t* error       = NULL;
  cugraph_paths_result_t* path = NULL;
  result |= expect_success(
    cugraph_dawn_bfs(handle, graph, source_view, SIZE_MAX, FALSE, &path, &error),
    error,
    "multi-source cugraph_dawn_bfs");
  if (result == 0) {
    int32_t expected[] = {0, 0, INF, 1, 1, 2};
    result |= fetch_distances_by_vertex(handle, path, distances, 6, "multi-source distances");
    result |= check_distances(distances, expected, 6, "multi-source distances");
  }

  if (path != NULL) { cugraph_paths_result_free(path); }
  cugraph_type_erased_device_array_view_free(source_view);
  cugraph_type_erased_device_array_free(source_array);
  cugraph_graph_free(graph);
  cugraph_free_resource_handle(handle);
  return result;
}

static int test_dawn_bfs_v64_e64(void)
{
  const size_t num_vertices = 6;
  const size_t num_edges    = 8;
  int64_t vertices[]        = {0, 1, 2, 3, 4, 5};
  int64_t src[]             = {0, 1, 1, 2, 2, 2, 3, 4};
  int64_t dst[]             = {1, 3, 4, 0, 1, 3, 5, 5};
  int64_t source[]          = {0};
  int64_t distances[6];
  int result                = 0;

  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(NULL);
  if (handle == NULL) { return 1; }

  cugraph_type_erased_device_array_t* vertices_array =
    make_device_i64(handle, vertices, num_vertices, "vertices64");
  cugraph_type_erased_device_array_t* src_array = make_device_i64(handle, src, num_edges, "src64");
  cugraph_type_erased_device_array_t* dst_array = make_device_i64(handle, dst, num_edges, "dst64");
  cugraph_type_erased_device_array_t* source_array = make_device_i64(handle, source, 1, "source64");
  cugraph_type_erased_device_array_view_t* vertices_view =
    cugraph_type_erased_device_array_view(vertices_array);
  cugraph_type_erased_device_array_view_t* src_view =
    cugraph_type_erased_device_array_view(src_array);
  cugraph_type_erased_device_array_view_t* dst_view =
    cugraph_type_erased_device_array_view(dst_array);
  cugraph_type_erased_device_array_view_t* source_view =
    cugraph_type_erased_device_array_view(source_array);

  cugraph_graph_properties_t properties;
  properties.is_symmetric  = FALSE;
  properties.is_multigraph = FALSE;

  cugraph_error_t* error = NULL;
  cugraph_graph_t* graph = NULL;
  result |= expect_success(cugraph_graph_create_sg(handle,
                                                   &properties,
                                                   vertices_view,
                                                   src_view,
                                                   dst_view,
                                                   NULL,
                                                   NULL,
                                                   NULL,
                                                   FALSE,
                                                   FALSE,
                                                   FALSE,
                                                   FALSE,
                                                   FALSE,
                                                   FALSE,
                                                   &graph,
                                                   &error),
                           error,
                           "graph64 create");

  cugraph_paths_result_t* path = NULL;
  result |= expect_success(cugraph_dawn_bfs(handle, graph, source_view, SIZE_MAX, FALSE, &path, &error),
                           error,
                           "v64/e64 cugraph_dawn_bfs");
  if (result == 0) {
    int64_t expected[] = {0, 1, INF64, 2, 2, 3};
    result |= fetch_distances_by_vertex64(handle, path, distances, num_vertices, "v64/e64 distances");
    result |= check_distances64(distances, expected, num_vertices, "v64/e64 distances");
  }

  if (path != NULL) { cugraph_paths_result_free(path); }
  cugraph_graph_free(graph);
  cugraph_type_erased_device_array_view_free(source_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_view_free(vertices_view);
  cugraph_type_erased_device_array_free(source_array);
  cugraph_type_erased_device_array_free(dst_array);
  cugraph_type_erased_device_array_free(src_array);
  cugraph_type_erased_device_array_free(vertices_array);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(error);
  return result;
}

static int run_test(int (*test)(void), const char* name)
{
  int result = test();
  printf("%s %s\n", result == 0 ? "PASS" : "FAIL", name);
  return result;
}

int main(int argc, char** argv)
{
  (void)argc;
  (void)argv;
  int result = 0;
  result |= run_test(test_dawn_bfs_non_renumbered, "test_dawn_bfs_non_renumbered");
  result |= run_test(test_dawn_bfs_depth_limit, "test_dawn_bfs_depth_limit");
  result |= run_test(test_dawn_bfs_distances_api, "test_dawn_bfs_distances_api");
  result |= run_test(test_dawn_bfs_renumbered_rejected, "test_dawn_bfs_renumbered_rejected");
  result |= run_test(test_dawn_bfs_store_transposed_rejected,
                     "test_dawn_bfs_store_transposed_rejected");
  result |= run_test(test_dawn_bfs_multi_source, "test_dawn_bfs_multi_source");
  result |= run_test(test_dawn_bfs_v64_e64, "test_dawn_bfs_v64_e64");
  return result;
}
