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

#include "c_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int generic_hits_test(vertex_t* h_src,
                      vertex_t* h_dst,
                      weight_t* h_wgt,
                      size_t num_vertices,
                      size_t num_edges,
                      vertex_t* h_initial_vertices,
                      weight_t* h_initial_hubs,
                      size_t num_initial_vertices,
                      weight_t* h_result_hubs,
                      weight_t* h_result_authorities,
                      bool_t store_transposed,
                      bool_t renumber,
                      bool_t normalize,
                      double epsilon,
                      size_t max_iterations)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle = NULL;
  cugraph_graph_t* p_graph            = NULL;
  cugraph_hits_result_t* p_result     = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, renumber, FALSE, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  if (h_initial_vertices == NULL) {
    ret_code = cugraph_hits(p_handle,
                            p_graph,
                            epsilon,
                            max_iterations,
                            NULL,
                            NULL,
                            normalize,
                            FALSE,
                            &p_result,
                            &ret_error);
  } else {
    cugraph_type_erased_device_array_t* initial_vertices;
    cugraph_type_erased_device_array_t* initial_hubs;
    cugraph_type_erased_device_array_view_t* initial_vertices_view;
    cugraph_type_erased_device_array_view_t* initial_hubs_view;

    ret_code = cugraph_type_erased_device_array_create(
      p_handle, num_initial_vertices, INT32, &initial_vertices, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "initial_vertices create failed.");

    ret_code = cugraph_type_erased_device_array_create(
      p_handle, num_initial_vertices, FLOAT32, &initial_hubs, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "initial_hubs create failed.");

    initial_vertices_view = cugraph_type_erased_device_array_view(initial_vertices);
    initial_hubs_view     = cugraph_type_erased_device_array_view(initial_hubs);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      p_handle, initial_vertices_view, (byte_t*)h_initial_vertices, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      p_handle, initial_hubs_view, (byte_t*)h_initial_hubs, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

    ret_code = cugraph_hits(p_handle,
                            p_graph,
                            epsilon,
                            max_iterations,
                            initial_vertices_view,
                            initial_hubs_view,
                            normalize,
                            FALSE,
                            &p_result,
                            &ret_error);
  }

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_hits failed.");

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* hubs;
  cugraph_type_erased_device_array_view_t* authorities;

  vertices                 = cugraph_hits_result_get_vertices(p_result);
  hubs                     = cugraph_hits_result_get_hubs(p_result);
  authorities              = cugraph_hits_result_get_authorities(p_result);
  double score_differences = cugraph_hits_result_get_hub_score_differences(p_result);
  size_t num_iterations    = cugraph_hits_result_get_number_of_iterations(p_result);

  vertex_t h_vertices[num_vertices];
  weight_t h_hubs[num_vertices];
  weight_t h_authorities[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code =
    cugraph_type_erased_device_array_view_copy_to_host(p_handle, (byte_t*)h_hubs, hubs, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_authorities, authorities, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result_hubs[h_vertices[i]], h_hubs[i], 0.001),
                "hubs results don't match");
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result_authorities[h_vertices[i]], h_authorities[i], 0.001),
                "authorities results don't match");
  }

  cugraph_hits_result_free(p_result);
  cugraph_sg_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_hits()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_hubs[]        = {0.347296, 0.532089, 1, 0.00003608, 0.00003608, 0};
  weight_t h_authorities[] = {0.652703, 0.879385, 0, 1, 0.347296, 0.00009136};

  double epsilon        = 0.0001;
  size_t max_iterations = 20;

  // hits wants store_transposed = TRUE
  return generic_hits_test(h_src,
                           h_dst,
                           h_wgt,
                           num_vertices,
                           num_edges,
                           NULL,
                           NULL,
                           0,
                           h_hubs,
                           h_authorities,
                           TRUE,
                           FALSE,
                           FALSE,
                           epsilon,
                           max_iterations);
}

int test_hits_with_transpose()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_hubs[]        = {0.347296, 0.532089, 1, 0.00003608, 0.00003608, 0};
  weight_t h_authorities[] = {0.652703, 0.879385, 0, 1, 0.347296, 0.00009136};

  double epsilon        = 0.0001;
  size_t max_iterations = 20;

  // Hits wants store_transposed = TRUE
  //    This call will force cugraph_hits to transpose the graph
  //    But we're passing src/dst backwards so the results will be the same
  return generic_hits_test(h_src,
                           h_dst,
                           h_wgt,
                           num_vertices,
                           num_edges,
                           NULL,
                           NULL,
                           0,
                           h_hubs,
                           h_authorities,
                           FALSE,
                           FALSE,
                           FALSE,
                           epsilon,
                           max_iterations);
}

int test_hits_with_initial()
{
  size_t num_edges        = 8;
  size_t num_vertices     = 6;
  size_t num_initial_hubs = 5;

  vertex_t h_src[]              = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]              = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[]              = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_hubs[]             = {0.347296, 0.532089, 1, 0.00000959, 0.00000959, 0};
  weight_t h_authorities[]      = {0.652704, 0.879385, 0, 1, 0.347296, 0.00002428};
  vertex_t h_initial_vertices[] = {0, 1, 2, 3, 4};
  weight_t h_initial_hubs[]     = {0.347296, 0.532089, 1, 0.00003608, 0.00003608};

  double epsilon        = 0.0001;
  size_t max_iterations = 20;

  return generic_hits_test(h_src,
                           h_dst,
                           h_wgt,
                           num_vertices,
                           num_edges,
                           h_initial_vertices,
                           h_initial_hubs,
                           num_initial_hubs,
                           h_hubs,
                           h_authorities,
                           FALSE,
                           FALSE,
                           FALSE,
                           epsilon,
                           max_iterations);
}

int test_hits_bigger()
{
  size_t num_edges    = 48;
  size_t num_vertices = 54;

  vertex_t h_src[] = {29, 45, 6,  8,  16, 45, 8,  16, 6,  38, 45, 45, 48, 45, 45, 45,
                      45, 48, 53, 45, 6,  45, 38, 45, 38, 45, 16, 45, 38, 16, 45, 45,
                      38, 6,  38, 45, 45, 45, 16, 38, 6,  45, 29, 45, 29, 6,  38, 6};
  vertex_t h_dst[] = {45, 45, 16, 45, 6,  45, 45, 16, 45, 38, 45, 6,  45, 38, 16, 45,
                      45, 45, 45, 53, 29, 16, 45, 8,  8,  16, 45, 38, 45, 6,  45, 45,
                      6,  6,  16, 38, 16, 45, 45, 6,  16, 6,  53, 16, 38, 45, 45, 16};
  weight_t h_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  weight_t h_hubs[] = {0, 0, 0, 0, 0,        0,        0.323569, 0,        0.156401, 0,        0,
                       0, 0, 0, 0, 0,        0.253312, 0,        0,        0,        0,        0,
                       0, 0, 0, 0, 0,        0,        0,        0.110617, 0,        0,        0,
                       0, 0, 0, 0, 0,        0.365733, 0,        0,        0,        0,        0,
                       0, 1, 0, 0, 0.156401, 0,        0,        0,        0,        0.0782005};
  weight_t h_authorities[] = {0, 0, 0, 0, 0, 0,        0.321874, 0,         0.123424, 0,       0,
                              0, 0, 0, 0, 0, 0.595522, 0,        0,         0,        0,       0,
                              0, 0, 0, 0, 0, 0,        0,        0.0292397, 0,        0,       0,
                              0, 0, 0, 0, 0, 0.314164, 0,        0,         0,        0,       0,
                              0, 1, 0, 0, 0, 0,        0,        0,         0,        0.100368};

  double epsilon        = 0.000001;
  size_t max_iterations = 100;

  return generic_hits_test(h_src,
                           h_dst,
                           h_wgt,
                           num_vertices,
                           num_edges,
                           NULL,
                           NULL,
                           0,
                           h_hubs,
                           h_authorities,
                           FALSE,
                           FALSE,
                           FALSE,
                           epsilon,
                           max_iterations);
}

int test_hits_bigger_unnormalized()
{
  size_t num_edges    = 48;
  size_t num_vertices = 54;

  vertex_t h_src[] = {29, 45, 6,  8,  16, 45, 8,  16, 6,  38, 45, 45, 48, 45, 45, 45,
                      45, 48, 53, 45, 6,  45, 38, 45, 38, 45, 16, 45, 38, 16, 45, 45,
                      38, 6,  38, 45, 45, 45, 16, 38, 6,  45, 29, 45, 29, 6,  38, 6};
  vertex_t h_dst[] = {45, 45, 16, 45, 6,  45, 45, 16, 45, 38, 45, 6,  45, 38, 16, 45,
                      45, 45, 45, 53, 29, 16, 45, 8,  8,  16, 45, 38, 45, 6,  45, 45,
                      6,  6,  16, 38, 16, 45, 45, 6,  16, 6,  53, 16, 38, 45, 45, 16};
  weight_t h_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  weight_t h_hubs[] = {0, 0, 0, 0, 0,        0,        0.323569, 0,        0.156401, 0,        0,
                       0, 0, 0, 0, 0,        0.253312, 0,        0,        0,        0,        0,
                       0, 0, 0, 0, 0,        0,        0,        0.110617, 0,        0,        0,
                       0, 0, 0, 0, 0,        0.365733, 0,        0,        0,        0,        0,
                       0, 1, 0, 0, 0.156401, 0,        0,        0,        0,        0.0782005};
  weight_t h_authorities[] = {0, 0, 0, 0, 0, 0,        0.321874, 0,         0.123424, 0,       0,
                              0, 0, 0, 0, 0, 0.595522, 0,        0,         0,        0,       0,
                              0, 0, 0, 0, 0, 0,        0,        0.0292397, 0,        0,       0,
                              0, 0, 0, 0, 0, 0.314164, 0,        0,         0,        0,       0,
                              0, 1, 0, 0, 0, 0,        0,        0,         0,        0.100368};

  double epsilon        = 0.000001;
  size_t max_iterations = 100;

  return generic_hits_test(h_src,
                           h_dst,
                           h_wgt,
                           num_vertices,
                           num_edges,
                           NULL,
                           NULL,
                           0,
                           h_hubs,
                           h_authorities,
                           FALSE,
                           FALSE,
                           FALSE,
                           epsilon,
                           max_iterations);
}

int test_hits_bigger_normalized()
{
  size_t num_edges    = 48;
  size_t num_vertices = 54;

  vertex_t h_src[] = {29, 45, 6,  8,  16, 45, 8,  16, 6,  38, 45, 45, 48, 45, 45, 45,
                      45, 48, 53, 45, 6,  45, 38, 45, 38, 45, 16, 45, 38, 16, 45, 45,
                      38, 6,  38, 45, 45, 45, 16, 38, 6,  45, 29, 45, 29, 6,  38, 6};
  vertex_t h_dst[] = {45, 45, 16, 45, 6,  45, 45, 16, 45, 38, 45, 6,  45, 38, 16, 45,
                      45, 45, 45, 53, 29, 16, 45, 8,  8,  16, 45, 38, 45, 6,  45, 45,
                      6,  6,  16, 38, 16, 45, 45, 6,  16, 6,  53, 16, 38, 45, 45, 16};
  weight_t h_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  weight_t h_hubs[] = {
    0, 0,         0,        0,        0, 0, 0.132381,  0, 0.0639876, 0, 0,        0,        0, 0,
    0, 0,         0.103637, 0,        0, 0, 0,         0, 0,         0, 0,        0,        0, 0,
    0, 0.0452563, 0,        0,        0, 0, 0,         0, 0,         0, 0.149631, 0,        0, 0,
    0, 0,         0,        0.409126, 0, 0, 0.0639876, 0, 0,         0, 0,        0.0319938};

  weight_t h_authorities[] = {
    0, 0,         0,        0,        0, 0, 0.129548, 0, 0.0496755, 0, 0,        0,        0, 0,
    0, 0,         0.239688, 0,        0, 0, 0,        0, 0,         0, 0,        0,        0, 0,
    0, 0.0117691, 0,        0,        0, 0, 0,        0, 0,         0, 0.126445, 0,        0, 0,
    0, 0,         0,        0.402479, 0, 0, 0,        0, 0,         0, 0,        0.0403963};

  double epsilon        = 0.000001;
  size_t max_iterations = 100;

  return generic_hits_test(h_src,
                           h_dst,
                           h_wgt,
                           num_vertices,
                           num_edges,
                           NULL,
                           NULL,
                           0,
                           h_hubs,
                           h_authorities,
                           FALSE,
                           FALSE,
                           TRUE,
                           epsilon,
                           max_iterations);
}
int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_hits);
  result |= RUN_TEST(test_hits_with_transpose);
  result |= RUN_TEST(test_hits_with_initial);
  result |= RUN_TEST(test_hits_bigger);
  result |= RUN_TEST(test_hits_bigger_normalized);
  result |= RUN_TEST(test_hits_bigger_unnormalized);
  return result;
}
