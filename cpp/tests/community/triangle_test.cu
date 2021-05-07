/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */
#include <utilities/base_fixture.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>

#include <rmm/thrust_rmm_allocator.h>

TEST(triangle, dolphin)
{
  std::vector<int> off_h = {0,   6,   14,  18,  21,  22,  26,  32,  37,  43,  50,  55,  56,
                            57,  65,  77,  84,  90,  99,  106, 110, 119, 125, 126, 129, 135,
                            138, 141, 146, 151, 160, 165, 166, 169, 179, 184, 185, 192, 203,
                            211, 213, 221, 226, 232, 239, 243, 254, 256, 262, 263, 265, 272,
                            282, 286, 288, 295, 297, 299, 308, 309, 314, 315, 318};
  std::vector<int> ind_h = {
    10, 14, 15, 40, 42, 47, 17, 19, 26, 27, 28, 36, 41, 54, 10, 42, 44, 61, 8,  14, 59, 51, 9,
    13, 56, 57, 9,  13, 17, 54, 56, 57, 19, 27, 30, 40, 54, 3,  20, 28, 37, 45, 59, 5,  6,  13,
    17, 32, 41, 57, 0,  2,  29, 42, 47, 51, 33, 5,  6,  9,  17, 32, 41, 54, 57, 0,  3,  16, 24,
    33, 34, 37, 38, 40, 43, 50, 52, 0,  18, 24, 40, 45, 55, 59, 14, 20, 33, 37, 38, 50, 1,  6,
    9,  13, 22, 25, 27, 31, 57, 15, 20, 21, 24, 29, 45, 51, 1,  7,  30, 54, 8,  16, 18, 28, 36,
    38, 44, 47, 50, 18, 29, 33, 37, 45, 51, 17, 36, 45, 51, 14, 15, 18, 29, 45, 51, 17, 26, 27,
    1,  25, 27, 1,  7,  17, 25, 26, 1,  8,  20, 30, 47, 10, 18, 21, 24, 35, 43, 45, 51, 52, 7,
    19, 28, 42, 47, 17, 9,  13, 60, 12, 14, 16, 21, 34, 37, 38, 40, 43, 50, 14, 33, 37, 44, 49,
    29, 1,  20, 23, 37, 39, 40, 59, 8,  14, 16, 21, 33, 34, 36, 40, 43, 45, 61, 14, 16, 20, 33,
    43, 44, 52, 58, 36, 57, 0,  7,  14, 15, 33, 36, 37, 52, 1,  9,  13, 54, 57, 0,  2,  10, 30,
    47, 50, 14, 29, 33, 37, 38, 46, 53, 2,  20, 34, 38, 8,  15, 18, 21, 23, 24, 29, 37, 50, 51,
    59, 43, 49, 0,  10, 20, 28, 30, 42, 57, 34, 46, 14, 16, 20, 33, 42, 45, 51, 4,  11, 18, 21,
    23, 24, 29, 45, 50, 55, 14, 29, 38, 40, 43, 61, 1,  6,  7,  13, 19, 41, 57, 15, 51, 5,  6,
    5,  6,  9,  13, 17, 39, 41, 48, 54, 38, 3,  8,  15, 36, 45, 32, 2,  37, 53};

  std::vector<float> w_h(ind_h.size(), float{1.0});

  int num_verts = off_h.size() - 1;
  int num_edges = ind_h.size();

  uint64_t expected{285};

  rmm::device_vector<int> offsets_v(off_h);
  rmm::device_vector<int> indices_v(ind_h);
  rmm::device_vector<float> weights_v(w_h);

  cugraph::GraphCSRView<int, int, float> graph_csr(
    offsets_v.data().get(), indices_v.data().get(), weights_v.data().get(), num_verts, num_edges);

  uint64_t count{0};

  try {
    count = cugraph::triangle::triangle_count<int32_t, int32_t, float>(graph_csr);
  } catch (std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }

  ASSERT_EQ(count, expected);
}

CUGRAPH_TEST_PROGRAM_MAIN()
