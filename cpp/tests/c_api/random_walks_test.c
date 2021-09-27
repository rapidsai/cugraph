/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cugraph_c/cugraph_api.h>

/* sample graph:
    0 --(.1)--> 1 --(1.1)--> 4
   /|\       /\ |            |
    |       /   |            |
   (5.1) (3.1)(2.1)        (3.2)
    |   /       |            |
    | /        \|/          \|/
    2 --(4.1)-->3 --(7.2)--> 5
*/

/* array of compressed paths,
   using 1-based indexing for vertices,
   to avoid confusion between, for example,
   `012` and `12`, which result in same number*/
#define N_PATHS 30
static int32_t c_ps_array[N_PATHS] = {1,    2,     3,     4,   5,   6,    12,   124, 125, 1246,
                                      1256, 24,    25,    246, 256, 31,   32,   34,  312, 3124,
                                      3125, 31246, 31256, 324, 325, 3246, 3256, 346, 46,  56};

/* linear search of `value` inside `p_cmprsd_path[max_num_paths]`*/
bool_t is_one_of(int32_t value, int32_t* p_cmprsd_path, int max_num_paths)
{
  int i = 0;
  for (; i < max_num_paths; ++i)
    if (value == p_cmprsd_path[i]) return 1;

  return 0;
}

/* check on host if all obtained paths are possible paths */
bool_t host_check_paths(int32_t* p_path_v, int32_t* p_path_sz, int num_paths)
{
  int i            = 0;
  int count_passed = 0;

  for (; i < num_paths; ++i) {
    int32_t crt_path_sz          = p_path_sz[i];
    int path_it                  = 0;
    int32_t crt_path_accumulator = 0;
    bool_t flag_passed           = 0;

    for (; path_it < crt_path_sz; ++path_it) {
      crt_path_accumulator = *p_path_v + 10 * crt_path_accumulator;
      ++p_path_v; /* iterate p_path_v*/
    }

    flag_passed = is_one_of(crt_path_accumulator, c_ps_array, N_PATHS);
    if (flag_passed) ++count_passed;
  }

  return (count_passed == num_paths);
}
int test_random_walks_1() { return 0; }

int test_random_walks_2() { return 0; }

int test_random_walks_3() { return 0; }

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_random_walks_1);
  result |= RUN_TEST(test_random_walks_2);
  result |= RUN_TEST(test_random_walks_3);
  return result;
}
