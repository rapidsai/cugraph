# Copyright (c) 2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
from itertools import product
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import pytest

import cudf
import cugraph
from cugraph.tests import utils
from scipy.optimize import linear_sum_assignment
import rmm

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx

def create_random_bipartite(v1, v2, size, dtype):
    #
    #   Create a full bipartite graph
    #
    df1 = cudf.DataFrame()
    df1['src'] = cudf.Series(range(0, v1, 1))
    df1['key'] = 1

    df2 = cudf.DataFrame()
    df2['dst'] = cudf.Series(range(v1, v1+v2, 1))
    df2['key'] = 1

    edges = df1.merge(df2, on='key')[['src', 'dst']]

    # Generate edge weights
    a = np.random.randint(1, high=size, size=(v1, v2)).astype(dtype)
    edges['weight'] = a.flatten()

    g = cugraph.Graph()
    g.from_cudf_edgelist(edges, source='src', destination='dst', edge_attr='weight', renumber=False)

    return df1['src'], g, a

SPARSE_SIZES = [ [3, 7, 100] ] #  , [100, 500, 10000]  ]
DENSE_SIZES = [ [5, 10], [20, 20] ] # , [50, 50], [100, 100], [500, 1000] ] # , [5000, 10000], [5000, 50000], [10000, 200000] ]

'''
# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('v1_size, v2_size, weight_limit', SPARSE_SIZES)
def test_hungarian(managed, pool, v1_size, v2_size, weight_limit):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    v1, g, m = create_random_bipartite(v1_size, v2_size, weight_limit, np.float)

    start = timer()
    matching = cugraph.hungarian(g, v1)
    end = timer()

    print('cugraph time: ', (end - start))

    start = timer()
    np_matching = linear_sum_assignment(m)
    end = timer()

    print('scipy time: ', (end - start))

    scipy_cost = m[np_matching[0], np_matching[1]].sum()

    # ONLY WORKS WITHOUT RENUMBERING:
    cugraph_cost = matching.merge(g.edgelist.edgelist_df,
                                  left_on=['vertices', 'assignment'],
                                  right_on=['src', 'dst'],
                                  how='left')['weights'].sum()

    print('m = ', m)
    print('scipy_cost = ', scipy_cost)
    print('cugraph_cost = ', cugraph_cost)

    cugraph_df = matching.merge(g.edgelist.edgelist_df,
                                left_on=['vertices', 'assignment'],
                                right_on=['src', 'dst'],
                                how='left')

    print('cugraph_df = \n', cugraph_df)

    # Have cases where cugraph_cost is smaller... seems like I must
    # be miscomputing this somehow

    assert(scipy_cost == cugraph_cost)
'''

# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('v_size, weight_limit', DENSE_SIZES)
def test_hungarian_dense(managed, pool, v_size, weight_limit):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    print('dense size: ', v_size)

    v1, g, m = create_random_bipartite(v_size, v_size, weight_limit, np.float)

    print('m = ', m)

    start = timer()
    cugraph_matching = cugraph.linear_sum_assignment(m)
    end = timer()

    print('cugraph time: ', (end - start))

    start = timer()
    np_matching = linear_sum_assignment(m)
    end = timer()

    print('scipy time: ', (end - start))

    scipy_cost = m[np_matching[0], np_matching[1]].sum()
    cugraph_cost = m[cugraph_matching[0].to_pandas(),
                     cugraph_matching[1].to_pandas()].sum()

    print('cugraph_matching = ', cugraph_matching)
    print('np_matching = ', np_matching)

    assert(scipy_cost == cugraph_cost)

'''
m =  [[12.  4.  5. 19.  5.  4.  7. 17.  3. 19.  1. 14.  3. 17. 10. 15.  9. 19. 14.  8.]
       [16. 13. 19.  9.  9.  6. 16.  4.  4.  4. 17.  4. 16.  4.  1. 18.  6.  7. 1.  8.]
       [10. 11.  6. 18. 18.  1. 17.  9.  6. 10.  2.  3.  2. 17. 17. 19.  9.  3. 11.  2.]
       [11.  6.  5.  7.  5.  9.  6.  2.  9. 16.  1.  2. 19. 12.  1. 12. 17.  5. 6.  4.]
       [18.  3.  3. 11.  3. 13.  1.  1. 17. 12. 14.  2. 13. 13. 18. 10. 15.  3. 9. 15.]
       [14.  4.  2. 19. 11. 12. 14. 15.  6. 19. 10.  4. 18. 14. 18. 15. 18.  5. 15. 13.]
       [ 9.  9. 10.  6. 16. 17.  4. 18.  6. 16. 14. 14.  1. 19. 15. 19.  1.  3. 14. 19.]
       [ 3. 12. 19. 14.  7. 17.  2.  4. 17. 17. 16.  4. 14.  7.  7. 18. 14. 14. 11. 13.]
       [12.  2.  1.  8. 16.  1.  3. 13.  8.  8.  1.  3. 15.  8. 13. 12. 18.  3. 19. 13.]
       [ 7. 17.  7. 14. 10.  2.  3. 16.  7. 16. 15.  5.  6. 10. 15. 10.  6.  8. 17.  2.]
       [14.  7. 14.  5.  1. 19.  9.  9.  4. 15. 16. 17. 15. 18.  6. 19. 14. 13. 13.  8.]
       [ 4. 11. 12. 12.  3. 19. 13. 11. 19. 14. 11.  8. 18. 13. 18. 12.  6.  2. 3.  3.]
       [12.  3. 14. 15.  8. 13. 18.  1. 16.  7. 16. 13.  1.  6.  7. 17.  9. 10. 12.  3.]
       [13.  4.  3. 14. 15. 14.  3.  2.  5. 15. 12.  6. 12.  4.  3.  3. 14. 17. 7. 10.]
       [19. 19. 16. 13. 17.  9.  3.  1.  3. 13. 11.  5. 12. 17.  1. 14. 17. 13. 6. 12.]
       [13.  6. 12. 19. 12.  8.  9. 17. 10. 18. 19.  9. 17. 12. 17. 11. 12. 15. 15. 12.]
       [19. 13.  7. 10. 17.  7.  6.  9.  3. 16.  1.  3. 11. 14.  6. 12.  2. 13. 15.  5.]
       [11. 12. 17. 19. 16. 18.  7.  6.  4. 18. 19.  8.  8. 12. 13.  6.  5.  8. 5.  6.]
       [17. 11.  3. 16.  2. 14. 17.  1.  3. 16. 14.  7.  1. 12. 12. 16. 14. 15. 1.  3.]
       [14.  8. 19.  2. 11.  9.  4.  8.  4.  8.  3. 19.  1. 17. 16.  1.  9. 19. 9. 16.]]
df =      assignment  vertices
0           10         0
1            9         1
2            5         2
3           14         3
4            6         4
5           11         5
6            3         6
7            0         7
8            2         8
9           19         9
10           4        10
11          17        11
12          12        12
13          13        13
14           7        14
15           1        15
16          16        16
17           8        17
18          18        18
19          15        19
F

47 ???
'''


'''
dense size:  5
m =  [[7. 6. 3. 6. 4.]
      [6. 9. 2. 9. 9.]
      [7. 5. 3. 8. 9.]
      [5. 3. 8. 3. 1.]
      [6. 2. 1. 1. 3.]]
Dense Matrix:
--------
      7      6      3      6      4
      6      9      2      9      9
      7      5      3      8      9
      5      3      8      3      1
      6      2      1      1      3
------
after subtracting rows and columns
Dense Matrix:
--------
      0      2      0      3      1
      0      6      0      7      7
      0      1      0      5      6
      0      1      7      2      0
      1      0      0      0      2
------
starting loop, num_assigned = 0, max_assigned = 5
*** inner loop, num_assigned = 0, max_assigned = 5, local_assigned = 1
assigned rows:
Vector:
--------
 5 5 5 5 5
------
assigned cols:
Vector:
--------
 5 5 5 5 5
------
zero count col:
Vector:
--------
 4 1 4 1 1
------
assigned rows:
Vector:
--------
 5 4 5 5 3
------
assigned cols:
Vector:
--------
 5 5 5 4 1
------
num_assigned = 2, local_assigned = 2
*** inner loop, num_assigned = 2, max_assigned = 5, local_assigned = 2
assigned rows:
Vector:
--------
 5 4 5 5 3
------
assigned cols:
Vector:
--------
 5 5 5 4 1
------
zero count col:
Vector:
--------
 3 0 3 0 0
------
assigned rows:
Vector:
--------
 5 4 5 5 3
------
assigned cols:
Vector:
--------
 5 5 5 4 1
------
num_assigned = 2, local_assigned = 0
computing h
Dense Matrix:
--------
      0      2      0      3      1
      0      6      0      7      7
      0      1      0      5      6
      0      1      7      2      0
      1      0      0      0      2
------
d_covered_row
Vector:
--------
 true true true true true
------
d_covered_col
Vector:
--------
 false false false false false
------
h = 1.79769e+308, num_assigned = 2, max_assigned = 5
adding noise
Dense Matrix:
--------
      0     12      2     13     11
      3     16      2     17     17
      2     11      3     15     16
      1     11     17     12      3
     11      0      0      4     12
------
starting loop, num_assigned = 2, max_assigned = 5
*** inner loop, num_assigned = 2, max_assigned = 5, local_assigned = 1
assigned rows:
Vector:
--------
 0 4 1 5 3
------
assigned cols:
Vector:
--------
 0 2 5 4 1
------
zero count col:
Vector:
--------
 0 0 0 0 0
------
assigned rows:
Vector:
--------
 0 4 1 5 3
------
assigned cols:
Vector:
--------
 0 2 5 4 1
------
num_assigned = 4, local_assigned = 2
*** inner loop, num_assigned = 4, max_assigned = 5, local_assigned = 2
assigned rows:
Vector:
--------
 0 4 1 5 3
------
assigned cols:
Vector:
--------
 0 2 5 4 1
------
zero count col:
Vector:
--------
 0 0 0 0 0
------
assigned rows:
Vector:
--------
 0 4 1 5 3
------
assigned cols:
Vector:
--------
 0 2 5 4 1
------
num_assigned = 4, local_assigned = 0
computing h
Dense Matrix:
--------
      0     12      2      9      9
      1     14      0     11     13
      0      9      1      9     12
      0     10     16      7      0
     11      0      0      0     10
------
d_covered_row
Vector:
--------
 false false false false true
------
d_covered_col
Vector:
--------
 true false true false true
------
h = 7, num_assigned = 4, max_assigned = 5
after applying h
Dense Matrix:
--------
      0      5      2      2      9
      1      7      0      4     13
      0      2      1      2     12
      0      3     16      0      0
     18      0      7      0     17
------
starting loop, num_assigned = 0, max_assigned = 5
*** inner loop, num_assigned = 0, max_assigned = 5, local_assigned = 1
assigned rows:
Vector:
--------
 0 5 1 5 5
------
assigned cols:
Vector:
--------
 0 2 5 5 5
------
zero count col:
Vector:
--------
 0 1 0 2 1
------
assigned rows:
Vector:
--------
 0 4 1 5 3
------
assigned cols:
Vector:
--------
 0 2 5 4 1
------
num_assigned = 4, local_assigned = 4
*** inner loop, num_assigned = 4, max_assigned = 5, local_assigned = 4
assigned rows:
Vector:
--------
 0 4 1 5 3
------
assigned cols:
Vector:
--------
 0 2 5 4 1
------
zero count col:
Vector:
--------
 0 0 0 0 0
------
assigned rows:
Vector:
--------
 0 4 1 5 3
------
assigned cols:
Vector:
--------
 0 2 5 4 1
------
num_assigned = 4, local_assigned = 0
computing h
Dense Matrix:
--------
      0      5      2      2      9
      1      7      0      4     13
      0      2      1      2     12
      0      3     16      0      0
     18      0      7      0     17
------
d_covered_row
Vector:
--------
 false false false true true
------
d_covered_col
Vector:
--------
 true false true false false
------
h = 2, num_assigned = 4, max_assigned = 5
after applying h
Dense Matrix:
--------
      0      3      2      0      7
      1      5      0      2     11
      0      0      1      0     10
      2      3     18      0      0
     20      0      9      0     17
------
starting loop, num_assigned = 0, max_assigned = 5
*** inner loop, num_assigned = 0, max_assigned = 5, local_assigned = 1
assigned rows:
Vector:
--------
 5 5 1 5 5
------
assigned cols:
Vector:
--------
 5 2 5 5 5
------
zero count col:
Vector:
--------
 2 2 0 4 1
------
assigned rows:
Vector:
--------
 5 5 1 5 3
------
assigned cols:
Vector:
--------
 5 2 5 4 5
------
num_assigned = 2, local_assigned = 2
*** inner loop, num_assigned = 2, max_assigned = 5, local_assigned = 2
assigned rows:
Vector:
--------
 5 5 1 5 3
------
assigned cols:
Vector:
--------
 5 2 5 4 5
------
zero count col:
Vector:
--------
 2 2 0 3 0
------
assigned rows:
Vector:
--------
 5 5 1 5 3
------
assigned cols:
Vector:
--------
 5 2 5 4 5
------
num_assigned = 2, local_assigned = 0
computing h
Dense Matrix:
--------
      0      3      2      0      7
      1      5      0      2     11
      0      0      1      0     10
      2      3     18      0      0
     20      0      9      0     17
------
d_covered_row
Vector:
--------
 true false true true true
------
d_covered_col
Vector:
--------
 false false true false false
------
h = 1, num_assigned = 2, max_assigned = 5
after applying h
Dense Matrix:
--------
      0      3      3      0      7
      0      4      0      1     10
      0      0      2      0     10
      2      3     19      0      0
     20      0     10      0     17
------
starting loop, num_assigned = 0, max_assigned = 5
*** inner loop, num_assigned = 0, max_assigned = 5, local_assigned = 1
assigned rows:
Vector:
--------
 5 5 5 5 5
------
assigned cols:
Vector:
--------
 5 5 5 5 5
------
zero count col:
Vector:
--------
 3 2 1 4 1
------
assigned rows:
Vector:
--------
 5 5 1 5 3
------
assigned cols:
Vector:
--------
 5 2 5 4 5
------
num_assigned = 2, local_assigned = 2
*** inner loop, num_assigned = 2, max_assigned = 5, local_assigned = 2
assigned rows:
Vector:
--------
 5 5 1 5 3
------
assigned cols:
Vector:
--------
 5 2 5 4 5
------
zero count col:
Vector:
--------
 2 2 0 3 0
------
assigned rows:
Vector:
--------
 5 5 1 5 3
------
assigned cols:
Vector:
--------
 5 2 5 4 5
------
num_assigned = 2, local_assigned = 0
computing h
Dense Matrix:
--------
      0      3      3      0      7
      0      4      0      1     10
      0      0      2      0     10
      2      3     19      0      0
     20      0     10      0     17
------
d_covered_row
Vector:
--------
 true true true true true
------
d_covered_col
Vector:
--------
 false false false false false
------
h = 1.79769e+308, num_assigned = 2, max_assigned = 5
adding noise
Dense Matrix:
--------
      3     13     13      2     17
      2     14      4     11     20
      1      2     12      4     20
     12     13     29      2      3
     30      1     20      3     27
------
starting loop, num_assigned = 2, max_assigned = 5
*** inner loop, num_assigned = 2, max_assigned = 5, local_assigned = 1
assigned rows:
Vector:
--------
 2 4 1 0 3
------
assigned cols:
Vector:
--------
 3 2 0 4 1
------
zero count col:
Vector:
--------
 0 0 0 0 0
------
assigned rows:
Vector:
--------
 2 4 1 0 3
------
assigned cols:
Vector:
--------
 3 2 0 4 1
------
num_assigned = 5, local_assigned = 3
df =     assignment  vertices
0           3         0
1           2         1
2           0         2
3           4         3
4           1         4
cugraph time:  0.03873412497341633
scipy time:  6.368383765220642e-05
cugraph_matching =  (0    0
1    1
2    2
3    3
4    4
Name: vertices, dtype: int64, 0    3
1    2
2    0
3    4
4    1
Name: assignment, dtype: int32)
np_matching =  (array([0, 1, 2, 3, 4]), array([0, 2, 1, 4, 3]))
F
'''
