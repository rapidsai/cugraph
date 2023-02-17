# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
from timeit import default_timer as timer

import numpy as np
import pytest

import cudf
import cugraph
from scipy.optimize import linear_sum_assignment


def create_random_bipartite(v1, v2, size, dtype):

    #
    #   Create a full bipartite graph
    #
    df1 = cudf.DataFrame()
    df1["src"] = cudf.Series(range(0, v1, 1))
    df1["key"] = 1

    df2 = cudf.DataFrame()
    df2["dst"] = cudf.Series(range(v1, v1 + v2, 1))
    df2["key"] = 1

    edges = df1.merge(df2, on="key")[["src", "dst"]]
    edges = edges.sort_values(["src", "dst"]).reset_index()

    # Generate edge weights
    a = np.random.randint(1, high=size, size=(v1, v2)).astype(dtype)
    edges["weight"] = a.flatten()

    g = cugraph.Graph()
    g.from_cudf_edgelist(
        edges, source="src", destination="dst", edge_attr="weight", renumber=False
    )

    return df1["src"], g, a


SPARSE_SIZES = [[5, 5, 100], [500, 500, 10000]]
DENSE_SIZES = [[5, 100], [500, 10000]]


def setup_function():
    gc.collect()


@pytest.mark.parametrize("v1_size, v2_size, weight_limit", SPARSE_SIZES)
def test_hungarian(v1_size, v2_size, weight_limit):
    v1, g, m = create_random_bipartite(v1_size, v2_size, weight_limit, np.float64)

    start = timer()
    cugraph_cost, matching = cugraph.hungarian(g, v1)
    end = timer()

    print("cugraph time: ", (end - start))

    start = timer()
    np_matching = linear_sum_assignment(m)
    end = timer()

    print("scipy time: ", (end - start))

    scipy_cost = m[np_matching[0], np_matching[1]].sum()

    assert scipy_cost == cugraph_cost


@pytest.mark.parametrize("n, weight_limit", DENSE_SIZES)
def test_dense_hungarian(n, weight_limit):
    C = np.random.uniform(0, weight_limit, size=(n, n)).round().astype(np.float32)

    C_series = cudf.Series(C.flatten())

    start = timer()
    cugraph_cost, matching = cugraph.dense_hungarian(C_series, n, n)
    end = timer()

    print("cugraph time: ", (end - start))

    start = timer()
    np_matching = linear_sum_assignment(C)
    end = timer()

    print("scipy time: ", (end - start))

    scipy_cost = C[np_matching[0], np_matching[1]].sum()

    assert scipy_cost == cugraph_cost
