# Copyright (c) 2021, NVIDIA CORPORATION.
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

from cugraph.tests import utils
import cudf
import cugraph
import gc
import numpy as np
import pytest

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx

print("Networkx version : {} ".format(nx.__version__))


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Helper functions
# =============================================================================
def load_tsp(filename=None):
    gdf = cudf.read_csv(filename,
                        delim_whitespace=True,
                        skiprows=6,
                        names=["vertex", "x", "y"],
                        dtypes={"vertex": "int32",
                                "x": "float32",
                                "y": "float32"}
                        )
    gdf = gdf.dropna()
    gdf['vertex'] = gdf['vertex'].str.strip()
    gdf['vertex'] = gdf['vertex'].astype("int32")
    return gdf


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize("tsplib_file, ref_cost", utils.DATASETS_TSPLIB)
def test_traveling_salesperson(gpubenchmark, tsplib_file, ref_cost):
    pos_list = load_tsp(tsplib_file)

    cu_route, cu_cost = gpubenchmark(cugraph.traveling_salesperson,
                                     pos_list,
                                     restarts=4096)

    print("Cugraph cost: ", cu_cost)
    print("Ref cost: ", ref_cost)
    error = np.abs(cu_cost - ref_cost) / ref_cost
    print("Approximation error is: {:.2f}%".format(error * 100))
    # Check we are within 5% of TSPLIB
    assert(error * 100 < 5.)
    assert(cu_route.nunique() == pos_list.shape[0])
    assert(cu_route.shape[0] == pos_list.shape[0])
    min_val = pos_list["vertex"].min()
    max_val = pos_list["vertex"].max()
    assert(cu_route.clip(min_val, max_val).shape[0] == cu_route.shape[0])
