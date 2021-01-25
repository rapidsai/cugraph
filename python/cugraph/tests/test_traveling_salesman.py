# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import pytest

import cugraph
from cugraph.tests import utils
import rmm
import cudf
import time
import numpy as np

TSPLIB_REF = "../datasets/ref/tsp/tsplib.csv"

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


def load_tsp(filename=None):
    gdf = cudf.read_csv(filename,
                        delim_whitespace=True,
                        skiprows=6,
                        names=["vertex", "x", "y"],
                        dtypes={"vertex" : "int32",
                            "x": "float32",
                            "y": "float32"}
                        )
    gdf = gdf.dropna()
    gdf['vertex'] = gdf['vertex'].str.strip()
    gdf['vertex'] = gdf['vertex'].astype("int32")
    return gdf

@pytest.mark.parametrize("tsplib_file", utils.DATASETS_TSPLIB)
def test_traveling_salesman_route(tsplib_file):
    gc.collect()
    pos_list = load_tsp(tsplib_file)
    # cugraph
    t1 = time.time()
    cu_route, cu_cost = cugraph.traveling_salesman(pos_list,
                                                   restarts=4096)
    t2 = time.time() - t1
    # Route goes through each vertex once
    cu_route.nunique == pos_list.shape[0]


@pytest.mark.parametrize("tsplib_file", utils.DATASETS_TSPLIB)
def test_traveling_salesman(tsplib_file):
    gc.collect()
    pos_list = load_tsp(tsplib_file)
    ref_gdf = cudf.read_csv(TSPLIB_REF, sep=",", names=["file", "cost"])
    # cugraph
    t1 = time.time()
    cu_route, cu_cost = cugraph.traveling_salesman(pos_list,
                                                   restarts=4096)
    t2 = time.time() - t1
    print("Cugraph time : " + str(t2))

    tsp_instance = tsplib_file.split('/')[-1].split('.')[0]
    ref_cost = int(ref_gdf.loc[ref_gdf["file"] == tsp_instance, 'cost'].values[0][0])
    print("Cugraph cost: ", cu_cost)
    print("Ref cost: ", ref_cost)

    error = np.abs(cu_cost - ref_cost) / ref_cost
    print("Approximation error is: {:.2f}%".format(error * 100))
    # Check we are within 5% of TSPLIB
    assert(error * 100 < 5.)
