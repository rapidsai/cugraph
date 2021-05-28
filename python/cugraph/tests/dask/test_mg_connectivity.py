# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import cugraph.dask as dcg
import gc
import pytest
import cugraph
import dask_cudf
import cudf
from cugraph.dask.common.mg_utils import (is_single_gpu,
                                          setup_local_dask_cluster,
                                          teardown_local_dask_cluster)


@pytest.fixture(scope="module")
def client_connection():
    (cluster, client) = setup_local_dask_cluster(p2p=True)
    yield client
    teardown_local_dask_cluster(cluster, client)


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
def test_dask_wcc(client_connection):
    gc.collect()

    # FIXME: update this to allow dataset to be parameterized and have dataset
    # part of test param id (see other tests)
    input_data_path = r"../datasets/netscience.csv"
    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, "src", "dst", renumber=True)

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf, "src", "dst")

    expected_dist = cugraph.weakly_connected_components(g)
    result_dist = dcg.weakly_connected_components(dg)

    result_dist = result_dist.compute()
    compare_dist = expected_dist.merge(
        result_dist, on="vertex", suffixes=["_local", "_dask"]
    )

    unique_local_labels = compare_dist['labels_local'].unique()

    for label in unique_local_labels.values.tolist():
        dask_labels_df = compare_dist[compare_dist['labels_local']==label]
        dask_labels = dask_labels_df['labels_dask']
        assert (dask_labels.iloc[0] == dask_labels).all()
