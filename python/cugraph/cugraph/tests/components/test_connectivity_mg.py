# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import pytest
import cugraph.dask as dcg
import gc

# import pytest
import cugraph
import dask_cudf
import cudf

# from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.testing.utils import RAPIDS_DATASET_ROOT_DIR_PATH

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# Directed graph is not currently supported
IS_DIRECTED = [False, True]


# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_wcc(dask_client, directed):

    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH / "netscience.csv").as_posix()
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

    g = cugraph.Graph(directed=directed)
    g.from_cudf_edgelist(df, "src", "dst", renumber=True)

    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst")

    if not directed:
        expected_dist = cugraph.weakly_connected_components(g)
        result_dist = dcg.weakly_connected_components(dg)

        result_dist = result_dist.compute()
        compare_dist = expected_dist.merge(
            result_dist, on="vertex", suffixes=["_local", "_dask"]
        )

        unique_local_labels = compare_dist["labels_local"].unique()

        for label in unique_local_labels.values.tolist():
            dask_labels_df = compare_dist[compare_dist["labels_local"] == label]
            dask_labels = dask_labels_df["labels_dask"]
            assert (dask_labels.iloc[0] == dask_labels).all()
    else:
        with pytest.raises(ValueError):
            cugraph.weakly_connected_components(g)
