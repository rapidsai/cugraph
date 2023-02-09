# Copyright (c) 2022, NVIDIA CORPORATION.
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
import pandas as pd
import dask_cudf
from pylibcugraph.testing.utils import gen_fixture_params_product

import cugraph
from cugraph.testing import utils


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def test_version():
    cugraph.__version__


def compare(ddf1, ddf2, src_col_name, dst_col_name, val_col_name):
    #
    #  We will do comparison computations by using dataframe
    #  merge functions (essentially doing fast joins).
    #  Check to see if all pairs in the original data frame
    #  still exist in the new data frame.  If we join (merge)
    #  the data frames where (src1[i]=src2[i]) and (dst1[i]=dst2[i])
    #  then we should get exactly the same number of entries in
    #  the data frame if we did not lose any data.
    #

    ddf1 = ddf1.add_suffix("_x")
    ddf2 = ddf2.add_suffix("_y")

    if not isinstance(src_col_name, list) and not isinstance(dst_col_name, list):
        src_col_name = [src_col_name]
        dst_col_name = [dst_col_name]

    # Column names for ddf1
    src_col_name1 = [f"{src}_x" for src in src_col_name]
    dst_col_name1 = [f"{dst}_x" for dst in dst_col_name]
    col_names1 = src_col_name1 + dst_col_name1

    # Column names for ddf2
    src_col_name2 = [f"{src}_y" for src in src_col_name]
    dst_col_name2 = [f"{dst}_y" for dst in dst_col_name]
    col_names2 = src_col_name2 + dst_col_name2

    if val_col_name is not None:
        val_col_name = [val_col_name]
        val_col_name1 = [f"{val}_x" for val in val_col_name]
        val_col_name2 = [f"{val}_y" for val in val_col_name]
        col_names1 += val_col_name1
        col_names2 += val_col_name2
    #
    #  Now check the symmetrized edges are present.  If the original
    #  data contains (u,v), we want to make sure that (v,u) is present
    #  in the new data frame.
    #
    #  We can accomplish this by doing the join (merge) where
    #  (src1[i] = dst2[i]) and (dst1[i] = src2[i]), and verifying
    #  that we get exactly the same number of entries in the data frame.
    #
    join = ddf1.merge(ddf2, left_on=[*col_names1], right_on=[*col_names2])

    if len(ddf1) != len(join):
        # The code below is for debugging purposes only. It will print
        # edges in the original dataframe that are missing from the symmetrize
        # dataframe
        join2 = ddf1.merge(
            ddf2, how="left", left_on=[*col_names1], right_on=[*col_names2]
        )
        # FIXME: Didn't find a cudf alternative for the function below
        pd.set_option("display.max_rows", 500)
        print(
            "join2 = \n",
            join2.sort_values([*col_names1])
            .compute()
            .to_pandas()
            .query(f"{src_col_name[0]}_y.isnull()", engine="python"),
        )

    assert len(ddf1) == len(join)

    #
    #  Finally, let's check (in both directions) backwards.
    #  We want to make sure that no edges were created in
    #  the symmetrize logic that didn't already exist in one
    #  direction or the other.  This is a bit more complicated.
    #
    #  The complication here is that the original data could,
    #  for some edge (u,v) ALREADY contain the edge (v,u).  The
    #  symmetrized graph will not duplicate any edges, so the edge
    #  (u,v) will only be present once.  So we can't simply check
    #  counts of ddf2 joined with ddf1.
    #
    #  join1 will contain the join (merge) of ddf2 to ddf1 in the
    #        forward direction
    #  join2 will contain the join (merge) of ddf2 to ddf1 in the
    #        reverse direction
    #
    #  Finally, we'll do an outer join of join1 and join2, which
    #  will combine any (u,v)/(v,u) pairs that might exist into
    #  a joined row while keeping any (u,v) pairs that don't exist
    #  in both data frames as single rows.  This gives us a data frame
    #  with the same number of rows as the symmetrized data.
    #

    swap_columns = dst_col_name1 + src_col_name1
    if val_col_name is not None:
        swap_columns += val_col_name1

    join1 = ddf2.merge(ddf1, left_on=[*col_names2], right_on=[*col_names1])
    join2 = ddf2.merge(ddf1, left_on=[*col_names2], right_on=[*swap_columns])

    # Ensure join2["weight_*"] and join1["weight"] are of the same type.
    # Failing to do that can trigger ddf to return a warning if the two ddf
    # being merge are of dofferent types
    join2 = join2.astype(join1.dtypes.to_dict())

    joinM = join1.merge(join2, how="outer", on=[*ddf2.columns])

    assert len(ddf2) == len(joinM)

    #
    #  Note, we don't need to check the reverse values... we checked
    #  them in both directions earlier.
    #


input_data_path = [
    utils.RAPIDS_DATASET_ROOT_DIR_PATH / "karate-asymmetric.csv"
] + utils.DATASETS_UNDIRECTED
datasets = [pytest.param(d.as_posix()) for d in input_data_path]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    ([True, False], "edgevals"),
    ([True, False], "multi_columns"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    return dict(zip(("graph_file", "edgevals", "multi_columns"), request.param))


@pytest.fixture(scope="module")
def read_datasets(input_combo):
    """
    This fixture reads the datasets and returns a dictionary containing all
    input params required to run the symmetrize function
    """

    graph_file = input_combo["graph_file"]
    edgevals = input_combo["edgevals"]
    multi_columns = input_combo["multi_columns"]

    ddf = utils.read_dask_cudf_csv_file(graph_file)

    src_col_name = "src"
    dst_col_name = "dst"
    val_col_name = None

    if edgevals:
        val_col_name = "weight"

    if multi_columns:
        # Generate multicolumn from the ddf
        ddf = ddf.rename(columns={"src": "src_0", "dst": "dst_0"})
        ddf["src_1"] = ddf["src_0"] + 100
        ddf["dst_1"] = ddf["dst_0"] + 100

        src_col_name = ["src_0", "src_1"]
        dst_col_name = ["dst_0", "dst_1"]

    input_combo["ddf"] = ddf
    input_combo["src_col_name"] = src_col_name
    input_combo["dst_col_name"] = dst_col_name
    input_combo["val_col_name"] = val_col_name

    return input_combo


# =============================================================================
# Tests
# =============================================================================
# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
def test_mg_symmetrize(dask_client, read_datasets):

    ddf = read_datasets["ddf"]
    src_col_name = read_datasets["src_col_name"]
    dst_col_name = read_datasets["dst_col_name"]
    val_col_name = read_datasets["val_col_name"]

    if val_col_name is not None:
        sym_src, sym_dst, sym_val = cugraph.symmetrize(
            ddf, src_col_name, dst_col_name, val_col_name
        )
    else:
        if not isinstance(src_col_name, list):
            vertex_col_names = [src_col_name, dst_col_name]
        else:
            vertex_col_names = src_col_name + dst_col_name
        ddf = ddf[[*vertex_col_names]]
        sym_src, sym_dst = cugraph.symmetrize(ddf, src_col_name, dst_col_name)

    # create a dask DataFrame from the dask Series
    if isinstance(sym_src, dask_cudf.Series):
        ddf2 = sym_src.to_frame()
        ddf2 = ddf2.rename(columns={sym_src.name: "src"})
        ddf2["dst"] = sym_dst
    else:
        ddf2 = dask_cudf.concat([sym_src, sym_dst], axis=1)

    if val_col_name is not None:
        ddf2["weight"] = sym_val

    compare(ddf, ddf2, src_col_name, dst_col_name, val_col_name)


# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
def test_mg_symmetrize_df(dask_client, read_datasets):
    ddf = read_datasets["ddf"]
    src_col_name = read_datasets["src_col_name"]
    dst_col_name = read_datasets["dst_col_name"]
    val_col_name = read_datasets["val_col_name"]

    sym_ddf = cugraph.symmetrize_ddf(ddf, src_col_name, dst_col_name, val_col_name)

    compare(ddf, sym_ddf, src_col_name, dst_col_name, val_col_name)
