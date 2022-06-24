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
import cugraph.dask as dcg
import cugraph
import dask_cudf
import cudf
from cugraph.testing import utils
from cugraph.dask import uniform_neighbor_sample
import random


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Pytest fixtures
# =============================================================================
IS_DIRECTED = [True, False]

datasets = utils.DATASETS_UNDIRECTED + \
    [utils.RAPIDS_DATASET_ROOT_DIR_PATH/"email-Eu-core.csv"]

fixture_params = utils.genFixtureParamsProduct(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
    ([False, True], "with_replacement"),
    (["int32", "float32"], "indices_type")
    )


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(zip(("graph_file",
                           "directed",
                           "with_replacement",
                           "indices_type"), request.param))

    indices_type = parameters["indices_type"]

    input_data_path = parameters["graph_file"]
    directed = parameters["directed"]

    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", indices_type],
    )

    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf, source='src', destination='dst', edge_attr='value')

    parameters["MGGraph"] = dg

    # sample k vertices from the cuGraph graph
    k = random.randint(1, 10)
    srcs = dg.input_df["src"]
    dsts = dg.input_df["dst"]

    vertices = dask_cudf.concat([srcs, dsts]).drop_duplicates().compute()
    start_list = vertices.sample(k).astype("int32")

    # Generate a random fanout_vals list of length random(1, k)
    fanout_vals = [random.randint(1, k) for _ in range(random.randint(1, k))]

    # These prints are for debugging purposes since the vertices and the
    # fanout_vals are randomly sampled/chosen
    print("\nstart_list: \n", start_list)
    print("fanout_vals: ", fanout_vals)

    parameters["start_list"] = start_list
    parameters["fanout_vals"] = fanout_vals

    return parameters


def test_mg_neighborhood_sampling_simple(dask_client, input_combo):

    dg = input_combo["MGGraph"]

    input_df = dg.input_df
    result_nbr = uniform_neighbor_sample(dg,
                                         input_combo["start_list"],
                                         input_combo["fanout_vals"],
                                         input_combo["with_replacement"])

    # multi edges are dropped to easily verify that each edge in the
    # results is present in the input dataframe
    result_nbr = result_nbr.drop_duplicates()

    # FIXME: The indices are not included in the comparison because garbage
    # value are intermittently retuned. This observation is observed when
    # passing float weights
    join = result_nbr.merge(
        input_df, left_on=[*result_nbr.columns[:2]],
        right_on=[*input_df.columns[:2]])

    if len(result_nbr) != len(join):
        join2 = input_df.merge(
            result_nbr, how='right', left_on=[*input_df.columns],
            right_on=[*result_nbr.columns])
        # The left part of the datasets shows which edge is missing from the
        # right part where the left and right part are respectively the
        # uniform-neighbor-sample results and the input dataframe.
        difference = join2.sort_values([*result_nbr.columns]) \
            .compute().to_pandas().query(
                'src.isnull()', engine='python')

        invalid_edge = difference[difference.columns[:3]]
        raise Exception(f"\nThe edges below from uniform-neighbor-sample "
                        f"are invalid\n {invalid_edge}")

    # Ensure the right indices type is returned
    assert result_nbr['indices'].dtype == input_combo["indices_type"]

    sampled_vertex_result = dask_cudf.concat(
        [result_nbr["sources"], result_nbr["destinations"]]). \
        drop_duplicates().compute().reset_index(drop=True)

    sampled_vertex_result = sampled_vertex_result.to_pandas()
    start_list = input_combo["start_list"].to_pandas()

    if not set(start_list).issubset(set(sampled_vertex_result)):
        missing_vertex = set(start_list) - set(sampled_vertex_result)
        missing_vertex = list(missing_vertex)
        # compute the out-degree of the missing vertices
        out_degree = dg.out_degree(missing_vertex)

        out_degree = out_degree[out_degree.degree != 0]
        # If the missing vertices have outgoing edges, return an error
        if len(out_degree) != 0:
            missing_vertex = out_degree["vertex"].compute(). \
                to_pandas().to_list()
            raise Exception(f"vertex {missing_vertex} is missing from "
                            f"uniform neighbor sampling results")


@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_mg_neighborhood_sampling_tree(dask_client, directed):

    input_data_path = (utils.RAPIDS_DATASET_ROOT_DIR_PATH /
                       "small_tree.csv").as_posix()
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    G = cugraph.Graph(directed=directed)
    G.from_dask_cudf_edgelist(ddf, "src", "dst", "value")

    # TODO: Incomplete, include more testing for tree graph as well as
    # for larger graphs
    start_list = cudf.Series([0, 0], dtype="int32")
    fanout_vals = [4, 1, 3]
    with_replacement = True
    result_nbr = uniform_neighbor_sample(G,
                                         start_list,
                                         fanout_vals,
                                         with_replacement)

    result_nbr = result_nbr.drop_duplicates()

    # input_df != ddf if 'directed = False' because ddf will be symmetrized
    # internally.
    input_df = G.input_df
    join = result_nbr.merge(
        input_df, left_on=[*result_nbr.columns[:2]],
        right_on=[*input_df.columns[:2]])

    assert len(join) == len(result_nbr)
    # Since the validity of results have (probably) been tested at both the C++
    # and C layers, simply test that the python interface and conversions were
    # done correctly.
    assert result_nbr['sources'].dtype == "int32"
    assert result_nbr['destinations'].dtype == "int32"
    assert result_nbr['indices'].dtype == "float32"

    result_nbr_vertices = dask_cudf.concat(
        [result_nbr["sources"], result_nbr["destinations"]]). \
        drop_duplicates().compute().reset_index(drop=True)

    result_nbr_vertices = result_nbr_vertices.to_pandas()
    start_list = start_list.to_pandas()

    # The vertices in start_list must be a subsets of the vertices
    # in the result
    assert set(start_list).issubset(set(result_nbr_vertices))
