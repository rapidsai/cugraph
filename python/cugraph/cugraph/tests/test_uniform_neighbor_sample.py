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
import random

import pytest
import cudf
from pylibcugraph.testing.utils import gen_fixture_params_product

import cugraph
from cugraph import uniform_neighbor_sample
from cugraph.experimental.datasets import DATASETS_UNDIRECTED, email_Eu_core, small_tree


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Pytest fixtures
# =============================================================================
IS_DIRECTED = [True, False]

datasets = DATASETS_UNDIRECTED + [email_Eu_core]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    (IS_DIRECTED, "directed"),
    ([False, True], "with_replacement"),
    (["int32", "float32"], "indices_type"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(
        zip(
            ("graph_file", "directed", "with_replacement", "indices_type"),
            request.param,
        )
    )

    indices_type = parameters["indices_type"]

    input_data_path = parameters["graph_file"].get_path()
    directed = parameters["directed"]

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", indices_type],
    )

    G = cugraph.Graph(directed=directed)
    G.from_cudf_edgelist(
        df, source="src", destination="dst", edge_attr="value", legacy_renum_only=True
    )

    parameters["Graph"] = G

    # sample k vertices from the cuGraph graph
    k = random.randint(1, 3)
    srcs = G.view_edge_list()["src"]
    dsts = G.view_edge_list()["dst"]

    vertices = cudf.concat([srcs, dsts]).drop_duplicates()

    start_list = vertices.sample(k).astype("int32")

    # Generate a random fanout_vals list of length random(1, k)
    fanout_vals = [random.randint(1, k) for _ in range(random.randint(1, k))]

    # These prints are for debugging purposes since the vertices and
    # the fanout_vals are randomly sampled/chosen
    print("\nstart_list: \n", start_list)
    print("fanout_vals: ", fanout_vals)

    parameters["start_list"] = start_list
    parameters["fanout_vals"] = fanout_vals

    return parameters


@pytest.fixture(scope="module")
def simple_unweighted_input_expected_output(request):
    """
    Fixture for providing the input for a uniform_neighbor_sample test using a
    small/simple unweighted graph and the corresponding expected output.
    """
    test_data = {}

    df = cudf.DataFrame(
        {"src": [0, 1, 2, 2, 0, 1, 4, 4], "dst": [3, 2, 1, 4, 1, 3, 1, 2]}
    )

    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source="src", destination="dst")
    test_data["Graph"] = G
    test_data["start_list"] = cudf.Series([0], dtype="int32")
    test_data["fanout_vals"] = [-1]
    test_data["with_replacement"] = True

    test_data["expected_src"] = [0, 0]
    test_data["expected_dst"] = [3, 1]

    return test_data


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.cugraph_ops
def test_uniform_neighbor_sample_simple(input_combo):

    G = input_combo["Graph"]

    #
    # Make sure the old C++ renumbering was skipped because:
    #    1) Pylibcugraph already does renumbering
    #    2) Uniform neighborhood sampling allows int32 weights
    #       which are not supported by the C++ renumbering
    # This should be 'True' only for string vertices and multi columns vertices
    #

    assert G.renumbered is False
    # Retrieve the input dataframe.
    # FIXME: in simpleGraph and simpleDistributedGraph, G.edgelist.edgelist_df
    # should be 'None' if the datasets was never renumbered
    input_df = G.edgelist.edgelist_df

    result_nbr = uniform_neighbor_sample(
        G,
        input_combo["start_list"],
        input_combo["fanout_vals"],
        input_combo["with_replacement"],
    )

    # multi edges are dropped to easily verify that each edge in the
    # results is present in the input dataframe
    result_nbr = result_nbr.drop_duplicates()

    # FIXME: The indices are not included in the comparison because garbage
    # value are intermittently retuned. This observation is observed
    # when passing float weights
    join = result_nbr.merge(
        input_df, left_on=[*result_nbr.columns[:2]], right_on=[*input_df.columns[:2]]
    )

    if len(result_nbr) != len(join):
        join2 = input_df.merge(
            result_nbr,
            how="right",
            left_on=[*input_df.columns],
            right_on=[*result_nbr.columns],
        )
        # The left part of the datasets shows which edge is missing from the
        # right part where the left and right part are respectively the
        # uniform-neighbor-sample results and the input dataframe.
        difference = (
            join2.sort_values([*result_nbr.columns])
            .to_pandas()
            .query("src.isnull()", engine="python")
        )

        invalid_edge = difference[difference.columns[:3]]
        raise Exception(
            f"\nThe edges below from uniform-neighbor-sample "
            f"are invalid\n {invalid_edge}"
        )

    # Ensure the right indices type is returned
    assert result_nbr["indices"].dtype == input_combo["indices_type"]

    sampled_vertex_result = (
        cudf.concat([result_nbr["sources"], result_nbr["destinations"]])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    sampled_vertex_result = sampled_vertex_result.to_pandas()
    start_list = input_combo["start_list"].to_pandas()

    if not set(start_list).issubset(set(sampled_vertex_result)):
        missing_vertex = set(start_list) - set(sampled_vertex_result)
        missing_vertex = list(missing_vertex)
        # compute the out-degree of the missing vertices
        out_degree = G.out_degree(missing_vertex)
        out_degree = out_degree[out_degree.degree != 0]
        # If the missing vertices have outgoing edges, return an error
        if len(out_degree) != 0:
            missing_vertex = out_degree["vertex"].to_pandas().to_list()
            raise Exception(
                f"vertex {missing_vertex} is missing from "
                f"uniform neighbor sampling results"
            )


@pytest.mark.cugraph_ops
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_uniform_neighbor_sample_tree(directed):

    input_data_path = small_tree.get_path()

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    G = cugraph.Graph(directed=directed)
    G.from_cudf_edgelist(df, "src", "dst", "value", legacy_renum_only=True)

    #
    # Make sure the old C++ renumbering was skipped because:
    #    1) Pylibcugraph already does renumbering
    #    2) Uniform neighborhood sampling allows int32 weights
    #       which are not supported by the C++ renumbering
    # This should be 'True' only for string vertices and multi columns vertices
    #

    assert G.renumbered is False

    # Retrieve the input dataframe.
    # input_df != df if 'directed = False' because df will be symmetrized
    # internally.
    input_df = G.edgelist.edgelist_df

    # TODO: Incomplete, include more testing for tree graph as well as
    # for larger graphs
    start_list = cudf.Series([0, 0], dtype="int32")
    fanout_vals = [4, 1, 3]
    with_replacement = True
    result_nbr = uniform_neighbor_sample(G, start_list, fanout_vals, with_replacement)

    result_nbr = result_nbr.drop_duplicates()

    join = result_nbr.merge(
        input_df, left_on=[*result_nbr.columns[:2]], right_on=[*input_df.columns[:2]]
    )

    assert len(join) == len(result_nbr)
    # Since the validity of results have (probably) been tested at both the C++
    # and C layers, simply test that the python interface and conversions were
    # done correctly.
    assert result_nbr["sources"].dtype == "int32"
    assert result_nbr["destinations"].dtype == "int32"
    assert result_nbr["indices"].dtype == "float32"

    result_nbr_vertices = (
        cudf.concat([result_nbr["sources"], result_nbr["destinations"]])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    assert set(start_list.to_pandas()).issubset(set(result_nbr_vertices.to_pandas()))


@pytest.mark.cugraph_ops
def test_uniform_neighbor_sample_unweighted(simple_unweighted_input_expected_output):
    test_data = simple_unweighted_input_expected_output

    sampling_results = uniform_neighbor_sample(
        test_data["Graph"],
        test_data["start_list"],
        test_data["fanout_vals"],
        test_data["with_replacement"],
    )

    actual_src = sampling_results.sources
    actual_src = actual_src.to_arrow().to_pylist()
    assert sorted(actual_src) == sorted(test_data["expected_src"])

    actual_dst = sampling_results.destinations
    actual_dst = actual_dst.to_arrow().to_pylist()
    assert sorted(actual_dst) == sorted(test_data["expected_dst"])
