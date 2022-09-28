# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
from cugraph.experimental import MGPropertyGraph
from cugraph.gnn import CuGraphStore


import numpy as np
import cudf
import cupy as cp
import dask_cudf


@pytest.fixture(scope="module")
def basic_mg_gs(dask_client):
    pg = MGPropertyGraph()
    gs = CuGraphStore(pg, backend_lib="cupy")

    df_1 = cudf.DataFrame(
        {
            "src": cudf.Series([0, 0, 1, 2, 2, 3], dtype="int32"),
            "dst": cudf.Series([1, 2, 4, 3, 4, 1], dtype="int32"),
            "edge_w1": cudf.Series([10, 20, 40, 10, 20, 40], dtype="int32"),
            "edge_w2": cudf.Series([11, 21, 41, 11, 21, 41], dtype="int32"),
        }
    )

    df_1 = dask_cudf.from_cudf(df_1, npartitions=2)

    gs.add_edge_data(df_1, vertex_col_names=["src", "dst"], feat_name="edge_w")

    df_2 = cudf.DataFrame(
        {
            "prop1": cudf.Series([100, 200, 300, 400, 500]),
            "prop2": cudf.Series([5, 4, 3, 2, 1]),
            "id": cudf.Series([0, 1, 2, 3, 4]),
        }
    )

    df_2 = df_2.astype(np.int32)

    df_2 = dask_cudf.from_cudf(df_2, npartitions=2)
    gs.add_node_data(df_2, node_col_name="id", feat_name="prop")
    return gs


def test_num_nodes(basic_mg_gs):
    assert basic_mg_gs.num_nodes() == 5


def test_num_edges(basic_mg_gs):
    assert basic_mg_gs.num_edges() == 6


def test_sampling(basic_mg_gs):
    seed_cap = cudf.Series([1], dtype="int32").to_dlpack()
    cap = basic_mg_gs.sample_neighbors(seed_cap)
    src_t, dst_t, eid_t = get_cudf_ser_from_cap_tup(cap)
    assert len(src_t) == 2


def test_get_node_storage(basic_mg_gs):
    result = basic_mg_gs.get_node_storage(feat_name="prop").fetch(
        indices=[2, 3]
    )
    expected_result = cp.asarray([[300, 3], [400, 2]]).astype(cp.int32)

    cp.testing.assert_array_equal(result, expected_result)


def test_get_edge_storage(basic_mg_gs):
    result = basic_mg_gs.get_edge_storage(feat_name="edge_w").fetch(
        indices=[1, 2]
    )
    expected_result = cp.asarray([[20, 21], [40, 41]]).astype(cp.int32)

    cp.testing.assert_array_equal(result, expected_result)


def test_sampling_homogeneous_gs_in_dir(dask_client):
    src_ser = cudf.Series([1, 1, 1, 1, 1, 2, 2, 3])
    dst_ser = cudf.Series([2, 3, 4, 5, 6, 3, 4, 7])
    df = cudf.DataFrame(
        {"src": src_ser, "dst": dst_ser, "edge_id": np.arange(len(src_ser))}
    )
    df = df.astype(np.int32)
    df = dask_cudf.from_cudf(df, npartitions=3)

    pg = MGPropertyGraph()
    gs = CuGraphStore(pg)
    gs.add_edge_data(df, ["src", "dst"], feat_name="edges")

    # below are obtained from dgl runs on the same graph
    expected_in = {
        1: ([], []),
        2: ([1], [2]),
        3: ([1, 2], [3, 3]),
        4: ([1, 2], [4, 4]),
    }

    for seed in expected_in.keys():
        seed_cap = cudf.Series([seed]).astype(np.int32).to_dlpack()
        sample_src, sample_dst, sample_eid = gs.sample_neighbors(
            seed_cap, fanout=9, edge_dir="in"
        )
        if sample_src is None:
            sample_src = cudf.Series([]).astype(np.int32)
            sample_dst = cudf.Series([]).astype(np.int32)
            sample_eid = cudf.Series([]).astype(np.int32)
        else:
            sample_src = cudf.from_dlpack(sample_src)
            sample_dst = cudf.from_dlpack(sample_dst)
            sample_eid = cudf.from_dlpack(sample_eid)

        output_df = cudf.DataFrame({"src": sample_src, "dst": sample_dst})
        output_df = output_df.sort_values(by=["src", "dst"])
        output_df = output_df.reset_index(drop=True)

        expected_df = cudf.DataFrame(
            {"src": expected_in[seed][0], "dst": expected_in[seed][1]}
        ).astype(np.int32)

        cudf.testing.assert_frame_equal(output_df, expected_df)


def test_sampling_homogeneous_gs_out_dir(dask_client):
    src_ser = cudf.Series([1, 1, 1, 1, 1, 2, 2, 3])
    dst_ser = cudf.Series([2, 3, 4, 5, 6, 3, 4, 7])
    df = cudf.DataFrame(
        {"src": src_ser, "dst": dst_ser, "edge_id": np.arange(len(src_ser))}
    )
    df = df.astype(np.int32)
    df = dask_cudf.from_cudf(df, npartitions=3)

    pg = MGPropertyGraph()
    gs = CuGraphStore(pg)
    gs.add_edge_data(df, ["src", "dst"], feat_name="edges")

    # below are obtained from dgl runs on the same graph
    expected_out = {
        1: ([1, 1, 1, 1, 1], [2, 3, 4, 5, 6]),
        2: ([2, 2], [3, 4]),
        3: ([3], [7]),
        4: ([], []),
    }

    for seed in expected_out.keys():
        seed_cap = cudf.Series([seed]).to_dlpack()
        sample_src, sample_dst, sample_eid = gs.sample_neighbors(
            seed_cap, fanout=9, edge_dir="out"
        )
        if sample_src is None:
            sample_src = cudf.Series([]).astype(np.int64)
            sample_dst = cudf.Series([]).astype(np.int64)
            sample_eid = cudf.Series([]).astype(np.int64)
        else:
            sample_src = cudf.from_dlpack(sample_src)
            sample_dst = cudf.from_dlpack(sample_dst)
            sample_eid = cudf.from_dlpack(sample_eid)

        output_df = cudf.DataFrame({"src": sample_src, "dst": sample_dst})
        output_df = output_df.sort_values(by=["src", "dst"])
        output_df = output_df.reset_index(drop=True)

        expected_df = cudf.DataFrame(
            {"src": expected_out[seed][0], "dst": expected_out[seed][1]}
        ).astype(np.int32)
        cudf.testing.assert_frame_equal(output_df, expected_df)


# @pytest.fixture(
#     params=[
#         'basic_mg_graph_gs',
#     ]
# )
# def cugraph_graphstore(request):
#     return request.getfixturevalue(request.param)

# test_num_nodes_gs
# test_num_edges_gs
# test_get_node_storage_gs
# test_get_edge_storage_gs
# test_sampling_homogeneous_gs_in_dir
# test_sampling_homogeneous_gs_out_dir
# test_sampling_gs_homogeneous_neg_one_fanout
# test_sampling_gs_heterogeneous_in_dir
# test_sampling_gs_heterogeneous_out_dir
# test_sampling_gs_heterogeneous_neg_one_fanout

# Util to help testing
def get_cudf_ser_from_cap_tup(cap_t):
    src_id, dst_id, e_id = cap_t
    if src_id is not None:
        src_id = cudf.from_dlpack(src_id)
        dst_id = cudf.from_dlpack(dst_id)
        e_id = cudf.from_dlpack(e_id)
    else:
        src_id = cudf.Series([]).astype(np.int32)
        dst_id = cudf.Series([]).astype(np.int32)
        e_id = cudf.Series([]).astype(np.int32)
    return src_id, dst_id, e_id
