# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

    gs.add_edge_data(
        df_1,
        node_col_names=["src", "dst"],
        feat_name="edge_w",
        contains_vector_features=True,
    )

    df_2 = cudf.DataFrame(
        {
            "prop1": cudf.Series([100, 200, 300, 400, 500]),
            "prop2": cudf.Series([5, 4, 3, 2, 1]),
            "id": cudf.Series([0, 1, 2, 3, 4]),
        }
    )

    df_2 = df_2.astype(np.int32)

    df_2 = dask_cudf.from_cudf(df_2, npartitions=2)
    gs.add_node_data(
        df_2, node_col_name="id", feat_name="prop", contains_vector_features=True
    )
    return gs


@pytest.fixture(scope="module")
def gs_heterogeneous_dgl_eg(dask_client):
    pg = MGPropertyGraph()
    gs = CuGraphStore(pg)
    # Changing npartitions is leading to errors
    npartitions = 2

    # Add Edge Data
    src_ser = [0, 1, 2, 0, 1, 2, 7, 9, 10, 11]
    dst_ser = [3, 4, 5, 6, 7, 8, 6, 6, 6, 6]
    etype_ser = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    edge_feat = [10, 10, 10, 11, 11, 11, 12, 12, 12, 13]

    etype_map = {
        0: "('nt.a', 'connects', 'nt.b')",
        1: "('nt.a', 'connects', 'nt.c')",
        2: "('nt.c', 'connects', 'nt.c')",
    }

    df = cudf.DataFrame(
        {
            "src": src_ser,
            "dst": dst_ser,
            "etype": etype_ser,
            "edge_feat": edge_feat,
        }
    )
    df = df.astype(np.int32)
    df = dask_cudf.from_cudf(df, npartitions=npartitions)
    for e in df["etype"].unique().compute().values_host:
        subset_df = df[df["etype"] == e][["src", "dst", "edge_feat"]].reset_index(
            drop=True
        )
        gs.add_edge_data(
            subset_df,
            ["src", "dst"],
            canonical_etype=etype_map[e],
        )

    # Add Node Data
    node_ser = np.arange(0, 12)
    node_type = ["nt.a"] * 3 + ["nt.b"] * 3 + ["nt.c"] * 6
    node_feat = np.arange(0, 12) * 10
    df = cudf.DataFrame(
        {"node_id": node_ser, "ntype": node_type, "node_feat": node_feat}
    )
    df = dask_cudf.from_cudf(df, npartitions=npartitions)
    for n in df["ntype"].unique().compute().values_host:
        subset_df = df[df["ntype"] == n][["node_id", "node_feat"]]
        gs.add_node_data(subset_df, "node_id", ntype=str(n))

    return gs


def test_num_nodes(basic_mg_gs):
    assert basic_mg_gs.num_nodes() == 5


def test_num_edges(basic_mg_gs):
    assert basic_mg_gs.num_edges() == 6


@pytest.mark.cugraph_ops
def test_sampling(basic_mg_gs):
    seed_cap = cudf.Series([1], dtype="int32").to_dlpack()
    cap = basic_mg_gs.sample_neighbors(seed_cap)
    src_t, dst_t, eid_t = get_cudf_ser_from_cap_tup(cap)
    assert len(src_t) == 2


def test_get_node_storage(basic_mg_gs):
    result = basic_mg_gs.get_node_storage(key="prop").fetch(indices=[2, 3])
    expected_result = cp.asarray([[300, 3], [400, 2]]).astype(cp.int32)

    cp.testing.assert_array_equal(result, expected_result)


def test_get_edge_storage(basic_mg_gs):
    result = basic_mg_gs.get_edge_storage(key="edge_w").fetch(indices=[1, 2])
    expected_result = cp.asarray([[20, 21], [40, 41]]).astype(cp.int32)

    cp.testing.assert_array_equal(result, expected_result)


@pytest.mark.cugraph_ops
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
    gs.add_edge_data(df, ["src", "dst"])

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

        cudf.testing.assert_frame_equal(
            output_df.astype(np.int32), expected_df.astype(np.int32)
        )


@pytest.mark.cugraph_ops
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
    gs.add_edge_data(df, ["src", "dst"])

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
        cudf.testing.assert_frame_equal(
            output_df.astype(np.int32), expected_df.astype(np.int32)
        )


@pytest.mark.cugraph_ops
def test_sampling_homogeneous_gs_neg_one_fanout(dask_client):

    src_ser = cudf.Series([1, 1, 1, 1, 1, 2, 2, 3])
    dst_ser = cudf.Series([2, 3, 4, 5, 6, 3, 4, 7])
    df = cudf.DataFrame(
        {"src": src_ser, "dst": dst_ser, "edge_id": np.arange(len(src_ser))}
    )
    df = df.astype(np.int32)
    df = dask_cudf.from_cudf(df, npartitions=3)

    pg = MGPropertyGraph()
    gs = CuGraphStore(pg)
    gs.add_edge_data(df, ["src", "dst"])

    # Results obtained by running DGL
    # sample_neighbors on the same graph
    expected_df_4_in = cudf.DataFrame(
        {"src": cudf.Series([1, 2]), "dst": cudf.Series([4, 4])}
    ).astype(np.int32)

    expected_df_1_out = cudf.DataFrame(
        {"src": [1, 1, 1, 1, 1], "dst": [2, 3, 4, 5, 6]}
    ).astype(np.int32)

    exprect_d = {"in": expected_df_4_in, "out": expected_df_1_out}
    for d, seed_n in [("in", 4), ("out", 1)]:
        seed_cap = cudf.Series(seed_n).astype(np.int32).to_dlpack()
        sample_src, sample_dst, sample_eid = gs.sample_neighbors(
            seed_cap, fanout=-1, edge_dir=d
        )
        sample_src = cudf.from_dlpack(sample_src)
        sample_dst = cudf.from_dlpack(sample_dst)
        sample_eid = cudf.from_dlpack(sample_eid)

        output_df = cudf.DataFrame({"src": sample_src, "dst": sample_dst})
        output_df = output_df.sort_values(by=["src", "dst"])
        output_df = output_df.reset_index(drop=True)

        cudf.testing.assert_frame_equal(output_df, exprect_d[d])


# Test against DGLs output
# See below notebook
# https://gist.github.com/VibhuJawa/f85fda8e1183886078f2a34c28c4638c
def test_sampling_dgl_heterogeneous_gs_m_fanouts(gs_heterogeneous_dgl_eg):
    gs = gs_heterogeneous_dgl_eg
    expected_output = {
        1: {
            "('nt.a', 'connects', 'nt.b')": 0,
            "('nt.a', 'connects', 'nt.c')": 1,
            "('nt.c', 'connects', 'nt.c')": 1,
        },
        2: {
            "('nt.a', 'connects', 'nt.b')": 0,
            "('nt.a', 'connects', 'nt.c')": 1,
            "('nt.c', 'connects', 'nt.c')": 2,
        },
        3: {
            "('nt.a', 'connects', 'nt.b')": 0,
            "('nt.a', 'connects', 'nt.c')": 1,
            "('nt.c', 'connects', 'nt.c')": 3,
        },
        -1: {
            "('nt.a', 'connects', 'nt.b')": 0,
            "('nt.a', 'connects', 'nt.c')": 1,
            "('nt.c', 'connects', 'nt.c')": 4,
        },
    }
    # taking a subgraph with non contiguous numbering
    # to help with cugraph testing
    for fanout in expected_output.keys():
        sampled_node = [6]
        sampled_node_p = cudf.Series(sampled_node).astype(np.int32).to_dlpack()

        sampled_g = gs.sample_neighbors({"nt.c": sampled_node_p}, fanout=fanout)
        sampled_g = convert_dlpack_dict_to_df(sampled_g)
        for etype, output_df in sampled_g.items():
            assert expected_output[fanout][etype] == len(output_df)


def test_sampling_gs_heterogeneous_in_dir(gs_heterogeneous_dgl_eg):
    gs = gs_heterogeneous_dgl_eg
    # DGL expected_output from
    # https://gist.github.com/VibhuJawa/f85fda8e1183886078f2a34c28c4638c
    expeced_val_d = {
        6: {
            "('nt.a', 'connects', 'nt.b')": (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
            "('nt.a', 'connects', 'nt.c')": (
                cudf.Series([0]),
                cudf.Series([6]),
            ),
            "('nt.c', 'connects', 'nt.c')": (
                cudf.Series([7, 9, 10, 11]),
                cudf.Series([6, 6, 6, 6]),
            ),
        },
        7: {
            "('nt.a', 'connects', 'nt.b')": (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
            "('nt.a', 'connects', 'nt.c')": (
                cudf.Series([1]),
                cudf.Series([7]),
            ),
            "('nt.c', 'connects', 'nt.c')": (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
        },
    }

    for seed in expeced_val_d.keys():
        fanout = 4
        sampled_node_p = cudf.Series(seed).astype(np.int32).to_dlpack()
        sampled_g = gs.sample_neighbors(
            {"nt.c": sampled_node_p}, fanout=fanout, edge_dir="in"
        )
        sampled_g = convert_dlpack_dict_to_df(sampled_g)
        for etype, df in sampled_g.items():
            output_df = (
                df[["src", "dst"]]
                .sort_values(by=["src", "dst"])
                .reset_index(drop=True)
                .astype(np.int32)
            )
            expected_df = cudf.DataFrame(
                {
                    "src": expeced_val_d[seed][etype][0],
                    "dst": expeced_val_d[seed][etype][1],
                }
            ).astype(np.int32)
            cudf.testing.assert_frame_equal(output_df, expected_df)


def test_sampling_gs_heterogeneous_out_dir(gs_heterogeneous_dgl_eg):
    gs = gs_heterogeneous_dgl_eg
    # DGL expected_output from
    # https://gist.github.com/VibhuJawa/f85fda8e1183886078f2a34c28c4638c
    expeced_val_d = {
        0: {
            "('nt.a', 'connects', 'nt.b')": (
                cudf.Series([0], dtype=np.int32),
                cudf.Series([3], dtype=np.int32),
            ),
            "('nt.a', 'connects', 'nt.c')": (
                cudf.Series([0]),
                cudf.Series([6]),
            ),
            "('nt.c', 'connects', 'nt.c')": (cudf.Series([]), cudf.Series([])),
        },
        1: {
            "('nt.a', 'connects', 'nt.b')": (
                cudf.Series([1], dtype=np.int32),
                cudf.Series([4], dtype=np.int32),
            ),
            "('nt.a', 'connects', 'nt.c')": (
                cudf.Series([1]),
                cudf.Series([7]),
            ),
            "('nt.c', 'connects', 'nt.c')": (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
        },
        2: {
            "('nt.a', 'connects', 'nt.b')": (
                cudf.Series([2], dtype=np.int32),
                cudf.Series([5], dtype=np.int32),
            ),
            "('nt.a', 'connects', 'nt.c')": (
                cudf.Series([2]),
                cudf.Series([8]),
            ),
            "('nt.c', 'connects', 'nt.c')": (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
        },
    }

    for seed in expeced_val_d.keys():
        fanout = 4
        sampled_node_p = cudf.Series(seed).astype(np.int32).to_dlpack()
        sampled_g = gs.sample_neighbors(
            {"nt.a": sampled_node_p}, fanout=fanout, edge_dir="out"
        )
        sampled_g = convert_dlpack_dict_to_df(sampled_g)
        for etype, df in sampled_g.items():
            output_df = (
                df[["src", "dst"]]
                .sort_values(by=["src", "dst"])
                .reset_index(drop=True)
                .astype(np.int32)
            )
            expected_df = cudf.DataFrame(
                {
                    "src": expeced_val_d[seed][etype][0],
                    "dst": expeced_val_d[seed][etype][1],
                }
            ).astype(np.int32)
            cudf.testing.assert_frame_equal(output_df, expected_df)


@pytest.mark.cugraph_ops
def test_sampling_with_out_of_index_seed(dask_client):
    pg = MGPropertyGraph()
    gs = CuGraphStore(pg)
    node_df = cudf.DataFrame()
    node_df["node_id"] = cudf.Series([0, 1, 2, 3, 4, 5]).astype("int32")
    node_df = dask_cudf.from_cudf(node_df, npartitions=2)
    gs.add_node_data(node_df, "node_id", "_N")

    edge_df = cudf.DataFrame()
    edge_df["src"] = cudf.Series([0, 1, 2]).astype("int32")
    edge_df["dst"] = cudf.Series([0, 0, 0]).astype("int32")
    edge_df = dask_cudf.from_cudf(edge_df, npartitions=2)
    gs.add_edge_data(edge_df, ["src", "dst"], canonical_etype="('_N', 'con.a', '_N')")

    edge_df = cudf.DataFrame()
    edge_df["src"] = cudf.Series([3, 4, 5]).astype("int32")
    edge_df["dst"] = cudf.Series([3, 3, 3]).astype("int32")
    edge_df = dask_cudf.from_cudf(edge_df, npartitions=2)
    gs.add_edge_data(edge_df, ["src", "dst"], canonical_etype="('_N', 'con.b', '_N')")

    output = gs.sample_neighbors(
        {"_N": cudf.Series([0, 1, 3, 5], "int32").to_dlpack()}, fanout=3
    )
    output_e1 = (
        cudf.from_dlpack(output["('_N', 'con.a', '_N')"][0])
        .sort_values()
        .reset_index(drop=True)
    )
    output_e2 = (
        cudf.from_dlpack(output["('_N', 'con.b', '_N')"][0])
        .sort_values()
        .reset_index(drop=True)
    )

    cudf.testing.assert_series_equal(
        output_e1, cudf.Series([0, 1, 2], dtype="int32", name=0)
    )
    cudf.testing.assert_series_equal(
        output_e2, cudf.Series([3, 4, 5], dtype="int32", name=0)
    )


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


def convert_dlpack_dict_to_df(d):
    df_d = {k: get_cudf_ser_from_cap_tup(v) for k, v in d.items()}
    df_d = {
        k: cudf.DataFrame({"src": s, "dst": d, "eids": e})
        for k, (s, d, e) in df_d.items()
    }
    return df_d
