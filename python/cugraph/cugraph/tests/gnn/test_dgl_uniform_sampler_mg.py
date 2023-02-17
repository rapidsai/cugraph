# Copyright (c) 2023, NVIDIA CORPORATION.
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
import dask_cudf
import cudf
import pandas as pd
import numpy as np
import cupy as cp

from cugraph.gnn.dgl_extensions.dgl_uniform_sampler import DGLUniformSampler


def assert_correct_eids(edge_df, sample_edge_id_df):
    # We test that all src, dst correspond to the correct
    # eids in the sample_edge_id_df
    # we do this by ensuring that the inner merge to edge_df
    # remains the same as sample_edge_id_df
    # if they don't correspond correctly
    # the inner merge would fail
    sample_edge_id_df = sample_edge_id_df.sort_values(by="_EDGE_ID_")
    sample_edge_id_df = sample_edge_id_df.reset_index(drop=True)

    sample_merged_df = sample_edge_id_df.merge(edge_df, how="inner")
    sample_merged_df = sample_merged_df.sort_values(by="_EDGE_ID_")
    sample_merged_df = sample_merged_df.reset_index(drop=True)
    assert sample_merged_df.equals(sample_edge_id_df)


def convert_array_dict_to_df(d):
    df_d = {
        k: cudf.DataFrame({"_SRC_": s, "_DST_": d, "_EDGE_ID_": e})
        for k, (s, d, e) in d.items()
    }
    return df_d


def test_sampling_homogeneous_gs_out_dir(dask_client):
    src_ser = cudf.Series([1, 1, 1, 1, 1, 2, 2, 3])
    dst_ser = cudf.Series([2, 3, 4, 5, 6, 3, 4, 7])
    df = cudf.DataFrame(
        {"_SRC_": src_ser, "_DST_": dst_ser, "_EDGE_ID_": np.arange(len(src_ser))}
    )
    df = dask_cudf.from_cudf(df, 4)
    sampler = DGLUniformSampler({("_N", "connects", "_N"): df}, {"_N": (0, 8)}, False)

    # below are obtained from dgl runs on the same graph
    expected_out = {
        1: ([1, 1, 1, 1, 1], [2, 3, 4, 5, 6]),
        2: ([2, 2], [3, 4]),
        3: ([3], [7]),
        4: ([], []),
    }

    for seed in expected_out.keys():
        seed_cap = cudf.Series([seed]).to_dlpack()
        sample_src, sample_dst, sample_eid = sampler.sample_neighbors(
            seed_cap, fanout=9, edge_dir="out"
        )
        output_df = cudf.DataFrame({"src": sample_src, "dst": sample_dst}).astype(
            np.int64
        )
        output_df = output_df.sort_values(by=["src", "dst"])
        output_df = output_df.reset_index(drop=True)

        expected_df = cudf.DataFrame(
            {"src": expected_out[seed][0], "dst": expected_out[seed][1]}
        ).astype(np.int64)
        cudf.testing.assert_frame_equal(output_df, expected_df)

        sample_edge_id_df = cudf.DataFrame(
            {"_SRC_": sample_src, "_DST_": sample_dst, "_EDGE_ID_": sample_eid}
        )

        assert_correct_eids(df.compute(), sample_edge_id_df)


def test_sampling_homogeneous_gs_in_dir(dask_client):
    src_ser = cudf.Series([1, 1, 1, 1, 1, 2, 2, 3])
    dst_ser = cudf.Series([2, 3, 4, 5, 6, 3, 4, 7])
    df = cudf.DataFrame(
        {"_SRC_": src_ser, "_DST_": dst_ser, "_EDGE_ID_": np.arange(len(src_ser))}
    )
    df = dask_cudf.from_cudf(df, 4)
    sampler = DGLUniformSampler({("_N", "connects", "_N"): df}, {"_N": (0, 8)}, False)

    # below are obtained from dgl runs on the same graph
    expected_in = {
        1: ([], []),
        2: ([1], [2]),
        3: ([1, 2], [3, 3]),
        4: ([1, 2], [4, 4]),
    }

    for seed in expected_in.keys():
        seed_cap = cudf.Series([seed]).to_dlpack()
        sample_src, sample_dst, sample_eid = sampler.sample_neighbors(
            seed_cap, fanout=9, edge_dir="in"
        )
        output_df = cudf.DataFrame({"_SRC_": sample_src, "_DST_": sample_dst}).astype(
            np.int64
        )
        output_df = output_df.sort_values(by=["_SRC_", "_DST_"])
        output_df = output_df.reset_index(drop=True)

        expected_df = cudf.DataFrame(
            {"_SRC_": expected_in[seed][0], "_DST_": expected_in[seed][1]}
        ).astype(np.int64)
        cudf.testing.assert_frame_equal(output_df, expected_df)

        sample_edge_id_df = cudf.DataFrame(
            {"_SRC_": sample_src, "_DST_": sample_dst, "_EDGE_ID_": sample_eid}
        )

        assert_correct_eids(df.compute(), sample_edge_id_df)


def create_gs_heterogeneous_dgl_sampler():
    # Add Edge Data
    src_ser = [0, 1, 2, 0, 1, 2, 7, 9, 10, 11]
    dst_ser = [3, 4, 5, 6, 7, 8, 6, 6, 6, 6]
    etype_ser = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

    etype_map = {
        0: ("nt.a", "connects", "nt.b"),
        1: ("nt.a", "connects", "nt.c"),
        2: ("nt.c", "connects", "nt.c"),
    }

    df = pd.DataFrame({"_SRC_": src_ser, "_DST_": dst_ser, "etype": etype_ser})

    etype_offset = 0
    edge_id_range_dict = {}
    edge_list_dict = {}
    for e in df["etype"].unique():
        subset_df = df[df["etype"] == e][["_SRC_", "_DST_"]]
        subset_df = cudf.DataFrame.from_pandas(subset_df)
        subset_df["_EDGE_ID_"] = cp.arange(0, len(subset_df)) + etype_offset
        subset_df = dask_cudf.from_cudf(subset_df, 4)
        edge_list_dict[etype_map[e]] = subset_df
        edge_id_range_dict[etype_map[e]] = (etype_offset, etype_offset + len(subset_df))
        etype_offset = etype_offset + len(subset_df)
    return DGLUniformSampler(edge_list_dict, edge_id_range_dict, False)


def test_sampling_gs_heterogeneous_out_dir(dask_client):
    sampler = create_gs_heterogeneous_dgl_sampler()
    # DGL expected_output from
    # https://gist.github.com/VibhuJawa/f85fda8e1183886078f2a34c28c4638c
    expeced_val_d = {
        0: {
            ("nt.a", "connects", "nt.b"): (
                cudf.Series([0], dtype=np.int32),
                cudf.Series([3], dtype=np.int32),
            ),
            ("nt.a", "connects", "nt.c"): (
                cudf.Series([0]),
                cudf.Series([6]),
            ),
            ("nt.c", "connects", "nt.c"): (cudf.Series([]), cudf.Series([])),
        },
        1: {
            ("nt.a", "connects", "nt.b"): (
                cudf.Series([1], dtype=np.int32),
                cudf.Series([4], dtype=np.int32),
            ),
            ("nt.a", "connects", "nt.c"): (
                cudf.Series([1]),
                cudf.Series([7]),
            ),
            ("nt.c", "connects", "nt.c"): (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
        },
        2: {
            ("nt.a", "connects", "nt.b"): (
                cudf.Series([2], dtype=np.int32),
                cudf.Series([5], dtype=np.int32),
            ),
            ("nt.a", "connects", "nt.c"): (
                cudf.Series([2]),
                cudf.Series([8]),
            ),
            ("nt.c", "connects", "nt.c"): (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
        },
    }

    for seed in expeced_val_d.keys():
        fanout = 4
        sampled_node_p = cudf.Series(seed).astype(np.int32).to_dlpack()
        sampled_g = sampler.sample_neighbors(
            {"nt.a": sampled_node_p}, fanout=fanout, edge_dir="out"
        )
        sampled_g = convert_array_dict_to_df(sampled_g)
        for etype, df in sampled_g.items():
            output_df = (
                df[["_SRC_", "_DST_"]]
                .sort_values(by=["_SRC_", "_DST_"])
                .reset_index(drop=True)
                .astype(np.int32)
            )
            expected_df = cudf.DataFrame(
                {
                    "_SRC_": expeced_val_d[seed][etype][0],
                    "_DST_": expeced_val_d[seed][etype][1],
                }
            ).astype(np.int32)
            cudf.testing.assert_frame_equal(output_df, expected_df)


def test_sampling_gs_heterogeneous_in_dir(dask_client):
    sampler = create_gs_heterogeneous_dgl_sampler()
    # DGL expected_output from
    # https://gist.github.com/VibhuJawa/f85fda8e1183886078f2a34c28c4638c
    expeced_val_d = {
        6: {
            ("nt.a", "connects", "nt.b"): (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
            ("nt.a", "connects", "nt.c"): (
                cudf.Series([0]),
                cudf.Series([6]),
            ),
            ("nt.c", "connects", "nt.c"): (
                cudf.Series([7, 9, 10, 11]),
                cudf.Series([6, 6, 6, 6]),
            ),
        },
        7: {
            ("nt.a", "connects", "nt.b"): (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
            ("nt.a", "connects", "nt.c"): (
                cudf.Series([1]),
                cudf.Series([7]),
            ),
            ("nt.c", "connects", "nt.c"): (
                cudf.Series([], dtype=np.int32),
                cudf.Series([], dtype=np.int32),
            ),
        },
    }

    for seed in expeced_val_d.keys():
        fanout = 4
        sampled_node_p = cudf.Series(seed).astype(np.int32).to_dlpack()
        sampled_g = sampler.sample_neighbors(
            {"nt.c": sampled_node_p}, fanout=fanout, edge_dir="in"
        )
        sampled_g = convert_array_dict_to_df(sampled_g)
        for etype, df in sampled_g.items():
            output_df = (
                df[["_SRC_", "_DST_"]]
                .sort_values(by=["_SRC_", "_DST_"])
                .reset_index(drop=True)
                .astype(np.int32)
            )
            expected_df = cudf.DataFrame(
                {
                    "_SRC_": expeced_val_d[seed][etype][0],
                    "_DST_": expeced_val_d[seed][etype][1],
                }
            ).astype(np.int32)
            cudf.testing.assert_frame_equal(output_df, expected_df)


def test_sampling_dgl_heterogeneous_gs_m_fanouts(dask_client):
    gs = create_gs_heterogeneous_dgl_sampler()
    # Test against DGLs output
    # See below notebook
    # https://gist.github.com/VibhuJawa/f85fda8e1183886078f2a34c28c4638c

    expected_output = {
        1: {
            ("nt.a", "connects", "nt.b"): 0,
            ("nt.a", "connects", "nt.c"): 1,
            ("nt.c", "connects", "nt.c"): 1,
        },
        2: {
            ("nt.a", "connects", "nt.b"): 0,
            ("nt.a", "connects", "nt.c"): 1,
            ("nt.c", "connects", "nt.c"): 2,
        },
        3: {
            ("nt.a", "connects", "nt.b"): 0,
            ("nt.a", "connects", "nt.c"): 1,
            ("nt.c", "connects", "nt.c"): 3,
        },
        -1: {
            ("nt.a", "connects", "nt.b"): 0,
            ("nt.a", "connects", "nt.c"): 1,
            ("nt.c", "connects", "nt.c"): 4,
        },
    }

    for fanout in [1, 2, 3, -1]:
        sampled_node = [6]
        sampled_node_p = cudf.Series(sampled_node)
        sampled_g = gs.sample_neighbors({"nt.c": sampled_node_p}, fanout=fanout)
        sampled_g = convert_array_dict_to_df(sampled_g)
        for etype, output_df in sampled_g.items():
            assert expected_output[fanout][etype] == len(output_df)
