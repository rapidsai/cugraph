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
#
# Copyright (c) 2015, Graphistry, Inc.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Graphistry, Inc nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL Graphistry, Inc BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import datetime as dt

import pandas as pd
import pytest
import cudf
from cudf.testing.testing import assert_frame_equal
import cugraph


simple_df = cudf.DataFrame.from_pandas(
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "a1": [1, 2, 3],
            "a2": ["red", "blue", "green"],
            "🙈": ["æski ēˈmōjē", "😋", "s"],
        }
    )
)

hyper_df = cudf.DataFrame.from_pandas(
    pd.DataFrame({"aa": [0, 1, 2], "bb": ["a", "b", "c"], "cc": ["b", "0", "1"]})
)


def test_complex_df():
    complex_df = pd.DataFrame(
        {
            "src": [0, 1, 2, 3],
            "dst": [1, 2, 3, 0],
            "colors": [1, 1, 2, 2],
            "bool": [True, False, True, True],
            "char": ["a", "b", "c", "d"],
            "str": ["a", "b", "c", "d"],
            "ustr": ["a", "b", "c", "d"],
            "emoji": ["😋", "😋😋", "😋", "😋"],
            "int": [0, 1, 2, 3],
            "num": [0.5, 1.5, 2.5, 3.5],
            "date_str": [
                "2018-01-01 00:00:00",
                "2018-01-02 00:00:00",
                "2018-01-03 00:00:00",
                "2018-01-05 00:00:00",
            ],
            "date": [
                dt.datetime(2018, 1, 1),
                dt.datetime(2018, 1, 1),
                dt.datetime(2018, 1, 1),
                dt.datetime(2018, 1, 1),
            ],
            "time": [
                pd.Timestamp("2018-01-05"),
                pd.Timestamp("2018-01-05"),
                pd.Timestamp("2018-01-05"),
                pd.Timestamp("2018-01-05"),
            ],
        }
    )

    for c in complex_df.columns:
        try:
            complex_df[c + "_cat"] = complex_df[c].astype("category")
        except Exception:
            # lists aren't categorical
            # print('could not make categorical', c)
            pass

    complex_df = cudf.DataFrame.from_pandas(complex_df)

    cugraph.hypergraph(complex_df)


@pytest.mark.parametrize("categorical_metadata", [False, True])
def test_hyperedges(categorical_metadata):

    h = cugraph.hypergraph(simple_df, categorical_metadata=categorical_metadata)

    assert len(h.keys()) == len(["entities", "nodes", "edges", "events", "graph"])

    edges = cudf.from_pandas(
        pd.DataFrame(
            {
                "event_id": [
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                ],
                "edge_type": [
                    "a1",
                    "a1",
                    "a1",
                    "a2",
                    "a2",
                    "a2",
                    "id",
                    "id",
                    "id",
                    "🙈",
                    "🙈",
                    "🙈",
                ],
                "attrib_id": [
                    "a1::1",
                    "a1::2",
                    "a1::3",
                    "a2::red",
                    "a2::blue",
                    "a2::green",
                    "id::a",
                    "id::b",
                    "id::c",
                    "🙈::æski ēˈmōjē",
                    "🙈::😋",
                    "🙈::s",
                ],
                "id": ["a", "b", "c"] * 4,
                "a1": [1, 2, 3] * 4,
                "a2": ["red", "blue", "green"] * 4,
                "🙈": ["æski ēˈmōjē", "😋", "s"] * 4,
            }
        )
    )

    if categorical_metadata:
        edges = edges.astype({"edge_type": "category"})

    assert_frame_equal(edges, h["edges"], check_dtype=False)
    for (k, v) in [("entities", 12), ("nodes", 15), ("edges", 12), ("events", 3)]:
        assert len(h[k]) == v


def test_hyperedges_direct():

    h = cugraph.hypergraph(hyper_df, direct=True)

    assert len(h["edges"]) == 9
    assert len(h["nodes"]) == 9


def test_hyperedges_direct_categories():

    h = cugraph.hypergraph(
        hyper_df,
        direct=True,
        categories={
            "aa": "N",
            "bb": "N",
            "cc": "N",
        },
    )

    assert len(h["edges"]) == 9
    assert len(h["nodes"]) == 6


def test_hyperedges_direct_manual_shaping():

    h1 = cugraph.hypergraph(
        hyper_df,
        direct=True,
        EDGES={"aa": ["cc"], "cc": ["cc"]},
    )
    assert len(h1["edges"]) == 6

    h2 = cugraph.hypergraph(
        hyper_df,
        direct=True,
        EDGES={"aa": ["cc", "bb", "aa"], "cc": ["cc"]},
    )
    assert len(h2["edges"]) == 12


@pytest.mark.parametrize("categorical_metadata", [False, True])
def test_drop_edge_attrs(categorical_metadata):

    h = cugraph.hypergraph(
        simple_df,
        columns=["id", "a1", "🙈"],
        drop_edge_attrs=True,
        categorical_metadata=categorical_metadata,
    )

    assert len(h.keys()) == len(["entities", "nodes", "edges", "events", "graph"])

    edges = cudf.DataFrame.from_pandas(
        pd.DataFrame(
            {
                "event_id": [
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                ],
                "edge_type": ["a1", "a1", "a1", "id", "id", "id", "🙈", "🙈", "🙈"],
                "attrib_id": [
                    "a1::1",
                    "a1::2",
                    "a1::3",
                    "id::a",
                    "id::b",
                    "id::c",
                    "🙈::æski ēˈmōjē",
                    "🙈::😋",
                    "🙈::s",
                ],
            }
        )
    )

    if categorical_metadata:
        edges = edges.astype({"edge_type": "category"})

    assert_frame_equal(edges, h["edges"], check_dtype=False)

    for (k, v) in [("entities", 9), ("nodes", 12), ("edges", 9), ("events", 3)]:
        assert len(h[k]) == v


@pytest.mark.parametrize("categorical_metadata", [False, True])
def test_drop_edge_attrs_direct(categorical_metadata):

    h = cugraph.hypergraph(
        simple_df,
        ["id", "a1", "🙈"],
        direct=True,
        drop_edge_attrs=True,
        EDGES={"id": ["a1"], "a1": ["🙈"]},
        categorical_metadata=categorical_metadata,
    )

    assert len(h.keys()) == len(["entities", "nodes", "edges", "events", "graph"])

    edges = cudf.DataFrame.from_pandas(
        pd.DataFrame(
            {
                "event_id": [
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                    "event_id::0",
                    "event_id::1",
                    "event_id::2",
                ],
                "edge_type": ["a1::🙈", "a1::🙈", "a1::🙈", "id::a1", "id::a1", "id::a1"],
                "src": ["a1::1", "a1::2", "a1::3", "id::a", "id::b", "id::c"],
                "dst": ["🙈::æski ēˈmōjē", "🙈::😋", "🙈::s", "a1::1", "a1::2", "a1::3"],
            }
        )
    )

    if categorical_metadata:
        edges = edges.astype({"edge_type": "category"})

    assert_frame_equal(edges, h["edges"], check_dtype=False)

    for (k, v) in [("entities", 9), ("nodes", 9), ("edges", 6), ("events", 0)]:
        assert len(h[k]) == v


def test_skip_hyper():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ["a", None, "b"], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    )

    hg = cugraph.hypergraph(df, SKIP=["c"], dropna=False)

    assert len(hg["graph"].nodes()) == 9
    assert len(hg["graph"].edges()) == 6


def test_skip_drop_na_hyper():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ["a", None, "b"], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    )

    hg = cugraph.hypergraph(df, SKIP=["c"], dropna=True)

    assert len(hg["graph"].nodes()) == 8
    assert len(hg["graph"].edges()) == 5


def test_skip_direct():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ["a", None, "b"], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    )

    hg = cugraph.hypergraph(df, SKIP=["c"], dropna=False, direct=True)

    assert len(hg["graph"].nodes()) == 6
    assert len(hg["graph"].edges()) == 3


def test_skip_drop_na_direct():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ["a", None, "b"], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    )

    hg = cugraph.hypergraph(df, SKIP=["c"], dropna=True, direct=True)

    assert len(hg["graph"].nodes()) == 4
    assert len(hg["graph"].edges()) == 2


def test_drop_na_hyper():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ["a", None, "c"], "i": [1, 2, None]})
    )

    hg = cugraph.hypergraph(df, dropna=True)

    assert len(hg["graph"].nodes()) == 7
    assert len(hg["graph"].edges()) == 4


def test_drop_na_direct():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"a": ["a", None, "a"], "i": [1, 1, None]})
    )

    hg = cugraph.hypergraph(df, dropna=True, direct=True)

    assert len(hg["graph"].nodes()) == 2
    assert len(hg["graph"].edges()) == 1


def test_skip_na_hyperedge():

    nans_df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"x": ["a", "b", "c"], "y": ["aa", None, "cc"]})
    )

    expected_hits = ["a", "b", "c", "aa", "cc"]

    skip_attr_h_edges = cugraph.hypergraph(nans_df, drop_edge_attrs=True)["edges"]

    assert len(skip_attr_h_edges) == len(expected_hits)

    default_h_edges = cugraph.hypergraph(nans_df)["edges"]
    assert len(default_h_edges) == len(expected_hits)


def test_hyper_to_pa_vanilla():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"x": ["a", "b", "c"], "y": ["d", "e", "f"]})
    )

    hg = cugraph.hypergraph(df)
    nodes_arr = hg["graph"].nodes().to_arrow()
    assert len(nodes_arr) == 9
    edges_err = hg["graph"].edges().to_arrow()
    assert len(edges_err) == 6


def test_hyper_to_pa_mixed():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]})
    )

    hg = cugraph.hypergraph(df)
    nodes_arr = hg["graph"].nodes().to_arrow()
    assert len(nodes_arr) == 9
    edges_err = hg["graph"].edges().to_arrow()
    assert len(edges_err) == 6


def test_hyper_to_pa_na():

    df = cudf.DataFrame.from_pandas(
        pd.DataFrame({"x": ["a", None, "c"], "y": [1, 2, None]})
    )

    hg = cugraph.hypergraph(df, dropna=False)
    print(hg["graph"].nodes())
    nodes_arr = hg["graph"].nodes().to_arrow()
    assert len(hg["graph"].nodes()) == 9
    assert len(nodes_arr) == 9
    edges_err = hg["graph"].edges().to_arrow()
    assert len(hg["graph"].edges()) == 6
    assert len(edges_err) == 6


def test_hyper_to_pa_all():
    hg = cugraph.hypergraph(simple_df, ["id", "a1", "🙈"])
    nodes_arr = hg["graph"].nodes().to_arrow()
    assert len(hg["graph"].nodes()) == 12
    assert len(nodes_arr) == 12
    edges_err = hg["graph"].edges().to_arrow()
    assert len(hg["graph"].edges()) == 9
    assert len(edges_err) == 9


def test_hyper_to_pa_all_direct():
    hg = cugraph.hypergraph(simple_df, ["id", "a1", "🙈"], direct=True)
    nodes_arr = hg["graph"].nodes().to_arrow()
    assert len(hg["graph"].nodes()) == 9
    assert len(nodes_arr) == 9
    edges_err = hg["graph"].edges().to_arrow()
    assert len(hg["graph"].edges()) == 9
    assert len(edges_err) == 9
