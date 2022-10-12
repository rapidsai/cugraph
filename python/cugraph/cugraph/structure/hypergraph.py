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

import cudf
import numpy as np
from cugraph.structure.graph_classes import Graph


def hypergraph(
    values,
    columns=None,
    dropna=True,
    direct=False,
    graph_class=Graph,
    categories=dict(),
    drop_edge_attrs=False,
    categorical_metadata=True,
    SKIP=None,
    EDGES=None,
    DELIM="::",
    SOURCE="src",
    TARGET="dst",
    WEIGHTS=None,
    NODEID="node_id",
    EVENTID="event_id",
    ATTRIBID="attrib_id",
    CATEGORY="category",
    NODETYPE="node_type",
    EDGETYPE="edge_type",
):
    """
    Creates a hypergraph out of the given dataframe, returning the graph
    components as dataframes. The transform reveals relationships between the
    rows and unique values. This transform is useful for lists of events,
    samples, relationships, and other structured high-dimensional data.
    The transform creates a node for every row, and turns a row's column
    entries into node attributes. If direct=False (default), every unique
    value within a column is also turned into a node. Edges are added to
    connect a row's nodes to each of its column nodes, or if direct=True, to
    one another. Nodes are given the attribute specified by NODETYPE
    that corresponds to the originating column name, or if a row EVENTID.
    Consider a list of events. Each row represents a distinct event, and each
    column some metadata about an event. If multiple events have common
    metadata, they will be transitively connected through those metadata
    values. Conversely, if an event has unique metadata, the unique metadata
    will turn into nodes that only have connections to the event node.
    For best results, set EVENTID to a row's unique ID, SKIP to all
    non-categorical columns (or columns to all categorical columns),
    and categories to group columns with the same kinds of values.



    Parameters
    ----------

    values : cudf.DataFrame
        The input Dataframe to transform into a hypergraph.

    columns : sequence, optional (default=None)
        An optional sequence of column names to process.

    dropna : bool, optional (default=True)
        If True, do not include "null" values in the graph.

    direct : bool, optional (default=False)
        If True, omit hypernodes and instead strongly connect nodes for each
        row with each other.

    graph_class : cugraph.Graph, optional (default=cugraph.Graph)
        Specify the type of Graph to create.

    categories : dict, optional (default=dict())
        Dictionary mapping column names to distinct categories. If the same
        value appears columns mapped to the same category, the transform will
        generate one node for it, instead of one for each column.

    drop_edge_attrs : bool, optional, (default=False)
        If True, exclude each row's attributes from its edges

    categorical_metadata : bool, optional (default=True)
        Whether to use cudf.CategoricalDtype for the ``CATEGORY``,
        ``NODETYPE``, and ``EDGETYPE`` columns. These columns are typically
        large string columns with with low cardinality, and using categorical
        dtypes can save a significant amount of memory.

    SKIP : sequence, optional
        A sequence of column names not to transform into nodes.

    EDGES : dict, optional
        When ``direct=True``, select column pairs instead of making all edges.

    DELIM : str, optional (default="::")
        The delimiter to use when joining column names, categories, and ids.

    SOURCE : str, optional (default="src")
        The name to use as the source column in the graph and edge DF.

    TARGET : str, optional (default="dst")
        The name to use as the target column in the graph and edge DF.

    WEIGHTS : str, optional (default=None)
        The column name from the input DF to map as the graph's edge weights.

    NODEID : str, optional (default="node_id")
        The name to use as the node id column in the graph and node DFs.

    EVENTID : str, optional (default="event_id")
        The name to use as the event id column in the graph and node DFs.

    ATTRIBID : str, optional (default="attrib_id")
        The name to use as the attribute id column in the graph and node DFs.

    CATEGORY : str, optional (default "category")
        The name to use as the category column in the graph and DFs.

    NODETYPE : str, optional (default="node_type")
        The name to use as the node type column in the graph and node DFs.

    EDGETYPE : str, optional (default="edge_type")
        The name to use as the edge type column in the graph and edge DF.

    Returns
    -------
    result : dict {"nodes", "edges", "graph", "events", "entities"}

        nodes : cudf.DataFrame
            A DataFrame of found entity and hyper node attributes.
        edges : cudf.DataFrame
            A DataFrame of edge attributes.
        graph : cugraph.Graph
            A Graph of the found entity nodes, hyper nodes, and edges.
        events : cudf.DataFrame
            If direct=True, a DataFrame of hyper node attributes, else empty.
        entities : cudf.DataFrame
            A DataFrame of the found entity node attributes.

    Examples
    --------
    >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                   names=['src', 'dst', 'weights'],
    ...                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> nodes, edges, G, events, entities = cugraph.hypergraph(M)

    """

    columns = values.columns if columns is None else columns
    columns = sorted(
        list(columns if SKIP is None else [x for x in columns if x not in SKIP])
    )

    events = values.copy(deep=False)
    events.reset_index(drop=True, inplace=True)

    if EVENTID not in events.columns:
        events[EVENTID] = cudf.core.index.RangeIndex(len(events))

    events[EVENTID] = _prepend_str(events[EVENTID], EVENTID + DELIM)
    events[NODETYPE] = (
        "event"
        if not categorical_metadata
        else _str_scalar_to_category(len(events), "event")
    )

    if not dropna:
        for key, col in events[columns].items():
            if cudf.api.types.is_string_dtype(col.dtype):
                events[key].fillna("null", inplace=True)

    edges = None
    nodes = None
    entities = _create_entity_nodes(
        events,
        columns,
        dropna=dropna,
        categories=categories,
        categorical_metadata=categorical_metadata,
        DELIM=DELIM,
        NODEID=NODEID,
        CATEGORY=CATEGORY,
        NODETYPE=NODETYPE,
    )

    if direct:
        edges = _create_direct_edges(
            events,
            columns,
            dropna=dropna,
            edge_shape=EDGES,
            categories=categories,
            drop_edge_attrs=drop_edge_attrs,
            categorical_metadata=categorical_metadata,
            DELIM=DELIM,
            SOURCE=SOURCE,
            TARGET=TARGET,
            EVENTID=EVENTID,
            CATEGORY=CATEGORY,
            EDGETYPE=EDGETYPE,
            NODETYPE=NODETYPE,
        )
        nodes = entities
        events = cudf.DataFrame()
    else:
        SOURCE = ATTRIBID
        TARGET = EVENTID
        edges = _create_hyper_edges(
            events,
            columns,
            dropna=dropna,
            categories=categories,
            drop_edge_attrs=drop_edge_attrs,
            categorical_metadata=categorical_metadata,
            DELIM=DELIM,
            EVENTID=EVENTID,
            ATTRIBID=ATTRIBID,
            CATEGORY=CATEGORY,
            EDGETYPE=EDGETYPE,
            NODETYPE=NODETYPE,
        )
        # Concatenate regular nodes and hyper nodes
        events = _create_hyper_nodes(
            events,
            NODEID=NODEID,
            EVENTID=EVENTID,
            CATEGORY=CATEGORY,
            NODETYPE=NODETYPE,
            categorical_metadata=categorical_metadata,
        )
        nodes = cudf.concat([entities, events])
        nodes.reset_index(drop=True, inplace=True)

    if WEIGHTS is not None:
        if WEIGHTS not in edges:
            WEIGHTS = None
        else:
            edges[WEIGHTS].fillna(0, inplace=True)

    graph = graph_class()
    graph.from_cudf_edgelist(
        edges,
        # force using renumber_from_cudf
        source=[SOURCE],
        destination=[TARGET],
        edge_attr=WEIGHTS,
        renumber=True,
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "graph": graph,
        "events": events,
        "entities": entities,
    }


def _create_entity_nodes(
    events,
    columns,
    dropna=True,
    categorical_metadata=False,
    categories=dict(),
    DELIM="::",
    NODEID="node_id",
    CATEGORY="category",
    NODETYPE="node_type",
):
    nodes = [
        cudf.DataFrame(
            dict(
                [
                    (NODEID, cudf.core.column.column_empty(0, "str")),
                    (
                        CATEGORY,
                        cudf.core.column.column_empty(
                            0, "str" if not categorical_metadata else _empty_cat_dt()
                        ),
                    ),
                    (
                        NODETYPE,
                        cudf.core.column.column_empty(
                            0, "str" if not categorical_metadata else _empty_cat_dt()
                        ),
                    ),
                ]
                + [
                    (key, cudf.core.column.column_empty(0, col.dtype))
                    for key, col in events[columns].items()
                ]
            )
        )
    ]

    for key, col in events[columns].items():
        cat = categories.get(key, key)
        col = col.unique()
        col = col.nans_to_nulls().dropna() if dropna else col
        if len(col) == 0:
            continue
        df = cudf.DataFrame(
            {
                key: cudf.core.column.as_column(col),
                NODEID: _prepend_str(col, cat + DELIM),
                CATEGORY: cat
                if not categorical_metadata
                else _str_scalar_to_category(len(col), cat),
                NODETYPE: key
                if not categorical_metadata
                else _str_scalar_to_category(len(col), key),
            }
        )
        df.reset_index(drop=True, inplace=True)
        nodes.append(df)

    nodes = cudf.concat(nodes)
    nodes = nodes.drop_duplicates(subset=[NODEID])
    nodes = nodes[[NODEID, NODETYPE, CATEGORY] + list(columns)]
    nodes.reset_index(drop=True, inplace=True)
    return nodes


def _create_hyper_nodes(
    events,
    categorical_metadata=False,
    NODEID="node_id",
    EVENTID="event_id",
    CATEGORY="category",
    NODETYPE="node_type",
):
    nodes = events.copy(deep=False)
    if NODEID in nodes:
        nodes.drop(columns=[NODEID], inplace=True)
    if NODETYPE in nodes:
        nodes.drop(columns=[NODETYPE], inplace=True)
    if CATEGORY in nodes:
        nodes.drop(columns=[CATEGORY], inplace=True)
    nodes[NODETYPE] = (
        EVENTID
        if not categorical_metadata
        else _str_scalar_to_category(len(nodes), EVENTID)
    )
    nodes[CATEGORY] = (
        "event"
        if not categorical_metadata
        else _str_scalar_to_category(len(nodes), "event")
    )
    nodes[NODEID] = nodes[EVENTID]
    nodes.reset_index(drop=True, inplace=True)
    return nodes


def _create_hyper_edges(
    events,
    columns,
    dropna=True,
    categories=dict(),
    drop_edge_attrs=False,
    categorical_metadata=False,
    DELIM="::",
    EVENTID="event_id",
    ATTRIBID="attrib_id",
    CATEGORY="category",
    EDGETYPE="edge_type",
    NODETYPE="node_type",
):
    edge_attrs = [x for x in events.columns if x != NODETYPE]
    edges = [
        cudf.DataFrame(
            dict(
                (
                    [
                        (EVENTID, cudf.core.column.column_empty(0, "str")),
                        (ATTRIBID, cudf.core.column.column_empty(0, "str")),
                        (
                            EDGETYPE,
                            cudf.core.column.column_empty(
                                0,
                                "str" if not categorical_metadata else _empty_cat_dt(),
                            ),
                        ),
                    ]
                )
                + (
                    []
                    if len(categories) == 0
                    else [
                        (
                            CATEGORY,
                            cudf.core.column.column_empty(
                                0,
                                "str" if not categorical_metadata else _empty_cat_dt(),
                            ),
                        )
                    ]
                )
                + (
                    []
                    if drop_edge_attrs
                    else [
                        (key, cudf.core.column.column_empty(0, col.dtype))
                        for key, col in events[edge_attrs].items()
                    ]
                )
            )
        )
    ]

    for key, col in events[columns].items():
        cat = categories.get(key, key)
        fs = [EVENTID] + ([key] if drop_edge_attrs else edge_attrs)
        df = events[fs].dropna(subset=[key]) if dropna else events[fs]
        if len(df) == 0:
            continue
        if len(categories) > 0:
            df[CATEGORY] = (
                key
                if not categorical_metadata
                else _str_scalar_to_category(len(df), key)
            )
        df[EDGETYPE] = (
            cat if not categorical_metadata else _str_scalar_to_category(len(df), cat)
        )
        df[ATTRIBID] = _prepend_str(col, cat + DELIM)
        df.reset_index(drop=True, inplace=True)
        edges.append(df)

    columns = [EVENTID, EDGETYPE, ATTRIBID]

    if len(categories) > 0:
        columns += [CATEGORY]

    if not drop_edge_attrs:
        columns += edge_attrs

    edges = cudf.concat(edges)[columns]
    edges.reset_index(drop=True, inplace=True)
    return edges


def _create_direct_edges(
    events,
    columns,
    dropna=True,
    categories=dict(),
    edge_shape=None,
    drop_edge_attrs=False,
    categorical_metadata=False,
    DELIM="::",
    SOURCE="src",
    TARGET="dst",
    EVENTID="event_id",
    CATEGORY="category",
    EDGETYPE="edge_type",
    NODETYPE="node_type",
):
    if edge_shape is None:
        edge_shape = {}
        for i, name in enumerate(columns):
            edge_shape[name] = columns[(i + 1) :]

    edge_attrs = [x for x in events.columns if x != NODETYPE]
    edges = [
        cudf.DataFrame(
            dict(
                (
                    [
                        (EVENTID, cudf.core.column.column_empty(0, "str")),
                        (SOURCE, cudf.core.column.column_empty(0, "str")),
                        (TARGET, cudf.core.column.column_empty(0, "str")),
                        (
                            EDGETYPE,
                            cudf.core.column.column_empty(
                                0,
                                "str" if not categorical_metadata else _empty_cat_dt(),
                            ),
                        ),
                    ]
                )
                + (
                    []
                    if len(categories) == 0
                    else [
                        (
                            CATEGORY,
                            cudf.core.column.column_empty(
                                0,
                                "str" if not categorical_metadata else _empty_cat_dt(),
                            ),
                        )
                    ]
                )
                + (
                    []
                    if drop_edge_attrs
                    else [
                        (key, cudf.core.column.column_empty(0, col.dtype))
                        for key, col in events[edge_attrs].items()
                    ]
                )
            )
        )
    ]

    for key1, col1 in events[sorted(edge_shape.keys())].items():
        cat1 = categories.get(key1, key1)

        if isinstance(edge_shape[key1], str):
            edge_shape[key1] = [edge_shape[key1]]
        elif isinstance(edge_shape[key1], dict):
            edge_shape[key1] = list(edge_shape[key1].keys())
        elif not isinstance(edge_shape[key1], (set, list, tuple)):
            raise ValueError("EDGES must be a dict of column name(s)")

        for key2, col2 in events[sorted(edge_shape[key1])].items():
            cat2 = categories.get(key2, key2)
            fs = [EVENTID] + ([key1, key2] if drop_edge_attrs else edge_attrs)
            df = events[fs].dropna(subset=[key1, key2]) if dropna else events[fs]
            if len(df) == 0:
                continue
            if len(categories) > 0:
                df[CATEGORY] = (
                    key1 + DELIM + key2
                    if not categorical_metadata
                    else _str_scalar_to_category(len(df), key1 + DELIM + key2)
                )
            df[EDGETYPE] = (
                cat1 + DELIM + cat2
                if not categorical_metadata
                else _str_scalar_to_category(len(df), cat1 + DELIM + cat2)
            )
            df[SOURCE] = _prepend_str(col1, cat1 + DELIM)
            df[TARGET] = _prepend_str(col2, cat2 + DELIM)
            df.reset_index(drop=True, inplace=True)
            edges.append(df)

    columns = [EVENTID, EDGETYPE, SOURCE, TARGET]

    if len(categories) > 0:
        columns += [CATEGORY]

    if not drop_edge_attrs:
        columns += edge_attrs

    edges = cudf.concat(edges)[columns]
    edges.reset_index(drop=True, inplace=True)
    return edges


def _str_scalar_to_category(size, val):
    return cudf.core.column.build_categorical_column(
        categories=cudf.core.column.as_column([val], dtype="str"),
        codes=cudf.core.column.column.full(size, 0, dtype=np.int32),
        mask=None,
        size=size,
        offset=0,
        null_count=0,
        ordered=False,
    )


def _prepend_str(col, val):
    return val + col.astype(str).fillna("null")


# Make an empty categorical string dtype
def _empty_cat_dt():
    return cudf.core.dtypes.CategoricalDtype(
        categories=np.array([], dtype="str"), ordered=False
    )
