# Copyright (c) 2020, NVIDIA CORPORATION.
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
from cugraph.structure.graph import Graph


_empty_cat_dt = cudf.core.dtypes.CategoricalDtype(ordered=False)


def hypergraph(
    values,
    columns=None,
    dropna=True,
    direct=False,
    graph_class=Graph,
    categories=dict(),
    drop_edge_attrs=False,
    categorical_metadata=False,
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
    columns = values.columns if columns is None else columns
    columns = sorted(list(columns if SKIP is None else [
        x for x in columns if x not in SKIP
    ]))

    events = values.copy(deep=False)
    events.reset_index(drop=True, inplace=True)

    for key, col in events[columns].iteritems():
        if cudf.utils.dtypes.is_categorical_dtype(col.dtype):
            events[columns] = col.astype(col.cat.categories.dtype)

    if EVENTID not in events.columns:
        events[EVENTID] = cudf.core.index.RangeIndex(len(events))

    events[EVENTID] = _prepend_str(events[EVENTID], EVENTID + DELIM)
    events[NODETYPE] = "event" if not categorical_metadata \
        else _str_scalar_to_category(len(events), "event")

    if not dropna:
        for key, col in events[columns].iteritems():
            if cudf.utils.dtypes.is_string_dtype(col.dtype):
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
        nodes = cudf.concat([entities, events], ignore_index=True)
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
    nodes = [cudf.DataFrame(dict([
        (NODEID, cudf.core.column.column_empty(0, "str")),
        (CATEGORY, cudf.core.column.column_empty(0, _empty_cat_dt)),
        (NODETYPE, cudf.core.column.column_empty(0, _empty_cat_dt))
    ] + [
        (key, cudf.core.column.column_empty(0, col.dtype))
        for key, col in events[columns].iteritems()
    ]))]

    for key, col in events[columns].iteritems():
        cat = categories.get(key, key)
        col = col.unique()
        col = col.nans_to_nulls().dropna() if dropna else col
        if len(col) == 0:
            continue
        nodes.append(cudf.DataFrame({
            key: cudf.core.column.as_column(col),
            NODEID: _prepend_str(col, cat + DELIM),
            CATEGORY: cat if not categorical_metadata
            else _str_scalar_to_category(len(events), cat),
            NODETYPE: key if not categorical_metadata
            else _str_scalar_to_category(len(col), key),
        }))

    nodes = cudf.concat(nodes, ignore_index=True)
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
        nodes.drop([NODEID], inplace=True)
    if NODETYPE in nodes:
        nodes.drop([NODETYPE], inplace=True)
    if CATEGORY in nodes:
        nodes.drop([CATEGORY], inplace=True)
    nodes[NODETYPE] = EVENTID if not categorical_metadata \
        else _str_scalar_to_category(len(nodes), EVENTID)
    nodes[CATEGORY] = "event" if not categorical_metadata \
        else _str_scalar_to_category(len(nodes), "event")
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
    edges = [cudf.DataFrame(dict(
        ([
            (EVENTID, cudf.core.column.column_empty(0, "str")),
            (ATTRIBID, cudf.core.column.column_empty(0, "str")),
            (EDGETYPE, cudf.core.column.column_empty(0, _empty_cat_dt))
        ]) +
        ([] if len(categories) == 0 else [
            (CATEGORY, cudf.core.column.column_empty(0, _empty_cat_dt))
        ]) +
        ([] if drop_edge_attrs else [
            (key, cudf.core.column.column_empty(0, col.dtype))
            for key, col in events[edge_attrs].iteritems()
        ])
    ))]

    for key, col in events[columns].iteritems():
        cat = categories.get(key, key)
        fs = [EVENTID] + ([key] if drop_edge_attrs else edge_attrs)
        df = events[fs].dropna(subset=[key]) if dropna else events[fs]
        if len(df) == 0:
            continue
        if len(categories) > 0:
            df[CATEGORY] = key if not categorical_metadata \
                else _str_scalar_to_category(len(df), key)
        df[EDGETYPE] = cat if not categorical_metadata \
            else _str_scalar_to_category(len(df), cat)
        df[ATTRIBID] = _prepend_str(col, cat + DELIM)
        edges.append(df)

    columns = [EVENTID, EDGETYPE, ATTRIBID]

    if len(categories) > 0:
        columns += [CATEGORY]

    if not drop_edge_attrs:
        columns += edge_attrs

    edges = cudf.concat(edges, ignore_index=True)[columns]
    edges.reset_index(drop=True, inplace=True)
    return edges


def _create_direct_edges(
    events,
    columns,
    dropna=True,
    categories=dict(),
    edge_shape=dict(),
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
            edge_shape[name] = columns[(i + 1):]

    edge_attrs = [x for x in events.columns if x != NODETYPE]
    edges = [cudf.DataFrame(dict(
        ([
            (EVENTID, cudf.core.column.column_empty(0, "str")),
            (SOURCE, cudf.core.column.column_empty(0, "str")),
            (TARGET, cudf.core.column.column_empty(0, "str")),
            (EDGETYPE, cudf.core.column.column_empty(0, _empty_cat_dt))
        ]) +
        ([] if len(categories) == 0 else [
            (CATEGORY, cudf.core.column.column_empty(0, _empty_cat_dt))
        ]) +
        ([] if drop_edge_attrs else [
            (key, cudf.core.column.column_empty(0, col.dtype))
            for key, col in events[edge_attrs].iteritems()
        ])
    ))]

    for key1, col1 in events[sorted(edge_shape.keys())].iteritems():
        cat1 = categories.get(key1, key1)

        if isinstance(edge_shape[key1], str):
            edge_shape[key1] = [edge_shape[key1]]
        elif isinstance(edge_shape[key1], dict):
            edge_shape[key1] = list(edge_shape[key1].keys())
        elif not isinstance(edge_shape[key1], (set, list, tuple)):
            raise ValueError("EDGES must be a dict of column name(s)")

        for key2, col2 in events[sorted(edge_shape[key1])].iteritems():
            cat2 = categories.get(key2, key2)
            fs = [EVENTID] + ([key1, key2] if drop_edge_attrs else edge_attrs)
            df = (
                events[fs].dropna(subset=[key1, key2])
                if dropna else events[fs]
            )
            if len(df) == 0:
                continue
            if len(categories) > 0:
                df[CATEGORY] = key1 + DELIM + key2 \
                    if not categorical_metadata \
                    else _str_scalar_to_category(
                        len(df), key1 + DELIM + key2
                    )
            df[EDGETYPE] = cat1 + DELIM + cat2 \
                if not categorical_metadata \
                else _str_scalar_to_category(
                    len(df), cat1 + DELIM + cat2
                )
            df[SOURCE] = _prepend_str(col1, cat1 + DELIM)
            df[TARGET] = _prepend_str(col2, cat2 + DELIM)
            edges.append(df)

    columns = [EVENTID, EDGETYPE, SOURCE, TARGET]

    if len(categories) > 0:
        columns += [CATEGORY]

    if not drop_edge_attrs:
        columns += edge_attrs

    edges = cudf.concat(edges, ignore_index=True)[columns]
    edges.reset_index(drop=True, inplace=True)
    return edges


def _str_scalar_to_category(size, val):
    return cudf.core.column.build_categorical_column(
        categories=cudf.core.column.as_column([val], dtype="str"),
        codes=cudf.utils.utils.scalar_broadcast_to(0, size, dtype=np.int32),
        mask=None,
        size=size,
        offset=0,
        null_count=0,
        ordered=False,
    )


def _prepend_str(col, val):
    col = cudf.core.column.as_column(col)
    if not cudf.utils.dtypes.is_string_dtype(col.dtype):
        col = col.astype("str")
    return cudf.Series(col.str().insert(0, val))
