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
from __future__ import annotations

import networkx as nx

import cugraph_nx as cnx

from . import algorithms


class Dispatcher:
    betweenness_centrality = algorithms.betweenness_centrality
    edge_betweenness_centrality = algorithms.edge_betweenness_centrality

    # Required conversions
    @staticmethod
    def convert_from_nx(
        graph, *args, edge_attrs=None, weight=None, is_testing=False, **kwargs
    ):
        if weight is not None:
            # For networkx 3.0 and 3.1 compatibility
            if edge_attrs is not None:
                raise TypeError(
                    "edge_attrs and weight arguments should not both be given"
                )
            edge_attrs = {weight: 1}
        if is_testing and isinstance(graph, nx.MultiGraph):
            return graph  # TODO: implement digraph support
        return cnx.from_networkx(graph, *args, edge_attrs=edge_attrs, **kwargs)

    @staticmethod
    def convert_to_nx(obj, *, name: str | None = None):
        if isinstance(obj, cnx.Graph):
            return cnx.to_networkx(obj)
        return obj

    @staticmethod
    def on_start_tests(items):
        try:
            import pytest
        except ModuleNotFoundError:
            return

        def key(testpath):
            filename, path = testpath.split(":")
            *names, testname = path.split(".")
            if names:
                [classname] = names
                return (testname, frozenset({classname, filename}))
            return (testname, frozenset({filename}))

        string_attribute = "unable to handle string attributes"

        skip = {
            key("test_pajek.py:TestPajek.test_ignored_attribute"): string_attribute,
            key(
                "test_agraph.py:TestAGraph.test_no_warnings_raised"
            ): "pytest.warn(None) deprecated",
        }
        for item in items:
            kset = set(item.keywords)
            for (test_name, keywords), reason in skip.items():
                if item.name == test_name and keywords.issubset(kset):
                    item.add_marker(pytest.mark.xfail(reason=reason))
