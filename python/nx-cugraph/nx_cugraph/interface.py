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

import nx_cugraph as nxcg


class BackendInterface:
    # Required conversions
    @staticmethod
    def convert_from_nx(graph, *args, edge_attrs=None, weight=None, **kwargs):
        if weight is not None:
            # MAINT: networkx 3.0, 3.1
            # For networkx 3.0 and 3.1 compatibility
            if edge_attrs is not None:
                raise TypeError(
                    "edge_attrs and weight arguments should not both be given"
                )
            edge_attrs = {weight: 1}
        return nxcg.from_networkx(graph, *args, edge_attrs=edge_attrs, **kwargs)

    @staticmethod
    def convert_to_nx(obj, *, name: str | None = None):
        if isinstance(obj, nxcg.Graph):
            return nxcg.to_networkx(obj)
        return obj

    @staticmethod
    def on_start_tests(items):
        """Modify pytest items after tests have been collected.

        This is called during ``pytest_collection_modifyitems`` phase of pytest.
        We use this to set `xfail` on tests we expect to fail. See:

        https://docs.pytest.org/en/stable/reference/reference.html#std-hook-pytest_collection_modifyitems
        """
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

        # Reasons for xfailing
        no_weights = "weighted implementation not currently supported"
        no_multigraph = "multigraphs not currently supported"
        louvain_different = (
            "Louvain may be different due to RNG or unsupported threshold parameter"
        )

        xfail = {}

        from packaging.version import parse

        nxver = parse(nx.__version__)
        if nxver.major == 3 and nxver.minor in {0, 1}:
            # MAINT: networkx 3.0, 3.1
            # NetworkX 3.2 added the ability to "fallback to nx" if backend algorithms
            # raise NotImplementedError or `can_run` returns False. The tests below
            # exercise behavior we have not implemented yet, so we mark them as xfail
            # for previous versions of NetworkX.
            xfail.update(
                {
                    key(
                        "test_agraph.py:TestAGraph.test_no_warnings_raised"
                    ): "pytest.warn(None) deprecated",
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedBetweennessCentrality.test_K5"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedBetweennessCentrality.test_P3_normalized"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedBetweennessCentrality.test_P3"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedBetweennessCentrality.test_krackhardt_kite_graph"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedBetweennessCentrality."
                        "test_krackhardt_kite_graph_normalized"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedBetweennessCentrality."
                        "test_florentine_families_graph"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedBetweennessCentrality.test_les_miserables_graph"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedBetweennessCentrality.test_ladder_graph"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedBetweennessCentrality.test_G"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedBetweennessCentrality.test_G2"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedBetweennessCentrality.test_G3"
                    ): no_multigraph,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedBetweennessCentrality.test_G4"
                    ): no_multigraph,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedEdgeBetweennessCentrality.test_K5"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedEdgeBetweennessCentrality.test_C4"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedEdgeBetweennessCentrality.test_P4"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedEdgeBetweennessCentrality.test_balanced_tree"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedEdgeBetweennessCentrality.test_weighted_graph"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedEdgeBetweennessCentrality."
                        "test_normalized_weighted_graph"
                    ): no_weights,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedEdgeBetweennessCentrality.test_weighted_multigraph"
                    ): no_multigraph,
                    key(
                        "test_betweenness_centrality.py:"
                        "TestWeightedEdgeBetweennessCentrality."
                        "test_normalized_weighted_multigraph"
                    ): no_multigraph,
                }
            )
        else:
            xfail.update(
                {
                    key(
                        "test_louvain.py:test_karate_club_partition"
                    ): louvain_different,
                    key("test_louvain.py:test_none_weight_param"): louvain_different,
                    key("test_louvain.py:test_multigraph"): louvain_different,
                    key("test_louvain.py:test_threshold"): louvain_different,
                }
            )

        for item in items:
            kset = set(item.keywords)
            for (test_name, keywords), reason in xfail.items():
                if item.name == test_name and keywords.issubset(kset):
                    item.add_marker(pytest.mark.xfail(reason=reason))

    @classmethod
    def can_run(cls, name, args, kwargs):
        """Can this backend run the specified algorithms with the given arguments?

        This is a proposed API to add to networkx dispatching machinery and may change.
        """
        return (
            hasattr(cls, name)
            and getattr(cls, name).can_run(*args, **kwargs)
            # We don't support MultiGraphs yet
            and not any(isinstance(x, nx.MultiGraph) for x in args)
            and not any(isinstance(x, nx.MultiGraph) for x in kwargs.values())
        )
