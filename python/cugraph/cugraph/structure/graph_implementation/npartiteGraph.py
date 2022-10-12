# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

from .simpleGraph import simpleGraphImpl
import cudf


class npartiteGraphImpl(simpleGraphImpl):
    def __init__(self, properties):
        super(npartiteGraphImpl, self).__init__(properties)
        self.properties.bipartite = properties.bipartite

    # API may change in future
    def __from_edgelist(
        self,
        input_df,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True,
    ):
        self._simpleGraphImpl__from_edgelist(
            input_df,
            source=source,
            destination=destination,
            edge_attr=edge_attr,
            renumber=renumber,
        )

    def sets(self):
        """
        Returns the bipartite set of nodes. This solely relies on the user's
        call of add_nodes_from with the bipartite parameter. This does not
        parse the graph to compute bipartite sets. If bipartite argument was
        not provided during add_nodes_from(), it raise an exception that the
        graph is not bipartite.
        """
        # TO DO: Call coloring algorithm
        set_names = [i for i in self._nodes.keys() if i != "all_nodes"]
        if self.properties.bipartite:
            top = self._nodes[set_names[0]]
            if len(set_names) == 2:
                bottom = self._nodes[set_names[1]]
            else:
                bottom = cudf.Series(
                    set(self.nodes().values_host) - set(top.values_host)
                )
            return top, bottom
        else:
            return {k: self._nodes[k] for k in set_names}

    # API may change in future
    def add_nodes_from(self, nodes, bipartite=None, multipartite=None):
        """
        Add nodes information to the Graph.

        Parameters
        ----------
        nodes : list or cudf.Series
            The nodes of the graph to be stored. If bipartite and multipartite
            arguments are not passed, the nodes are considered to be a list of
            all the nodes present in the Graph.
        bipartite : str, optional (default=None)
            Sets the Graph as bipartite. The nodes are stored as a set of nodes
            of the partition named as bipartite argument.
        multipartite : str, optional (default=None)
            Sets the Graph as multipartite. The nodes are stored as a set of
            nodes of the partition named as multipartite argument.
        """
        if bipartite is None and multipartite is None:
            raise Exception("Partition not provided")
        else:
            set_names = [i for i in self._nodes.keys() if i != "all_nodes"]
            if multipartite is not None:
                if self.properties.bipartite:
                    raise Exception(
                        "The Graph is bipartite. " "Use bipartite option instead."
                    )
            elif bipartite is not None:
                if not self.properties.bipartite:
                    raise Exception(
                        "The Graph is set as npartite. "
                        "Use multipartite option instead."
                    )
                multipartite = bipartite
                if multipartite not in set_names and len(set_names) == 2:
                    raise Exception(
                        "The Graph is set as bipartite and "
                        "already has two partitions initialized."
                    )
            self._nodes[multipartite] = cudf.Series(nodes)
