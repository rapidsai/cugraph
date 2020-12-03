# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

from cugraph.structure import graph_primtypes_wrapper
from cugraph.structure.symmetrize import symmetrize
from cugraph.structure.number_map import NumberMap
from cugraph.dask.common.input_utils import get_local_data
import cugraph.dask.common.mg_utils as mg_utils
import cudf
import dask_cudf
import cugraph.comms.comms as Comms
import pandas as pd
import numpy as np
from cugraph.dask.structure import replication


def null_check(col):
    if col.null_count != 0:
        raise ValueError("Series contains NULL values")


class Graph:
    class EdgeList:
        def __init__(self, *args):
            if len(args) == 1:
                self.__from_dask_cudf(*args)
            else:
                self.__from_cudf(*args)

        def __from_cudf(self, source, destination, edge_attr=None):
            self.edgelist_df = cudf.DataFrame()
            self.edgelist_df["src"] = source
            self.edgelist_df["dst"] = destination
            self.weights = False
            if edge_attr is not None:
                self.weights = True
                if type(edge_attr) is dict:
                    for k in edge_attr.keys():
                        self.edgelist_df[k] = edge_attr[k]
                else:
                    self.edgelist_df["weights"] = edge_attr

        def __from_dask_cudf(self, ddf):
            self.edgelist_df = ddf
            self.weights = False
            # FIXME: Edge Attribute not handled

    class AdjList:
        def __init__(self, offsets, indices, value=None):
            self.offsets = offsets
            self.indices = indices
            self.weights = value  # Should be a dataframe for multiple weights

    class transposedAdjList:
        def __init__(self, offsets, indices, value=None):
            Graph.AdjList.__init__(self, offsets, indices, value)

    """
    cuGraph graph class containing basic graph creation and transformation
    operations.
    """

    def __init__(
        self,
        m_graph=None,
        edge_attr=None,
        symmetrized=False,
        bipartite=False,
        multi=False,
        dynamic=False,
    ):
        """
        Returns
        -------
        G : cuGraph.Graph.

        Examples
        --------
        >>> import cuGraph
        >>> G = cuGraph.Graph()

        """
        self.symmetrized = symmetrized
        self.renumbered = False
        self.renumber_map = None
        self.bipartite = False
        self.multipartite = False
        self._nodes = {}
        self.multi = multi
        self.distributed = False
        self.dynamic = dynamic
        self.self_loop = False
        self.edgelist = None
        self.adjlist = None
        self.transposedadjlist = None
        self.edge_count = None
        self.node_count = None

        # MG - Batch
        self.batch_enabled = False
        self.batch_edgelists = None
        self.batch_adjlists = None
        self.batch_transposed_adjlists = None

        if m_graph is not None:
            if (type(self) is Graph and type(m_graph) is MultiGraph) or (
                type(self) is DiGraph and type(m_graph) is MultiDiGraph
            ):
                self.from_cudf_edgelist(
                    m_graph.edgelist.edgelist_df,
                    source="src",
                    destination="dst",
                    edge_attr=edge_attr,
                )
                self.renumbered = m_graph.renumbered
                self.renumber_map = m_graph.renumber_map
            else:
                msg = (
                    "Graph can be initialized using MultiGraph "
                    "and DiGraph can be initialized using MultiDiGraph"
                )
                raise Exception(msg)
        # self.number_of_vertices = None

    def enable_batch(self):
        client = mg_utils.get_client()
        comms = Comms.get_comms()

        if client is None or comms is None:
            msg = (
                "MG Batch needs a Dask Client and the "
                "Communicator needs to be initialized."
            )
            raise Exception(msg)

        self.batch_enabled = True

        if self.edgelist is not None:
            if self.batch_edgelists is None:
                self._replicate_edgelist()

        if self.adjlist is not None:
            if self.batch_adjlists is None:
                self._replicate_adjlist()

        if self.transposedadjlist is not None:
            if self.batch_transposed_adjlists is None:
                self._replicate_transposed_adjlist()

    def _replicate_edgelist(self):
        client = mg_utils.get_client()
        comms = Comms.get_comms()

        # FIXME: There  might be a better way to control it
        if client is None:
            return
        work_futures = replication.replicate_cudf_dataframe(
            self.edgelist.edgelist_df, client=client, comms=comms
        )

        self.batch_edgelists = work_futures

    def _replicate_adjlist(self):
        client = mg_utils.get_client()
        comms = Comms.get_comms()

        # FIXME: There  might be a better way to control it
        if client is None:
            return

        weights = None
        offsets_futures = replication.replicate_cudf_series(
            self.adjlist.offsets, client=client, comms=comms
        )
        indices_futures = replication.replicate_cudf_series(
            self.adjlist.indices, client=client, comms=comms
        )

        if self.adjlist.weights is not None:
            weights = replication.replicate_cudf_series(self.adjlist.weights)
        else:
            weights = {worker: None for worker in offsets_futures}

        merged_futures = {
            worker: [
                offsets_futures[worker],
                indices_futures[worker],
                weights[worker],
            ]
            for worker in offsets_futures
        }
        self.batch_adjlists = merged_futures

    # FIXME: Not implemented yet
    def _replicate_transposed_adjlist(self):
        self.batch_transposed_adjlists = True

    def clear(self):
        """
        Empty this graph. This function is added for NetworkX compatibility.
        """
        self.edgelist = None
        self.adjlist = None
        self.transposedadjlist = None

        self.batch_edgelists = None
        self.batch_adjlists = None
        self.batch_transposed_adjlists = None

    def add_nodes_from(self, nodes, bipartite=None, multipartite=None):
        """
        Add nodes information to the Graph.

        Parameters
        ----------
        nodes : list or cudf.Series
            The nodes of the graph to be stored. If bipartite and multipartite
            arguments are not passed, the nodes are considered to be a list of
            all the nodes present in the Graph.
        bipartite : str
            Sets the Graph as bipartite. The nodes are stored as a set of nodes
            of the partition named as bipartite argument.
        multipartite : str
            Sets the Graph as multipartite. The nodes are stored as a set of
            nodes of the partition named as multipartite argument.
        """
        if bipartite is None and multipartite is None:
            self._nodes["all_nodes"] = cudf.Series(nodes)
        else:
            set_names = [i for i in self._nodes.keys() if i != "all_nodes"]
            if multipartite is not None:
                if self.bipartite:
                    raise Exception(
                        "The Graph is already set as bipartite. "
                        "Use bipartite option instead."
                    )
                self.multipartite = True
            elif bipartite is not None:
                if self.multipartite:
                    raise Exception(
                        "The Graph is set as multipartite. "
                        "Use multipartite option instead."
                    )
                self.bipartite = True
                multipartite = bipartite
                if multipartite not in set_names and len(set_names) == 2:
                    raise Exception(
                        "The Graph is set as bipartite and "
                        "already has two partitions initialized."
                    )
            self._nodes[multipartite] = cudf.Series(nodes)

    def is_bipartite(self):
        """
        Checks if Graph is bipartite. This solely relies on the user call of
        add_nodes_from with the bipartite parameter. This does not parse the
        graph to check if it is bipartite.
        """
        # TO DO: Call coloring algorithm
        return self.bipartite

    def is_multipartite(self):
        """
        Checks if Graph is multipartite. This solely relies on the user call
        of add_nodes_from with the partition parameter. This does not parse
        the graph to check if it is multipartite.
        """
        # TO DO: Call coloring algorithm
        return self.multipartite or self.bipartite

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
        if self.bipartite:
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

    def from_cudf_edgelist(
        self,
        input_df,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True,
    ):
        """
        Initialize a graph from the edge list. It is an error to call this
        method on an initialized Graph object. The passed input_df argument
        wraps gdf_column objects that represent a graph using the edge list
        format. source argument is source column name and destination argument
        is destination column name.

        By default, renumbering is enabled to map the source and destination
        vertices into an index in the range [0, V) where V is the number
        of vertices.  If the input vertices are a single column of integers
        in the range [0, V), renumbering can be disabled and the original
        external vertex ids will be used.

        If weights are present, edge_attr argument is the weights column name.

        Parameters
        ----------
        input_df : cudf.DataFrame or dask_cudf.DataFrame
            A DataFrame that contains edge information
            If a dask_cudf.DataFrame is passed it will be reinterpreted as
            a cudf.DataFrame. For the distributed path please use
            from_dask_cudf_edgelist.
        source : str or array-like
            source column name or array of column names
        destination : str or array-like
            destination column name or array of column names
        edge_attr : str or None
            the weights column name. Default is None
        renumber : bool
            Indicate whether or not to renumber the source and destination
            vertex IDs. Default is True.

        Examples
        --------
        >>> df = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(df, source='0', destination='1',
                                 edge_attr='2', renumber=False)

        """
        if self.edgelist is not None or self.adjlist is not None:
            raise Exception("Graph already has values")

        s_col = source
        d_col = destination
        if not isinstance(s_col, list):
            s_col = [s_col]
        if not isinstance(d_col, list):
            d_col = [d_col]
        if not (
            set(s_col).issubset(set(input_df.columns))
            and set(d_col).issubset(set(input_df.columns))
        ):
            raise Exception(
                "source column names and/or destination column "
                "names not found in input. Recheck the source and "
                "destination parameters"
            )

        # FIXME: update for smaller GPUs
        # Consolidation
        if isinstance(input_df, cudf.DataFrame):
            if len(input_df[source]) > 2147483100:
                raise Exception(
                    "cudf dataFrame edge list is too big "
                    "to fit in a single GPU"
                )
            elist = input_df
        elif isinstance(input_df, dask_cudf.DataFrame):
            if len(input_df[source]) > 2147483100:
                raise Exception(
                    "dask_cudf dataFrame edge list is too big "
                    "to fit in a single GPU"
                )
            elist = input_df.compute().reset_index(drop=True)
        else:
            raise Exception(
                "input should be a cudf.DataFrame or "
                "a dask_cudf dataFrame"
            )

        renumber_map = None
        if renumber:
            # FIXME: Should SG do lazy evaluation like MG?
            elist, renumber_map = NumberMap.renumber(
                elist, source, destination, store_transposed=False
            )
            source = "src"
            destination = "dst"
            self.renumbered = True
            self.renumber_map = renumber_map
        else:
            if type(source) is list and type(destination) is list:
                raise Exception("set renumber to True for multi column ids")

        if (elist[source] == elist[destination]).any():
            self.self_loop = True
        source_col = elist[source]
        dest_col = elist[destination]

        if self.multi:
            if type(edge_attr) is not list:
                raise Exception("edge_attr should be a list of column names")
            value_col = {}
            for col_name in edge_attr:
                value_col[col_name] = elist[col_name]
        elif edge_attr is not None:
            value_col = elist[edge_attr]
        else:
            value_col = None

        if not self.symmetrized and not self.multi:
            if value_col is not None:
                source_col, dest_col, value_col = symmetrize(
                    source_col, dest_col, value_col
                )
            else:
                source_col, dest_col = symmetrize(source_col, dest_col)

        self.edgelist = Graph.EdgeList(source_col, dest_col, value_col)

        if self.batch_enabled:
            self._replicate_edgelist()

        self.renumber_map = renumber_map

    def from_pandas_edgelist(
        self,
        pdf,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True,
    ):
        """
        Initialize a graph from the edge list. It is an error to call this
        method on an initialized Graph object. Source argument is source
        column name and destination argument is destination column name.

        By default, renumbering is enabled to map the source and destination
        vertices into an index in the range [0, V) where V is the number
        of vertices.  If the input vertices are a single column of integers
        in the range [0, V), renumbering can be disabled and the original
        external vertex ids will be used.

        If weights are present, edge_attr argument is the weights column name.

        Parameters
        ----------
        input_df : pandas.DataFrame
            A DataFrame that contains edge information
        source : str or array-like
            source column name or array of column names
        destination : str or array-like
            destination column name or array of column names
        edge_attr : str or None
            the weights column name. Default is None
        renumber : bool
            Indicate whether or not to renumber the source and destination
            vertex IDs. Default is True.

        Examples
        --------
        >>> df = pandas.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_pandas_edgelist(df, source='0', destination='1',
                                 edge_attr='2', renumber=False)

        """
        gdf = cudf.DataFrame.from_pandas(pdf)
        self.from_cudf_edgelist(gdf, source=source, destination=destination,
                                edge_attr=edge_attr, renumber=renumber)

    def to_pandas_edgelist(self, source='source', destination='destination'):
        """
        Returns the graph edge list as a Pandas DataFrame.

        Parameters
        ----------
        source : str or array-like
            source column name or array of column names
        destination : str or array-like
            destination column name or array of column names

        Returns
        -------
        df : pandas.DataFrame
        """

        gdf = self.view_edge_list()
        return gdf.to_pandas()

    def from_pandas_adjacency(self, pdf):
        """
        Initializes the graph from pandas adjacency matrix
        """
        np_array = pdf.to_numpy()
        columns = pdf.columns
        self.from_numpy_array(np_array, columns)

    def to_pandas_adjacency(self):
        """
        Returns the graph adjacency matrix as a Pandas DataFrame.
        """

        np_array_data = self.to_numpy_array()
        pdf = pd.DataFrame(np_array_data)
        if self.renumbered:
            nodes = self.renumber_map.implementation.df['0'].\
                    values_host.tolist()
        pdf.columns = nodes
        pdf.index = nodes
        return pdf

    def to_numpy_array(self):
        """
        Returns the graph adjacency matrix as a NumPy array.
        """

        nlen = self.number_of_nodes()
        elen = self.number_of_edges()
        df = self.edgelist.edgelist_df
        np_array = np.full((nlen, nlen), 0.0)
        for i in range(0, elen):
            np_array[df['src'].iloc[i], df['dst'].iloc[i]] = df['weights'].\
                                                             iloc[i]
        return np_array

    def to_numpy_matrix(self):
        """
        Returns the graph adjacency matrix as a NumPy matrix.
        """
        np_array = self.to_numpy_array()
        return np.asmatrix(np_array)

    def from_numpy_array(self, np_array, nodes=None):
        """
        Initializes the graph from numpy array containing adjacency matrix.
        """
        src, dst = np_array.nonzero()
        weight = np_array[src, dst]
        df = cudf.DataFrame()
        if nodes is not None:
            df['src'] = nodes[src]
            df['dst'] = nodes[dst]
        else:
            df['src'] = src
            df['dst'] = dst
        df['weight'] = weight
        self.from_cudf_edgelist(df, 'src', 'dst', edge_attr='weight')

    def from_numpy_matrix(self, np_matrix):
        """
        Initializes the graph from numpy matrix containing adjacency matrix.
        """
        np_array = np.asarray(np_matrix)
        self.from_numpy_array(np_array)

    def from_dask_cudf_edgelist(
        self,
        input_ddf,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True,
    ):
        """
        Initializes the distributed graph from the dask_cudf.DataFrame
        edgelist. Undirected Graphs are not currently supported.

        By default, renumbering is enabled to map the source and destination
        vertices into an index in the range [0, V) where V is the number
        of vertices.  If the input vertices are a single column of integers
        in the range [0, V), renumbering can be disabled and the original
        external vertex ids will be used.

        Note that the graph object will store a reference to the
        dask_cudf.DataFrame provided.

        Parameters
        ----------
        input_ddf : dask_cudf.DataFrame
            The edgelist as a dask_cudf.DataFrame
        source : str or array-like
            source column name or array of column names
        destination : str
            destination column name or array of column names
        edge_attr : str
            weights column name.
        renumber : bool
            If source and destination indices are not in range 0 to V where V
            is number of vertices, renumber argument should be True.
        """
        if self.edgelist is not None or self.adjlist is not None:
            raise Exception("Graph already has values")
        if not isinstance(input_ddf, dask_cudf.DataFrame):
            raise Exception("input should be a dask_cudf dataFrame")
        if type(self) is Graph:
            raise Exception("Undirected distributed graph not supported")

        s_col = source
        d_col = destination
        if not isinstance(s_col, list):
            s_col = [s_col]
        if not isinstance(d_col, list):
            d_col = [d_col]
        if not (
            set(s_col).issubset(set(input_ddf.columns))
            and set(d_col).issubset(set(input_ddf.columns))
        ):
            raise Exception(
                "source column names and/or destination column "
                "names not found in input. Recheck the source "
                "and destination parameters"
            )
        ddf_columns = s_col + d_col
        if edge_attr is not None:
            if not (set([edge_attr]).issubset(set(input_ddf.columns))):
                raise Exception(
                    "edge_attr column name not found in input."
                    "Recheck the edge_attr parameter")
            ddf_columns = ddf_columns + [edge_attr]
        input_ddf = input_ddf[ddf_columns]

        if edge_attr is not None:
            input_ddf = input_ddf.rename(columns={edge_attr: 'value'})

        #
        # Keep all of the original parameters so we can lazily
        # evaluate this function
        #

        # FIXME: Edge Attribute not handled
        self.distributed = True
        self.local_data = None
        self.edgelist = None
        self.adjlist = None
        self.renumbered = renumber
        self.input_df = input_ddf
        self.source_columns = source
        self.destination_columns = destination
        self.store_tranposed = None

    def compute_local_data(self, by, load_balance=True):
        """
        Compute the local edges, vertices and offsets for a distributed
        graph stored as a dask-cudf dataframe and initialize the
        communicator. Performs global sorting and load_balancing.

        Parameters
        ----------
        by : str
            by argument is the column by which we want to sort and
            partition. It should be the source column name for generating
            CSR format and destination column name for generating CSC
            format.
        load_balance : bool
            Set as True to perform load_balancing after global sorting of
            dask-cudf DataFrame. This ensures that the data is uniformly
            distributed among multiple GPUs to avoid over-loading.
        """
        if self.distributed:
            data = get_local_data(self, by, load_balance)
            self.local_data = {}
            self.local_data["data"] = data
            self.local_data["by"] = by
        else:
            raise Exception("Graph should be a distributed graph")

    def view_edge_list(self):
        """
        Display the edge list. Compute it if needed.

        NOTE: If the graph is of type Graph() then the displayed undirected
        edges are the same as displayed by networkx Graph(), but the direction
        could be different i.e. an edge displayed by cugraph as (src, dst)
        could be displayed as (dst, src) by networkx.

        cugraph.Graph stores symmetrized edgelist internally. For displaying
        undirected edgelist for a Graph the upper trianglar matrix of the
        symmetrized edgelist is returned.

        networkx.Graph renumbers the input and stores the upper triangle of
        this renumbered input. Since the internal renumbering of networx and
        cugraph is different, the upper triangular matrix of networkx
        renumbered input may not be the same as cugraph's upper trianglar
        matrix of the symmetrized edgelist. Hence the displayed source and
        destination pairs in both will represent the same edge but node values
        could be swapped.

        Returns
        -------
        df : cudf.DataFrame
            This cudf.DataFrame wraps source, destination and weight

            df[src] : cudf.Series
                contains the source index for each edge
            df[dst] : cudf.Series
                contains the destination index for each edge
            df[weight] : cusd.Series
                Column is only present for weighted Graph,
                then containing the weight value for each edge
        """
        if self.distributed:
            if self.edgelist is None:
                raise Exception("Graph has no Edgelist.")
            return self.edgelist.edgelist_df
        if self.edgelist is None:
            src, dst, weights = graph_primtypes_wrapper.view_edge_list(self)
            self.edgelist = self.EdgeList(src, dst, weights)

        edgelist_df = self.edgelist.edgelist_df

        if self.renumbered:
            edgelist_df = self.unrenumber(edgelist_df, "src")
            edgelist_df = self.unrenumber(edgelist_df, "dst")

        if type(self) is Graph:
            edgelist_df = edgelist_df[edgelist_df["src"] <= edgelist_df["dst"]]
            edgelist_df = edgelist_df.reset_index(drop=True)
            self.edge_count = len(edgelist_df)

        return edgelist_df

    def delete_edge_list(self):
        """
        Delete the edge list.
        """
        # decrease reference count to free memory if the referenced objects are
        # no longer used.
        self.edgelist = None

    def from_cudf_adjlist(self, offset_col, index_col, value_col=None):
        """
        Initialize a graph from the adjacency list. It is an error to call this
        method on an initialized Graph object. The passed offset_col and
        index_col arguments wrap gdf_column objects that represent a graph
        using the adjacency list format.
        If value_col is None, an unweighted graph is created. If value_col is
        not None, a weighted graph is created.
        Undirected edges must be stored as directed edges in both directions.

        Parameters
        ----------
        offset_col : cudf.Series
            This cudf.Series wraps a gdf_column of size V + 1 (V: number of
            vertices).
            The gdf column contains the offsets for the vertices in this graph.
            Offsets must be in the range [0, E] (E: number of edges).
        index_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the destination index for each edge.
            Destination indices must be in the range [0, V) (V: number of
            vertices).
        value_col : cudf.Series, optional
            This pointer can be ``None``.
            If not, this cudf.Series wraps a gdf_column of size E (E: number of
            edges).
            The gdf column contains the weight value for each edge.
            The expected type of the gdf_column element is floating point
            number.

        Examples
        --------
        >>> gdf = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> M = gdf.to_pandas()
        >>> M = scipy.sparse.coo_matrix((M['2'],(M['0'],M['1'])))
        >>> M = M.tocsr()
        >>> offsets = cudf.Series(M.indptr)
        >>> indices = cudf.Series(M.indices)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_adjlist(offsets, indices, None)

        """
        if self.edgelist is not None or self.adjlist is not None:
            raise Exception("Graph already has values")
        self.adjlist = Graph.AdjList(offset_col, index_col, value_col)

        if self.batch_enabled:
            self._replicate_adjlist()

    def compute_renumber_edge_list(self, transposed=False):
        """
        Compute a renumbered edge list

        This function works in the MNMG pipeline and will transform
        the input dask_cudf.DataFrame into a renumbered edge list
        in the prescribed direction.

        This function will be called by the algorithms to ensure
        that the graph is renumbered properly.  The graph object will
        cache the most recent renumbering attempt.  For benchmarking
        purposes, this function can be called prior to calling a
        graph algorithm so we can measure the cost of computing
        the renumbering separately from the cost of executing the
        algorithm.

        When creating a CSR-like structure, set transposed to False.
        When creating a CSC-like structure, set transposed to True.

        Parameters
        ----------
        transposed : (optional) bool
            If True, renumber with the intent to make a CSC-like
            structure.  If False, renumber with the intent to make
            a CSR-like structure.  Defaults to False.
        """
        # FIXME:  What to do about edge_attr???
        #         currently ignored for MNMG

        if not self.distributed:
            raise Exception(
                "compute_renumber_edge_list should only be used "
                "for distributed graphs"
            )

        if not self.renumbered:
            self.edgelist = self.EdgeList(self.input_df)
            self.renumber_map = None
        else:
            if self.edgelist is not None:
                if type(self) is Graph:
                    return

                if self.store_transposed == transposed:
                    return

                del self.edgelist

            renumbered_ddf, number_map = NumberMap.renumber(
                self.input_df,
                self.source_columns,
                self.destination_columns,
                store_transposed=transposed,
            )
            self.edgelist = self.EdgeList(renumbered_ddf)
            self.renumber_map = number_map
            self.store_transposed = transposed

    def view_adj_list(self):
        """
        Display the adjacency list. Compute it if needed.

        Returns
        -------
        offset_col : cudf.Series
            This cudf.Series wraps a gdf_column of size V + 1 (V: number of
            vertices).
            The gdf column contains the offsets for the vertices in this graph.
            Offsets are in the range [0, E] (E: number of edges).
        index_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the destination index for each edge.
            Destination indices are in the range [0, V) (V: number of
            vertices).
        value_col : cudf.Series or ``None``
            This pointer is ``None`` for unweighted graphs.
            For weighted graphs, this cudf.Series wraps a gdf_column of size E
            (E: number of edges).
            The gdf column contains the weight value for each edge.
            The expected type of the gdf_column element is floating point
            number.
        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")

        if self.adjlist is None:
            if self.transposedadjlist is not None and type(self) is Graph:
                off, ind, vals = (
                    self.transposedadjlist.offsets,
                    self.transposedadjlist.indices,
                    self.transposedadjlist.weights,
                )
            else:
                off, ind, vals = graph_primtypes_wrapper.view_adj_list(self)
            self.adjlist = self.AdjList(off, ind, vals)

            if self.batch_enabled:
                self._replicate_adjlist()

        return self.adjlist.offsets, self.adjlist.indices, self.adjlist.weights

    def view_transposed_adj_list(self):
        """
        Display the transposed adjacency list. Compute it if needed.

        Returns
        -------
        offset_col : cudf.Series
            This cudf.Series wraps a gdf_column of size V + 1 (V: number of
            vertices).
            The gdf column contains the offsets for the vertices in this graph.
            Offsets are in the range [0, E] (E: number of edges).
        index_col : cudf.Series
            This cudf.Series wraps a gdf_column of size E (E: number of edges).
            The gdf column contains the destination index for each edge.
            Destination indices are in the range [0, V) (V: number of
            vertices).
        value_col : cudf.Series or ``None``
            This pointer is ``None`` for unweighted graphs.
            For weighted graphs, this cudf.Series wraps a gdf_column of size E
            (E: number of edges).
            The gdf column contains the weight value for each edge.
            The expected type of the gdf_column element is floating point
            number.

        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        if self.transposedadjlist is None:
            if self.adjlist is not None and type(self) is Graph:
                off, ind, vals = (
                    self.adjlist.offsets,
                    self.adjlist.indices,
                    self.adjlist.weights,
                )
            else:
                (
                    off,
                    ind,
                    vals,
                ) = graph_primtypes_wrapper.view_transposed_adj_list(self)
            self.transposedadjlist = self.transposedAdjList(off, ind, vals)

            if self.batch_enabled:
                self._replicate_transposed_adjlist()

        return (
            self.transposedadjlist.offsets,
            self.transposedadjlist.indices,
            self.transposedadjlist.weights,
        )

    def delete_adj_list(self):
        """
        Delete the adjacency list.
        """
        self.adjlist = None

    def get_two_hop_neighbors(self):
        """
        Compute vertex pairs that are two hops apart. The resulting pairs are
        sorted before returning.

        Returns
        -------
        df : cudf.DataFrame
            df[first] : cudf.Series
                the first vertex id of a pair, if an external vertex id
                is defined by only one column
            df[second] : cudf.Series
                the second vertex id of a pair, if an external vertex id
                is defined by only one column
        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        df = graph_primtypes_wrapper.get_two_hop_neighbors(self)
        if self.renumbered is True:
            df = self.unrenumber(df, "first")
            df = self.unrenumber(df, "second")

        return df

    def number_of_vertices(self):
        """
        Get the number of nodes in the graph.

        """
        if self.node_count is None:
            if self.distributed:
                if self.edgelist is not None:
                    ddf = self.edgelist.edgelist_df[["src", "dst"]]
                    self.node_count = ddf.max().max().compute() + 1
                else:
                    raise Exception("Graph is Empty")
            elif self.adjlist is not None:
                self.node_count = len(self.adjlist.offsets) - 1
            elif self.transposedadjlist is not None:
                self.node_count = len(self.transposedadjlist.offsets) - 1
            elif self.edgelist is not None:
                df = self.edgelist.edgelist_df[["src", "dst"]]
                self.node_count = df.max().max() + 1
            else:
                raise Exception("Graph is Empty")
        return self.node_count

    def number_of_nodes(self):
        """
        An alias of number_of_vertices(). This function is added for NetworkX
        compatibility.

        """
        return self.number_of_vertices()

    def number_of_edges(self, directed_edges=False):
        """
        Get the number of edges in the graph.

        """
        if self.distributed:
            if self.edgelist is not None:
                return len(self.edgelist.edgelist_df)
            else:
                raise ValueError("Graph is Empty")
        if directed_edges and self.edgelist is not None:
            return len(self.edgelist.edgelist_df)
        if self.edge_count is None:
            if self.edgelist is not None:
                if type(self) is Graph:
                    self.edge_count = len(
                        self.edgelist.edgelist_df[
                            self.edgelist.edgelist_df["src"]
                            >= self.edgelist.edgelist_df["dst"]
                        ]
                    )
                else:
                    self.edge_count = len(self.edgelist.edgelist_df)
            elif self.adjlist is not None:
                self.edge_count = len(self.adjlist.indices)
            elif self.transposedadjlist is not None:
                self.edge_count = len(self.transposedadjlist.indices)
            else:
                raise ValueError("Graph is Empty")
        return self.edge_count

    def in_degree(self, vertex_subset=None):
        """
        Compute vertex in-degree. Vertex in-degree is the number of edges
        pointing into the vertex. By default, this method computes vertex
        degrees for the entire set of vertices. If vertex_subset is provided,
        this method optionally filters out all but those listed in
        vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf.Series or iterable container, optional
            A container of vertices for displaying corresponding in-degree.
            If not set, degrees are computed for the entire set of vertices.

        Returns
        -------
        df : cudf.DataFrame
            GPU DataFrame of size N (the default) or the size of the given
            vertices (vertex_subset) containing the in_degree. The ordering is
            relative to the adjacency list, or that given by the specified
            vertex_subset.

            df[vertex] : cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df[degree] : cudf.Series
                The computed in-degree of the corresponding vertex.

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, '0', '1')
        >>> df = G.in_degree([0,9,12])

        """
        return self._degree(vertex_subset, x=1)

    def out_degree(self, vertex_subset=None):
        """
        Compute vertex out-degree. Vertex out-degree is the number of edges
        pointing out from the vertex. By default, this method computes vertex
        degrees for the entire set of vertices. If vertex_subset is provided,
        this method optionally filters out all but those listed in
        vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf.Series or iterable container, optional
            A container of vertices for displaying corresponding out-degree.
            If not set, degrees are computed for the entire set of vertices.

        Returns
        -------
        df : cudf.DataFrame
            GPU DataFrame of size N (the default) or the size of the given
            vertices (vertex_subset) containing the out_degree. The ordering is
            relative to the adjacency list, or that given by the specified
            vertex_subset.

            df[vertex] : cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df[degree] : cudf.Series
                The computed out-degree of the corresponding vertex.

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, '0', '1')
        >>> df = G.out_degree([0,9,12])

        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        return self._degree(vertex_subset, x=2)

    def degree(self, vertex_subset=None):
        """
        Compute vertex degree, which is the total number of edges incident
        to a vertex (both in and out edges). By default, this method computes
        degrees for the entire set of vertices. If vertex_subset is provided,
        then this method optionally filters out all but those listed in
        vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf.Series or iterable container, optional
            a container of vertices for displaying corresponding degree. If not
            set, degrees are computed for the entire set of vertices.

        Returns
        -------
        df : cudf.DataFrame
            GPU DataFrame of size N (the default) or the size of the given
            vertices (vertex_subset) containing the degree. The ordering is
            relative to the adjacency list, or that given by the specified
            vertex_subset.

            df['vertex'] : cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df['degree'] : cudf.Series
                The computed degree of the corresponding vertex.

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, '0', '1')
        >>> all_df = G.degree()
        >>> subset_df = G.degree([0,9,12])

        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        return self._degree(vertex_subset)

    # FIXME:  vertex_subset could be a DataFrame for multi-column vertices
    def degrees(self, vertex_subset=None):
        """
        Compute vertex in-degree and out-degree. By default, this method
        computes vertex degrees for the entire set of vertices. If
        vertex_subset is provided, this method optionally filters out all but
        those listed in vertex_subset.

        Parameters
        ----------
        vertex_subset : cudf.Series or iterable container, optional
            A container of vertices for displaying corresponding degree. If not
            set, degrees are computed for the entire set of vertices.

        Returns
        -------
        df : cudf.DataFrame
            GPU DataFrame of size N (the default) or the size of the given
            vertices (vertex_subset) containing the degrees. The ordering is
            relative to the adjacency list, or that given by the specified
            vertex_subset.

            df['vertex'] : cudf.Series
                The vertex IDs (will be identical to vertex_subset if
                specified).
            df['in_degree'] : cudf.Series
                The in-degree of the vertex.
            df['out_degree'] : cudf.Series
                The out-degree of the vertex.

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, '0', '1')
        >>> df = G.degrees([0,9,12])

        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        (
            vertex_col,
            in_degree_col,
            out_degree_col,
        ) = graph_primtypes_wrapper._degrees(self)

        df = cudf.DataFrame()
        df["vertex"] = vertex_col
        df["in_degree"] = in_degree_col
        df["out_degree"] = out_degree_col

        if self.renumbered is True:
            df = self.unrenumber(df, "vertex")

        if vertex_subset is not None:
            df = df[df['vertex'].isin(vertex_subset)]

        return df

    def _degree(self, vertex_subset, x=0):
        vertex_col, degree_col = graph_primtypes_wrapper._degree(self, x)
        df = cudf.DataFrame()
        df["vertex"] = vertex_col
        df["degree"] = degree_col

        if self.renumbered is True:
            df = self.unrenumber(df, "vertex")

        if vertex_subset is not None:
            df = df[df['vertex'].isin(vertex_subset)]

        return df

    def to_directed(self):
        """
        Return a directed representation of the graph.
        This function sets the type of graph as DiGraph() and returns the
        directed view.

        Returns
        -------
        G : DiGraph
            A directed graph with the same nodes, and each edge (u,v,weights)
            replaced by two directed edges (u,v,weights) and (v,u,weights).

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(M, '0', '1')
        >>> DiG = G.to_directed()

        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        if type(self) is DiGraph:
            return self
        if type(self) is Graph:
            DiG = DiGraph()
            DiG.renumbered = self.renumbered
            DiG.renumber_map = self.renumber_map
            DiG.edgelist = self.edgelist
            DiG.adjlist = self.adjlist
            DiG.transposedadjlist = self.transposedadjlist
            return DiG

    def to_undirected(self):
        """
        Return an undirected copy of the graph.

        Returns
        -------
        G : Graph
            A undirected graph with the same nodes, and each directed edge
            (u,v,weights) replaced by an undirected edge (u,v,weights).

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> DiG = cugraph.DiGraph()
        >>> DiG.from_cudf_edgelist(M, '0', '1')
        >>> G = DiG.to_undirected()

        """

        if type(self) is Graph:
            return self
        if type(self) is DiGraph:
            G = Graph()
            df = self.edgelist.edgelist_df
            G.renumbered = self.renumbered
            G.renumber_map = self.renumber_map
            G.multi = self.multi
            if self.edgelist.weights:
                source_col, dest_col, value_col = symmetrize(
                    df["src"], df["dst"], df["weights"]
                )
            else:
                source_col, dest_col = symmetrize(df["src"], df["dst"])
                value_col = None
            G.edgelist = Graph.EdgeList(source_col, dest_col, value_col)

            return G

    def is_directed(self):
        if type(self) is DiGraph:
            return True
        else:
            return False

    def has_node(self, n):
        """
        Returns True if the graph contains the node n.
        """
        if self.edgelist is None:
            raise Exception("Graph has no Edgelist.")
        if self.distributed:
            ddf = self.edgelist.edgelist_df[["src", "dst"]]
            return (ddf == n).any().any().compute()
        if self.renumbered:
            tmp = self.renumber_map.to_internal_vertex_id(cudf.Series([n]))
            return tmp[0] is not cudf.NA and tmp[0] >= 0
        else:
            df = self.edgelist.edgelist_df[["src", "dst"]]
            return (df == n).any().any()

    def has_edge(self, u, v):
        """
        Returns True if the graph contains the edge (u,v).
        """
        if self.edgelist is None:
            raise Exception("Graph has no Edgelist.")
        if self.renumbered:
            tmp = cudf.DataFrame({"src": [u, v]})
            tmp = tmp.astype({"src": "int"})
            tmp = self.add_internal_vertex_id(
                tmp, "id", "src", preserve_order=True
            )

            u = tmp["id"][0]
            v = tmp["id"][1]

        df = self.edgelist.edgelist_df
        if self.distributed:
            return ((df["src"] == u) & (df["dst"] == v)).any().compute()
        return ((df["src"] == u) & (df["dst"] == v)).any()

    def edges(self):
        """
        Returns all the edges in the graph as a cudf.DataFrame containing
        sources and destinations. It does not return the edge weights.
        For viewing edges with weights use view_edge_list()
        """
        return self.view_edge_list()[["src", "dst"]]

    def nodes(self):
        """
        Returns all the nodes in the graph as a cudf.Series
        """
        if self.distributed:
            raise Exception("Not supported for distributed graph")
        if self.edgelist is not None:
            df = self.edgelist.edgelist_df
            if self.renumbered:
                # FIXME: If vertices are multicolumn
                #        this needs to return a dataframe
                # FIXME: This relies on current implementation
                #        of NumberMap, should not really expose
                #        this, perhaps add a method to NumberMap
                return self.renumber_map.implementation.df["0"]
            else:
                return cudf.concat([df["src"], df["dst"]]).unique()
        if "all_nodes" in self._nodes.keys():
            return self._nodes["all_nodes"]
        else:
            n = cudf.Series(dtype="int")
            set_names = [i for i in self._nodes.keys() if i != "all_nodes"]
            for k in set_names:
                n = n.append(self._nodes[k])
            return n

    def neighbors(self, n):
        if self.edgelist is None:
            raise Exception("Graph has no Edgelist.")
        if self.distributed:
            ddf = self.edgelist.edgelist_df
            return ddf[ddf["src"] == n]["dst"].reset_index(drop=True)
        if self.renumbered:
            node = self.renumber_map.to_internal_vertex_id(cudf.Series([n]))
            if len(node) == 0:
                return cudf.Series(dtype="int")
            n = node[0]

        df = self.edgelist.edgelist_df
        neighbors = df[df["src"] == n]["dst"].reset_index(drop=True)
        if self.renumbered:
            # FIXME:  Multi-column vertices
            return self.renumber_map.from_internal_vertex_id(neighbors)["0"]
        else:
            return neighbors

    def unrenumber(self, df, column_name, preserve_order=False):
        """
        Given a DataFrame containing internal vertex ids in the identified
        column, replace this with external vertex ids.  If the renumbering
        is from a single column, the output dataframe will use the same
        name for the external vertex identifiers.  If the renumbering is from
        a multi-column input, the output columns will be labeled 0 through
        n-1 with a suffix of _column_name.

        Note that this function does not guarantee order in single GPU mode,
        and does not guarantee order or partitioning in multi-GPU mode.  If you
        wish to preserve ordering, add an index column to df and sort the
        return by that index column.

        Parameters
        ----------
        df: cudf.DataFrame or dask_cudf.DataFrame
            A DataFrame containing internal vertex identifiers that will be
            converted into external vertex identifiers.

        column_name: string
            Name of the column containing the internal vertex id.

        preserve_order: (optional) bool
            If True, preserve the order of the rows in the output
            DataFrame to match the input DataFrame

        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            The original DataFrame columns exist unmodified.  The external
            vertex identifiers are added to the DataFrame, the internal
            vertex identifier column is removed from the dataframe.
        """
        return self.renumber_map.unrenumber(df, column_name, preserve_order)

    def lookup_internal_vertex_id(self, df, column_name=None):
        """
        Given a DataFrame containing external vertex ids in the identified
        columns, or a Series containing external vertex ids, return a
        Series with the internal vertex ids.

        Note that this function does not guarantee order in single GPU mode,
        and does not guarantee order or partitioning in multi-GPU mode.

        Parameters
        ----------
        df: cudf.DataFrame, cudf.Series, dask_cudf.DataFrame, dask_cudf.Series
            A DataFrame containing external vertex identifiers that will be
            converted into internal vertex identifiers.

        column_name: (optional) string
            Name of the column containing the external vertex ids

        Returns
        ---------
        series : cudf.Series or dask_cudf.Series
            The internal vertex identifiers
        """
        return self.renumber_map.to_internal_vertex_id(df, column_name)

    def add_internal_vertex_id(
        self,
        df,
        internal_column_name,
        external_column_name,
        drop=True,
        preserve_order=False,
    ):
        """
        Given a DataFrame containing external vertex ids in the identified
        columns, return a DataFrame containing the internal vertex ids as the
        specified column name.  Optionally drop the external vertex id columns.
        Optionally preserve the order of the original DataFrame.

        Parameters
        ----------
        df: cudf.DataFrame or dask_cudf.DataFrame
            A DataFrame containing external vertex identifiers that will be
            converted into internal vertex identifiers.

        internal_column_name: string
            Name of column to contain the internal vertex id

        external_column_name: string or list of strings
            Name of the column(s) containing the external vertex ids

        drop: (optional) bool, defaults to True
            Drop the external columns from the returned DataFrame

        preserve_order: (optional) bool, defaults to False
            Preserve the order of the data frame (requires an extra sort)

        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            Original DataFrame with new column containing internal vertex
            id
        """
        return self.renumber_map.add_internal_vertex_id(
            df,
            internal_column_name,
            external_column_name,
            drop,
            preserve_order,
        )


class DiGraph(Graph):
    def __init__(self, m_graph=None, edge_attr=None):
        super().__init__(
            m_graph=m_graph, edge_attr=edge_attr, symmetrized=True
        )


class MultiGraph(Graph):
    def __init__(self, renumbered=True):
        super().__init__(multi=True)


class MultiDiGraph(Graph):
    def __init__(self, renumbered=True):
        super().__init__(symmetrized=True, multi=True)
