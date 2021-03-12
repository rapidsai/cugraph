from .graph_implentation import *

class Graph:
    def __init__(self, multi_edge = False, directed = False, tree = False):
        self.Base = None
        self.multi_edge = multi_edge
        self.directed = directed
        self.tree = tree

    def __getattr__(self, name):
        if self.Base is None:
            raise Exception("Graph is Empty")
        if hasattr(self.Base, name):
            return getattr(self.Base, name)
        else:
            raise Exception("method not found")

    def __dir__(self):
        return dir(self.base)

    def from_cudf_edgelist(
        self,
        input_df,
        source="source",
        destination="destination",
        edge_attr=None,
        renumber=True
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
        self.Base = simpleGraphImpl()
        self.Base.from_edgelist(input_df,
                                source="source",
                                destination="destination",
                                edge_attr=None,
                                renumber=True)

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
        self.Base = simpleDistributedGraphImpl()
        self.Base.from_edgelist(input_ddf,
                                source="source",
                                destination="destination",
                                edge_attr=None,
                                renumber=True)

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
    def __init__(self):
        super(Digraph, self).__init__(directed=True)

class MultiGraph(Graph):
    def __init__(self, directed=False):
        super(MultiGraph, self).__init__(directed=directed, multi_edge=True)

class MultiDiGraph(MultiGraph):
    def __init__(self):
        super(MultiDiGraph, self).__init__(directed=True)

class Tree(Graph):
    def __init__(self, directed=False):
        super(Tree, self).__init__(directed=directed, multi_edge=False, tree=True)

class DiTree(Tree):
    def __init__(self):
        super(DiTree, self).__init__(directed=True)

