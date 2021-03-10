from .graph_implentation import *

class Graph:
    def __init__(self):
        self.Base = None
        self.multi = False
        self.directed = False
        self.tree = False

    def __getattr__(self, name):
        if hasattr(self.base, name):
            return getattr(self.base, name)
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
        self.Base = simpleGraphImpl
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
        self.Base = simpleDistributedGraphImpl
        self.Base.from_edgelist(input_ddf,
                                source="source",
                                destination="destination",
                                edge_attr=None,
                                renumber=True)

class DiGraph(Graph):
    def __init__(self):
        super(Digraph, self).__init__()
        self.directed = True

class MultiGraph(Graph):
    def __init__(self):
        super(MultiGraph, self).__init__()
        self.multi = True

class MultiDiGraph(MultiGraph, DiGraph):
    def __init__(self):
        super(MultiDiGraph, self).__init__()
        self.directed = True
        self.multi = True

class Tree(Graph):
    def __init__(self):
        super(Tree, self).__init__()
        self.Tree = True

class DiTree(Tree, DiGraph):
    def __init__(self):
        super(DiTree, self).__init__()
        self.directed = True
        self.multi = True


