import numpy as np
import cupy
import importlib


class MissingModule:
    """
    Raises RuntimeError when any attribute is accessed on instances of this
    class.

    Instances of this class are returned by import_optional() when a module
    cannot be found, which allows for code to import optional dependencies, and
    have only the code paths that use the module affected.
    """

    def __init__(self, mod_name):
        self.name = mod_name

    def __getattr__(self, attr):
        raise RuntimeError(f"This feature requires the {self.name} " "package/module")


try:
    cudf = importlib.import_module("cudf")
except ModuleNotFoundError:
    cudf = MissingModule("cudf")


class RemoteGraph:
    def __init__(self, cgs_client, cgs_graph_id):
        self.__client = cgs_client
        self.__cgs_graph_id = cgs_graph_id

    def is_remote(self):
        return True

    def is_multigraph(self):
        return self.__multigraph


class RemotePropertyGraph:
    """
    Supports method-by-method selection of backend type (cupy, cudf, etc.)
    to avoid costly conversion such as row-major to column-major transformation.
    """

    # column name constants used in internal DataFrames
    vertex_col_name = "_VERTEX_"
    src_col_name = "_SRC_"
    dst_col_name = "_DST_"
    type_col_name = "_TYPE_"
    edge_id_col_name = "_EDGE_ID_"
    weight_col_name = "_WEIGHT_"
    _default_type_name = ""

    def __init__(self, cgs_client, cgs_graph_id):
        self.__client = cgs_client
        self.__graph_id = cgs_graph_id

    def __transform_to_backend_dtype(self, data, column_names, backend):
        """
        data : cupy.ndarray, np.ndarray
            The raw ndarray that will be transformed to the backend type.
        """

        if backend == "cupy":
            if isinstance(data, np.ndarray):
                data = cupy.array(data)
            return data
        else:
            # cudf
            return cudf.DataFrame.from_records(data, columns=column_names)

        # TODO support torch

    @property
    def graph_info(self):
        return self.__client.get_graph_info(graph_id=self.__graph_id)

    @property
    def edges(self, _backend="cudf"):
        np_edges = self.__client.get_graph_edge_data(
            -1,
            graph_id=self.__graph_id,
            property_keys=[self.src_col_name, self.dst_col_name],
        )

        return self.__transform_to_backend_dtype(
            np_edges,
            [
                self.edge_id_col_name,
                self.src_col_name,
                self.dst_col_name,
                self.type_col_name,
            ],
            _backend,
        )

    @property
    def vertex_property_names(self):
        """
        Return a Python list of vertex property names.
        """
        np_names = self.__client.get_graph_vertex_property_names(self.__graph_id)
        return np_names

    @property
    def edge_property_names(self):
        """
        Return a Python list of edge property names.
        """
        np_names = self.__client.get_graph_edge_property_names(self.__graph_id)
        return np_names

    @property
    def vertex_types(self):
        """The set of vertex type names"""
        raise NotImplementedError("not implemented")

    @property
    def edge_types(self):
        """The set of edge type names"""
        raise NotImplementedError("not implemented")

    @property
    def _vertex_type_value_counts(self):
        """A Series of the counts of types in __vertex_prop_dataframe"""
        raise NotImplementedError("not implemented")

    @property
    def _edge_type_value_counts(self):
        """A Series of the counts of types in __edge_prop_dataframe"""
        raise NotImplementedError("not implemented")

    def get_num_vertices(self, type=None, *, include_edge_data=True):
        """Return the number of all vertices or vertices of a given type.

        Parameters
        ----------
        type : string, optional
            If type is None (the default), return the total number of vertices,
            otherwise return the number of vertices of the specified type.
        include_edge_data : bool (default True)
            If True, include vertices that were added in vertex and edge data.
            If False, only include vertices that were added in vertex data.
            Note that vertices that only exist in edge data are assumed to have
            the default type.

        See Also
        --------
        RemotePropertyGraph.get_num_edges
        """
        raise NotImplementedError("not implemented")

    def get_num_edges(self, type=None):
        """Return the number of all edges or edges of a given type.

        Parameters
        ----------
        type : string, optional
            If type is None (the default), return the total number of edges,
            otherwise return the number of edges of the specified type.

        See Also
        --------
        PropertyGraph.get_num_vertices
        """
        raise NotImplementedError("not implemented")

    def get_vertices(self, selection=None):
        """
        Return a Series containing the unique vertex IDs contained in both
        the vertex and edge property data.
        """
        raise NotImplementedError("not implemented")

    def vertices_ids(self):
        """
        Alias for get_vertices()
        """
        return self.get_vertices()

    def add_vertex_data(
        self, dataframe, vertex_col_name, type_name=None, property_columns=None
    ):
        """
        Add a dataframe describing vertex properties to the PropertyGraph.

        Parameters
        ----------
        dataframe : DataFrame-compatible instance
            A DataFrame instance with a compatible Pandas-like DataFrame
            interface.
        vertex_col_name : string
            The column name that contains the values to be used as vertex IDs.
        type_name : string
            The name to be assigned to the type of property being added. For
            example, if dataframe contains data about users, type_name might be
            "users". If not specified, the type of properties will be added as
            the empty string, "".
        property_columns : list of strings
            List of column names in dataframe to be added as properties. All
            other columns in dataframe will be ignored. If not specified, all
            columns in dataframe are added.

        Returns
        -------
        None

        Examples
        --------
        >>>
        """
        raise NotImplementedError("not implemented")

    def get_vertex_data(self, vertex_ids=None, types=None, columns=None):
        """
        Return a dataframe containing vertex properties for only the specified
        vertex_ids, columns, and/or types, or all vertex IDs if not specified.
        """
        raise NotImplementedError("not implemented")

    def add_edge_data(
        self,
        dataframe,
        vertex_col_names,
        edge_id_col_name=None,
        type_name=None,
        property_columns=None,
    ):
        """
        Add a dataframe describing edge properties to the PropertyGraph.

        Parameters
        ----------
        dataframe : DataFrame-compatible instance
            A DataFrame instance with a compatible Pandas-like DataFrame
            interface.
        vertex_col_names : list of strings
            The column names that contain the values to be used as the source
            and destination vertex IDs for the edges.
        edge_id_col_name : string, optional
            The column name that contains the values to be used as edge IDs.
            If unspecified, edge IDs will be automatically assigned.
            Currently, all edge data must be added with the same method: either
            with automatically generated IDs, or from user-provided edge IDs.
        type_name : string
            The name to be assigned to the type of property being added. For
            example, if dataframe contains data about transactions, type_name
            might be "transactions". If not specified, the type of properties
            will be added as the empty string "".
        property_columns : list of strings
            List of column names in dataframe to be added as properties. All
            other columns in dataframe will be ignored. If not specified, all
            columns in dataframe are added.

        Returns
        -------
        None

        Examples
        --------
        >>>
        """
        raise NotImplementedError("not implemented")

    def get_edge_data(self, edge_ids=None, types=None, columns=None):
        """
        Return a dataframe containing edge properties for only the specified
        edge_ids, columns, and/or edge type, or all edge IDs if not specified.
        """
        raise NotImplementedError("not implemented")

    def select_vertices(self, expr, from_previous_selection=None):
        """
        Evaluate expr and return a PropertySelection object representing the
        vertices that match the expression.

        Parameters
        ----------
        expr : string
            A python expression using property names and operators to select
            specific vertices.
        from_previous_selection : PropertySelection
            A PropertySelection instance returned from a prior call to
            select_vertices() that can be used to select a subset of vertices
            to evaluate the expression against. This allows for a selection of
            the intersection of vertices of multiple types (eg. all vertices
            that are both type A and type B)

        Returns
        -------
        PropertySelection instance to be used for calls to extract_subgraph()
        in order to construct a Graph containing only specific vertices.

        Examples
        --------
        >>>
        """
        raise NotImplementedError("not implemented")

    def select_edges(self, expr):
        """
        Evaluate expr and return a PropertySelection object representing the
        edges that match the expression.

        Parameters
        ----------
        expr : string
            A python expression using property names and operators to select
            specific edges.

        Returns
        -------
        PropertySelection instance to be used for calls to extract_subgraph()
        in order to construct a Graph containing only specific edges.

        Examples
        --------
        >>>
        """
        raise NotImplementedError("not implemented")

    def extract_subgraph(
        self,
        create_using=None,
        selection=None,
        edge_weight_property=None,
        default_edge_weight=None,
        check_multi_edges=True,
        renumber_graph=True,
        add_edge_data=True,
    ):
        """
        Return a subgraph of the overall PropertyGraph containing vertices
        and edges that match a selection.

        Parameters
        ----------
        create_using : cugraph Graph type or instance, optional
            Creates a Graph to return using the type specified. If an instance
            is specified, the type of the instance is used to construct the
            return Graph, and all relevant attributes set on the instance are
            copied to the return Graph (eg. directed). If not specified the
            returned Graph will be a directed cugraph.MultiGraph instance.
        selection : PropertySelection
            A PropertySelection returned from one or more calls to
            select_vertices() and/or select_edges(), used for creating a Graph
            with only the selected properties. If not speciied the returned
            Graph will have all properties. Note, this could result in a Graph
            with multiple edges, which may not be supported based on the value
            of create_using.
        edge_weight_property : string
            The name of the property whose values will be used as weights on
            the returned Graph. If not specified, the returned Graph will be
            unweighted.
        check_multi_edges : bool (default is True)
            When True and create_using argument is given and not a MultiGraph,
            this will perform an expensive check to verify that the edges in
            the edge dataframe do not form a multigraph with duplicate edges.
        renumber_graph : bool (default is True)
            If True, return a Graph that has been renumbered for use by graph
            algorithms. If False, the returned graph will need to be manually
            renumbered prior to calling graph algos.
        add_edge_data : bool (default is True)
            If True, add meta data about the edges contained in the extracted
            graph which are required for future calls to annotate_dataframe().

        Returns
        -------
        A Graph instance of the same type as create_using containing only the
        vertices and edges resulting from applying the selection to the set of
        vertex and edge property data.

        Examples
        --------
        >>>
        """
        raise NotImplementedError("not implemented")

    def annotate_dataframe(self, df, G, edge_vertex_col_names):
        """
        Add properties to df that represent the vertices and edges in graph G.

        Parameters
        ----------
        df : cudf.DataFrame or pandas.DataFrame
            A DataFrame containing edges identified by edge_vertex_col_names
            which will have properties for those edges added to it.
        G : cugraph.Graph (or subclass of) instance.
            Graph containing the edges specified in df. The Graph instance must
            have been generated from a prior call to extract_subgraph() in
            order to have the edge meta-data used to look up the correct
            properties.
        edge_vertex_col_names : tuple of strings
            The column names in df that represent the source and destination
            vertices, used for identifying edges.

        Returns
        -------
        A copy of df with additional columns corresponding to properties for
        the edge in the row.
        FIXME: also provide the ability to annotate vertex data.

        Examples
        --------
        >>>
        """
        raise NotImplementedError("not ipmlemented")

    def edge_props_to_graph(
        self,
        edge_prop_df,
        create_using,
        edge_weight_property=None,
        default_edge_weight=None,
        check_multi_edges=True,
        renumber_graph=True,
        add_edge_data=True,
    ):
        """
        Create and return a Graph from the edges in edge_prop_df.
        """
        raise NotImplementedError("not implemented")

    def renumber_vertices_by_type(self):
        """Renumber vertex IDs to be contiguous by type.

        Returns a DataFrame with the start and stop IDs for each vertex type.
        Stop is *inclusive*.
        """
        raise NotImplementedError("not implemented")

    def renumber_edges_by_type(self):
        """Renumber edge IDs to be contiguous by type.

        Returns a DataFrame with the start and stop IDs for each edge type.
        Stop is *inclusive*.
        """
        raise NotImplementedError("not implemented")
