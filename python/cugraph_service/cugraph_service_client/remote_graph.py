# Copyright (c) 2022, NVIDIA CORPORATION.
#
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

import numpy as np
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

try:
    cupy = importlib.import_module("cupy")
except ModuleNotFoundError:
    cupy = MissingModule("cupy")

try:
    pandas = importlib.import_module("pandas")
except ModuleNotFoundError:
    pandas = MissingModule("pandas")

try:
    torch = importlib.import_module("torch")
except ModuleNotFoundError:
    torch = MissingModule("torch")


def _transform_to_backend_dtype(data, column_names, backend="numpy", dtypes=None):
    """
    Supports method-by-method selection of backend type (cupy, cudf, etc.)
    to avoid costly conversion such as row-major to column-major transformation.
    If using an array or tensor backend, this method will likely be followed with
    one or more stack() operations to create a matrix or matrices.

    Note: If using inferred dtypes, the returned dataframes, arrays, or tensors may
    infer a different dtype than what was originally on the server (i.e promotion
    of int32 to int64).  In the future, the server may also return dtype to prevent
    this from occurring.

    data : numpy.ndarray
        The raw ndarray that will be transformed to the backend type.
    column_names : list[string]
        The names of the columns, if creating a dataframe.
    backend : ('numpy', 'pandas', 'cupy', 'cudf', 'torch', 'torch:<device>')
              [default = 'cudf']
        The data backend to convert the provided data to.
    dtypes : ('int32', 'int64', 'float32', etc.)
        Optional.  The data type to use when storing data in a dataframe or array.
        If not set, it will be inferred for dataframe backends, and assumed as float64
        for array and tensor backends.
        May be a list, or dictionary corresponding to column names.  Unspecified
        columns in the dictionary will have their dtype inferred.  Note: for array
        and tensor backends, the inferred type is always 'float64' which will result
        in a error for non-numeric inputs.
        i.e. ['int32', 'int64', 'int32', 'float64']
        i.e. {'col1':'int32', 'col2': 'int64', 'col3': 'float64'}
    """

    default_dtype = None if backend in ["cudf", "pandas"] else "float64"

    if dtypes is None:
        dtypes = [default_dtype] * data.shape[1]
    elif isinstance(dtypes, (list, tuple)):
        if len(dtypes) != data.shape[1]:
            raise ValueError("Datatype array length must match number of columns!")
    elif isinstance(dtypes, dict):
        dtypes = [
            dtypes[name] if name in dtypes else default_dtype for name in column_names
        ]
    else:
        raise ValueError("dtypes must be None, a list/tuple, or a dict")

    if not isinstance(data, np.ndarray):
        raise TypeError("Numpy ndarray expected")

    if backend == "cupy":
        return [cupy.array(data[:, c], dtype=dtypes[c]) for c in range(data.shape[1])]
    elif backend == "numpy":
        print("dtypes:", dtypes)
        return [np.array(data[:, c], dtype=dtypes[c]) for c in range(data.shape[1])]

    elif backend == "pandas" or backend == "cudf":
        from_records = (
            pandas.DataFrame.from_records
            if backend == "pandas"
            else cudf.DataFrame.from_records
        )
        df = from_records(data, columns=column_names)
        for i, t in enumerate(dtypes):
            if t is not None:
                df[column_names[i]] = df[column_names[i]].astype(t)
        return df
    elif backend == "torch":
        return [
            torch.tensor(data[:, c].astype(dtypes[c])) for c in range(data.shape[1])
        ]

    backend = backend.split(":")
    if backend[0] == "torch":
        try:
            device = int(backend[1])
        except ValueError:
            device = backend[1]
        return [
            torch.tensor(data[:, c].astype(dtypes[c]), device=device)
            for c in range(data.shape[1])
        ]

    raise ValueError(f"invalid backend {backend[0]}")


class RemoteGraph:
    def __init__(self, cgs_client, cgs_graph_id, backend="numpy"):
        self.__client = cgs_client
        self.__cgs_graph_id = cgs_graph_id
        self.__backend = backend

    def is_remote(self):
        return True

    def is_bipartite(self):
        return False

    def is_multipartite(self):
        return False

    def is_directed(self):
        return True

    def is_multigraph(self):
        return True

    def is_weighted(self):
        return True

    def has_isolated_vertices(self):
        raise NotImplementedError("not implemented")

    def to_directed(self):
        raise NotImplementedError("not implemented")

    def to_undirected(self):
        raise NotImplementedError("not implemented")

    @property
    def backend(self):
        return self.__backend

    @property
    def edgelist(self):
        raise NotImplementedError("not implemented")

    @property
    def adjlist(self):
        raise NotImplementedError("not implemented")


class RemotePropertyGraph:
    # column name constants used in internal DataFrames
    vertex_col_name = "_VERTEX_"
    src_col_name = "_SRC_"
    dst_col_name = "_DST_"
    type_col_name = "_TYPE_"
    edge_id_col_name = "_EDGE_ID_"
    weight_col_name = "_WEIGHT_"
    _default_type_name = ""

    def __init__(self, cgs_client, cgs_graph_id, backend="numpy"):
        self.__client = cgs_client
        self.__graph_id = cgs_graph_id
        self.__vertex_categorical_dtype = None
        self.__edge_categorical_dtype = None
        self.__backend = backend

    @property
    def backend(self):
        return self.__backend

    @property
    def _vertex_categorical_dtype(self):
        if self.__vertex_categorical_dtype is None:
            cats = self.vertex_types
            if self.__backend == "cudf":
                self.__vertex_categorical_dtype = cudf.CategoricalDtype(cats)
            elif self.__backend == "pandas":
                self.__vertex_categorical_dtype = pandas.CategoricalDtype(cats)
            else:
                self.__vertex_categorical_dtype = {cat: i for i, cat in enumerate(cats)}
        return self.__vertex_categorical_dtype

    @property
    def _edge_categorical_dtype(self):
        if self.__edge_categorical_dtype is None:
            cats = self.edge_types
            if self.__backend == "cudf":
                self.__edge_categorical_dtype = cudf.CategoricalDtype(cats)
            elif self.__backend == "pandas":
                self.__edge_categorical_dtype = pandas.CategoricalDtype(cats)
            else:
                self.__edge_categorical_dtype = {cat: i for i, cat in enumerate(cats)}
        return self.__edge_categorical_dtype

    @property
    def graph_info(self):
        return self.__client.get_graph_info(graph_id=self.__graph_id)

    @property
    def edges(self):
        """
        Returns the edge list for this property graph as a dataframe,
        array, or tensor containing edge ids, source vertex,
        destination vertex, and edge type.
        """
        np_edges = self.__client.get_graph_edge_data(
            -1,
            graph_id=self.__graph_id,
            property_keys=[self.src_col_name, self.dst_col_name],
        )

        # Convert edge type to numeric if necessary
        if self.__backend not in ["cudf", "pandas"]:
            edge_cat_types = self._edge_categorical_dtype
            np_edges[:, 3] = np.array([edge_cat_types[t] for t in np_edges[:, 3]])
            cat_dtype = "int32"
        else:
            cat_dtype = self._edge_categorical_dtype

        return _transform_to_backend_dtype(
            np_edges,
            [
                self.edge_id_col_name,
                self.src_col_name,
                self.dst_col_name,
                self.type_col_name,
            ],
            self.__backend,
            dtypes=["int64", "int64", "int64", cat_dtype],
        )

    @property
    def vertex_property_names(self):
        """
        Return a Python list of vertex property names.
        """
        return self.__client.get_graph_vertex_property_names(self.__graph_id)

    @property
    def edge_property_names(self):
        """
        Return a Python list of edge property names.
        """
        return self.__client.get_graph_edge_property_names(self.__graph_id)

    @property
    def vertex_types(self):
        """The set of vertex type names"""
        return set(self.__client.get_graph_vertex_types(self.__graph_id))

    @property
    def edge_types(self):
        """The set of edge type names"""
        return set(self.__client.get_graph_edge_types(self.__graph_id))

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
        return self.__client.get_num_vertices(type, include_edge_data, self.__graph_id)

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
        return self.__client.get_num_edges(type, self.__graph_id)

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
        # FIXME expose na handling

        if columns is None:
            columns = self.vertex_property_names

        vertex_data = self.__client.get_graph_vertex_data(
            id_or_ids=vertex_ids or -1,
            property_keys=columns,
            types=types,
            graph_id=self.__graph_id,
        )

        # Convert type to numeric if necessary
        if self.__backend not in ["cudf", "pandas"]:
            vertex_cat_types = self._vertex_categorical_dtype
            vertex_data[:, 1] = np.array(
                [vertex_cat_types[t] for t in vertex_data[:, 1]]
            )
            cat_dtype = "int32"
        else:
            cat_dtype = self._vertex_categorical_dtype

        column_names = [self.vertex_col_name, self.type_col_name] + list(columns)
        return _transform_to_backend_dtype(
            vertex_data,
            column_names,
            self.__backend,
            dtypes={self.type_col_name: cat_dtype},
        )

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

        # FIXME expose na handling

        if columns is None:
            columns = self.edge_property_names

        edge_data = self.__client.get_graph_edge_data(
            id_or_ids=edge_ids or -1,
            property_keys=columns,
            types=types,
            graph_id=self.__graph_id,
        )

        # Convert edge type to numeric if necessary
        if self.__backend not in ["cudf", "pandas"]:
            edge_cat_types = self._edge_categorical_dtype
            edge_data[:, 3] = np.array([edge_cat_types[t] for t in edge_data[:, 3]])
            cat_dtype = "int32"
        else:
            cat_dtype = self._edge_categorical_dtype

        column_names = [
            self.edge_id_col_name,
            self.src_col_name,
            self.dst_col_name,
            self.type_col_name,
        ] + list(columns)
        return _transform_to_backend_dtype(
            edge_data,
            column_names,
            self.__backend,
            dtypes={self.type_col_name: cat_dtype},
        )

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
