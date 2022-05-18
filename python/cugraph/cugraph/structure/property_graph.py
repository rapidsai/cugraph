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

import cudf

import cugraph
from cugraph.utilities.utils import import_optional, MissingModule

pd = import_optional("pandas")

_dataframe_types = [cudf.DataFrame]
if not isinstance(pd, MissingModule):
    _dataframe_types.append(pd.DataFrame)


class EXPERIMENTAL__PropertySelection:
    """
    Instances of this class are returned from the PropertyGraph.select_*()
    methods and can be used by the PropertyGraph.extract_subgraph() method to
    extrac a Graph containing vertices and edges with only the selected
    properties.
    """
    def __init__(self,
                 vertex_selection_series=None,
                 edge_selection_series=None):
        self.vertex_selections = vertex_selection_series
        self.edge_selections = edge_selection_series

    def __add__(self, other):
        """
        Add either the vertex_selections, edge_selections, or both to this
        instance from "other" if either are not already set.
        """
        vs = self.vertex_selections
        if vs is None:
            vs = other.vertex_selections
        es = self.edge_selections
        if es is None:
            es = other.edge_selections
        return EXPERIMENTAL__PropertySelection(vs, es)


# FIXME: remove leading __ when no longer experimental
class EXPERIMENTAL__PropertyGraph:
    """
    Class which stores vertex and edge properties that can be used to construct
    Graphs from individual property selections and used later to annotate graph
    algorithm results with corresponding properties.
    """
    # column name constants used in internal DataFrames
    vertex_col_name = "_VERTEX_"
    src_col_name = "_SRC_"
    dst_col_name = "_DST_"
    type_col_name = "_TYPE_"
    edge_id_col_name = "_EDGE_ID_"
    vertex_id_col_name = "_VERTEX_ID_"
    weight_col_name = "_WEIGHT_"

    def __init__(self):
        # The dataframe containing the properties for each vertex.
        # Each vertex occupies a row, and individual properties are maintained
        # in individual columns. The table contains a column for each property
        # of each vertex. If a vertex does not contain a property, it will have
        # a NaN value in that property column. Each vertex will also have a
        # "type_name" that can be assigned by the caller to describe the type
        # of the vertex for a given application domain. If no type_name is
        # provided, the default type_name is "".
        # Example:
        # vertex | type_name | propA | propB | propC
        # ------------------------------------------
        #      3 | "user"    | 22    | NaN   | 11
        #     88 | "service" | NaN   | 3.14  | 21
        #      9 | ""        | NaN   | NaN   | 2
        self.__vertex_prop_dataframe = None

        # The dataframe containing the properties for each edge.
        # The description is identical to the vertex property dataframe, except
        # edges are identified by ordered pairs of vertices (src and dst).
        # Example:
        # src | dst | type_name | propA | propB | propC
        # ---------------------------------------------
        #   3 |  88 | "started" | 22    | NaN   | 11
        #  88 |   9 | "called"  | NaN   | 3.14  | 21
        #   9 |  88 | ""        | NaN   | NaN   | 2
        self.__edge_prop_dataframe = None

        # The var:value dictionaries used during evaluation of filter/query
        # expressions for vertices and edges. These dictionaries contain
        # entries for each column name in their respective dataframes which
        # are mapped to instances of PropertyColumn objects.
        #
        # When filter/query expressions are evaluated, PropertyColumn objects
        # are used in place of DataFrame columns in order to support string
        # comparisons when cuDF DataFrames are used. This approach also allows
        # expressions to contain var names that can be used in expressions that
        # are different than those in the actual internal tables, allowing for
        # the tables to contain additional or different column names than what
        # can be used in expressions.
        #
        # Example: "type_name == 'user' & propC > 10"
        #
        # The above would be evaluated and "type_name" and "propC" would be
        # PropertyColumn instances which support specific operators used in
        # queries.
        self.__vertex_prop_eval_dict = {}
        self.__edge_prop_eval_dict = {}

        # The types used for DataFrames and Series, typically Pandas (for host
        # storage) or cuDF (device storage), but this need not strictly be one
        # of those if the type supports the Pandas-like API. These are used for
        # constructing other DataFrames and Series of the same type, as well as
        # for enforing that both vertex and edge properties are the same type.
        self.__dataframe_type = None
        self.__series_type = None

        # The dtypes for each column in each DataFrame.  This is required since
        # merge operations can often change the dtypes to accommodate NaN
        # values (eg. int64 to float64, since NaN is a float).
        self.__vertex_prop_dtypes = {}
        self.__edge_prop_dtypes = {}

        # Add unique edge IDs to the __edge_prop_dataframe by simply
        # incrementing this counter.
        self.__last_edge_id = None

        # Cached property values
        self.__num_vertices = None

    # PropertyGraph read-only attributes
    @property
    def num_vertices(self):
        if self.__num_vertices is not None:
            return self.__num_vertices

        self.__num_vertices = 0
        vert_sers = self.__get_all_vertices_series()
        if vert_sers:
            if self.__series_type is cudf.Series:
                self.__num_vertices = cudf.concat(vert_sers).nunique()
            else:
                self.__num_vertices = pd.concat(vert_sers).nunique()

        return self.__num_vertices

    @property
    def num_edges(self):
        if self.__edge_prop_dataframe is not None:
            return len(self.__edge_prop_dataframe)
        else:
            return 0

    @property
    def edges(self):
        if self.__edge_prop_dataframe is not None:
            return self.__edge_prop_dataframe[[self.src_col_name,
                                               self.dst_col_name]]
        return None

    @property
    def vertex_property_names(self):
        if self.__vertex_prop_dataframe is not None:
            props = list(self.__vertex_prop_dataframe.columns)
            props.remove(self.vertex_col_name)
            props.remove(self.type_col_name)  # should "type" be removed?
            return props
        return []

    @property
    def edge_property_names(self):
        if self.__edge_prop_dataframe is not None:
            props = list(self.__edge_prop_dataframe.columns)
            props.remove(self.src_col_name)
            props.remove(self.dst_col_name)
            props.remove(self.edge_id_col_name)
            props.remove(self.type_col_name)  # should "type" be removed?
            return props
        return []

    # PropertyGraph read-only attributes for debugging
    @property
    def _vertex_prop_dataframe(self):
        return self.__vertex_prop_dataframe

    @property
    def _edge_prop_dataframe(self):
        return self.__edge_prop_dataframe

    def get_vertices(self, selection=None):
        """
        Return a Series containing the unique vertex IDs contained in both
        the vertex and edge property data.
        """
        vert_sers = self.__get_all_vertices_series()
        if vert_sers:
            if self.__series_type is cudf.Series:
                return self.__series_type(cudf.concat(vert_sers).unique())
            else:
                return self.__series_type(pd.concat(vert_sers).unique())
        return self.__series_type()

    def vertices_ids(self):
        """
        Alias for get_vertices()
        """
        return self.get_vertices()

    def add_vertex_data(self,
                        dataframe,
                        vertex_col_name,
                        type_name=None,
                        property_columns=None
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
            None or NA
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
        if type(dataframe) not in _dataframe_types:
            raise TypeError("dataframe must be one of the following types: "
                            f"{_dataframe_types}, got: {type(dataframe)}")
        if vertex_col_name not in dataframe.columns:
            raise ValueError(f"{vertex_col_name} is not a column in "
                             f"dataframe: {dataframe.columns}")
        if (type_name is not None) and not(isinstance(type_name, str)):
            raise TypeError("type_name must be a string, got: "
                            f"{type(type_name)}")
        if property_columns:
            if type(property_columns) is not list:
                raise TypeError("property_columns must be a list, got: "
                                f"{type(property_columns)}")
            invalid_columns = \
                set(property_columns).difference(dataframe.columns)
            if invalid_columns:
                raise ValueError("property_columns contains column(s) not "
                                 "found in dataframe: "
                                 f"{list(invalid_columns)}")

        # Save the DataFrame and Series types for future instantiations
        if (self.__dataframe_type is None) or (self.__series_type is None):
            self.__dataframe_type = type(dataframe)
            self.__series_type = type(dataframe[dataframe.columns[0]])
        else:
            if type(dataframe) is not self.__dataframe_type:
                raise TypeError(f"dataframe is type {type(dataframe)} but "
                                "the PropertyGraph was already initialized "
                                f"using type {self.__dataframe_type}")

        # Clear the cached value for num_vertices since more could be added in
        # this method.
        self.__num_vertices = None

        # Initialize the __vertex_prop_dataframe if necessary using the same
        # type as the incoming dataframe.
        default_vertex_columns = [self.vertex_col_name, self.type_col_name]
        if self.__vertex_prop_dataframe is None:
            self.__vertex_prop_dataframe = \
                self.__dataframe_type(columns=default_vertex_columns)
            # Initialize the new columns to the same dtype as the appropriate
            # column in the incoming dataframe, since the initial merge may not
            # result in the same dtype. (see
            # https://github.com/rapidsai/cudf/issues/9981)
            self.__update_dataframe_dtypes(
                self.__vertex_prop_dataframe,
                {self.vertex_col_name: dataframe[vertex_col_name].dtype})

        # Ensure that both the predetermined vertex ID column name and vertex
        # type column name are present for proper merging.

        # NOTE: This copies the incoming DataFrame in order to add the new
        # columns. The copied DataFrame is then merged (another copy) and then
        # deleted when out-of-scope.
        tmp_df = dataframe.copy(deep=True)
        tmp_df[self.vertex_col_name] = tmp_df[vertex_col_name]
        # FIXME: handle case of a type_name column already being in tmp_df
        tmp_df[self.type_col_name] = type_name

        if property_columns:
            # all columns
            column_names_to_drop = set(tmp_df.columns)
            # remove the ones to keep
            column_names_to_drop.difference_update(property_columns +
                                                   default_vertex_columns)
            tmp_df = tmp_df.drop(labels=column_names_to_drop, axis=1)

        # Save the original dtypes for each new column so they can be restored
        # prior to constructing subgraphs (since column dtypes may get altered
        # during merge to accommodate NaN values).
        new_col_info = self.__get_new_column_dtypes(
                           tmp_df, self.__vertex_prop_dataframe)
        self.__vertex_prop_dtypes.update(new_col_info)

        self.__vertex_prop_dataframe = \
            self.__vertex_prop_dataframe.merge(tmp_df, how="outer")

        # Update the vertex eval dict with the latest column instances
        latest = dict([(n, self.__vertex_prop_dataframe[n])
                       for n in self.__vertex_prop_dataframe.columns])
        self.__vertex_prop_eval_dict.update(latest)

    def add_edge_data(self,
                      dataframe,
                      vertex_col_names,
                      type_name=None,
                      property_columns=None
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
        type_name : string
            The name to be assigned to the type of property being added. For
            example, if dataframe contains data about transactions, type_name
            might be "transactions". If not specified, the type of properties
            will be added as None or NA
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
        if type(dataframe) not in _dataframe_types:
            raise TypeError("dataframe must be one of the following types: "
                            f"{_dataframe_types}, got: {type(dataframe)}")
        if type(vertex_col_names) not in [list, tuple]:
            raise TypeError("vertex_col_names must be a list or tuple, got: "
                            f"{type(vertex_col_names)}")
        invalid_columns = set(vertex_col_names).difference(dataframe.columns)
        if invalid_columns:
            raise ValueError("vertex_col_names contains column(s) not found "
                             f"in dataframe: {list(invalid_columns)}")
        if (type_name is not None) and not(isinstance(type_name, str)):
            raise TypeError("type_name must be a string, got: "
                            f"{type(type_name)}")
        if property_columns:
            if type(property_columns) is not list:
                raise TypeError("property_columns must be a list, got: "
                                f"{type(property_columns)}")
            invalid_columns = \
                set(property_columns).difference(dataframe.columns)
            if invalid_columns:
                raise ValueError("property_columns contains column(s) not "
                                 "found in dataframe: "
                                 f"{list(invalid_columns)}")

        # Save the DataFrame and Series types for future instantiations
        if (self.__dataframe_type is None) or (self.__series_type is None):
            self.__dataframe_type = type(dataframe)
            self.__series_type = type(dataframe[dataframe.columns[0]])
        else:
            if type(dataframe) is not self.__dataframe_type:
                raise TypeError(f"dataframe is type {type(dataframe)} but "
                                "the PropertyGraph was already initialized "
                                f"using type {self.__dataframe_type}")

        # Clear the cached value for num_vertices since more could be added in
        # this method.
        self.__num_vertices = None

        default_edge_columns = [self.src_col_name,
                                self.dst_col_name,
                                self.edge_id_col_name,
                                self.type_col_name]
        if self.__edge_prop_dataframe is None:
            self.__edge_prop_dataframe = \
                self.__dataframe_type(columns=default_edge_columns)
            # Initialize the new columns to the same dtype as the appropriate
            # column in the incoming dataframe, since the initial merge may not
            # result in the same dtype. (see
            # https://github.com/rapidsai/cudf/issues/9981)
            self.__update_dataframe_dtypes(
                self.__edge_prop_dataframe,
                {self.src_col_name: dataframe[vertex_col_names[0]].dtype,
                 self.dst_col_name: dataframe[vertex_col_names[1]].dtype,
                 self.edge_id_col_name: "Int64"})

        # NOTE: This copies the incoming DataFrame in order to add the new
        # columns. The copied DataFrame is then merged (another copy) and then
        # deleted when out-of-scope.
        tmp_df = dataframe.copy(deep=True)
        tmp_df[self.src_col_name] = tmp_df[vertex_col_names[0]]
        tmp_df[self.dst_col_name] = tmp_df[vertex_col_names[1]]
        # FIXME: handle case of a type_name column already being in tmp_df
        tmp_df[self.type_col_name] = type_name

        if property_columns:
            # all columns
            column_names_to_drop = set(tmp_df.columns)
            # remove the ones to keep
            column_names_to_drop.difference_update(property_columns +
                                                   default_edge_columns)
            tmp_df = tmp_df.drop(labels=column_names_to_drop, axis=1)

        # Save the original dtypes for each new column so they can be restored
        # prior to constructing subgraphs (since column dtypes may get altered
        # during merge to accommodate NaN values).
        new_col_info = self.__get_new_column_dtypes(
            tmp_df, self.__edge_prop_dataframe)
        self.__edge_prop_dtypes.update(new_col_info)

        self.__edge_prop_dataframe = \
            self.__edge_prop_dataframe.merge(tmp_df, how="outer")

        self.__add_edge_ids()

        # Update the vertex eval dict with the latest column instances
        latest = dict([(n, self.__edge_prop_dataframe[n])
                       for n in self.__edge_prop_dataframe.columns])
        self.__edge_prop_eval_dict.update(latest)

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
        # FIXME: check types

        # Check if the expr is to be evaluated in the context of properties
        # from only the previously selected vertices (as opposed to all
        # properties from all vertices)
        if (from_previous_selection is not None) and \
           (from_previous_selection.vertex_selections is not None):
            previously_selected_rows = self.__vertex_prop_dataframe[
                from_previous_selection.vertex_selections]
            verts_from_previously_selected_rows = \
                previously_selected_rows[self.vertex_col_name]
            # get all the rows from the entire __vertex_prop_dataframe that
            # contain those verts
            rows_with_verts = \
                self.__vertex_prop_dataframe[self.vertex_col_name]\
                    .isin(verts_from_previously_selected_rows)
            rows_to_eval = self.__vertex_prop_dataframe[rows_with_verts]
            locals = dict([(n, rows_to_eval[n])
                           for n in rows_to_eval.columns])
        else:
            locals = self.__vertex_prop_eval_dict

        globals = {}
        selected_col = eval(expr, globals, locals)

        num_rows = len(self.__vertex_prop_dataframe)
        # Ensure the column is the same size as the DataFrame, then replace any
        # NA values with False to represent rows that should not be selected.
        # This ensures the selected column can be applied to the entire
        # __vertex_prop_dataframe to determine which rows to use when creating
        # a Graph from a query.
        if num_rows != len(selected_col):
            selected_col = selected_col.reindex(range(num_rows), copy=False)
            selected_col.fillna(False, inplace=True)

        return EXPERIMENTAL__PropertySelection(
            vertex_selection_series=selected_col)

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
        # FIXME: check types
        globals = {}
        locals = self.__edge_prop_eval_dict

        selected_col = eval(expr, globals, locals)
        return EXPERIMENTAL__PropertySelection(
            edge_selection_series=selected_col)

    def extract_subgraph(self,
                         create_using=cugraph.Graph,
                         selection=None,
                         edge_weight_property=None,
                         default_edge_weight=None,
                         allow_multi_edges=False
                         ):
        """
        Return a subgraph of the overall PropertyGraph containing vertices
        and edges that match a selection.

        Parameters
        ----------
        create_using : cugraph Graph type or instance
            Creates a Graph to return using the type specified. If an instance
            is specified, the type of the instance is used to construct the
            return Graph, and all relevant attributes set on the instance are
            copied to the return Graph (eg. directed). If not specified the
            returned Graph will be a cugraph.Graph instance.
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
        allow_multi_edges : bool
            If True, multiple edges should be used to create the return Graph,
            otherwise multiple edges will be detected and an exception raised.

        Returns
        -------
        A Graph instance of the same type as create_using containing only the
        vertices and edges resulting from applying the selection to the set of
        vertex and edge property data.

        Examples
        --------
        >>>
        """
        if (selection is not None) and \
           not isinstance(selection, EXPERIMENTAL__PropertySelection):
            raise TypeError("selection must be an instance of "
                            f"PropertySelection, got {type(selection)}")

        # NOTE: the expressions passed in to extract specific edges and
        # vertices assume the original dtypes in the user input have been
        # preserved. However, merge operations on the DataFrames can change
        # dtypes (eg. int64 to float64 in order to add NaN entries). This
        # should not be a problem since the conversions do not change the
        # values.
        if (selection is not None) and \
           (selection.vertex_selections is not None):
            selected_vertex_dataframe = \
                self.__vertex_prop_dataframe[selection.vertex_selections]
        else:
            selected_vertex_dataframe = None

        if (selection is not None) and \
           (selection.edge_selections is not None):
            selected_edge_dataframe = \
                self.__edge_prop_dataframe[selection.edge_selections]
        else:
            selected_edge_dataframe = self.__edge_prop_dataframe

        # FIXME: check that self.__edge_prop_dataframe is set!

        # If vertices were specified, select only the edges that contain the
        # selected verts in both src and dst
        if (selected_vertex_dataframe is not None) and \
           not(selected_vertex_dataframe.empty):
            selected_verts = selected_vertex_dataframe[self.vertex_col_name]
            has_srcs = selected_edge_dataframe[self.src_col_name]\
                .isin(selected_verts)
            has_dsts = selected_edge_dataframe[self.dst_col_name]\
                .isin(selected_verts)
            edges = selected_edge_dataframe[has_srcs & has_dsts]
        else:
            edges = selected_edge_dataframe

        # The __*_prop_dataframes have likely been merged several times and
        # possibly had their dtypes converted in order to accommodate NaN
        # values. Restore the original dtypes in the resulting edges df prior
        # to creating a Graph.
        self.__update_dataframe_dtypes(edges, self.__edge_prop_dtypes)

        return self.edge_props_to_graph(
            edges,
            create_using=create_using,
            edge_weight_property=edge_weight_property,
            default_edge_weight=default_edge_weight,
            allow_multi_edges=allow_multi_edges)

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
        # FIXME: check all args
        (src_col_name, dst_col_name) = edge_vertex_col_names

        df_type = type(df)
        if df_type is not self.__dataframe_type:
            raise TypeError(f"df type {df_type} does not match DataFrame type "
                            f"{self.__dataframe_type} used in PropertyGraph")

        if hasattr(G, "edge_data"):
            edge_info_df = G.edge_data
        else:
            raise AttributeError("Graph G does not have attribute 'edge_data'")

        # New result includes only properties from the src/dst edges identified
        # by edge IDs. All other data in df is merged based on src/dst values.
        # NOTE: results from MultiGraph graphs will have to include edge IDs!
        edge_props_df = edge_info_df.merge(self.__edge_prop_dataframe,
                                           how="inner")

        # FIXME: also allow edge ID col to be passed in and renamed.
        new_df = df.rename(columns={src_col_name: self.src_col_name,
                                    dst_col_name: self.dst_col_name})
        new_df = new_df.merge(edge_props_df)
        # restore the original src/dst column names
        new_df.rename(columns={self.src_col_name: src_col_name,
                               self.dst_col_name: dst_col_name},
                      inplace=True)

        # restore the original dtypes
        self.__update_dataframe_dtypes(new_df, self.__edge_prop_dtypes)
        for col in df.columns:
            new_df[col] = new_df[col].astype(df[col].dtype)

        # FIXME: consider removing internal columns (_EDGE_ID_, etc.) and
        # columns from edge types not included in the edges in df.
        return new_df

    def edge_props_to_graph(self,
                            edge_prop_df,
                            create_using,
                            edge_weight_property=None,
                            default_edge_weight=None,
                            allow_multi_edges=False):
        """
        Create and return a Graph from the edges in edge_prop_df.
        """
        # FIXME: check default_edge_weight is valid

        if edge_weight_property:
            if edge_weight_property not in edge_prop_df.columns:
                raise ValueError("edge_weight_property "
                                 f'"{edge_weight_property}" was not found in '
                                 "edge_prop_df")

            # Ensure a valid edge_weight_property can be used for applying
            # weights to the subgraph, and if a default_edge_weight was
            # specified, apply it to all NAs in the weight column.
            prop_col = edge_prop_df[edge_weight_property]
            if prop_col.count() != prop_col.size:
                if default_edge_weight is None:
                    raise ValueError("edge_weight_property "
                                     f'"{edge_weight_property}" '
                                     "contains NA values in the subgraph and "
                                     "default_edge_weight is not set")
                else:
                    prop_col.fillna(default_edge_weight, inplace=True)
            edge_attr = edge_weight_property

        # If a default_edge_weight was specified but an edge_weight_property
        # was not, a new edge weight column must be added.
        elif default_edge_weight:
            edge_attr = self.__gen_unique_name(edge_prop_df.columns,
                                               prefix=self.weight_col_name)
            edge_prop_df[edge_attr] = default_edge_weight

        else:
            edge_attr = None

        # Set up the new Graph to return
        if isinstance(create_using, cugraph.Graph):
            # FIXME: extract more attrs from the create_using instance
            attrs = {"directed": create_using.is_directed()}
            G = type(create_using)(**attrs)
        # FIXME: this allows anything to be instantiated does not check that
        # the type is a valid Graph type.
        elif type(create_using) is type(type):
            G = create_using()
        else:
            raise TypeError("create_using must be a cugraph.Graph "
                            "(or subclass) type or instance, got: "
                            f"{type(create_using)}")

        # Prevent duplicate edges (if not allowed) since applying them to
        # non-MultiGraphs would result in ambiguous edge properties.
        # FIXME: make allow_multi_edges accept "auto" for use with MultiGraph
        if (allow_multi_edges is False) and \
           self.has_duplicate_edges(edge_prop_df):
            if create_using:
                if type(create_using) is type:
                    t = create_using.__name__
                else:
                    t = type(create_using).__name__
                msg = f"'{t}' graph type specified by create_using"
            else:
                msg = "default Graph graph type"
            raise RuntimeError("query resulted in duplicate edges which "
                               f"cannot be represented with the {msg}")

        create_args = {"source": self.src_col_name,
                       "destination": self.dst_col_name,
                       "edge_attr": edge_attr,
                       "renumber": True,
                       }
        if type(edge_prop_df) is cudf.DataFrame:
            G.from_cudf_edgelist(edge_prop_df, **create_args)
        else:
            G.from_pandas_edgelist(edge_prop_df, **create_args)

        # Set the edge_data on the resulting Graph to a DataFrame containing
        # the edges and the edge ID for each. Edge IDs are needed for future
        # calls to annotate_dataframe() in order to associate edges with their
        # properties, since the PG can contain multiple edges between vertrices
        # with different properties.
        G.edge_data = self.__create_property_lookup_table(edge_prop_df)
        # FIXME: also add vertex_data

        return G

    @classmethod
    def has_duplicate_edges(cls, df):
        """
        Return True if df has >1 of the same src, dst pair
        """
        # FIXME: this can be very expensive for large DataFrames
        if df.empty:
            return False

        def has_duplicate_dst(df):
            return df[cls.dst_col_name].nunique() != \
                df[cls.dst_col_name].size

        return df.groupby(cls.src_col_name).apply(has_duplicate_dst).any()

    def __create_property_lookup_table(self, edge_prop_df):
        """
        Returns a DataFrame containing the src vertex, dst vertex, and edge_id
        values from edge_prop_df.
        """
        src = edge_prop_df[self.src_col_name]
        dst = edge_prop_df[self.dst_col_name]
        edge_id = edge_prop_df[self.edge_id_col_name]
        return self.__dataframe_type({self.src_col_name: src,
                                      self.dst_col_name: dst,
                                      self.edge_id_col_name: edge_id})

    def __add_edge_ids(self):
        """
        Replace nans with unique edge IDs. Edge IDs are simply numbers
        incremented by 1 for each edge.
        """
        prev_eid = -1 if self.__last_edge_id is None else self.__last_edge_id
        nans = self.__edge_prop_dataframe[self.edge_id_col_name].isna()

        if nans.any():
            indices = nans.index[nans]
            num_indices = len(indices)
            starting_eid = prev_eid + 1
            new_eids = self.__series_type(
                range(starting_eid, starting_eid + num_indices))

            self.__edge_prop_dataframe[self.edge_id_col_name]\
                .iloc[indices] = new_eids

            self.__last_edge_id = starting_eid + num_indices - 1

    def __get_all_vertices_series(self):
        """
        Return a list of all Series objects that contain vertices from all
        tables.
        """
        vpd = self.__vertex_prop_dataframe
        epd = self.__edge_prop_dataframe
        vert_sers = []
        if vpd is not None:
            vert_sers.append(vpd[self.vertex_col_name])
        if epd is not None:
            vert_sers.append(epd[self.src_col_name])
            vert_sers.append(epd[self.dst_col_name])
        return vert_sers

    @staticmethod
    def __gen_unique_name(current_names, prefix="col"):
        """
        Helper function to generate a currently unused name.
        """
        name = prefix
        counter = 2
        while name in current_names:
            name = f"{prefix}{counter}"
            counter += 1
        return name

    @staticmethod
    def __get_new_column_dtypes(from_df, to_df):
        """
        Returns a list containing tuples of (column name, dtype) for each
        column in from_df that is not present in to_df.
        """
        new_cols = set(from_df.columns) - set(to_df.columns)
        return [(col, from_df[col].dtype) for col in new_cols]

    @staticmethod
    def __update_dataframe_dtypes(df, column_dtype_dict):
        """
        Set the dtype for columns in df using the dtypes in column_dtype_dict.
        This also handles converting standard integer dtypes to nullable
        integer dtypes, needed to accommodate NA values in columns.
        """
        for (col, dtype) in column_dtype_dict.items():
            # If the DataFrame is Pandas and the dtype is an integer type,
            # ensure a nullable integer array is used by specifying the correct
            # dtype. The alias for these dtypes is simply a capitalized string
            # (eg. "Int64")
            # https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html#integer-dtypes-and-missing-data
            dtype_str = str(dtype)
            if dtype_str in ["int32", "int64"]:
                dtype_str = dtype_str.title()
            if str(df[col].dtype) != dtype_str:
                # Assigning to df[col] produces a (false?) warning with Pandas,
                # but assigning to df.loc[:,col] does not update the df in
                # cudf, so do one or the other based on type.
                if type(df) is cudf.DataFrame:
                    df[col] = df[col].astype(dtype_str)
                else:
                    df.loc[:, col] = df[col].astype(dtype_str)
