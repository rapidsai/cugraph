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


class PropertyGraph:
    """
    FIXME: fill this in
    """
    # column name constants used in internal DataFrames
    __vertex_col_name = "_VERTEX_"
    __src_col_name = "_SRC_"
    __dst_col_name = "_DST_"
    __type_col_name = "_TYPE_"
    __edge_id_col_name = "_EDGE_ID_"
    __vertex_id_col_name = "_VERTEX_ID_"

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

        # Remember the type used for DataFrames and Series, typically Pandas
        # (for host storage) or cuDF (device storage), but this need not
        # strictly be one of those if the type supports the Pandas-like API.
        self.__dataframe_type = None
        self.__series_type = None

        # Keep track of dtypes for each column in each DataFrame.  This is
        # required since merge operations can often change the dtypes to
        # accommodate NaN values (eg. int64 to float64, since NaN is a float)
        self.__vertex_prop_dtypes = {}
        self.__edge_prop_dtypes = {}

        # Add unique edge IDs to the __edge_prop_dataframe by simply
        # incrementing a counter
        self.__last_edge_id = None

    # PropertyGraph read-only attributes
    @property
    def num_vertices(self):
        # Create a Series of the appropriate type (cudf.Series, pandas.Series,
        # etc.) based on the type currently in use, then use it to gather all
        # unique vertices.
        vpd = self.__vertex_prop_dataframe
        epd = self.__edge_prop_dataframe
        if (vpd is None) and (epd is None):
            return 0

        # Assume __series_type is set if this point reached!
        verts = self.__series_type(dtype="object")
        if vpd is not None:
            verts = verts.append(vpd[self.__vertex_col_name])
        if epd is not None:
            # pandas.Series.unique() can return an ndarray, which cannot be
            # appended to a Series. Always construct an appropriate series_type
            # from the unique values prior to appending.
            verts = verts.append(
                self.__series_type(epd[self.__src_col_name].unique()))
            verts = verts.append(
                self.__series_type(epd[self.__dst_col_name].unique()))
            verts = verts.unique()
        return len(verts)

    @property
    def num_edges(self):
        if self.__edge_prop_dataframe is not None:
            return len(self.__edge_prop_dataframe)
        else:
            return 0

    @property
    def vertex_property_names(self):
        if self.__vertex_prop_dataframe is not None:
            props = list(self.__vertex_prop_dataframe.columns)
            props.remove(self.__vertex_col_name)
            props.remove(self.__type_col_name)  # should "type" be removed?
            return props
        return []

    @property
    def edge_property_names(self):
        if self.__edge_prop_dataframe is not None:
            props = list(self.__edge_prop_dataframe.columns)
            props.remove(self.__src_col_name)
            props.remove(self.__dst_col_name)
            props.remove(self.__edge_id_col_name)
            props.remove(self.__type_col_name)  # should "type" be removed?
            return props
        return []

    # PropertyGraph read-only attributes for debugging
    @property
    def _vertex_prop_dataframe(self):
        return self.__vertex_prop_dataframe

    @property
    def _edge_prop_dataframe(self):
        return self.__edge_prop_dataframe

    def add_vertex_data(self,
                        dataframe,
                        vertex_id_column,
                        type_name=None,
                        property_columns=None
                        ):
        """
        Add a dataframe describing vertex properties to the PropertyGraph.
        dataframe must be
        FIXME: finish this description

        Parameters
        ----------
        dataframe : DataFrame-compatible instance
            A DataFrame instance with a compatible Pandas-like DataFrame
            interface.
            FIXME: finish this description
        vertex_id_column : string
            FIXME: finish this description

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
        if vertex_id_column not in dataframe.columns:
            raise ValueError(f"{vertex_id_column} is not a column in "
                             f"dataframe: {dataframe.columns}")
        if type(type_name) is not str:
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

        # Initialize the __vertex_prop_dataframe if necessary using the same
        # type as the incoming dataframe.
        default_vertex_columns = [self.__vertex_col_name, self.__type_col_name]
        if self.__vertex_prop_dataframe is None:
            self.__vertex_prop_dataframe = \
                self.__dataframe_type(columns=default_vertex_columns)
            # Initialize the new columns to the same dtype as the appropriate
            # column in the incoming dataframe, since the initial merge may not
            # result in the same dtype. (see
            # https://github.com/rapidsai/cudf/issues/9981)
            self.__update_dataframe_dtypes(
                self.__vertex_prop_dataframe,
                {self.__vertex_col_name: dataframe[vertex_id_column].dtype})

        # Ensure that both the predetermined vertex ID column name and vertex
        # type column name are present for proper merging.

        # NOTE: This copies the incoming DataFrame in order to add the new
        # columns. The copied DataFrame is then merged (another copy) and then
        # deleted when out-of-scope.
        tmp_df = dataframe.copy(deep=True)
        tmp_df[self.__vertex_col_name] = tmp_df[vertex_id_column]
        # FIXME: handle case of a type_name column already being in tmp_df
        tmp_df[self.__type_col_name] = type_name

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
                      vertex_id_columns,
                      type_name=None,
                      property_columns=None
                      ):
        """
        Add a dataframe describing edge properties to the PropertyGraph.
        dataframe must be
        FIXME: finish this description

        Parameters
        ----------
        dataframe : DataFrame-compatible instance
            A DataFrame instance with a compatible Pandas-like DataFrame
            interface.
            FIXME: finish this description
        vertex_id_column : string
            FIXME: finish this description

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
        if type(vertex_id_columns) not in [list, tuple]:
            raise TypeError("vertex_id_columns must be a list or tuple, got: "
                            f"{type(vertex_id_columns)}")
        invalid_columns = set(vertex_id_columns).difference(dataframe.columns)
        if invalid_columns:
            raise ValueError("vertex_id_columns contains column(s) not found "
                             f"in dataframe: {list(invalid_columns)}")
        if type(type_name) is not str:
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

        default_edge_columns = [self.__src_col_name,
                                self.__dst_col_name,
                                self.__edge_id_col_name,
                                self.__type_col_name]
        if self.__edge_prop_dataframe is None:
            self.__edge_prop_dataframe = \
                self.__dataframe_type(columns=default_edge_columns)
            # Initialize the new columns to the same dtype as the appropriate
            # column in the incoming dataframe, since the initial merge may not
            # result in the same dtype. (see
            # https://github.com/rapidsai/cudf/issues/9981)
            self.__update_dataframe_dtypes(
                self.__edge_prop_dataframe,
                {self.__src_col_name: dataframe[vertex_id_columns[0]].dtype,
                 self.__dst_col_name: dataframe[vertex_id_columns[1]].dtype,
                 self.__edge_id_col_name: "Int64"})

        # NOTE: This copies the incoming DataFrame in order to add the new
        # columns. The copied DataFrame is then merged (another copy) and then
        # deleted when out-of-scope.
        tmp_df = dataframe.copy(deep=True)
        tmp_df[self.__src_col_name] = tmp_df[vertex_id_columns[0]]
        tmp_df[self.__dst_col_name] = tmp_df[vertex_id_columns[1]]
        # FIXME: handle case of a type_name column already being in tmp_df
        tmp_df[self.__type_col_name] = type_name

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

    def extract_subgraph(self,
                         create_using=None,
                         edge_property_condition=None,
                         vertex_property_condition=None,
                         edge_weight_property=None,
                         default_edge_weight=None,
                         allow_multi_edges=False
                         ):
        """
        Return a subgraph of the overall PropertyGraph containing vertices
        and edges that match the criteria specified.
        FIXME: finish this description

        Parameters
        ----------
        create_using : cugraph Graph type or instance
            FIXME: finish this description
        edge_property_condition : string
            FIXME: finish this description

        Returns
        -------
        None

        Examples
        --------
        >>>
        """
        # NOTE: the expressions passed in to extract specific edges and
        # vertices assume the original dtypes in the user input have been
        # preserved. However, merge operations on the DataFrames can change
        # dtypes (eg. int64 to float64 in order to add NaN entries). This
        # should not be a problem since this the conversions do not change
        # the values.
        globals = {}
        if vertex_property_condition:
            locals = self.__vertex_prop_eval_dict
            filter_column = eval(vertex_property_condition, globals, locals)
            matching_indices = filter_column.index[filter_column]
            filtered_vertex_dataframe = \
                self.__vertex_prop_dataframe.loc[matching_indices]
        else:
            filtered_vertex_dataframe = self.__vertex_prop_dataframe

        if edge_property_condition:
            locals = self.__edge_prop_eval_dict
            filter_column = eval(edge_property_condition, globals, locals)
            matching_indices = filter_column.index[filter_column]
            filtered_edge_dataframe = \
                self.__edge_prop_dataframe.loc[matching_indices]
        else:
            filtered_edge_dataframe = self.__edge_prop_dataframe

        # FIXME: check that self.__edge_prop_dataframe is set!

        # If vertices were specified, filter the edges that contain any of the
        # filtered verts in both src and dst
        if (filtered_vertex_dataframe is not None) and \
           not(filtered_vertex_dataframe.empty):
            filtered_verts = filtered_vertex_dataframe[self.__vertex_col_name]
            src_filter = filtered_edge_dataframe[self.__src_col_name]\
                .isin(filtered_verts)
            dst_filter = filtered_edge_dataframe[self.__dst_col_name]\
                .isin(filtered_verts)
            edge_filter = src_filter & dst_filter
            edges = filtered_edge_dataframe.loc[
                filtered_edge_dataframe.index[edge_filter]]
        else:
            edges = filtered_edge_dataframe

        if edge_weight_property:
            if edge_weight_property not in edges.columns:
                raise ValueError("edge_weight_property "
                                 f'"{edge_weight_property}" was not found in '
                                 "the properties of the subgraph")

            # Ensure a valid edge_weight_property can be used for applying
            # weights to the subgraph, and if a default_edge_weight was
            # specified, apply it to all NAs in the weight column.
            prop_col = edges[edge_weight_property]
            if prop_col.count() != prop_col.size:
                if default_edge_weight is None:
                    raise ValueError("edge_weight_property "
                                     f'"{edge_weight_property}" '
                                     "contains NA values in the subgraph and "
                                     "default_edge_weight is not set")
                else:
                    prop_col.fillna(default_edge_weight, inplace=True)

        # The __*_prop_dataframes have likely been merged several times and
        # possibly had their dtypes converted in order to accommodate NaN
        # values. Restore the original dtypes in the resulting edges df prior
        # to creating a Graph.
        self.__update_dataframe_dtypes(edges, self.__edge_prop_dtypes)

        return self.edge_props_to_graph(
            edges,
            create_using=create_using,
            edge_weight_property=edge_weight_property,
            allow_multi_edges=allow_multi_edges)

    def annotate_dataframe(self, df, G, edge_vertex_id_columns):
        """
        FIXME: fill this in
        """
        # FIXME: all check args
        (src_col_name, dst_col_name) = edge_vertex_id_columns

        # FIXME: check that G has edge_data attr

        # Add the src, dst, edge_id info from the Graph to a DataFrame
        edge_info_df = self.__dataframe_type(columns=[self.__src_col_name,
                                                      self.__dst_col_name,
                                                      self.__edge_id_col_name],
                                             data=G.edge_data)

        # New result includes only properties from the src/dst edges identified
        # by edge IDs. All other data in df is merged based on src/dst values.
        # NOTE: results from MultiGraph graphs will have to include edge IDs!
        edge_props_df = edge_info_df.merge(self.__edge_prop_dataframe,
                                           how="inner")

        # FIXME: also allow edge ID col to be passed in and renamed.
        new_df = df.rename(columns={src_col_name: self.__src_col_name,
                                    dst_col_name: self.__dst_col_name})
        new_df = new_df.merge(edge_props_df)
        # restore the original src/dst column names
        new_df.rename(columns={self.__src_col_name: src_col_name,
                               self.__dst_col_name: dst_col_name},
                      inplace=True)
        # FIXME: consider removing internal columns (_EDGE_ID_, etc.) and
        # columns from edge types not included in the edges in df.
        return new_df

    @classmethod
    def get_edge_tuples(cls, edge_prop_df):
        """
        Returns a list of (src vertex, dst vertex, edge_id) tuples present in
        edge_prop_df.
        """
        if cls.__src_col_name not in edge_prop_df.columns:
            raise ValueError(f"column {cls.__src_col_name} missing from "
                             "edge_prop_df")
        if cls.__dst_col_name not in edge_prop_df.columns:
            raise ValueError(f"column {cls.__dst_col_name} missing from "
                             "edge_prop_df")
        if cls.__edge_id_col_name not in edge_prop_df.columns:
            raise ValueError(f"column {cls.__edge_id_col_name} missing "
                             "from edge_prop_df")
        src = edge_prop_df[cls.__src_col_name]
        dst = edge_prop_df[cls.__dst_col_name]
        edge_id = edge_prop_df[cls.__edge_id_col_name]
        retlist = [(src.iloc[i], dst.iloc[i], edge_id.iloc[i])
                   for i in range(len(src))]
        return retlist

    @classmethod
    def edge_props_to_graph(cls, edge_prop_df,
                            create_using=None,
                            edge_weight_property=None,
                            allow_multi_edges=False):
        """
        Create and return a Graph from the edges in edge_prop_df.
        """
        if edge_weight_property and \
           (edge_weight_property not in edge_prop_df.columns):
            raise ValueError("edge_weight_property "
                             f'"{edge_weight_property}" was not found in '
                             "edge_prop_df")

        # Set up the new Graph to return
        if create_using is None:
            G = cugraph.Graph()
        elif isinstance(create_using, cugraph.Graph):
            # FIXME: extract more attrs from the create_using instance
            attrs = {"directed": create_using.is_directed()}
            G = type(create_using)(**attrs)
        elif type(create_using) is type(type):
            G = create_using()
        else:
            raise TypeError("create_using must be a cugraph.Graph "
                            "(or subclass) type or instance, got: "
                            f"{type(create_using)}")

        # Ensure no repeat edges, since the individual edge used would be
        # ambiguous.
        # FIXME: make allow_multi_edges accept "auto" for use with MultiGraph
        if (allow_multi_edges is False) and \
           cls.has_duplicate_edges(edge_prop_df):
            if create_using:
                if type(create_using) is type:
                    t = create_using.__name__
                else:
                    t = type(create_using).__name__
                msg = f"{t} graph type specified by create_using"
            else:
                msg = "default Graph graph type"
            raise RuntimeError("query resulted in duplicate edges which "
                               f"cannot be represented with a {msg}")

        create_args = {"source": cls.__src_col_name,
                       "destination": cls.__dst_col_name,
                       "edge_attr": edge_weight_property,
                       "renumber": True,
                       }
        if type(edge_prop_df) is cudf.DataFrame:
            G.from_cudf_edgelist(edge_prop_df, **create_args)
        else:
            G.from_pandas_edgelist(edge_prop_df, **create_args)

        # Set the edge_data on the resulting Graph to the list of edge tuples,
        # which includes the unique edge IDs. Edge IDs are needed for future
        # calls to annotate_dataframe() in order to apply properties from the
        # correct edges.
        # FIXME: this could be a very large list of tuples if the number of
        # edges in G is large (eg. a large MNMG graph that cannot fit in host
        # memory). Consider adding the edge IDs to the edgelist DataFrame in G
        # instead.
        G.edge_data = cls.get_edge_tuples(edge_prop_df)
        # FIXME: also add vertex_data

        return G

    @classmethod
    def has_duplicate_edges(cls, df):
        """
        Return True if df has >1 of the same src, dst pair
        """
        if df.empty:
            return False

        def has_duplicate_dst(df):
            return df[cls.__dst_col_name].nunique() != \
                df[cls.__dst_col_name].size

        return df.groupby(cls.__src_col_name).apply(has_duplicate_dst).any()

    def __add_edge_ids(self):
        """
        Replace nans with unique edge IDs. Edge IDs are simply numbers
        incremented by 1 for each edge.
        """
        prev_eid = -1 if self.__last_edge_id is None else self.__last_edge_id
        nans = self.__edge_prop_dataframe[self.__edge_id_col_name].isna()

        if nans.any():
            indices = nans.index[nans]
            num_indices = len(indices)
            starting_eid = prev_eid + 1
            new_eids = self.__series_type(
                range(starting_eid, starting_eid + num_indices))

            self.__edge_prop_dataframe[self.__edge_id_col_name]\
                .iloc[indices] = new_eids

            self.__last_edge_id = starting_eid + num_indices - 1

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
                df[col] = df[col].astype(dtype_str)
