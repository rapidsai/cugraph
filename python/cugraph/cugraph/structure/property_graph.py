# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
import numpy as np

import cugraph
from cugraph.utilities.utils import (
    import_optional,
    MissingModule,
    create_list_series_from_2d_ar,
)

from typing import Union

pd = import_optional("pandas")

_dataframe_types = [cudf.DataFrame]
if not isinstance(pd, MissingModule):
    _dataframe_types.append(pd.DataFrame)


# FIXME: remove leading EXPERIMENTAL__ when no longer experimental
class EXPERIMENTAL__PropertySelection:
    """
    Instances of this class are returned from the PropertyGraph.select_*()
    methods and can be used by the PropertyGraph.extract_subgraph() method to
    extract a Graph containing vertices and edges with only the selected
    properties.
    """

    def __init__(self, vertex_selection_series=None, edge_selection_series=None):
        """
        Create a PropertySelection out of one or two Series objects containing
        booleans representing whether or not a specific row in a PropertyGraph
        internal vertex DataFrame (vertex_selection_series) or
        edge DataFrame (edge_selection_series) is selected.

        Parameters
        ----------
        vertex_selection_series : cudf or pandas series, optional
            Contains booleans representing selected vertices
        edge_selection_series : cudf or pandas series, optional
            Contains booleans representing selected edges
        """
        self.vertex_selections = vertex_selection_series
        self.edge_selections = edge_selection_series

    def __add__(self, other):
        """
        Add either the vertex_selections, edge_selections, or both to this
        instance from "other" if either are not already set.

        Parameters
        ----------
        other : PropertySelection to add

        Returns
        -------
        PropertySelection
            New PropertySelection instance containing the selection
            Series objects from either the current instance if present,
            or instances from "other" only if those instances are not already
            present in the current instance.
        """
        vs = self.vertex_selections
        if vs is None:
            vs = other.vertex_selections
        es = self.edge_selections
        if es is None:
            es = other.edge_selections
        return EXPERIMENTAL__PropertySelection(vs, es)


# FIXME: remove leading EXPERIMENTAL__ when no longer experimental
class EXPERIMENTAL__PropertyGraph:
    """
    Class which stores vertex and edge properties that can be used to construct
    Graphs from individual property selections and used later to annotate graph
    algorithm results with corresponding properties.
    """

    # column name constants used in internal DataFrames
    vertex_col_name = "_VERTEX_"
    """
    Column containing the vertex id.
    """

    src_col_name = "_SRC_"
    """
    Column containing the id of the edge source
    """

    dst_col_name = "_DST_"
    """
    Column containing the id of the edge destination
    """

    type_col_name = "_TYPE_"
    """
    Column containing the type of the edge or vertex
    """

    edge_id_col_name = "_EDGE_ID_"
    """
    Column containing the edge identifier
    """

    weight_col_name = "_WEIGHT_"
    """
    Column containing the edge weight if the graph is weighted.
    """

    _default_type_name = ""

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

        # Lengths of the properties that are vectors
        self.__vertex_vector_property_lengths = {}
        self.__edge_vector_property_lengths = {}

        # Add unique edge IDs to the __edge_prop_dataframe by simply
        # incrementing this counter. Remains None if user provides edge IDs.
        self.__last_edge_id = None

        # Are edge IDs automatically generated sequentially by PG (True),
        # provided by the user (False), or no edges added yet (None).
        self.__is_edge_id_autogenerated = None

        # Cached property values
        self.__num_vertices = None
        self.__vertex_type_value_counts = None
        self.__edge_type_value_counts = None

    # PropertyGraph read-only attributes
    @property
    def edges(self):
        """
        All the edges in the graph as a DataFrame containing
        sources and destinations. It does not return the edge properties.
        """
        if self.__edge_prop_dataframe is not None:
            return self.__edge_prop_dataframe[
                [self.src_col_name, self.dst_col_name]
            ].reset_index()
        return None

    @property
    def vertex_property_names(self):
        """
        Names of all the vertex properties excluding type.
        """
        if self.__vertex_prop_dataframe is not None:
            props = list(self.__vertex_prop_dataframe.columns)
            props.remove(self.type_col_name)  # should "type" be removed?
            return props
        return []

    @property
    def edge_property_names(self):
        """
        List containing each edge property name in the PropertyGraph instance.
        """
        if self.__edge_prop_dataframe is not None:
            props = list(self.__edge_prop_dataframe.columns)
            props.remove(self.src_col_name)
            props.remove(self.dst_col_name)
            props.remove(self.type_col_name)  # should "type" be removed?
            if self.weight_col_name in props:
                props.remove(self.weight_col_name)
            return props
        return []

    @property
    def vertex_types(self):
        """
        The set of vertex type names
        """
        value_counts = self._vertex_type_value_counts
        if value_counts is None:
            names = set()
        elif self.__series_type is cudf.Series:
            names = set(value_counts.index.to_arrow().to_pylist())
        else:
            names = set(value_counts.index)
        default = self._default_type_name
        if default not in names and self.get_num_vertices(default) > 0:
            # include "" from vertices that only exist in edge data
            names.add(default)
        return names

    @property
    def edge_types(self):
        """
        Series containing the set of edge type names
        """
        value_counts = self._edge_type_value_counts
        if value_counts is None:
            return set()
        elif self.__series_type is cudf.Series:
            return set(value_counts.index.to_arrow().to_pylist())
        else:
            return set(value_counts.index)

    # PropertyGraph read-only attributes for debugging
    @property
    def _vertex_prop_dataframe(self):
        return self.__vertex_prop_dataframe

    @property
    def _edge_prop_dataframe(self):
        """
        Dataframe containing the edge properties.
        """
        return self.__edge_prop_dataframe

    @property
    def _vertex_type_value_counts(self):
        """
        A Series of the counts of types in __vertex_prop_dataframe
        """
        if self.__vertex_prop_dataframe is None:
            return
        if self.__vertex_type_value_counts is None:
            # Types should all be strings; what should we do if we see NaN?
            self.__vertex_type_value_counts = self.__vertex_prop_dataframe[
                self.type_col_name
            ].value_counts(sort=False, dropna=False)
        return self.__vertex_type_value_counts

    @property
    def _edge_type_value_counts(self):
        """
        Series of the counts of types in __edge_prop_dataframe
        """
        if self.__edge_prop_dataframe is None:
            return
        if self.__edge_type_value_counts is None:
            # Types should all be strings; what should we do if we see NaN?
            self.__edge_type_value_counts = self.__edge_prop_dataframe[
                self.type_col_name
            ].value_counts(sort=False, dropna=False)
        return self.__edge_type_value_counts

    def get_num_vertices(self, type=None, *, include_edge_data=True):
        """
        Return the number of all vertices or vertices of a given type.

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

        Returns
        -------
        int
            The number of vertices in the graph constrained by the type parameter.

        See Also
        --------
        PropertyGraph.get_num_edges

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                     data=[(99, 22, "a"),
        ...                           (98, 34, "b"),
        ...                           (97, 56, "c"),
        ...                           (96, 88, "d"),
        ...                          ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, vertex_col_names=("src", "dst"))
        >>> pG.get_num_vertices()
        8
        """
        if type is None:
            if not include_edge_data:
                if self.__vertex_prop_dataframe is None:
                    return 0
                return len(self.__vertex_prop_dataframe)
            if self.__num_vertices is not None:
                return self.__num_vertices
            self.__num_vertices = 0
            vert_sers = self.__get_all_vertices_series()
            if vert_sers:
                if self.__series_type is cudf.Series:
                    self.__num_vertices = cudf.concat(
                        vert_sers, ignore_index=True
                    ).nunique()
                else:
                    self.__num_vertices = pd.concat(
                        vert_sers, ignore_index=True
                    ).nunique()
            return self.__num_vertices

        value_counts = self._vertex_type_value_counts
        if type == self._default_type_name and include_edge_data:
            # The default type, "", can refer to both vertex and edge data
            if self.__vertex_prop_dataframe is None:
                return self.get_num_vertices()
            return (
                self.get_num_vertices()
                - len(self.__vertex_prop_dataframe)
                + (value_counts[type] if type in value_counts else 0)
            )
        if self.__vertex_prop_dataframe is None:
            return 0
        return value_counts[type] if type in value_counts else 0

    def get_num_edges(self, type=None):
        """
        Return the number of all edges or edges of a given type.

        Parameters
        ----------
        type : string, optional
            Edge type or None, if None then all edges are counted

        Returns
        -------
        int
            If type is None (the default), returns the total number of edges,
            otherwise return the number of edges of the specified type.

        See Also
        --------
        PropertyGraph.get_num_vertices

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                     data=[(99, 22, "a"),
        ...                           (98, 34, "b"),
        ...                           (97, 56, "c"),
        ...                           (96, 88, "d"),
        ...                          ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
        >>> pG.get_num_edges()
        4
        """
        if type is None:
            if self.__edge_prop_dataframe is not None:
                return len(self.__edge_prop_dataframe)
            else:
                return 0
        if self.__edge_prop_dataframe is None:
            return 0
        value_counts = self._edge_type_value_counts
        return value_counts[type] if type in value_counts else 0

    def get_vertices(self, selection=None):
        """
        Return a Series containing the unique vertex IDs contained in both
        the vertex and edge property data.
        Selection is not yet supported.

        Parameters
        ----------
        selection : PropertySelection, optional
            A PropertySelection returned from one or more calls to
            select_vertices() and/or select_edges()

        Returns
        -------
        cudf series or pandas series, optional
            Contains vertices that match the selection or all

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                     data=[(99, 22, "a"),
        ...                           (98, 34, "b"),
        ...                           (97, 56, "c"),
        ...                           (96, 88, "d"),
        ...                          ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
        >>> pG.get_vertices()
        0    22
        1    34
        2    56
        3    88
        4    96
        5    97
        6    98
        7    99
        dtype: int64
        """
        vert_sers = self.__get_all_vertices_series()
        if vert_sers:
            if self.__series_type is cudf.Series:
                return self.__series_type(
                    cudf.concat(vert_sers, ignore_index=True).unique()
                )
            else:
                return self.__series_type(
                    pd.concat(vert_sers, ignore_index=True).unique()
                )
        return self.__series_type()

    def vertices_ids(self):
        """
        Alias for get_vertices()

        Returns
        -------
        cudf Series or pandas Series
            Series containing the unique vertex IDs in both the
            vertex and edge property data. Return type is based
            on if the PropertyGraph instance was created/updated
            using cudf or pandas DataFrames.

        See Also
        --------
        PropertyGraph.get_vertices
        """
        return self.get_vertices()

    def vertex_types_from_numerals(
        self, nums: Union[cudf.Series, pd.Series]
    ) -> Union[cudf.Series, pd.Series]:
        """
        Returns the string vertex type names given the numeric category labels.

        Parameters
        ----------
        nums: Union[cudf.Series, pandas.Series] (Required)
            The list of numeric category labels to convert.

        Returns
        -------
        Union[cudf.Series, pd.Series]
            The string type names converted from the input numerals.
        """
        return self.__vertex_prop_dataframe[self.type_col_name].dtype.categories[nums]

    def edge_types_from_numerals(
        self, nums: Union[cudf.Series, pd.Series]
    ) -> Union[cudf.Series, pd.Series]:
        """
        Returns the string edge type names given the numeric category labels.

        Parameters
        ----------
        nums: Union[cudf.Series, pandas.Series] (Required)
            The list of numeric category labels to convert.

        Returns
        -------
        Union[cudf.Series, pd.Series]
            The string type names converted from the input numerals.
        """
        return self.__edge_prop_dataframe[self.type_col_name].dtype.categories[nums]

    def add_vertex_data(
        self,
        dataframe,
        vertex_col_name,
        type_name=None,
        property_columns=None,
        vector_properties=None,
        vector_property=None,
    ):
        """
        Add a dataframe describing vertex properties to the PropertyGraph.
        Can contain additional vertices that will not have associated edges.

        Parameters
        ----------
        dataframe : DataFrame-compatible instance
            A DataFrame instance with a compatible Pandas-like DataFrame
            interface.
        vertex_col_name : string
            The column name that contains the values to be used as vertex IDs.
        type_name : string, optional
            The name to be assigned to the type of property being added. For
            example, if dataframe contains data about users, type_name might be
            "users". If not specified, the type of properties will be added as
            the empty string, "".
        property_columns : list of strings, optional
            List of column names in dataframe to be added as properties. All
            other columns in the dataframe will be ignored. If not specified, all
            columns in dataframe are added.
        vector_properties : dict of string to list of strings, optional
            A dict of vector properties to create from columns in the dataframe.
            Each vector property stores an array for each vertex.
            The dict keys are the new vector property names, and the dict values
            should be Python lists of column names from which to create the vector
            property. Columns used to create vector properties won't be added to
            the property graph by default, but may be included as properties by
            including them in the property_columns argument.
            Use ``PropertyGraph.vertex_vector_property_to_array`` to convert a
            vertex vector property to an array.
        vector_property : string, optional
            If provided, all columns not included in other arguments will be used
            to create a vector property with the given name. This is often used
            for convenience instead of ``vector_properties`` when all input
            properties should be converted to a vector property.

        Returns
        -------
        None

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                     data=[(99, 22, "a"),
        ...                           (98, 34, "b"),
        ...                           (97, 56, "c"),
        ...                           (96, 88, "d"),
        ...                          ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
        >>> vert_df = cudf.DataFrame({"vert_id": [99, 22, 98, 34, 97, 56, 96, 88],
        ...                           "v_prop": [1, 2, 3, 4, 5, 6, 7, 8]})
        >>> pG.add_vertex_data(vert_df, type_name="vtype", vertex_col_name="vert_id")
        >>> pG.get_vertex_data().sort_index(axis=1)
        _TYPE_  _VERTEX_  v_prop
        0  vtype        99       1
        1  vtype        22       2
        2  vtype        98       3
        3  vtype        34       4
        4  vtype        97       5
        5  vtype        56       6
        6  vtype        96       7
        7  vtype        88       8
        """
        if type(dataframe) not in _dataframe_types:
            raise TypeError(
                "dataframe must be one of the following types: "
                f"{_dataframe_types}, got: {type(dataframe)}"
            )
        if vertex_col_name not in dataframe.columns:
            raise ValueError(
                f"{vertex_col_name} is not a column in "
                f"dataframe: {dataframe.columns}"
            )
        if type_name is not None and not isinstance(type_name, str):
            raise TypeError(f"type_name must be a string, got: {type(type_name)}")
        if type_name is None:
            type_name = self._default_type_name
        if property_columns:
            if type(property_columns) is not list:
                raise TypeError(
                    f"property_columns must be a list, got: {type(property_columns)}"
                )
            invalid_columns = set(property_columns).difference(dataframe.columns)
            if invalid_columns:
                raise ValueError(
                    "property_columns contains column(s) not found in dataframe: "
                    f"{list(invalid_columns)}"
                )
            existing_vectors = (
                set(property_columns) & self.__vertex_vector_property_lengths.keys()
            )
            if existing_vectors:
                raise ValueError(
                    "Non-vector property columns cannot be added to existing "
                    f"vector properties: {', '.join(sorted(existing_vectors))}"
                )

        # Save the DataFrame and Series types for future instantiations
        if self.__dataframe_type is None or self.__series_type is None:
            self.__dataframe_type = type(dataframe)
            self.__series_type = type(dataframe[dataframe.columns[0]])
        else:
            if type(dataframe) is not self.__dataframe_type:
                raise TypeError(
                    f"dataframe is type {type(dataframe)} but "
                    "the PropertyGraph was already initialized "
                    f"using type {self.__dataframe_type}"
                )
        TCN = self.type_col_name
        if vector_properties is not None:
            invalid_keys = {self.vertex_col_name, TCN}
            if property_columns:
                invalid_keys.update(property_columns)
            self._check_vector_properties(
                dataframe,
                vector_properties,
                self.__vertex_vector_property_lengths,
                invalid_keys,
            )
        if vector_property is not None:
            invalid_keys = {self.vertex_col_name, TCN, vertex_col_name}
            if property_columns:
                invalid_keys.update(property_columns)
            if vector_properties:
                invalid_keys.update(*vector_properties.values())
            d = {
                vector_property: [
                    col for col in dataframe.columns if col not in invalid_keys
                ]
            }
            invalid_keys.remove(vertex_col_name)
            self._check_vector_properties(
                dataframe,
                d,
                self.__vertex_vector_property_lengths,
                invalid_keys,
            )
            # Update vector_properties, but don't mutate the original
            if vector_properties is not None:
                d.update(vector_properties)
            vector_properties = d

        # Clear the cached values related to the number of vertices since more
        # could be added in this method.
        self.__num_vertices = None
        self.__vertex_type_value_counts = None  # Could update instead

        # Add `type_name` to the TYPE categorical dtype if necessary
        is_first_data = self.__vertex_prop_dataframe is None
        if is_first_data:
            # Initialize the __vertex_prop_dataframe using the same type
            # as the incoming dataframe.
            self.__vertex_prop_dataframe = self.__dataframe_type(
                columns=[self.vertex_col_name, TCN]
            )
            # Initialize the new columns to the same dtype as the appropriate
            # column in the incoming dataframe, since the initial merge may not
            # result in the same dtype. (see
            # https://github.com/rapidsai/cudf/issues/9981)
            self.__vertex_prop_dataframe = self.__update_dataframe_dtypes(
                self.__vertex_prop_dataframe,
                {self.vertex_col_name: dataframe[vertex_col_name].dtype},
            )
            self.__vertex_prop_dataframe.set_index(self.vertex_col_name, inplace=True)

            # Use categorical dtype for the type column
            if self.__series_type is cudf.Series:
                cat_class = cudf.CategoricalDtype
            else:
                cat_class = pd.CategoricalDtype
            cat_dtype = cat_class([type_name], ordered=False)
        else:
            cat_dtype = self.__update_categorical_dtype(
                self.__vertex_prop_dataframe, TCN, type_name
            )

        # NOTE: This copies the incoming DataFrame in order to add the new
        # columns. The copied DataFrame is then merged (another copy) and then
        # deleted when out-of-scope.

        # Ensure that both the predetermined vertex ID column name and vertex
        # type column name are present for proper merging.
        tmp_df = dataframe.copy(deep=True)
        tmp_df[self.vertex_col_name] = tmp_df[vertex_col_name]
        # FIXME: handle case of a type_name column already being in tmp_df

        if self.__series_type is cudf.Series:
            # cudf does not yet support initialization with a scalar
            tmp_df[TCN] = cudf.Series(
                cudf.Series([type_name], dtype=cat_dtype).repeat(len(tmp_df)),
                index=tmp_df.index,
            )
        else:
            # pandas is oddly slow if dtype is passed to the constructor here
            tmp_df[TCN] = pd.Series(type_name, index=tmp_df.index).astype(cat_dtype)

        if property_columns:
            # all columns
            column_names_to_drop = set(tmp_df.columns)
            # remove the ones to keep
            column_names_to_drop.difference_update(
                property_columns + [self.vertex_col_name, TCN]
            )
        else:
            column_names_to_drop = {vertex_col_name}
        if vector_properties:
            # Drop vector property source columns by default
            more_to_drop = set().union(*vector_properties.values())
            if property_columns is not None:
                more_to_drop.difference_update(property_columns)
            column_names_to_drop |= more_to_drop
            column_names_to_drop -= vector_properties.keys()
            self._create_vector_properties(tmp_df, vector_properties)

        tmp_df.drop(labels=column_names_to_drop, axis=1, inplace=True)

        # Save the original dtypes for each new column so they can be restored
        # prior to constructing subgraphs (since column dtypes may get altered
        # during merge to accommodate NaN values).
        if is_first_data:
            new_col_info = tmp_df.dtypes.items()
        else:
            new_col_info = self.__get_new_column_dtypes(
                tmp_df, self.__vertex_prop_dataframe
            )
        self.__vertex_prop_dtypes.update(new_col_info)

        # TODO: allow tmp_df to come in with vertex id already as index
        tmp_df.set_index(self.vertex_col_name, inplace=True)
        tmp_df = self.__update_dataframe_dtypes(tmp_df, self.__vertex_prop_dtypes)

        if is_first_data:
            self.__vertex_prop_dataframe = tmp_df
        else:
            # Join on vertex ids (the index)
            # TODO: can we automagically determine when we to use concat?
            df = self.__vertex_prop_dataframe.join(tmp_df, how="outer", rsuffix="_NEW_")
            cols = self.__vertex_prop_dataframe.columns.intersection(
                tmp_df.columns
            ).to_list()
            rename_cols = {f"{col}_NEW_": col for col in cols}
            new_cols = list(rename_cols)
            sub_df = df[new_cols].rename(columns=rename_cols)
            df.drop(columns=new_cols, inplace=True)
            # This only adds data--it doesn't replace existing data
            df.fillna(sub_df, inplace=True)
            self.__vertex_prop_dataframe = df

        # Update the vertex eval dict with the latest column instances
        if self.__series_type is cudf.Series:
            latest = {
                n: self.__vertex_prop_dataframe[n]
                for n in self.__vertex_prop_dataframe.columns
            }
        else:
            latest = self.__vertex_prop_dataframe.to_dict("series")
        self.__vertex_prop_eval_dict.update(latest)
        self.__vertex_prop_eval_dict[
            self.vertex_col_name
        ] = self.__vertex_prop_dataframe.index

    def get_vertex_data(self, vertex_ids=None, types=None, columns=None):
        """
        Gets a DataFrame containing vertex properties

        Parameters
        ----------
        vertex_ids : one or a collection of integers, optional
            single, list, slice, pandas array, or series of integers which
            are the vertices to include in the returned dataframe
        types : str or collection of str, optional
            types of the vertices to include in the returned data.
            Default is to return all vertex types.
        columns : str or list of str, optional
            property or properties to include in returned data.
            Default includes all properties.

        Returns
        -------
        DataFrame
            containing vertex properties for only the specified
            vertex_ids, columns, and/or types, or all vertex IDs if not specified.

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                      data=[(99, 22, "a"),
        ...                            (98, 34, "b"),
        ...                            (97, 56, "c"),
        ...                            (96, 88, "d"),
        ...                           ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
        >>> vert_df = cudf.DataFrame({"vert_id": [99, 22, 98, 34, 97, 56, 96, 88],
        ...                           "v_prop": [1, 2, 3, 4, 5, 6, 7, 8]})
        >>> pG.add_vertex_data(vert_df, type_name="vtype", vertex_col_name="vert_id")
        >>> pG.get_vertex_data().sort_index(axis=1)
        _TYPE_  _VERTEX_  v_prop
        0  vtype        99       1
        1  vtype        22       2
        2  vtype        98       3
        3  vtype        34       4
        4  vtype        97       5
        5  vtype        56       6
        6  vtype        96       7
        7  vtype        88       8
        """
        if self.__vertex_prop_dataframe is not None:
            df = self.__vertex_prop_dataframe
            if vertex_ids is not None:
                if isinstance(vertex_ids, int):
                    vertex_ids = [vertex_ids]

                try:
                    df = df.loc[vertex_ids]
                except TypeError:
                    raise TypeError(
                        "vertex_ids needs to be a list-like type "
                        f"compatible with DataFrame.loc[], got {type(vertex_ids)}"
                    )

            if types is not None:
                if isinstance(types, str):
                    df_mask = df[self.type_col_name] == types
                else:
                    df_mask = df[self.type_col_name].isin(types)
                df = df.loc[df_mask]

            # The "internal" pG.vertex_col_name and pG.type_col_name columns
            # are also included/added since they are assumed to be needed by
            # the caller.
            if columns is not None:
                # FIXME: invalid columns will result in a KeyError, should a
                # check be done here and a more PG-specific error raised?
                df = df[[self.type_col_name] + columns]

            # Should not drop to ensure vertex ids are returned as a column.
            df_out = df.reset_index(drop=False)

            # Preserve the dtype (vertex id type) to avoid cugraph algorithms
            # throwing errors due to a dtype mismatch
            index_dtype = self.__vertex_prop_dataframe.index.dtype
            df_out.index = df_out.index.astype(index_dtype)

            return df_out
        return None

    def add_edge_data(
        self,
        dataframe,
        vertex_col_names,
        edge_id_col_name=None,
        type_name=None,
        property_columns=None,
        vector_properties=None,
        vector_property=None,
    ):
        """
        Add a dataframe describing edge properties to the PropertyGraph.
        Columns not specified as vertex columns are considered properties.

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
        type_name : string, optional
            The name to be assigned to the type of property being added. For
            example, if dataframe contains data about transactions, type_name
            might be "transactions". If not specified, the type of properties
            will be added as the empty string "".
        property_columns : list of strings, optional
            List of column names in the dataframe to be added as properties. All
            other columns in dataframe will be ignored. If not specified, all
            property columns in the dataframe are added.
        vector_properties : dict of string to list of strings, optional
            A dict of vector properties to create from columns in the dataframe.
            Each vector property stores an array for each edge.
            The dict keys are the new vector property names, and the dict values
            should be Python lists of column names from which to create the vector
            property. Columns used to create vector properties won't be added to
            the property graph by default, but may be included as properties by
            including them in the property_columns argument.
            Use ``PropertyGraph.edge_vector_property_to_array`` to convert an
            edge vector property to an array.
        vector_property : string, optional
            If provided, all columns not included in other arguments will be used
            to create a vector property with the given name. This is often used
            for convenience instead of ``vector_properties`` when all input
            properties should be converted to a vector property.

        Returns
        -------
        None

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                     data=[(99, 22, "a"),
        ...                           (98, 34, "b"),
        ...                           (97, 56, "c"),
        ...                           (96, 88, "d"),
        ...                          ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, vertex_col_names=("src", "dst"))
        >>> pG.get_num_vertices()
        8
        """
        if type(dataframe) not in _dataframe_types:
            raise TypeError(
                "dataframe must be one of the following types: "
                f"{_dataframe_types}, got: {type(dataframe)}"
            )
        if type(vertex_col_names) not in [list, tuple]:
            raise TypeError(
                "vertex_col_names must be a list or tuple, got: "
                f"{type(vertex_col_names)}"
            )
        if edge_id_col_name is not None:
            if not isinstance(edge_id_col_name, str):
                raise TypeError(
                    "edge_id_col_name must be a string, got: "
                    f"{type(edge_id_col_name)}"
                )
            if edge_id_col_name not in dataframe.columns:
                raise ValueError(
                    "edge_id_col_name argument not in columns, "
                    f"got {edge_id_col_name!r}"
                )
        invalid_columns = set(vertex_col_names).difference(dataframe.columns)
        if invalid_columns:
            raise ValueError(
                "vertex_col_names contains column(s) not found "
                f"in dataframe: {list(invalid_columns)}"
            )
        if type_name is not None and not isinstance(type_name, str):
            raise TypeError(f"type_name must be a string, got: {type(type_name)}")
        if type_name is None:
            type_name = self._default_type_name
        if property_columns:
            if type(property_columns) is not list:
                raise TypeError(
                    f"property_columns must be a list, got: {type(property_columns)}"
                )
            invalid_columns = set(property_columns).difference(dataframe.columns)
            if invalid_columns:
                raise ValueError(
                    "property_columns contains column(s) not found in dataframe: "
                    f"{list(invalid_columns)}"
                )
            existing_vectors = (
                set(property_columns) & self.__vertex_vector_property_lengths.keys()
            )
            if existing_vectors:
                raise ValueError(
                    "Non-vector property columns cannot be added to existing "
                    f"vector properties: {', '.join(sorted(existing_vectors))}"
                )

        # Save the DataFrame and Series types for future instantiations
        if self.__dataframe_type is None or self.__series_type is None:
            self.__dataframe_type = type(dataframe)
            self.__series_type = type(dataframe[dataframe.columns[0]])
        else:
            if type(dataframe) is not self.__dataframe_type:
                raise TypeError(
                    f"dataframe is type {type(dataframe)} but "
                    "the PropertyGraph was already initialized "
                    f"using type {self.__dataframe_type}"
                )
        if self.__is_edge_id_autogenerated is False and edge_id_col_name is None:
            raise NotImplementedError(
                "Unable to automatically generate edge IDs. "
                "`edge_id_col_name` must be specified if edge data has been "
                "previously added with edge_id_col_name."
            )
        if self.__is_edge_id_autogenerated is True and edge_id_col_name is not None:
            raise NotImplementedError(
                "Invalid use of `edge_id_col_name`. Edge data has already "
                "been added with automatically generated IDs, so now all "
                "edge data must be added using automatically generated IDs."
            )

        TCN = self.type_col_name
        if vector_properties is not None:
            invalid_keys = {self.src_col_name, self.dst_col_name, TCN}
            if property_columns:
                invalid_keys.update(property_columns)
            self._check_vector_properties(
                dataframe,
                vector_properties,
                self.__edge_vector_property_lengths,
                invalid_keys,
            )
        if vector_property is not None:
            invalid_keys = {
                self.src_col_name,
                self.dst_col_name,
                TCN,
                vertex_col_names[0],
                vertex_col_names[1],
            }
            if property_columns:
                invalid_keys.update(property_columns)
            if vector_properties:
                invalid_keys.update(*vector_properties.values())
            d = {
                vector_property: [
                    col for col in dataframe.columns if col not in invalid_keys
                ]
            }
            invalid_keys.difference_update(vertex_col_names)
            self._check_vector_properties(
                dataframe,
                d,
                self.__edge_vector_property_lengths,
                invalid_keys,
            )
            # Update vector_properties, but don't mutate the original
            if vector_properties is not None:
                d.update(vector_properties)
            vector_properties = d

        # Clear the cached value for num_vertices since more could be added in
        # this method. This method cannot affect __node_type_value_counts
        self.__num_vertices = None
        self.__edge_type_value_counts = None  # Could update instead

        # Add `type_name` to the categorical dtype if necessary
        is_first_data = self.__edge_prop_dataframe is None
        if is_first_data:
            self.__edge_prop_dataframe = self.__dataframe_type(
                columns=[self.src_col_name, self.dst_col_name, TCN]
            )
            # Initialize the new columns to the same dtype as the appropriate
            # column in the incoming dataframe, since the initial merge may not
            # result in the same dtype. (see
            # https://github.com/rapidsai/cudf/issues/9981)
            self.__edge_prop_dataframe = self.__update_dataframe_dtypes(
                self.__edge_prop_dataframe,
                {
                    self.src_col_name: dataframe[vertex_col_names[0]].dtype,
                    self.dst_col_name: dataframe[vertex_col_names[1]].dtype,
                },
            )
            self.__edge_prop_dataframe.index.name = self.edge_id_col_name

            # Use categorical dtype for the type column
            if self.__series_type is cudf.Series:
                cat_class = cudf.CategoricalDtype
            else:
                cat_class = pd.CategoricalDtype
            cat_dtype = cat_class([type_name], ordered=False)
            self.__is_edge_id_autogenerated = edge_id_col_name is None
        else:
            cat_dtype = self.__update_categorical_dtype(
                self.__edge_prop_dataframe, TCN, type_name
            )

        # NOTE: This copies the incoming DataFrame in order to add the new
        # columns. The copied DataFrame is then merged (another copy) and then
        # deleted when out-of-scope.
        tmp_df = dataframe.copy(deep=True)
        tmp_df[self.src_col_name] = tmp_df[vertex_col_names[0]]
        tmp_df[self.dst_col_name] = tmp_df[vertex_col_names[1]]

        if self.__series_type is cudf.Series:
            # cudf does not yet support initialization with a scalar
            tmp_df[TCN] = cudf.Series(
                cudf.Series([type_name], dtype=cat_dtype).repeat(len(tmp_df)),
                index=tmp_df.index,
            )
        else:
            # pandas is oddly slow if dtype is passed to the constructor here
            tmp_df[TCN] = pd.Series(type_name, index=tmp_df.index).astype(cat_dtype)

        # Add unique edge IDs to the new rows. This is just a count for each
        # row starting from the last edge ID value, with initial edge ID 0.
        if edge_id_col_name is None:
            start_eid = 0 if self.__last_edge_id is None else self.__last_edge_id
            end_eid = start_eid + len(tmp_df)  # exclusive
            if self.__series_type is cudf.Series:
                index_class = cudf.RangeIndex
            else:
                index_class = pd.RangeIndex
            tmp_df.index = index_class(start_eid, end_eid, name=self.edge_id_col_name)
            self.__last_edge_id = end_eid
        else:
            tmp_df.set_index(edge_id_col_name, inplace=True)
            tmp_df.index.name = self.edge_id_col_name

        if property_columns:
            # all columns
            column_names_to_drop = set(tmp_df.columns)
            # remove the ones to keep
            column_names_to_drop.difference_update(
                property_columns + [self.src_col_name, self.dst_col_name, TCN]
            )
        else:
            column_names_to_drop = {vertex_col_names[0], vertex_col_names[1]}

        if vector_properties:
            # Drop vector property source columns by default
            more_to_drop = set().union(*vector_properties.values())
            if property_columns is not None:
                more_to_drop.difference_update(property_columns)
            column_names_to_drop |= more_to_drop
            column_names_to_drop -= vector_properties.keys()
            self._create_vector_properties(tmp_df, vector_properties)

        tmp_df.drop(labels=column_names_to_drop, axis=1, inplace=True)

        # Save the original dtypes for each new column so they can be restored
        # prior to constructing subgraphs (since column dtypes may get altered
        # during merge to accommodate NaN values).
        if is_first_data:
            new_col_info = tmp_df.dtypes.items()
        else:
            new_col_info = self.__get_new_column_dtypes(
                tmp_df, self.__edge_prop_dataframe
            )
        self.__edge_prop_dtypes.update(new_col_info)

        # TODO: allow tmp_df to come in with edge id already as index
        tmp_df = self.__update_dataframe_dtypes(tmp_df, self.__edge_prop_dtypes)

        if is_first_data:
            self.__edge_prop_dataframe = tmp_df
        else:
            # Join on edge ids (the index)
            # TODO: can we automagically determine when we to use concat?
            df = self.__edge_prop_dataframe.join(tmp_df, how="outer", rsuffix="_NEW_")
            cols = self.__edge_prop_dataframe.columns.intersection(
                tmp_df.columns
            ).to_list()
            rename_cols = {f"{col}_NEW_": col for col in cols}
            new_cols = list(rename_cols)
            sub_df = df[new_cols].rename(columns=rename_cols)
            df.drop(columns=new_cols, inplace=True)
            # This only adds data--it doesn't replace existing data
            df.fillna(sub_df, inplace=True)
            self.__edge_prop_dataframe = df

        # Update the edge eval dict with the latest column instances
        if self.__series_type is cudf.Series:
            latest = {
                n: self.__edge_prop_dataframe[n]
                for n in self.__edge_prop_dataframe.columns
            }
        else:
            latest = self.__edge_prop_dataframe.to_dict("series")
        self.__edge_prop_eval_dict.update(latest)
        self.__edge_prop_eval_dict[
            self.edge_id_col_name
        ] = self.__edge_prop_dataframe.index

    def get_edge_data(self, edge_ids=None, types=None, columns=None):
        """
        Return a dataframe containing edge properties for only the specified
        edge_ids, columns, and/or edge type, or all edge IDs if not specified.

        Parameters
        ----------
        edge_ids : int or collection of int, optional
            The list of edges to include in the edge data
        types : list, optional
            List of edge types to include in returned dataframe.
            None is the default and will return all edge types.
        columns : which edge columns will be returned, optional
            None is the default and will result in all columns being returned

        Returns
        -------
        Dataframe
            Containing edge ids, type edge source, destination
            and all the columns specified in the columns parameter

        Examples
        --------
        >>> import cudf
        >>> import cugraph
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                     data=[(99, 22, "a"),
        ...                           (98, 34, "b"),
        ...                           (97, 56, "c"),
        ...                           (96, 88, "d"),
        ...                          ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
        >>> pG.get_edge_data(types="etype").sort_index(axis=1)
        _DST_  _EDGE_ID_  _SRC_ _TYPE_ some_property
        0     22          0     99  etype             a
        1     34          1     98  etype             b
        2     56          2     97  etype             c
        3     88          3     96  etype             d
        """
        if self.__edge_prop_dataframe is not None:
            df = self.__edge_prop_dataframe
            if edge_ids is not None:
                if isinstance(edge_ids, int):
                    edge_ids = [edge_ids]

                try:
                    df = df.loc[edge_ids]
                except TypeError:
                    raise TypeError(
                        "edge_ids needs to be a list-like type "
                        f"compatible with DataFrame.loc[], got {type(edge_ids)}"
                    )

            if types is not None:
                if isinstance(types, str):
                    df_mask = df[self.type_col_name] == types
                else:
                    df_mask = df[self.type_col_name].isin(types)
                df = df.loc[df_mask]

            # The "internal" src, dst, edge_id, and type columns are also
            # included/added since they are assumed to be needed by the caller.
            if columns is None:
                # remove the "internal" weight column if one was added
                all_columns = list(self.__edge_prop_dataframe.columns)
                if self.weight_col_name in all_columns:
                    all_columns.remove(self.weight_col_name)
                df = df[all_columns]
            else:
                # FIXME: invalid columns will result in a KeyError, should a
                # check be done here and a more PG-specific error raised?
                df = df[
                    [self.src_col_name, self.dst_col_name, self.type_col_name] + columns
                ]

            # Should not drop so the edge ids are returned as a column.
            df_out = df.reset_index()

            # Preserve the dtype (edge id type) to avoid cugraph algorithms
            # throwing errors due to a dtype mismatch
            index_dtype = self.__edge_prop_dataframe.index.dtype
            df_out.index = df_out.index.astype(index_dtype)

            return df_out

        return None

    def fillna_vertices(self, val=0):
        """
        Fills empty vertex property values with the given value, zero by default.
        Fills in-place.

        Parameters
        ----------
        val : object, Series, or dict
            The object that will replace "na". Default = 0.  If a dict or
            Series is passed, the index or keys are the columns to fill
            and the values are the fill value for the corresponding column.
        """
        self.__vertex_prop_dataframe.fillna(val, inplace=True)

    def fillna_edges(self, val=0):
        """
        Fills empty edge property values with the given value, zero by default.
        Fills in-place.

        Parameters
        ----------
        val : object, Series, or dict
            The object that will replace "na". Default = 0.  If a dict or
            Series is passed, the index or keys are the columns to fill
            and the values are the fill value for the corresponding column.
        """

        self.__edge_prop_dataframe.fillna(val, inplace=True)

    def select_vertices(self, expr, from_previous_selection=None):
        """
        Evaluate expr and return a PropertySelection object representing the
        vertices that match the expression.

        Parameters
        ----------
        expr : string
            A python expression using property names and operators to select
            specific vertices.
        from_previous_selection : PropertySelection, optional
            A PropertySelection instance returned from a prior call to
            select_vertices() that can be used to select a subset of vertices
            to evaluate the expression against. This allows for a selection of
            the intersection of vertices of multiple types (eg. all vertices
            that are both type A and type B)

        Returns
        -------
        PropertySelection
            used for calls to extract_subgraph()
            in order to construct a Graph containing only specific vertices.

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                     data=[(99, 22, "a"),
        ...                           (98, 34, "b"),
        ...                           (97, 56, "c"),
        ...                           (96, 88, "d"),
        ...                          ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
        >>> vert_df = cudf.DataFrame({"vert_id": [99, 22, 98, 34, 97, 56, 96, 88],
        ...                           "v_prop": [1, 2, 3, 4, 5, 6, 7, 8]})
        >>> pG.add_vertex_data(vert_df, type_name="vtype", vertex_col_name="vert_id")
        >>> selection = pG.select_vertices("(_TYPE_ == 'vtype') & (v_prop > 4)")
        >>> G = pG.extract_subgraph(selection=selection)
        >>> print (G.number_of_vertices())
        4
        """
        # FIXME: check types

        # Check if the expr is to be evaluated in the context of properties
        # from only the previously selected vertices (as opposed to all
        # properties from all vertices)
        if (
            from_previous_selection is not None
            and from_previous_selection.vertex_selections is not None
        ):
            previously_selected_rows = self.__vertex_prop_dataframe[
                from_previous_selection.vertex_selections
            ]

            rows_to_eval = self.__vertex_prop_dataframe.loc[
                previously_selected_rows.index
            ]

            locals = dict([(n, rows_to_eval[n]) for n in rows_to_eval.columns])
            locals[self.vertex_col_name] = rows_to_eval.index
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
            selected_col = selected_col.reindex(
                self.__vertex_prop_dataframe.index, fill_value=False, copy=False
            )

        return EXPERIMENTAL__PropertySelection(vertex_selection_series=selected_col)

    def select_edges(self, expr):
        """
        Evaluate expr and return a PropertySelection object representing the
        edges that match the expression selection criteria.

        Parameters
        ----------
        expr : string
            A python expression using property names and operators to select
            specific edges.

        Returns
        -------
        PropertySelection
            Can be used for calls to extract_subgraph()
            in order to construct a Graph containing only specific edges.

        Examples
        --------
        >>> import cudf
        >>> import cugraph
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                     data=[(99, 22, "a"),
        ...                           (98, 34, "b"),
        ...                           (97, 56, "c"),
        ...                           (96, 88, "d"),
        ...                          ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
        >>> vert_df = cudf.DataFrame({"vert_id": [99, 22, 98, 34, 97, 56, 96, 88],
        ...                           "v_prop": [1, 2, 3, 4, 5, 6, 7, 8]})
        >>> pG.add_vertex_data(vert_df, type_name="vtype", vertex_col_name="vert_id")
        >>> selection = pG.select_edges("(_TYPE_ == 'etype') & (some_property == 'd')")
        >>> G = pG.extract_subgraph(selection=selection,
        ...                         create_using=cugraph.Graph(directed=True),
        ...                         renumber_graph=False)
        >>> print (G.edges())
        src  dst
        0   96   88
        """
        # FIXME: check types
        globals = {}
        locals = self.__edge_prop_eval_dict

        selected_col = eval(expr, globals, locals)
        return EXPERIMENTAL__PropertySelection(edge_selection_series=selected_col)

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
        selection : PropertySelection, optional
            A PropertySelection returned from one or more calls to
            select_vertices() and/or select_edges(), used for creating a Graph
            with only the selected properties. If not specified the returned
            Graph will have all properties. Note, this could result in a Graph
            with multiple edges, which may not be supported based on the value
            of create_using.
        edge_weight_property : string, optional
            The name of the property whose values will be used as weights on
            the returned Graph. If not specified, the returned Graph will be
            unweighted.
        default_edge_weight : float64, optional
            Value that replaces empty weight property fields
        check_multi_edges : bool (default True)
            When True and create_using argument is given and not a MultiGraph,
            this will perform an expensive check to verify that the edges in
            the edge dataframe do not form a multigraph with duplicate edges.
        renumber_graph : bool (default True)
            If True, return a Graph that has been renumbered for use by graph
            algorithms. If False, the returned graph will need to be manually
            renumbered prior to calling graph algos.
        add_edge_data : bool (default True)
            If True, add meta data about the edges contained in the extracted
            graph which are required for future calls to annotate_dataframe().

        Returns
        -------
        A Graph instance of the same type as create_using containing only the
        vertices and edges resulting from applying the selection to the set of
        vertex and edge property data.

        Examples
        --------
        >>> import cugraph
        >>> from cugraph.experimental import PropertyGraph
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                     data=[(99, 22, "a"),
        ...                           (98, 34, "b"),
        ...                           (97, 56, "c"),
        ...                           (96, 88, "d"),
        ...                          ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
        >>> vert_df = cudf.DataFrame({"vert_id": [99, 22, 98, 34, 97, 56, 96, 88],
        ...                           "v_prop": [1, 2, 3, 4, 5, 6, 7, 8]})
        >>> pG.add_vertex_data(vert_df, type_name="vtype", vertex_col_name="vert_id")
        >>> selection = pG.select_edges("(_TYPE_ == 'etype') & (some_property == 'd')")
        >>> G = pG.extract_subgraph(selection=selection,
        ...                         create_using=cugraph.Graph(directed=True),
        ...                         renumber_graph=False)
        >>> print (G.edges())
        src  dst
        0   96   88
        """
        if selection is not None and not isinstance(
            selection, EXPERIMENTAL__PropertySelection
        ):
            raise TypeError(
                "selection must be an instance of "
                f"PropertySelection, got {type(selection)}"
            )

        # NOTE: the expressions passed in to extract specific edges and
        # vertices assume the original dtypes in the user input have been
        # preserved. However, merge operations on the DataFrames can change
        # dtypes (eg. int64 to float64 in order to add NaN entries). This
        # should not be a problem since the conversions do not change the
        # values.
        if selection is not None and selection.vertex_selections is not None:
            selected_vertex_dataframe = self.__vertex_prop_dataframe[
                selection.vertex_selections
            ]
        else:
            selected_vertex_dataframe = None

        if selection is not None and selection.edge_selections is not None:
            selected_edge_dataframe = self.__edge_prop_dataframe[
                selection.edge_selections
            ]
        else:
            selected_edge_dataframe = self.__edge_prop_dataframe

        # FIXME: check that self.__edge_prop_dataframe is set!

        # If vertices were specified, select only the edges that contain the
        # selected verts in both src and dst
        if (
            selected_vertex_dataframe is not None
            and not selected_vertex_dataframe.empty
        ):
            has_srcs = selected_edge_dataframe[self.src_col_name].isin(
                selected_vertex_dataframe.index
            )
            has_dsts = selected_edge_dataframe[self.dst_col_name].isin(
                selected_vertex_dataframe.index
            )
            edges = selected_edge_dataframe[has_srcs & has_dsts]
            # Alternative to benchmark
            # edges = selected_edge_dataframe.merge(
            #     selected_vertex_dataframe[[]],
            #     left_on=self.src_col_name,
            #     right_index=True,
            # ).merge(
            #     selected_vertex_dataframe[[]],
            #     left_on=self.dst_col_name,
            #     right_index=True,
            # )
        else:
            edges = selected_edge_dataframe

        # The __*_prop_dataframes have likely been merged several times and
        # possibly had their dtypes converted in order to accommodate NaN
        # values. Restore the original dtypes in the resulting edges df prior
        # to creating a Graph.
        edges = self.__update_dataframe_dtypes(edges, self.__edge_prop_dtypes)

        # Default create_using set here instead of function signature to
        # prevent cugraph from running on import. This may help diagnose errors
        if create_using is None:
            create_using = cugraph.MultiGraph(directed=True)

        return self.edge_props_to_graph(
            edges,
            create_using=create_using,
            edge_weight_property=edge_weight_property,
            default_edge_weight=default_edge_weight,
            check_multi_edges=check_multi_edges,
            renumber_graph=renumber_graph,
            add_edge_data=add_edge_data,
        )

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

        Examples
        --------
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                      data=[(99, 22, "a"),
        ...                            (98, 34, "b"),
        ...                            (97, 56, "c"),
        ...                            (96, 88, "d"),
        ...                           ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
        >>> G = pG.extract_subgraph(create_using=cugraph.Graph(directed=True))
        >>> # Represents results of an algorithm run on the graph returning a dataframe
        >>> algo_result = cudf.DataFrame({"from":df.src,
        ...                                "to":df.dst,
        ...                                "result": range(len(df.src))})
        >>> algo_result2 = pG.annotate_dataframe(algo_result,
        ...                                       G,
        ...                                       edge_vertex_col_names=("from", "to"))
        >>> print (algo_result2.sort_index(axis=1))
        _EDGE_ID_ _TYPE_  from  result some_property  to
        0          0  etype    99       0             a  22
        1          1  etype    98       1             b  34
        2          2  etype    97       2             c  56
        3          3  etype    96       3             d  88
        """
        # FIXME: check all args
        # FIXME: also provide the ability to annotate vertex data.
        (src_col_name, dst_col_name) = edge_vertex_col_names

        df_type = type(df)
        if df_type is not self.__dataframe_type:
            raise TypeError(
                f"df type {df_type} does not match DataFrame type "
                f"{self.__dataframe_type} used in PropertyGraph"
            )

        if hasattr(G, "edge_data"):
            edge_info_df = G.edge_data
        else:
            raise AttributeError("Graph G does not have attribute 'edge_data'")

        # Join on shared columns and the indices
        cols = self.__edge_prop_dataframe.columns.intersection(
            edge_info_df.columns
        ).to_list()
        cols.append(self.edge_id_col_name)

        # New result includes only properties from the src/dst edges identified
        # by edge IDs. All other data in df is merged based on src/dst values.
        # NOTE: results from MultiGraph graphs will have to include edge IDs!
        edge_props_df = edge_info_df.merge(
            self.__edge_prop_dataframe, on=cols, how="inner"
        )

        # FIXME: also allow edge ID col to be passed in and renamed.
        new_df = df.rename(
            columns={src_col_name: self.src_col_name, dst_col_name: self.dst_col_name}
        )
        new_df = new_df.merge(edge_props_df)
        # restore the original src/dst column names
        new_df.rename(
            columns={self.src_col_name: src_col_name, self.dst_col_name: dst_col_name},
            inplace=True,
        )

        # restore the original dtypes
        new_df = self.__update_dataframe_dtypes(new_df, self.__edge_prop_dtypes)
        for col in df.columns:
            new_df[col] = new_df[col].astype(df.dtypes[col])

        # FIXME: consider removing internal columns (_EDGE_ID_, etc.) and
        # columns from edge types not included in the edges in df.
        return new_df

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
        Create a Graph from the edges in edge_prop_df.

        Parameters
        ----------
        edge_prop_df : cudf.DataFrame or pandas.DataFrame
            conains the edge data with properties
        create_using : cugraph.Graph (or subclass of) instance.
            Attributes of the graph are passed to the returned graph.
        edge_weight_property : string, optional
            Property used to weight the returned graph.
        default_edge_weight : float64, optional
            Value used to replace NA in the specified weight column
        check_multi_edges : bool, optional (default=True)
            Prevent duplicate edges (if not allowed)
        renumber_graph : bool, optional (default=True)
            If True renumber edge Ids to start at 0, otherwise
            maintain the original ids
        add_edge_data bool, optional(default=True)

        Returns
        -------
        A CuGraph or Networkx Graph
            contains the edges in edge_prop_df

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                     data=[(99, 22, "a"),
        ...                           (98, 34, "b"),
        ...                           (97, 56, "c"),
        ...                           (96, 88, "d"),
        ...                          ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
        >>> G = pG.edge_props_to_graph(pG.edges,
        ...                        create_using=cugraph.Graph(),
        ...                        renumber_graph=False)
        >>> G.edges()
        src  dst
        0   88   96
        1   22   99
        2   56   97
        3   34   98
        """
        # Don't mutate input data, and ensure DataFrame is not a view
        edge_prop_df = edge_prop_df.copy()
        # FIXME: check default_edge_weight is valid
        if edge_weight_property:
            if (
                edge_weight_property not in edge_prop_df.columns
                and edge_prop_df.index.name != edge_weight_property
            ):
                raise ValueError(
                    "edge_weight_property "
                    f'"{edge_weight_property}" was not found in '
                    "edge_prop_df"
                )

            # Ensure a valid edge_weight_property can be used for applying
            # weights to the subgraph, and if a default_edge_weight was
            # specified, apply it to all NAs in the weight column.
            # Also allow the type column to be specified as the edge weight
            # property so that uniform_neighbor_sample can be called with
            # the weights interpreted as types.
            if edge_weight_property == self.type_col_name:
                prop_col = edge_prop_df[self.type_col_name].cat.codes.astype("float32")
                edge_prop_df["_temp_type_col"] = prop_col
                edge_weight_property = "_temp_type_col"
            elif edge_weight_property in edge_prop_df.columns:
                prop_col = edge_prop_df[edge_weight_property]
            else:
                prop_col = edge_prop_df.index.to_series()
            if prop_col.count() != prop_col.size:
                if default_edge_weight is None:
                    raise ValueError(
                        f'edge_weight_property "{edge_weight_property}" '
                        "contains NA values in the subgraph and "
                        "default_edge_weight is not set"
                    )
                prop_col = prop_col.fillna(default_edge_weight)
                if edge_weight_property in edge_prop_df.columns:
                    edge_prop_df[edge_weight_property] = prop_col
                else:
                    edge_prop_df.index = prop_col
            edge_attr = edge_weight_property

        # If a default_edge_weight was specified but an edge_weight_property
        # was not, a new edge weight column must be added.
        elif default_edge_weight:
            edge_attr = self.weight_col_name
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
            raise TypeError(
                "create_using must be a cugraph.Graph "
                "(or subclass) type or instance, got: "
                f"{type(create_using)}"
            )

        # Prevent duplicate edges (if not allowed) since applying them to
        # non-MultiGraphs would result in ambiguous edge properties.
        if (
            check_multi_edges
            and not G.is_multigraph()
            and self.is_multigraph(edge_prop_df)
        ):
            if create_using:
                if type(create_using) is type:
                    t = create_using.__name__
                else:
                    t = type(create_using).__name__
                msg = f"'{t}' graph type specified by create_using"
            else:
                msg = "default Graph graph type"
            raise RuntimeError(
                "query resulted in duplicate edges which "
                f"cannot be represented with the {msg}"
            )

        create_args = {
            "source": self.src_col_name,
            "destination": self.dst_col_name,
            "edge_attr": edge_attr,
            "renumber": renumber_graph,
        }
        if type(edge_prop_df) is cudf.DataFrame:
            G.from_cudf_edgelist(edge_prop_df.reset_index(), **create_args)
        else:
            G.from_pandas_edgelist(edge_prop_df.reset_index(), **create_args)

        if add_edge_data:
            # Set the edge_data on the resulting Graph to a DataFrame
            # containing the edges and the edge ID for each. Edge IDs are
            # needed for future calls to annotate_dataframe() in order to
            # associate edges with their properties, since the PG can contain
            # multiple edges between vertrices with different properties.
            # FIXME: also add vertex_data
            G.edge_data = self.__create_property_lookup_table(edge_prop_df)

        return G

    def renumber_vertices_by_type(self, prev_id_column=None):
        """
        Renumber vertex IDs to be contiguous by type.

        Parameters
        ----------
        prev_id_column : str, optional
            Column name to save the vertex ID before renumbering.

        Returns
        -------
        a DataFrame with the start and stop IDs for each vertex type.
        Stop is *inclusive*.

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                     data=[(99, 22, "a"),
        ...                           (98, 34, "b"),
        ...                           (97, 56, "c"),
        ...                           (96, 88, "d"),
        ...                          ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
        >>> vert_df1 = cudf.DataFrame({"vert_id": [99, 22, 98, 34],
        ...                            "v_prop": [1 ,2 ,3, 4]})
        >>> pG.add_vertex_data(vert_df1, type_name="vtype1", vertex_col_name="vert_id")
        >>> vert_df2 = cudf.DataFrame({"vert_id": [97, 56, 96, 88],
        ...                            "v_prop": [ 5, 6, 7, 8]})
        >>> pG.add_vertex_data(vert_df2, type_name="vtype2", vertex_col_name="vert_id")
        >>> pG.renumber_vertices_by_type()
                start  stop
        vtype1      0     3
        vtype2      4     7
        """
        # Check if some vertex IDs exist only in edge data
        TCN = self.type_col_name
        default = self._default_type_name
        if self.__edge_prop_dataframe is not None and self.get_num_vertices(
            default, include_edge_data=True
        ) != self.get_num_vertices(default, include_edge_data=False):
            raise NotImplementedError(
                "Currently unable to renumber vertices when some vertex "
                "IDs only exist in edge data"
            )
        if self.__vertex_prop_dataframe is None:
            return None
        if (
            prev_id_column is not None
            and prev_id_column in self.__vertex_prop_dataframe
        ):
            raise ValueError(
                f"Can't save previous IDs to existing column {prev_id_column!r}"
            )

        # Use categorical dtype for the type column
        if self.__series_type is cudf.Series:
            cat_class = cudf.CategoricalDtype
        else:
            cat_class = pd.CategoricalDtype

        is_cat = isinstance(self.__vertex_prop_dataframe.dtypes[TCN], cat_class)
        if not is_cat:
            cat_dtype = cat_class([TCN], ordered=False)
            self.__vertex_prop_dataframe[TCN] = self.__vertex_prop_dataframe[
                TCN
            ].astype(cat_dtype)

        index_dtype = self.__vertex_prop_dataframe.index.dtype
        df = self.__vertex_prop_dataframe.reset_index().sort_values(by=TCN)
        df.index = df.index.astype(index_dtype)
        if self.__edge_prop_dataframe is not None:
            mapper = self.__series_type(df.index, index=df[self.vertex_col_name])
            self.__edge_prop_dataframe[self.src_col_name] = self.__edge_prop_dataframe[
                self.src_col_name
            ].map(mapper)
            self.__edge_prop_dataframe[self.dst_col_name] = self.__edge_prop_dataframe[
                self.dst_col_name
            ].map(mapper)
        if prev_id_column is None:
            df.drop(columns=[self.vertex_col_name], inplace=True)
        else:
            df.rename(columns={self.vertex_col_name: prev_id_column}, inplace=True)
        df.index.name = self.vertex_col_name
        self.__vertex_prop_dataframe = df
        rv = self._vertex_type_value_counts.sort_index().cumsum().to_frame("stop")
        rv["start"] = rv["stop"].shift(1, fill_value=0)
        rv["stop"] -= 1  # Make inclusive
        return rv[["start", "stop"]]

    def renumber_edges_by_type(self, prev_id_column=None):
        """
        Renumber edge IDs to be contiguous by type.

        Parameters
        ----------
        prev_id_column : str, optional
            Column name to save the edge ID before renumbering.

        Returns
        -------
        DataFrame
            with the start and stop IDs for each edge type. Stop is *inclusive*.

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> pG = PropertyGraph()
        >>> df = cudf.DataFrame(columns=["src", "dst", "edge_ids" ,"some_property"],
        ...                     data=[(99, 22, 3, "a"),
        ...                           (98, 34, 5, "b"),
        ...                           (97, 56, 7, "c"),
        ...                           (96, 88, 11, "d"),
        ...                          ])
        >>> df2 = cudf.DataFrame(columns=["src", "dst", "edge_ids" ,"some_property"],
        ...                      data=[(95, 24, 2, "a"),
        ...                            (94, 36, 4, "b"),
        ...                            (93, 88, 8, "d"),
        ...                           ])
        >>> pG.add_edge_data(df,
        ...                  type_name="etype1",
        ...                  vertex_col_names=("src", "dst"),
        ...                  edge_id_col_name="edge_ids")
        >>> pG.add_edge_data(df2,
        ...                  type_name="etype2",
        ...                  vertex_col_names=("src", "dst"),
        ...                  edge_id_col_name="edge_ids")
        >>> pG.renumber_edges_by_type()
                start  stop
        etype1      0     3
        etype2      4     6
        """
        TCN = self.type_col_name

        if self.__edge_prop_dataframe is None:
            return None
        if prev_id_column is not None and prev_id_column in self.__edge_prop_dataframe:
            raise ValueError(
                f"Can't save previous IDs to existing column {prev_id_column!r}"
            )

        # Use categorical dtype for the type column
        if self.__series_type is cudf.Series:
            cat_class = cudf.CategoricalDtype
        else:
            cat_class = pd.CategoricalDtype

        is_cat = isinstance(self.__edge_prop_dataframe.dtypes[TCN], cat_class)
        if not is_cat:
            cat_dtype = cat_class([TCN], ordered=False)
            self.__edge_prop_dataframe[TCN] = self.__edge_prop_dataframe[TCN].astype(
                cat_dtype
            )

        df = self.__edge_prop_dataframe
        index_dtype = df.index.dtype
        if prev_id_column is None:
            df = df.sort_values(by=TCN, ignore_index=True)
        else:
            df = df.sort_values(by=TCN)
            df.index.name = prev_id_column
            df.reset_index(inplace=True)
        df.index = df.index.astype(index_dtype)
        df.index.name = self.edge_id_col_name
        self.__edge_prop_dataframe = df
        rv = self._edge_type_value_counts.sort_index().cumsum().to_frame("stop")
        rv["start"] = rv["stop"].shift(1, fill_value=0)
        rv["stop"] -= 1  # Make inclusive
        return rv[["start", "stop"]]

    def vertex_vector_property_to_array(
        self, df, col_name, fillvalue=None, *, missing="ignore"
    ):
        """Convert a known vertex vector property in a DataFrame to an array.

        Parameters
        ----------
        df : cudf.DataFrame or pandas.DataFrame
            If cudf.DataFrame, the result will be a cupy.ndarray.
            If pandas.DataFrame, the result will be a numpy.ndarray.
        col_name : str
            The column name in the DataFrame to convert to an array.
            This vector property should have been created by PropertyGraph.
        fillvalue : scalar or list, optional (default None)
            Fill value for rows with missing vector data.  If it is a list,
            it must be the correct size of the vector property.  If fillvalue is None,
            then behavior if missing data is controlled by ``missing`` keyword.
            Leave this as None for better performance if all rows should have data.
        missing : {"ignore", "error"}
            If "ignore", empty or null rows without vector data will be skipped
            when creating the array, so output array shape will be
            [# of non-empty rows] by [size of vector property].
            When "error", RuntimeError will be raised if there are any empty rows.
            Ignored if fillvalue is given.

        Returns
        -------
        cupy.ndarray or numpy.ndarray
        """
        if col_name not in self.__vertex_vector_property_lengths:
            raise ValueError(f"{col_name!r} is not a known vertex vector property")
        length = self.__vertex_vector_property_lengths[col_name]
        return self._get_vector_property(df, col_name, length, fillvalue, missing)

    def edge_vector_property_to_array(
        self, df, col_name, fillvalue=None, *, missing="ignore"
    ):
        """Convert a known edge vector property in a DataFrame to an array.

        Parameters
        ----------
        df : cudf.DataFrame or pandas.DataFrame
            If cudf.DataFrame, the result will be a cupy.ndarray.
            If pandas.DataFrame, the result will be a numpy.ndarray.
        col_name : str
            The column name in the DataFrame to convert to an array.
            This vector property should have been created by PropertyGraph.
        fillvalue : scalar or list, optional (default None)
            Fill value for rows with missing vector data.  If it is a list,
            it must be the correct size of the vector property.  If fillvalue is None,
            then behavior if missing data is controlled by ``missing`` keyword.
            Leave this as None for better performance if all rows should have data.
        missing : {"ignore", "error"}
            If "ignore", empty or null rows without vector data will be skipped
            when creating the array, so output array shape will be
            [# of non-empty rows] by [size of vector property].
            When "error", RuntimeError will be raised if there are any empty rows.
            Ignored if fillvalue is given.

        Returns
        -------
        cupy.ndarray or numpy.ndarray
        """
        if col_name not in self.__edge_vector_property_lengths:
            raise ValueError(f"{col_name!r} is not a known edge vector property")
        length = self.__edge_vector_property_lengths[col_name]
        return self._get_vector_property(df, col_name, length, fillvalue, missing)

    def _check_vector_properties(
        self, df, vector_properties, vector_property_lengths, invalid_keys
    ):
        """Check if vector_properties is valid and update vector_property_lengths"""
        df_cols = set(df.columns)
        for key, columns in vector_properties.items():
            if key in invalid_keys:
                raise ValueError(
                    "Cannot assign new vector property to existing "
                    f"non-vector property: {key}"
                )
            if isinstance(columns, str):
                # If df[columns] is a ListDtype column, should we allow it?
                raise TypeError(
                    f"vector property columns for {key!r} should be a list; "
                    f"got a str ({columns!r})"
                )
            if not df_cols.issuperset(columns):
                missing = ", ".join(set(columns) - df_cols)
                raise ValueError(
                    f"Dataframe does not have columns for vector property {key!r}:"
                    f"{missing}"
                )
            if not columns:
                raise ValueError("Empty vector property columns for {key!r}!")
            if vector_property_lengths.get(key, len(columns)) != len(columns):
                prev_length = vector_property_lengths[key]
                new_length = len(columns)
                raise ValueError(
                    f"Wrong size for vector property {key}; got {new_length}, but "
                    f"this vector property already exists with size {prev_length}"
                )
        for key, columns in vector_properties.items():
            vector_property_lengths[key] = len(columns)

    @staticmethod
    def _create_vector_properties(df, vector_properties):
        vectors = {}
        for key, columns in vector_properties.items():
            values = df[columns].values
            if isinstance(df, cudf.DataFrame):
                vectors[key] = create_list_series_from_2d_ar(values, index=df.index)
            else:
                vectors[key] = [
                    np.squeeze(vec, 0)
                    for vec in np.split(
                        np.ascontiguousarray(values, like=values), len(df)
                    )
                ]
        for key, vec in vectors.items():
            df[key] = vec

    def _get_vector_property(self, df, col_name, length, fillvalue, missing):
        if type(df) is not self.__dataframe_type:
            raise TypeError(
                f"Expected type {self.__dataframe_type}; got type {type(df)}"
            )
        if col_name not in df.columns:
            raise ValueError(f"Column name {col_name} is not in the columns of df")
        if missing not in {"error", "ignore"}:
            raise ValueError(
                f'missing keyword must be one of "error" or "ignore"; got {missing!r}'
            )
        if fillvalue is not None:
            try:
                fill = list(fillvalue)
            except Exception:
                fill = [fillvalue] * length
            else:
                if len(fill) != length:
                    raise ValueError(
                        f"Wrong size of list as fill value; got {len(fill)}, "
                        f"expected {length}"
                    )
            s = df[col_name].copy()  # copy b/c we mutate below
        else:
            s = df[col_name]
        if self.__series_type is cudf.Series:
            if df.dtypes[col_name] != "list":
                raise TypeError(
                    "Wrong dtype for vector property; expected 'list', "
                    f"got {df.dtypes[col_name]}"
                )
            if fillvalue is not None:
                s[s.isnull()] = fill
            # This returns a writable view (i.e., no copies!)
            rv = s._data.columns[0].children[-1].values.reshape(-1, length)
        else:
            if df.dtypes[col_name] != object:
                raise TypeError(
                    "Wrong dtype for vector property; expected 'object', "
                    f"got {df.dtypes[col_name]}"
                )
            if fillvalue is not None:
                a = np.empty(1, dtype=object)
                a[0] = np.array(fill)
                s[s.isnull()] = a
            else:
                s = s[s.notnull()]
            rv = np.vstack(s.to_numpy())
        if fillvalue is None and missing == "error" and rv.shape[0] != len(df):
            raise RuntimeError(
                f"Vector property {col_name!r} has empty rows! "
                'Provide a fill value or use `missing="ignore"` to ignore empty rows.'
            )
        return rv

    def is_multi_gpu(self):
        """
        Return True if this is a multi-gpu graph.  Always returns False for
        PropertyGraph.
        """
        return False

    @classmethod
    def is_multigraph(cls, df):
        """
        Parameters
        ----------
        df : dataframe
            Containing edge data

        Returns
        -------
        bool
            True if df has one or more edges with the same source, destination pair

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> pG = PropertyGraph()
        >>> df = cudf.DataFrame(columns=["src", "dst", "edge_ids", "some_property"],
        ...                     data=[(99, 22, 3, "a"),
        ...                           (98, 34, 5, "b"),
        ...                           (98, 34, 7, "c"),
        ...                           (96, 88, 11, "d"),
        ...                          ])
        >>> pG.add_edge_data(df, type_name="etype",
        ...                  vertex_col_names=("src", "dst"),
        ...                  edge_id_col_name="edge_ids")
        >>> pG.is_multigraph(pG.get_edge_data())
        True
        """
        return cls._has_duplicates(df, [cls.src_col_name, cls.dst_col_name])

    @classmethod
    def has_duplicate_edges(cls, df, columns=None):
        """
        Return True if df has rows with the same src, dst, type, and columns

        Parameters
        ----------
        df : dataframe
            Containing the edges to test test for duplicates
        columns : list of strings, optional
            List of column names to use when testing for duplicate edges in
            addition to source, destination and type.

        Returns
        -------
        bool
            True if df has multiple rows with the same source, destination and type
            plus columns that are specified.

        Examples
        --------
        >>> import cugraph
        >>> import cudf
        >>> from cugraph.experimental import PropertyGraph
        >>> df = cudf.DataFrame(columns=["src", "dst", "some_property"],
        ...                     data=[(99, 22, "a"),
        ...                           (98, 34, "b"),
        ...                           (97, 56, "c"),
        ...                           (96, 88, "d"),
        ...                          ])
        >>> pG = PropertyGraph()
        >>> pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
        >>> PropertyGraph.has_duplicate_edges(pG.get_edge_data())
        False
        """
        cols = [cls.src_col_name, cls.dst_col_name, cls.type_col_name]
        if columns:
            cols.extend(columns)
        return cls._has_duplicates(df, cols)

    @classmethod
    def _has_duplicates(cls, df, cols):
        """
        Checks for duplicate edges in the dataframe with the
        provided columns being equal as the criteria.
        """
        if df.empty:
            return False
        unique_pair_len = len(df[cols].drop_duplicates(ignore_index=True))
        # if unique_pairs == len(df)
        # then no duplicate edges
        return unique_pair_len != len(df)

    def __create_property_lookup_table(self, edge_prop_df):
        """
        a DataFrame containing the src vertex, dst vertex, and edge_id
        values from edge_prop_df.
        """
        src = edge_prop_df[self.src_col_name]
        dst = edge_prop_df[self.dst_col_name]
        return self.__dataframe_type(
            {self.src_col_name: src, self.dst_col_name: dst}
        ).reset_index()

    def __get_all_vertices_series(self):
        """
        Returns a list of all Series objects that contain vertices from all
        tables.
        """
        vpd = self.__vertex_prop_dataframe
        epd = self.__edge_prop_dataframe
        vert_sers = []
        if vpd is not None:
            vert_sers.append(vpd.index.to_series())
        if epd is not None:
            vert_sers.append(epd[self.src_col_name])
            vert_sers.append(epd[self.dst_col_name])
        return vert_sers

    @staticmethod
    def __get_new_column_dtypes(from_df, to_df):
        """
        Returns a list containing tuples of (column name, dtype) for each
        column in from_df that is not present in to_df.
        """
        new_cols = set(from_df.columns) - set(to_df.columns)
        return [(col, from_df.dtypes[col]) for col in new_cols]

    @staticmethod
    def __update_dataframe_dtypes(df, column_dtype_dict):
        """
        Set the dtype for columns in df using the dtypes in column_dtype_dict.
        This also handles converting standard integer dtypes to nullable
        integer dtypes, needed to accommodate NA values in columns.
        """
        update_cols = {}
        for (col, dtype) in column_dtype_dict.items():
            if col not in df.columns:
                continue
            # If the DataFrame is Pandas and the dtype is an integer type,
            # ensure a nullable integer array is used by specifying the correct
            # dtype. The alias for these dtypes is simply a capitalized string
            # (eg. "Int64")
            # https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html#integer-dtypes-and-missing-data
            dtype_str = str(dtype)
            if dtype_str in ["int32", "int64"]:
                dtype_str = dtype_str.title()
            if str(df.dtypes[col]) != dtype_str:
                # Assigning to df[col] produces a (false?) warning with Pandas,
                # but assigning to df.loc[:,col] does not update the df in
                # cudf, so do one or the other based on type.
                update_cols[col] = df[col].astype(dtype_str)
        if not update_cols:
            return df
        # Use df.assign to avoid assignment into df in case df is a view:
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
        # #returning-a-view-versus-a-copy
        # Note that this requires all column names to be strings.
        return df.assign(**update_cols)

    def __update_categorical_dtype(self, df, column, val):
        """
        Add a new category to a categorical dtype column of a dataframe.
        Returns the new categorical dtype.
        """
        # Add `val` to the categorical dtype if necessary
        if val in df.dtypes[column].categories:
            # No need to change the categorical dtype
            pass
        elif self.__series_type is cudf.Series:
            # cudf isn't as fast as pandas; does it scan through the data?
            # inplace is supported in cudf, but is deprecated in pandas.
            df[column].cat.add_categories([val], inplace=True)
        else:
            # Very fast in pandas
            df[column] = df[column].cat.add_categories([val])
        return df.dtypes[column]
