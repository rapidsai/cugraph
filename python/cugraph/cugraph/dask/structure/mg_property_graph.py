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
import cupy
import cugraph
import dask_cudf
import cugraph.dask as dcg
from cugraph.utilities.utils import import_optional, create_list_series_from_2d_ar

from typing import Union

pd = import_optional("pandas")


class EXPERIMENTAL__MGPropertySelection:
    """
    Instances of this class are returned from the PropertyGraph.select_*()
    methods and can be used by the PropertyGraph.extract_subgraph() method to
    extract a Graph containing vertices and edges with only the selected
    properties.
    """

    def __init__(self, vertex_selection_series=None, edge_selection_series=None):
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
        return EXPERIMENTAL__MGPropertySelection(vs, es)


# FIXME: remove leading __ when no longer experimental
class EXPERIMENTAL__MGPropertyGraph:
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
    weight_col_name = "_WEIGHT_"
    _default_type_name = ""

    def __init__(self, num_workers=None):
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

        self.__dataframe_type = dask_cudf.DataFrame
        self.__series_type = dask_cudf.Series

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

        # number of gpu's to use
        if num_workers is None:
            self.__num_workers = dcg.get_n_workers()
        else:
            self.__num_workers = num_workers

    # PropertyGraph read-only attributes
    @property
    def edges(self):
        if self.__edge_prop_dataframe is not None:
            return self.__edge_prop_dataframe[
                [self.src_col_name, self.dst_col_name]
            ].reset_index()
        return None

    @property
    def vertex_property_names(self):
        if self.__vertex_prop_dataframe is not None:
            props = list(self.__vertex_prop_dataframe.columns)
            props.remove(self.type_col_name)  # should "type" be removed?
            return props
        return []

    @property
    def edge_property_names(self):
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
        """The set of vertex type names"""
        value_counts = self._vertex_type_value_counts
        if value_counts is None:
            names = set()
        elif self.__series_type is dask_cudf.Series:
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
        """The set of edge type names"""
        value_counts = self._edge_type_value_counts
        if value_counts is None:
            return set()
        elif self.__series_type is dask_cudf.Series:
            return set(value_counts.index.to_arrow().to_pylist())
        else:
            return set(value_counts.index)

    # PropertyGraph read-only attributes for debugging
    @property
    def _vertex_prop_dataframe(self):
        return self.__vertex_prop_dataframe

    @property
    def _edge_prop_dataframe(self):
        return self.__edge_prop_dataframe

    @property
    def _vertex_type_value_counts(self):
        """A Series of the counts of types in __vertex_prop_dataframe"""
        if self.__vertex_prop_dataframe is None:
            return
        if self.__vertex_type_value_counts is None:
            # Types should all be strings; what should we do if we see NaN?
            self.__vertex_type_value_counts = (
                self.__vertex_prop_dataframe[self.type_col_name]
                .value_counts(sort=False, dropna=False)
                .compute()
            )
        return self.__vertex_type_value_counts

    @property
    def _edge_type_value_counts(self):
        """A Series of the counts of types in __edge_prop_dataframe"""
        if self.__edge_prop_dataframe is None:
            return
        if self.__edge_type_value_counts is None:
            # Types should all be strings; what should we do if we see NaN?
            self.__edge_type_value_counts = (
                self.__edge_prop_dataframe[self.type_col_name]
                .value_counts(sort=False, dropna=False)
                .compute()
            )
        return self.__edge_type_value_counts

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
        PropertyGraph.get_num_edges
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
                if self.__series_type is dask_cudf.Series:
                    vert_count = dask_cudf.concat(
                        vert_sers, ignore_index=True
                    ).nunique()
                    self.__num_vertices = vert_count.compute()
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
        """
        vert_sers = self.__get_all_vertices_series()
        if vert_sers:
            if self.__series_type is dask_cudf.Series:
                return dask_cudf.concat(vert_sers, ignore_index=True).unique()
            else:
                raise TypeError("dataframe must be a CUDF Dask dataframe.")
        return self.__series_type()

    def vertices_ids(self):
        """
        Alias for get_vertices()
        """
        return self.get_vertices()

    def vertex_types_from_numerals(
        self, nums: Union[cudf.Series, pd.Series]
    ) -> Union[cudf.Series, pd.Series]:
        """
        Returns the string vertex type names given the numeric category labels.
        Note: Does not accept or return dask_cudf Series.

        Parameters
        ----------
        nums: Union[cudf.Series, pandas.Series] (Required)
            The list of numeric category labels to convert.

        Returns
        -------
        Union[cudf.Series, pd.Series]
            The string type names converted from the input numerals.
        """
        return (
            self.__vertex_prop_dataframe[self.type_col_name]
            .dtype.categories.to_series()
            .iloc[nums]
            .reset_index(drop=True)
        )

    def edge_types_from_numerals(
        self, nums: Union[cudf.Series, pd.Series]
    ) -> Union[cudf.Series, pd.Series]:
        """
        Returns the string edge type names given the numeric category labels.
        Note: Does not accept or return dask_cudf Series.

        Parameters
        ----------
        nums: Union[cudf.Series, pandas.Series] (Required)
            The list of numeric category labels to convert.

        Returns
        -------
        Union[cudf.Series, pd.Series]
            The string type names converted from the input numerals.
        """
        return (
            self.__edge_prop_dataframe[self.type_col_name]
            .dtype.categories.to_series()
            .iloc[nums]
            .reset_index(drop=True)
        )

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
        vector_properties : dict of string to list of strings, optional
            A dict of vector properties to create from columns in the dataframe.
            Each vector property stores an array for each vertex.
            The dict keys are the new vector property names, and the dict values
            should be Python lists of column names from which to create the vector
            property. Columns used to create vector properties won't be added to
            the property graph by default, but may be included as properties by
            including them in the property_columns argument.
            Use ``MGPropertyGraph.vertex_vector_property_to_array`` to convert a
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
        >>>
        """
        if type(dataframe) is not dask_cudf.DataFrame:
            raise TypeError("dataframe must be a Dask dataframe.")
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
            # Initialize the __vertex_prop_dataframe using the same
            # type as the incoming dataframe.
            temp_dataframe = cudf.DataFrame(columns=[self.vertex_col_name, TCN])
            self.__vertex_prop_dataframe = dask_cudf.from_cudf(
                temp_dataframe, npartitions=self.__num_workers
            )
            # Initialize the new columns to the same dtype as the appropriate
            # column in the incoming dataframe, since the initial merge may not
            # result in the same dtype. (see
            # https://github.com/rapidsai/cudf/issues/9981)
            self.__update_dataframe_dtypes(
                self.__vertex_prop_dataframe,
                {self.vertex_col_name: dataframe[vertex_col_name].dtype},
            )
            self.__vertex_prop_dataframe = self.__vertex_prop_dataframe.set_index(
                self.vertex_col_name
            )

            # Use categorical dtype for the type column
            if self.__series_type is dask_cudf.Series:
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
        tmp_df = dataframe.copy()
        tmp_df[self.vertex_col_name] = tmp_df[vertex_col_name]
        # FIXME: handle case of a type_name column already being in tmp_df

        # FIXME: We should do categorization first
        # Related issue: https://github.com/rapidsai/cugraph/issues/2903
        tmp_df[TCN] = type_name
        tmp_df[TCN] = tmp_df[TCN].astype(cat_dtype)

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
            tmp_df = self._create_vector_properties(tmp_df, vector_properties)

        tmp_df = tmp_df.drop(labels=column_names_to_drop, axis=1)

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
        tmp_df = tmp_df.persist().set_index(self.vertex_col_name).persist()
        self.__update_dataframe_dtypes(tmp_df, self.__vertex_prop_dtypes)

        if is_first_data:
            self.__vertex_prop_dataframe = tmp_df
        else:
            # Join on vertex ids (the index)
            # TODO: can we automagically determine when we to use concat?
            df = self.__vertex_prop_dataframe.join(
                tmp_df,
                how="outer",
                rsuffix="_NEW_",
                # npartitions=self.__num_workers  # TODO: see how this behaves
            ).persist()
            cols = self.__vertex_prop_dataframe.columns.intersection(
                tmp_df.columns
            ).to_list()
            rename_cols = {f"{col}_NEW_": col for col in cols}
            new_cols = list(rename_cols)
            sub_df = df[new_cols].rename(columns=rename_cols)
            # This only adds data--it doesn't replace existing data
            df = df.drop(columns=new_cols).fillna(sub_df).persist()
            if df.npartitions > 4 * self.__num_workers:
                # TODO: better understand behavior of npartitions argument in join
                df = df.repartition(npartitions=2 * self.__num_workers).persist()
            self.__vertex_prop_dataframe = df

        # Update the vertex eval dict with the latest column instances
        latest = {
            n: self.__vertex_prop_dataframe[n]
            for n in self.__vertex_prop_dataframe.columns
        }
        self.__vertex_prop_eval_dict.update(latest)
        self.__vertex_prop_eval_dict[
            self.vertex_col_name
        ] = self.__vertex_prop_dataframe.index

    def get_vertex_data(self, vertex_ids=None, types=None, columns=None):
        """
        Return a dataframe containing vertex properties for only the specified
        vertex_ids, columns, and/or types, or all vertex IDs if not specified.
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
            df_out = df.reset_index()

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
        vector_properties : dict of string to list of strings, optional
            A dict of vector properties to create from columns in the dataframe.
            Each vector property stores an array for each edge.
            The dict keys are the new vector property names, and the dict values
            should be Python lists of column names from which to create the vector
            property. Columns used to create vector properties won't be added to
            the property graph by default, but may be included as properties by
            including them in the property_columns argument.
            Use ``MGPropertyGraph.edge_vector_property_to_array`` to convert an
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
        >>>
        """
        if type(dataframe) is not dask_cudf.DataFrame:
            raise TypeError("dataframe must be a Dask dataframe.")
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
            temp_dataframe = cudf.DataFrame(
                columns=[self.src_col_name, self.dst_col_name, TCN]
            )
            self.__update_dataframe_dtypes(
                temp_dataframe,
                {
                    self.src_col_name: dataframe[vertex_col_names[0]].dtype,
                    self.dst_col_name: dataframe[vertex_col_names[1]].dtype,
                },
            )
            temp_dataframe.index = temp_dataframe.index.rename(self.edge_id_col_name)
            if edge_id_col_name is not None:
                temp_dataframe.index = temp_dataframe.index.astype(
                    dataframe[edge_id_col_name].dtype
                )

            # Use categorical dtype for the type column
            if self.__series_type is dask_cudf.Series:
                cat_class = cudf.CategoricalDtype
            else:
                cat_class = pd.CategoricalDtype
            cat_dtype = cat_class([type_name], ordered=False)
            self.__is_edge_id_autogenerated = edge_id_col_name is None
            self.__edge_prop_dataframe = temp_dataframe
        else:
            cat_dtype = self.__update_categorical_dtype(
                self.__edge_prop_dataframe, TCN, type_name
            )

        # NOTE: This copies the incoming DataFrame in order to add the new
        # columns. The copied DataFrame is then merged (another copy) and then
        # deleted when out-of-scope.
        tmp_df = dataframe.copy()
        tmp_df[self.src_col_name] = tmp_df[vertex_col_names[0]]
        tmp_df[self.dst_col_name] = tmp_df[vertex_col_names[1]]

        # FIXME: We should do categorization first
        # Related issue: https://github.com/rapidsai/cugraph/issues/2903

        tmp_df[TCN] = type_name
        tmp_df[TCN] = tmp_df[TCN].astype(cat_dtype)

        # Add unique edge IDs to the new rows. This is just a count for each
        # row starting from the last edge ID value, with initial edge ID 0.
        if edge_id_col_name is None:
            # FIXME: can we assign index instead of column?
            starting_eid = -1 if self.__last_edge_id is None else self.__last_edge_id
            tmp_df[self.edge_id_col_name] = 1
            tmp_df[self.edge_id_col_name] = (
                tmp_df[self.edge_id_col_name].cumsum() + starting_eid
            )
            tmp_df = tmp_df.persist().set_index(self.edge_id_col_name).persist()
            self.__last_edge_id = starting_eid + len(tmp_df)
        else:
            tmp_df = (
                tmp_df.rename(columns={edge_id_col_name: self.edge_id_col_name})
                .persist()
                .set_index(self.edge_id_col_name)
                .persist()
            )
            tmp_df.index = tmp_df.index.astype(dataframe[edge_id_col_name].dtype)

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
            tmp_df = self._create_vector_properties(tmp_df, vector_properties)

        tmp_df = tmp_df.drop(labels=column_names_to_drop, axis=1)

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
        self.__update_dataframe_dtypes(tmp_df, self.__edge_prop_dtypes)

        if is_first_data:
            self.__edge_prop_dataframe = tmp_df
        else:
            # Join on edge ids (the index)
            # TODO: can we automagically determine when we to use concat?
            df = self.__edge_prop_dataframe.join(
                tmp_df,
                how="outer",
                rsuffix="_NEW_",
                # npartitions=self.__num_workers  # TODO: see how this behaves
            ).persist()
            cols = self.__edge_prop_dataframe.columns.intersection(
                tmp_df.columns
            ).to_list()
            rename_cols = {f"{col}_NEW_": col for col in cols}
            new_cols = list(rename_cols)
            sub_df = df[new_cols].rename(columns=rename_cols)
            # This only adds data--it doesn't replace existing data
            df = df.drop(columns=new_cols).fillna(sub_df).persist()
            if df.npartitions > 4 * self.__num_workers:
                # TODO: better understand behavior of npartitions argument in join
                df = df.repartition(npartitions=2 * self.__num_workers).persist()
            self.__edge_prop_dataframe = df

        # Update the edge eval dict with the latest column instances
        latest = dict(
            [
                (n, self.__edge_prop_dataframe[n])
                for n in self.__edge_prop_dataframe.columns
            ]
        )
        self.__edge_prop_eval_dict.update(latest)
        self.__edge_prop_eval_dict[
            self.edge_id_col_name
        ] = self.__edge_prop_dataframe.index

    def get_edge_data(self, edge_ids=None, types=None, columns=None):
        """
        Return a dataframe containing edge properties for only the specified
        edge_ids, columns, and/or edge type, or all edge IDs if not specified.
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
        self.__vertex_prop_dataframe = self.__vertex_prop_dataframe.fillna(
            val
        ).persist()

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

        self.__edge_prop_dataframe = self.__edge_prop_dataframe.fillna(val).persist()

    def select_vertices(self, expr, from_previous_selection=None):
        raise NotImplementedError

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
        return EXPERIMENTAL__MGPropertySelection(edge_selection_series=selected_col)

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
        default_edge_weight : float64, optional
            Value that replaces empty weight property fields
        check_multi_edges : bool (default is True)
            When True and create_using argument is given and not a MultiGraph,
            this will perform a check to verify that the edges in the edge
            dataframe do not form a multigraph with duplicate edges.
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
        if selection is not None and not isinstance(
            selection, EXPERIMENTAL__MGPropertySelection
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
        self.__update_dataframe_dtypes(edges, self.__edge_prop_dtypes)

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
        raise NotImplementedError()

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
        # Don't mutate input data
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
                edge_prop_df[edge_weight_property] = prop_col
            if prop_col.count().compute() != prop_col.size:
                if default_edge_weight is None:
                    raise ValueError(
                        f'edge_weight_property "{edge_weight_property}" '
                        "contains NA values in the subgraph and "
                        "default_edge_weight is not set"
                    )
                else:
                    prop_col.fillna(default_edge_weight, inplace=True)
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
            and self.is_multigraph(edge_prop_df).compute()
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

        # FIXME: This forces the renumbering code to run a python-only
        # renumbering without the newer C++ renumbering step.  This is
        # required since the newest graph algos which are using the
        # pylibcugraph library will crash if passed data renumbered using the
        # C++ renumbering.  The consequence of this is that these extracted
        # subgraphs can only be used with newer pylibcugraph-based MG algos.
        #
        # NOTE: if the vertices are integers (int32 or int64), renumbering is
        # actually skipped with the assumption that the C renumbering will
        # take place. The C renumbering only occurs for pylibcugraph algos,
        # hence the reason these extracted subgraphs only work with PLC algos.
        if renumber_graph is False:
            raise ValueError("currently, renumber_graph must be set to True for MG")
        legacy_renum_only = True

        col_names = [self.src_col_name, self.dst_col_name]
        if edge_attr is not None:
            col_names.append(edge_attr)

        edge_prop_df = edge_prop_df.reset_index().drop(
            [col for col in edge_prop_df if col not in col_names], axis=1
        )
        edge_prop_df = edge_prop_df.repartition(
            npartitions=self.__num_workers * 4
        ).persist()

        G.from_dask_cudf_edgelist(
            edge_prop_df,
            source=self.src_col_name,
            destination=self.dst_col_name,
            edge_attr=edge_attr,
            renumber=renumber_graph,
            legacy_renum_only=legacy_renum_only,
        )

        if add_edge_data:
            # Set the edge_data on the resulting Graph to a DataFrame
            # containing the edges and the edge ID for each. Edge IDs are
            # needed for future calls to annotate_dataframe() in order to
            # associate edges with their properties, since the PG can contain
            # multiple edges between vertrices with different properties.
            # FIXME: also add vertex_data
            G.edge_data = self.__create_property_lookup_table(edge_prop_df)

        del edge_prop_df

        return G

    def renumber_vertices_by_type(self, prev_id_column=None):
        """Renumber vertex IDs to be contiguous by type.

        Parameters
        ----------
        prev_id_column : str, optional
            Column name to save the vertex ID before renumbering.

        Returns a DataFrame with the start and stop IDs for each vertex type.
        Stop is *inclusive*.
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
        if self.__series_type is dask_cudf.Series:
            cat_class = cudf.CategoricalDtype
        else:
            cat_class = pd.CategoricalDtype

        is_cat = isinstance(self.__vertex_prop_dataframe.dtypes[TCN], cat_class)
        if not is_cat:
            cat_dtype = cat_class([TCN], ordered=False)
            self.__vertex_prop_dataframe[TCN] = self.__vertex_prop_dataframe[
                TCN
            ].astype(cat_dtype)

        df = self.__vertex_prop_dataframe
        index_dtype = df.index.dtype

        # FIXME DASK_CUDF: https://github.com/rapidsai/cudf/issues/11795
        cat_dtype = df.dtypes[TCN]
        df[TCN] = df[TCN].astype(str)

        # Include self.vertex_col_name when sorting by values to ensure we can
        # evenly distribute the data across workers.
        df = df.reset_index().persist()
        df = df.sort_values(by=[TCN, self.vertex_col_name], ignore_index=True).persist()
        if self.__edge_prop_dataframe is not None:
            new_name = f"new_{self.vertex_col_name}"
            df[new_name] = 1
            df[new_name] = df[new_name].cumsum() - 1
            mapper = df[[self.vertex_col_name, new_name]]
            edge_index_dtype = self.__edge_prop_dataframe.index.dtype
            self.__edge_prop_dataframe = (
                self.__edge_prop_dataframe
                # map src_col_name IDs
                .merge(mapper, left_on=self.src_col_name, right_on=self.vertex_col_name)
                .drop(columns=[self.src_col_name, self.vertex_col_name])
                .rename(columns={new_name: self.src_col_name})
                # map dst_col_name IDs
                .merge(mapper, left_on=self.dst_col_name, right_on=self.vertex_col_name)
                .drop(columns=[self.dst_col_name, self.vertex_col_name])
                .rename(columns={new_name: self.dst_col_name})
            )
            self.__edge_prop_dataframe.index = self.__edge_prop_dataframe.index.astype(
                edge_index_dtype
            )
            self.__edge_prop_dataframe.index = self.__edge_prop_dataframe.index.rename(
                self.edge_id_col_name
            )
            if prev_id_column is None:
                df[self.vertex_col_name] = df[new_name]
                del df[new_name]
            else:
                df = df.rename(
                    columns={
                        new_name: self.vertex_col_name,
                        self.vertex_col_name: prev_id_column,
                    }
                )
        else:
            if prev_id_column is not None:
                df[prev_id_column] = df[self.vertex_col_name]
            df[self.vertex_col_name] = 1
            df[self.vertex_col_name] = df[self.vertex_col_name].cumsum() - 1

        # FIXME DASK_CUDF: https://github.com/rapidsai/cudf/issues/11795
        df[TCN] = df[TCN].astype(cat_dtype)

        df[self.vertex_col_name] = df[self.vertex_col_name].astype(index_dtype)
        self.__vertex_prop_dataframe = (
            df.persist().set_index(self.vertex_col_name, sorted=True).persist()
        )

        # FIXME DASK_CUDF: https://github.com/rapidsai/cudf/issues/11795
        df = self._vertex_type_value_counts
        cat_dtype = df.index.dtype
        df.index = df.index.astype(str)

        # self._vertex_type_value_counts
        rv = df.sort_index().cumsum().to_frame("stop")

        # FIXME DASK_CUDF: https://github.com/rapidsai/cudf/issues/11795
        df.index = df.index.astype(cat_dtype)

        rv["start"] = rv["stop"].shift(1, fill_value=0)
        rv["stop"] -= 1  # Make inclusive
        return rv[["start", "stop"]]

    def renumber_edges_by_type(self, prev_id_column=None):
        """Renumber edge IDs to be contiguous by type.

        Parameters
        ----------
        prev_id_column : str, optional
            Column name to save the edge ID before renumbering.

        Returns a DataFrame with the start and stop IDs for each edge type.
        Stop is *inclusive*.
        """
        # TODO: keep track if edges are already numbered correctly.
        if self.__edge_prop_dataframe is None:
            return None
        if prev_id_column is not None and prev_id_column in self.__edge_prop_dataframe:
            raise ValueError(
                f"Can't save previous IDs to existing column {prev_id_column!r}"
            )
        df = self.__edge_prop_dataframe
        index_dtype = df.index.dtype

        # FIXME DASK_CUDF: https://github.com/rapidsai/cudf/issues/11795
        cat_dtype = df.dtypes[self.type_col_name]
        df[self.type_col_name] = df[self.type_col_name].astype(str)

        # Include self.edge_id_col_name when sorting by values to ensure we can
        # evenly distribute the data across workers.
        df = df.reset_index().persist()
        df = df.sort_values(
            by=[self.type_col_name, self.edge_id_col_name], ignore_index=True
        ).persist()
        if prev_id_column is not None:
            df[prev_id_column] = df[self.edge_id_col_name]

        # FIXME DASK_CUDF: https://github.com/rapidsai/cudf/issues/11795
        df[self.type_col_name] = df[self.type_col_name].astype(cat_dtype)

        df[self.edge_id_col_name] = 1
        df[self.edge_id_col_name] = df[self.edge_id_col_name].cumsum() - 1
        df[self.edge_id_col_name] = df[self.edge_id_col_name].astype(index_dtype)
        self.__edge_prop_dataframe = (
            df.persist().set_index(self.edge_id_col_name, sorted=True).persist()
        )

        # FIXME DASK_CUDF: https://github.com/rapidsai/cudf/issues/11795
        df = self._edge_type_value_counts
        assert df.index.dtype == cat_dtype
        df.index = df.index.astype(str)

        # self._edge_type_value_counts
        rv = df.sort_index().cumsum().to_frame("stop")

        # FIXME DASK_CUDF: https://github.com/rapidsai/cudf/issues/11795
        df.index = df.index.astype(cat_dtype)

        rv["start"] = rv["stop"].shift(1, fill_value=0)
        rv["stop"] -= 1  # Make inclusive
        return rv[["start", "stop"]]

    def vertex_vector_property_to_array(
        self, df, col_name, fillvalue=None, *, missing="ignore"
    ):
        """Convert a known vertex vector property in a DataFrame to an array.

        Parameters
        ----------
        df : dask_cudf.DataFrame
        col_name : str
            The column name in the DataFrame to convert to an array.
            This vector property should have been created by MGPropertyGraph.
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
        dask.array (of cupy.ndarray)
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
        df : dask_cudf.DataFrame
        col_name : str
            The column name in the DataFrame to convert to an array.
            This vector property should have been created by MGPropertyGraph.
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
        dask.array (of cupy.ndarray)
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

    def _create_vector_properties(self, df, vector_properties):
        return df.map_partitions(
            self._create_vector_properties_partition, vector_properties
        )

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
                fillvalue = list(fillvalue)
            except Exception:
                fillvalue = [fillvalue] * length
            else:
                if len(fillvalue) != length:
                    raise ValueError(
                        f"Wrong size of list as fill value; got {len(fillvalue)}, "
                        f"expected {length}"
                    )
        if df.dtypes[col_name] != "list":
            raise TypeError(
                "Wrong dtype for vector property; expected 'list', "
                f"got {df.dtypes[col_name]}"
            )
        s = df[col_name]
        meta = self._vector_series_to_array_partition(
            s._meta, length, fillvalue, "ignore"
        )
        return s.map_partitions(
            self._vector_series_to_array_partition,
            length,
            fillvalue,
            missing,
            meta=meta,
        )

    def is_multi_gpu(self):
        """
        Return True if this is a multi-gpu graph.  Always returns True for
        MGPropertyGraph.
        """
        return True

    @classmethod
    def is_multigraph(cls, df):
        """
        Return True if df has >1 of the same src, dst pair
        """
        return cls._has_duplicates(df, [cls.src_col_name, cls.dst_col_name])

    @classmethod
    def has_duplicate_edges(cls, df, columns=None):
        """
        Return True if df has rows with the same src, dst, type, and columns
        """
        cols = [cls.src_col_name, cls.dst_col_name, cls.type_col_name]
        if columns:
            cols.extend(columns)
        return cls._has_duplicates(df, cols)

    @classmethod
    def _has_duplicates(cls, df, cols):
        # empty not supported by dask
        if len(df.columns) == 0:
            return False

        unique_pair_len = df.drop_duplicates(
            split_out=df.npartitions, ignore_index=True
        ).shape[0]
        # if unique_pairs == len(df)
        # then no duplicate edges
        return unique_pair_len != df.shape[0]

    def __create_property_lookup_table(self, edge_prop_df):
        """
        Returns a DataFrame containing the src vertex, dst vertex, and edge_id
        values from edge_prop_df.
        """
        return edge_prop_df[[self.src_col_name, self.dst_col_name]].reset_index()

    def __get_all_vertices_series(self):
        """
        Return a list of all Series objects that contain vertices from all
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
        # `dask_cudf.concat` doesn't work when the index dtypes are different
        # See: https://github.com/rapidsai/cudf/issues/11741
        if len(vert_sers) > 1 and not all(
            cudf.api.types.is_dtype_equal(vert_sers[0].index.dtype, s.index.dtype)
            for s in vert_sers
        ):
            vert_sers = [s.reset_index(drop=True) for s in vert_sers]
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
                df[col] = df[col].astype(dtype_str)

    def __update_categorical_dtype(self, df, column, val):
        """Add a new category to a categorical dtype column of a dataframe.

        Returns the new categorical dtype.
        """
        # Add `val` to the categorical dtype if necessary
        if val in df.dtypes[column].categories:
            # No need to change the categorical dtype
            pass
        else:
            # dask_cudf doesn't support inplace here like cudf does
            df[column] = df[column].cat.add_categories([val])
        return df.dtypes[column]

    @staticmethod
    def _create_vector_properties_partition(df, vector_properties):
        # Make each vector contigous and 1-d
        new_cols = {}
        for key, columns in vector_properties.items():
            values = df[columns].values
            new_cols[key] = create_list_series_from_2d_ar(values, index=df.index)
        return df.assign(**new_cols)

    @staticmethod
    def _vector_series_to_array_partition(s, length, fillvalue, missing):
        # This returns a writable view (i.e., no copies!)
        if len(s) == 0:
            # TODO: fix bug in cudf; operating on dask_cudf dataframes nests list dtype
            dtype = s.dtype
            while dtype == "list":
                dtype = dtype.element_type
            return cupy.empty((0, length), dtype=dtype)
        if fillvalue is not None:
            s = s.copy()  # copy b/c we mutate below
            s[s.isnull()] = fillvalue
        rv = s._data.columns[0].children[-1].values.reshape(-1, length)
        if fillvalue is None and missing == "error" and rv.shape[0] != len(s):
            raise RuntimeError(
                f"Vector property {s.name!r} has empty rows! "
                'Provide a fill value or use `missing="ignore"` to ignore empty rows.'
            )
        return rv
