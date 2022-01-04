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


class PropertyColumn:
    """
    FIXME: fill this in
    """
    def __init__(self, series):
        self.series = series

    def __eq__(self, val):
        if isinstance(val, PropertyColumn):
            val = val.series
        return PropertyColumn(self.series == val)

    def __ne__(self, val):
        if isinstance(val, PropertyColumn):
            val = val.series
        return PropertyColumn(self.series != val)

    def __gt__(self, val):
        if isinstance(val, PropertyColumn):
            val = val.series
        return PropertyColumn(self.series > val)

    def __lt__(self, val):
        if isinstance(val, PropertyColumn):
            val = val.series
        return PropertyColumn(self.series < val)

    def __ge__(self, val):
        if isinstance(val, PropertyColumn):
            val = val.series
        return PropertyColumn(self.series >= val)

    def __le__(self, val):
        if isinstance(val, PropertyColumn):
            val = val.series
        return PropertyColumn(self.series <= val)

    def __and__(self, val):
        if not isinstance(val, PropertyColumn):
            raise TypeError("bitwise operators are not supported")
        return PropertyColumn(self.series & val.series)

    def __or__(self, val):
        if not isinstance(val, PropertyColumn):
            raise TypeError("bitwise operators are not supported")
        return PropertyColumn(self.series | val.series)

    def __bool__(self):
        raise TypeError("use bitwise operators here")


class PropertyGraph:
    """
    FIXME: fill this in
    """
    # column name constants used in internal DataFrames
    __vertex_col_name = "__vertex__"
    __src_col_name = "__src__"
    __dst_col_name = "__dst__"
    __type_col_name = "__type__"

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

    # PropertyGraph read-only attributes
    @property
    def num_vertices(self):
        verts = cudf.Series()
        if self.__vertex_prop_dataframe:
            verts = verts.append(
                self.__vertex_prop_dataframe[self.__vertex_col_name])
        if self.__edge_prop_dataframe:
            verts = verts.append(
                self.__edge_prop_dataframe[self.__src_col_name].unique())
            verts = verts.append(
                self.__edge_prop_dataframe[self.__dst_col_name].unique())
            verts = verts.unique()
        return len(verts)

    @property
    def num_edges(self):
        return len(self.__edge_prop_dataframe or [])

    @property
    def vertex_property_names(self):
        if self.__vertex_prop_dataframe:
            props = list(self.__vertex_prop_dataframe.columns)
            props.remove(self.__vertex_col_name)
            props.remove(self.__type_col_name)  # should "type" be removed?
            return props
        return []

    @property
    def edge_property_names(self):
        if self.__edge_prop_dataframe:
            props = list(self.__edge_prop_dataframe.columns)
            props.remove(self.__src_col_name)
            props.remove(self.__dst_col_name)
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

        # Initialize the __vertex_prop_dataframe if necessary using the same
        # type as the incoming dataframe.
        default_vertex_columns = [self.__vertex_col_name, self.__type_col_name]
        if self.__vertex_prop_dataframe is None:
            self.__vertex_prop_dataframe = \
                type(dataframe)(columns=default_vertex_columns)

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

        self.__vertex_prop_dataframe = \
            self.__vertex_prop_dataframe.merge(tmp_df, how="outer")

        # Update the vertex eval dict with PropertyColumn objs
        latest = dict([(n, PropertyColumn(self.__vertex_prop_dataframe[n]))
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

        # Initialize the __vertex_prop_dataframe if necessary using the same
        # type as the incoming dataframe.
        default_edge_columns = [self.__src_col_name,
                                self.__dst_col_name,
                                self.__type_col_name]
        if self.__edge_prop_dataframe is None:
            self.__edge_prop_dataframe = \
                type(dataframe)(columns=default_edge_columns)

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

        self.__edge_prop_dataframe = \
            self.__edge_prop_dataframe.merge(tmp_df, how="outer")

        # Update the prop eval dict with PropertyColumn objs
        latest = dict([(n, PropertyColumn(self.__edge_prop_dataframe[n]))
                       for n in self.__edge_prop_dataframe.columns])
        self.__edge_prop_eval_dict.update(latest)

    def extract_subgraph(self,
                         create_using=None,
                         edge_property_condition=None,
                         vertex_property_condition=None,
                         edge_weight_property=None,
                         default_edge_weight=None
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

        globals = {}
        if vertex_property_condition:
            locals = self.__vertex_prop_eval_dict
            filter_column = eval(vertex_property_condition, globals, locals)
            matching_indices = filter_column.series.index[filter_column.series]
            filtered_vertex_dataframe = \
                self.__vertex_prop_dataframe.loc[matching_indices]
        else:
            filtered_vertex_dataframe = self.__vertex_prop_dataframe

        if edge_property_condition:
            locals = self.__edge_prop_eval_dict
            filter_column = eval(edge_property_condition, globals, locals)
            matching_indices = filter_column.series.index[filter_column.series]
            filtered_edge_dataframe = \
                self.__edge_prop_dataframe.loc[matching_indices]
        else:
            filtered_edge_dataframe = self.__edge_prop_dataframe

        # FIXME: check that self.__edge_prop_dataframe is set!

        # filter the edges that only contain the filtered verts in src and dst
        filtered_verts = filtered_vertex_dataframe[self.__vertex_col_name]
        src_filter = \
            filtered_edge_dataframe[self.__src_col_name].isin(filtered_verts)
        dst_filter = \
            filtered_edge_dataframe[self.__dst_col_name].isin(filtered_verts)
        filter = src_filter & dst_filter
        edges = \
            filtered_edge_dataframe.loc[filtered_edge_dataframe.index[filter]]

        if edge_weight_property and \
           (edge_weight_property not in edges.columns):
            raise ValueError(f'edge_weight_property "{edge_weight_property}" '
                             "was not found in the properties of the subgraph")

        # Ensure a valid edge_weight_property can be used for applying weights
        # to the subgraph, and if a default_edge_weight was specified, apply it
        # to all NAs in the weight column.
        if edge_weight_property:
            prop_col = edges[edge_weight_property]
            if prop_col.count() != prop_col.size:
                if default_edge_weight is None:
                    raise ValueError("edge_weight_property "
                                     f'"{edge_weight_property}" '
                                     "contains NA values in the subgraph and "
                                     "default_edge_weight is not set")
                else:
                    prop_col.fillna(default_edge_weight, inplace=True)

        # FIXME: skip if empty edges
        create_args = {"source": self.__src_col_name,
                       "destination": self.__dst_col_name,
                       "edge_attr": edge_weight_property,
                       "renumber": True,
                       }
        if type(edges) is cudf.DataFrame:
            G.from_cudf_edgelist(edges, **create_args)
        else:
            G.from_pandas_edgelist(edges, **create_args)

        return G
