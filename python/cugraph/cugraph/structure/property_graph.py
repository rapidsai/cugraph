# Copyright (c) 2021, NVIDIA CORPORATION.
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
    def __init__(self):
        # The dataframe containing the properties for each vertex.
        # Each vertex occupies a row, and individual properties are maintained
        # in individual columns. The table contains a column for each property
        # of each vertex. If a vertex does not contain a property, it will have
        # a NaN value in that property column. Each vertex will also have a
        # "type_name" that can be assigned by the caller to describe the type of
        # the vertex for a given application domain. If no type_name is
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
        # expressions for vertices and edges. These dictionaries contain entries
        # for each column name in their respective dataframes which are mapped
        # to instances of PropertyColumn objects.
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
            A DataFrame instance with a compatible Pandas-like DataFrame interface.
            FIXME: finish this description
        vertex_id_column : string
            FIXME: finish this description

        Returns
        -------
        None

        Examples
        --------
        >>> gdf = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
        >>> pr = cugraph.pagerank(G, alpha = 0.85, max_iter = 500, tol = 1.0e-05)
        """
        # FIXME: check args

        # Initialize the __vertex_prop_dataframe if necessary using the same
        # type as the incoming dataframe.
        if self.__vertex_prop_dataframe is None:
            self.__vertex_prop_dataframe = type(dataframe)(columns=["vertex", "type_name"])

        tmp_df = dataframe.rename(columns={vertex_id_column : "vertex"})
        # FIXME: handle case of a type_name column already being in tmp_df
        tmp_df["type_name"] = type_name
        if property_columns:
            # use a set to remove the columns not specified in property_columns, then call drop on that result.
            raise NotImplementedError
        self.__vertex_prop_dataframe = self.__vertex_prop_dataframe.merge(tmp_df, how="outer")

        # Update the vertex eval dict
        self.__vertex_prop_eval_dict.update(dict([(n, PropertyColumn(self.__vertex_prop_dataframe[n])) for n in self.__vertex_prop_dataframe.columns]))


    def add_edge_data(self,
                      dataframe,
                      edge_vertices_columns,
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
            A DataFrame instance with a compatible Pandas-like DataFrame interface.
            FIXME: finish this description
        vertex_id_column : string
            FIXME: finish this description

        Returns
        -------
        None

        Examples
        --------
        >>> gdf = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
        >>> pr = cugraph.pagerank(G, alpha = 0.85, max_iter = 500, tol = 1.0e-05)
        """
        # FIXME: check args

        # Initialize the __vertex_prop_dataframe if necessary using the same
        # type as the incoming dataframe.
        if self.__edge_prop_dataframe is None:
            self.__edge_prop_dataframe = type(dataframe)(columns=["src", "dst", "type_name"])

        tmp_df = dataframe.rename(columns={edge_vertices_columns[0] : "src", edge_vertices_columns[1] : "dst"})
        # FIXME: handle case of a type_name column already being in tmp_df
        tmp_df["type_name"] = type_name
        if property_columns:
            # use a set to remove the columns not specified in property_columns, then call drop on that result.
            raise NotImplementedError
        self.__edge_prop_dataframe = self.__edge_prop_dataframe.merge(tmp_df, how="outer")

        # Update the vertex eval dict
        self.__edge_prop_eval_dict.update(dict([(n, PropertyColumn(self.__edge_prop_dataframe[n])) for n in self.__edge_prop_dataframe.columns]))


    def extract_subgraph(self,
                         create_using,
                         edge_property_condition=None,
                         vertex_property_condition=None,
                         ):
        """
        Return a subgraph of the overall PropertyGraph containing vertices and edges
        that match the criteria specified.
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
        >>> gdf = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
        >>> pr = cugraph.pagerank(G, alpha = 0.85, max_iter = 500, tol = 1.0e-05)
        """
        globals = {}
        if vertex_property_condition:
            locals = self.__vertex_prop_eval_dict
            filter_column = eval(vertex_property_condition, globals, locals)
            matching_indices = filter_column.series.index[filter_column.series]
            filtered_vertex_dataframe = self.__vertex_prop_dataframe.loc[matching_indices]
        else:
            filtered_vertex_dataframe = self.__vertex_prop_dataframe

        if edge_property_condition:
            locals = self.__edge_prop_eval_dict
            filter_column = eval(edge_property_condition, globals, locals)
            matching_indices = filter_column.series.index[filter_column.series]
            filtered_edge_dataframe = self.__edge_prop_dataframe.loc[matching_indices]
        else:
            filtered_edge_dataframe = self.__edge_prop_dataframe

        # FIXME: check that self.__edge_prop_dataframe is set!

        # filter the edges that only contain the filtered vertices in src and dst
        filtered_vertices = filtered_vertex_dataframe["vertex"]
        src_filter = filtered_edge_dataframe["src"].isin(filtered_vertices)
        dst_filter = filtered_edge_dataframe["dst"].isin(filtered_vertices)
        filter = src_filter & dst_filter
        result = filtered_edge_dataframe.loc[filtered_edge_dataframe.index[filter]]

        if type(create_using) is type(type):
            G = create_using()
        else:
            # FIXME: extract more attrs from the create_using instance
            attrs = {"directed" : create_using.is_directed()}
            G = type(create_using)(**attrs)

        # FIXME: check for dataframe type
        # FIXME: assign weights!
        if type(result) is type(cudf.DataFrame):
            G.from_cudf_edgelist(result, source="src", destination="dst")
        else:
            G.from_pandas_edgelist(result, source="src", destination="dst")

        return G
