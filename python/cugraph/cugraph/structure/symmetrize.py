# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from cugraph.structure import graph_classes as csg
import cudf
import dask_cudf
from cugraph.comms import comms as Comms


def symmetrize_df(df, src_name, dst_name, multi=False, symmetrize=True):
    """
    Take a COO stored in a DataFrame, along with the column names of
    the source and destination columns and create a new data frame
    using the same column names that symmetrize the graph so that all
    edges appear in both directions.
    Note that if other columns exist in the data frame (e.g. edge weights)
    the other columns will also be replicated.  That is, if (u,v,data)
    represents the source value (u), destination value (v) and some
    set of other columns (data) in the input data, then the output
    data will contain both (u,v,data) and (v,u,data) with matching
    data.
    If (u,v,data1) and (v,u,data2) exist in the input data where data1
    != data2 then this code will arbitrarily pick the smaller data
    element to keep, if this is not desired then the caller should
    should correct the data prior to calling symmetrize.

    Parameters
    ----------
    df : cudf.DataFrame
        Input data frame containing COO.  Columns should contain source
        ids, destination ids and any properties associated with the
        edges.

    src_name : string
        Name of the column in the data frame containing the source ids

    dst_name : string
        Name of the column in the data frame containing the destination ids

    multi : bool, optional (default=False)
        Set to True if graph is a Multi(Di)Graph. This allows multiple
        edges instead of dropping them.

    symmetrize : bool, optional (default=True)
        Default is True to perform symmetrization. If False only duplicate
        edges are dropped.

    Examples
    --------
    >>> from cugraph.structure.symmetrize import symmetrize_df
    >>> # Download dataset from https://github.com/rapidsai/cugraph/datasets/..
    >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sym_df = symmetrize_df(M, '0', '1')

    """
    #
    #  Now append the columns.  We add sources to the end of destinations,
    #  and destinations to the end of sources.  Otherwise we append a
    #  column onto itself.
    #
    if symmetrize:
        gdf = cudf.DataFrame()
        for idx, name in enumerate(df.columns):
            if name == src_name:
                gdf[src_name] = df[src_name].append(
                    df[dst_name], ignore_index=True
                )
            elif name == dst_name:
                gdf[dst_name] = df[dst_name].append(
                    df[src_name], ignore_index=True
                )
            else:
                gdf[name] = df[name].append(df[name], ignore_index=True)
    else:
        gdf = df
    if multi:
        return gdf
    else:
        return gdf.groupby(by=[src_name, dst_name], as_index=False).min()


def symmetrize_ddf(df, src_name, dst_name, weight_name=None):
    """
    Take a COO stored in a distributed DataFrame, and the column names of
    the source and destination columns and create a new data frame
    using the same column names that symmetrize the graph so that all
    edges appear in both directions.

    Note that if other columns exist in the data frame (e.g. edge weights)
    the other columns will also be replicated.  That is, if (u,v,data)
    represents the source value (u), destination value (v) and some
    set of other columns (data) in the input data, then the output
    data will contain both (u,v,data) and (v,u,data) with matching
    data.

    If (u,v,data1) and (v,u,data2) exist in the input data where data1
    != data2 then this code will arbitrarily pick the smaller data
    element to keep, if this is not desired then the caller should
    should correct the data prior to calling symmetrize.

    Parameters
    ----------
    df : dask_cudf.DataFrame
        Input data frame containing COO.  Columns should contain source
        ids, destination ids and any properties associated with the
        edges.

    src_name : string
        Name of the column in the data frame containing the source ids

    dst_name : string
        Name of the column in the data frame containing the destination ids

    weight_name : string, optional (default=None)
        Name of the column in the data frame containing the weights

    Examples
    --------
    >>> # import cugraph.dask as dcg
    >>> # from cugraph.structure.symmetrize import symmetrize_ddf
    >>> # Init a DASK Cluster
    >>> # Download dataset from https://github.com/rapidsai/cugraph/datasets/..
    >>> # chunksize = dcg.get_chunksize(datasets / 'karate.csv')
    >>> # ddf = dask_cudf.read_csv(datasets/'karate.csv', chunksize=chunksize,
    >>> #                          delimiter=' ',
    >>> #                          names=['src', 'dst', 'weight'],
    >>> #                          dtype=['int32', 'int32', 'float32'])
    >>> # sym_ddf = symmetrize_ddf(ddf, "src", "dst", "weight")

    """
    # FIXME: Uncomment out the above (broken) example

    if weight_name:
        ddf2 = df[[dst_name, src_name, weight_name]]
        ddf2.columns = [src_name, dst_name, weight_name]
    else:
        ddf2 = df[[dst_name, src_name]]
        ddf2.columns = [src_name, dst_name]
    worker_list = Comms.get_workers()
    num_workers = len(worker_list)
    ddf = df.append(ddf2).reset_index(drop=True)
    result = ddf.shuffle(on=[
        src_name, dst_name], ignore_index=True, npartitions=num_workers)
    result = result.map_partitions(lambda x: x.groupby(
        by=[src_name, dst_name], as_index=False).min().reset_index(drop=True))

    return result


def symmetrize(source_col, dest_col, value_col=None, multi=False,
               symmetrize=True):
    """
    Take a COO set of source destination pairs along with associated values
    stored in a single GPU or distributed
    create a new COO set of source destination pairs along with values where
    all edges exist in both directions.

    Return from this call will be a COO stored as two cudf Series or
    dask_cudf.Series -the symmetrized source column and the symmetrized dest
    column, along with
    an optional cudf Series containing the associated values (only if the
    values are passed in).

    Parameters
    ----------
    source_col : cudf.Series or dask_cudf.Series
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains the source index for each edge.
        Source indices must be an integer type.

    dest_col : cudf.Series or dask_cudf.Series
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains the destination index for each edge.
        Destination indices must be an integer type.

    value_col : cudf.Series or dask_cudf.Series, optional (default=None)
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains values associated with this edge.
        For this function the values can be any type, they are not
        examined, just copied.

    multi : bool, optional (default=False)
        Set to True if graph is a Multi(Di)Graph. This allows multiple
        edges instead of dropping them.

    symmetrize : bool, optional
        Default is True to perform symmetrization. If False only duplicate
        edges are dropped.


    Examples
    --------
    >>> from cugraph.structure.symmetrize import symmetrize
    >>> # Download dataset from https://github.com/rapidsai/cugraph/datasets/..
    >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> values = cudf.Series(M['2'])
    >>> src, dst, val = symmetrize(sources, destinations, values)

    """

    input_df = None
    weight_name = None
    if type(source_col) is dask_cudf.Series:
        # FIXME convoluted way of just wrapping dask cudf Series in a ddf
        input_df = source_col.to_frame()
        input_df = input_df.rename(columns={source_col.name: "source"})
        input_df["destination"] = dest_col
    else:
        input_df = cudf.DataFrame(
            {"source": source_col, "destination": dest_col}
        )
        csg.null_check(source_col)
        csg.null_check(dest_col)
    if value_col is not None:
        if isinstance(value_col, cudf.Series):
            weight_name = "value"
            input_df.insert(len(input_df.columns), "value", value_col)
        elif isinstance(value_col, cudf.DataFrame):
            input_df = cudf.concat([input_df, value_col], axis=1)

    output_df = None
    if type(source_col) is dask_cudf.Series:
        output_df = symmetrize_ddf(
            input_df, "source", "destination", weight_name
        ).persist()
    else:
        output_df = symmetrize_df(input_df, "source", "destination", multi,
                                  symmetrize)
    if value_col is not None:
        if isinstance(value_col, cudf.Series):
            return (
                output_df["source"],
                output_df["destination"],
                output_df["value"],
            )
        elif isinstance(value_col, cudf.DataFrame):
            return (
                output_df["source"],
                output_df["destination"],
                output_df[value_col.columns],
            )
    return output_df["source"], output_df["destination"]
