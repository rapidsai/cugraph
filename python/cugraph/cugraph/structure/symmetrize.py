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
from cugraph.dask.comms import comms as Comms


def symmetrize_df(df, src_name, dst_name, weight_name=None, multi=False, symmetrize=True):
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
    
    weight_name : string, optional (default=None)
        Name of the column in the data frame containing the weight ids

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


def symmetrize_ddf(df, src_name, dst_name, weight_name=None, multi=False, symmetrize=True):
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
        Name of the column in the data frame containing the weight ids
    
    multi : bool, optional (default=False)
        Set to True if graph is a Multi(Di)Graph. This allows multiple
        edges instead of dropping them.

    symmetrize : bool, optional (default=True)
        Default is True to perform symmetrization. If False only duplicate
        edges are dropped.

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

    worker_list = Comms.get_workers()
    num_workers = len(worker_list)
    if symmetrize:
        if weight_name:
            ddf2 = df[[dst_name, src_name, weight_name]]
            ddf2.columns = [src_name, dst_name, weight_name]
        else:
            ddf2 = df[[dst_name, src_name]]
            ddf2.columns = [src_name, dst_name]
        result = df.append(ddf2).reset_index(drop=True)
    else:
        result = df
    if multi:
        # FIXME: Repartition the dask_cudf to n num_workers
        return result
    else:
        result = result.drop_duplicates(
            subset=[src_name, dst_name], split_out=num_workers).reset_index(drop=True)

        return result



# FIXME: This function requires the dataframe to be renumbered in order to 
# to support multi columns
def symmetrize(input_df, source_col_name, dest_col_name, value_col_name=None,
               multi=False, symmetrize=True):
    """
    Take a dataframe of source destination pairs along with associated
    values stored in a single GPU or distributed
    create a COO set of source destination pairs along with values where
    all edges exist in both directions.

    Return from this call will be a COO stored as two/three cudf or
    dask_cudf Series -the symmetrized source column and the symmetrized dest
    column, along with an optional cudf Series containing the associated
    values (only if the values are passed in).

    Parameters
    ----------
    input_df : cudf.DataFrame or dask_cudf.DataFrame
        The edgelist as a cudf.DataFrame or dask_cudf.DataFrame

    source_col_name : str
        source column name.

    dest_col_name : str
        destination column name.

    value_col_name : str or None
        weights column name.

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
    >>> df = cudf.DataFrame()
    >>> df['sources'] = cudf.Series(M['0'])
    >>> df['destinations'] = cudf.Series(M['1'])
    >>> df['values'] = cudf.Series(M['2'])
    >>> src, dst, val = symmetrize(df, 'sources', 'destinations', 'values')

    """

    if isinstance(source_col_name, list) or isinstance(dest_col_name, list):
        raise ValueError("multi column ids is not yet supported")

    input_df = input_df.rename(columns={source_col_name:"source", dest_col_name:"destination"})
    
    if value_col_name is not None:
        input_df = input_df.rename(columns={value_col_name:"value"})
        value_col_name = 'value'
    
    # FIXME: Also check for NULL values in a dask dataframe
    if isinstance(input_df, cudf.DataFrame):
        csg.null_check(input_df["source"])
        csg.null_check(input_df["destination"])

    if isinstance(input_df, dask_cudf.DataFrame):
        output_df = symmetrize_ddf(
            input_df, "source", "destination", value_col_name, multi, symmetrize,
            )
    else:
        output_df = symmetrize_df(
            input_df, "source", "destination", value_col_name, multi, symmetrize,
            )
    if value_col_name is not None:
        value_col = output_df[value_col_name]
        if isinstance(value_col, (cudf.Series, dask_cudf.Series)):
            return (
                output_df["source"],
                output_df["destination"],
                output_df["value"],
            )
        # FIXME: which case to get a multi column value?
        elif isinstance(value_col, cudf.DataFrame):
            return (
                output_df["source"],
                output_df["destination"],
                output_df[value_col.columns],
            )
    return output_df["source"], output_df["destination"]
