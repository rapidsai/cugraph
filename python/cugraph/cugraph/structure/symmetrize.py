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


def symmetrize_df(
    df, src_name, dst_name, weight_name=None, multi=False, symmetrize=True
):
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
    element to keep, if this is not desired then the caller
    should correct the data prior to calling symmetrize.

    Parameters
    ----------
    df : cudf.DataFrame
        Input data frame containing COO.  Columns should contain source
        ids, destination ids and any properties associated with the
        edges.

    src_name : str or list
        Name(s) of the column(s) in the data frame containing the source ids

    dst_name : str or list
        Name(s) of the column(s) in the data frame containing
        the destination ids

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
    if not isinstance(src_name, list):
        src_name = [src_name]
    if not isinstance(dst_name, list):
        dst_name = [dst_name]
    if weight_name is not None and not isinstance(weight_name, list):
        weight_name = [weight_name]

    if symmetrize:
        if weight_name:
            df2 = df[[*dst_name, *src_name, *weight_name]]
            df2.columns = [*src_name, *dst_name, *weight_name]
        else:
            df2 = df[[*dst_name, *src_name]]
            df2.columns = [*src_name, *dst_name]
        result = cudf.concat([df, df2]).reset_index(drop=True)
    else:
        result = df
    if multi:
        return result
    else:
        vertex_col_name = src_name + dst_name
        result = result.groupby(by=[*vertex_col_name], as_index=False).min()
        return result


def symmetrize_ddf(
    ddf, src_name, dst_name, weight_name=None, multi=False, symmetrize=True
):
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
    element to keep, if this is not desired then the caller
    should correct the data prior to calling symmetrize.

    Parameters
    ----------
    ddf : dask_cudf.DataFrame
        Input data frame containing COO.  Columns should contain source
        ids, destination ids and any properties associated with the
        edges.

    src_name : str or list
        Name(s) of the column(s) in the data frame containing the source ids

    dst_name : str or list
        Name(s) of the column(s) in the data frame containing
        the destination ids

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

    if not isinstance(src_name, list):
        src_name = [src_name]
    if not isinstance(dst_name, list):
        dst_name = [dst_name]
    if weight_name is not None and not isinstance(weight_name, list):
        weight_name = [weight_name]

    if symmetrize:
        if weight_name:
            ddf2 = ddf[[*dst_name, *src_name, *weight_name]]
            ddf2.columns = [*src_name, *dst_name, *weight_name]
        else:
            ddf2 = ddf[[*dst_name, *src_name]]
            ddf2.columns = [*src_name, *dst_name]
        result = dask_cudf.concat([ddf, ddf2]).reset_index(drop=True)
    else:
        result = ddf
    if multi:
        # The concat call doubles the number of partitions therefore,
        # repartition the result so that the number of partitions equals
        # the number of workers
        result = result.repartition(npartitions=ddf.npartitions)
        return result
    else:
        vertex_col_name = src_name + dst_name
        result = (
            result.groupby(by=[*vertex_col_name])
            .min(split_out=ddf.npartitions)
            .reset_index()
        )

        return result


def symmetrize(
    input_df,
    source_col_name,
    dest_col_name,
    value_col_name=None,
    multi=False,
    symmetrize=True,
):
    """
    Take a dataframe of source destination pairs along with associated
    values stored in a single GPU or distributed
    create a COO set of source destination pairs along with values where
    all edges exist in both directions.

    Return from this call will be a COO stored as two/three cudf/dask_cudf
    Series/Dataframe -the symmetrized source column and the symmetrized dest
    column, along with an optional cudf/dask_cudf Series/DataFrame containing
    the associated values (only if the values are passed in).

    Parameters
    ----------
    input_df : cudf.DataFrame or dask_cudf.DataFrame
        The edgelist as a cudf.DataFrame or dask_cudf.DataFrame

    source_col_name : str or list
        source column name.

    dest_col_name : str or list
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

    csg.null_check(input_df[source_col_name])
    csg.null_check(input_df[dest_col_name])

    if isinstance(input_df, dask_cudf.DataFrame):
        output_df = symmetrize_ddf(
            input_df,
            source_col_name,
            dest_col_name,
            value_col_name,
            multi,
            symmetrize,
        )
    else:
        output_df = symmetrize_df(
            input_df,
            source_col_name,
            dest_col_name,
            value_col_name,
            multi,
            symmetrize,
        )
    if value_col_name is not None:
        value_col = output_df[value_col_name]
        if isinstance(value_col, (cudf.Series, dask_cudf.Series)):
            return (
                output_df[source_col_name],
                output_df[dest_col_name],
                output_df[value_col_name],
            )
        elif isinstance(value_col, (cudf.DataFrame, dask_cudf.DataFrame)):
            return (
                output_df[source_col_name],
                output_df[dest_col_name],
                output_df[value_col.columns],
            )

    return output_df[source_col_name], output_df[dest_col_name]
