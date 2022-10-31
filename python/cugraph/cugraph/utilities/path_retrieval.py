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

import numpy as np
import cudf

from cugraph.structure.symmetrize import symmetrize
from cugraph.structure.number_map import NumberMap
from cugraph.utilities import path_retrieval_wrapper


def get_traversed_cost(df, source, source_col, dest_col, value_col):
    """
    Take the DataFrame result from a BFS or SSSP function call and sums
    the given weights along the path to the starting vertex.
    The source_col, dest_col identifiers need to match with the vertex and
    predecessor columns of df.

    Input Parameters
    ----------
    df : cudf.DataFrame
        The dataframe containing the results of a BFS or SSSP call
    source: int
        Index of the source vertex.
    source_col : cudf.DataFrame
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains the source index for each edge.
        Source indices must be an integer type.
    dest_col : cudf.Series
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains the destination index for each edge.
        Destination indices must be an integer type.
    value_col : cudf.Series
        This cudf.Series wraps a gdf_column of size E (E: number of edges).
        The gdf column contains values associated with this edge.
        Weight should be a floating type.

    Returns
    ---------
    df : cudf.DataFrame
        DataFrame containing two columns 'vertex' and 'info'.
        Unreachable vertices will have value the max value of the weight type.
    """

    if "vertex" not in df.columns:
        raise ValueError(
            "DataFrame does not appear to be a BFS or "
            "SSP result - 'vertex' column missing"
        )
    if "distance" not in df.columns:
        raise ValueError(
            "DataFrame does not appear to be a BFS or "
            "SSP result - 'distance' column missing"
        )
    if "predecessor" not in df.columns:
        raise ValueError(
            "DataFrame does not appear to be a BFS or "
            "SSP result - 'predecessor' column missing"
        )

    src, dst, val = symmetrize(source_col, dest_col, value_col)

    symmetrized_df = cudf.DataFrame()
    symmetrized_df["source"] = src
    symmetrized_df["destination"] = dst
    symmetrized_df["weights"] = val

    input_df = df.merge(
        symmetrized_df,
        left_on=["vertex", "predecessor"],
        right_on=["source", "destination"],
        how="left",
    )

    # Set unreachable vertex weights to max float and source vertex weight to 0
    max_val = np.finfo(val.dtype).max
    input_df[["weights"]] = input_df[["weights"]].fillna(max_val)
    input_df.loc[input_df["vertex"] == source, "weights"] = 0

    # Renumber
    renumbered_gdf, renumber_map = NumberMap.renumber(
        input_df, ["vertex"], ["predecessor"], preserve_order=True
    )
    renumbered_gdf = renumbered_gdf.rename(
        columns={"src": "vertex", "dst": "predecessor"}
    )
    stop_vertex = renumber_map.to_internal_vertex_id(cudf.Series(-1)).values[0]

    out_df = path_retrieval_wrapper.get_traversed_cost(renumbered_gdf, stop_vertex)

    # Unrenumber
    out_df["vertex"] = renumber_map.unrenumber(
        renumbered_gdf, "vertex", preserve_order=True
    )["vertex"]
    return out_df
