# SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cugraph.structure.graph_classes import (
    Graph,
    MultiGraph,
    BiPartiteGraph,
)
from cugraph.structure.graph_classes import (
    is_weighted,
    is_directed,
    is_multigraph,
    is_bipartite,
    is_multipartite,
)
from cugraph.structure.number_map import NumberMap
from cugraph.structure.symmetrize import symmetrize, symmetrize_df, symmetrize_ddf
from cugraph.structure.replicate_edgelist import (
    replicate_edgelist,
    replicate_cudf_dataframe,
    replicate_cudf_series,
)
from cugraph.structure.convert_matrix import (
    from_edgelist,
    from_cudf_edgelist,
    from_pandas_edgelist,
    to_pandas_edgelist,
    from_pandas_adjacency,
    to_pandas_adjacency,
    from_numpy_array,
    to_numpy_array,
    from_numpy_matrix,
    to_numpy_matrix,
    from_adjlist,
)
from cugraph.structure.hypergraph import hypergraph
