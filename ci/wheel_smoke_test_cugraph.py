# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf
import cugraph

if __name__ == "__main__":
    edgelist = cudf.DataFrame({"source": ["a", "b", "c"], "destination": ["b", "c", "d"]})

    # directed graph
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(edgelist, store_transposed=True)
    result_df = cugraph.pagerank(G)

    assert(result_df["pagerank"].sum() == 1.0)
    assert(result_df.sort_values(by="pagerank")["vertex"].values_host.tolist()
           == ["a", "b", "c", "d"])

    # undirected graph
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(edgelist, store_transposed=True)
    result_df = cugraph.pagerank(G)

    assert(result_df["pagerank"].sum() == 1.0)
    result_df.set_index("vertex", inplace=True)
    assert(result_df.loc["a", "pagerank"] == result_df.loc["d", "pagerank"])
    assert(result_df.loc["b", "pagerank"] == result_df.loc["c", "pagerank"])
