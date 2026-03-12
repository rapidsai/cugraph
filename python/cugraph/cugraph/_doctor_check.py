# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke check for `rapids doctor` (RAPIDS CLI).

See: https://github.com/rapidsai/rapids-cli#check-plugins
"""


def cugraph_smoke_check(**kwargs):
    """
    A quick check to ensure cugraph can be imported and a minimal graph
    operation runs on the GPU.
    """
    try:
        import cudf
        import cugraph
    except ImportError as e:
        raise ImportError(
            "cuGraph or its dependencies (e.g. cudf) could not be imported. "
            "Tip: install with `pip install cugraph` or use a RAPIDS conda environment."
        ) from e

    # Build a tiny graph and run a trivial operation
    df = cudf.DataFrame({"src": [0, 1], "dst": [1, 2]})
    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source="src", destination="dst")
    n_vertices = G.number_of_vertices()
    if n_vertices != 3:
        raise AssertionError(
            f"cuGraph smoke check failed: expected 3 vertices, got {n_vertices}"
        )
