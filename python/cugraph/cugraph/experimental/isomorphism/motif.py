# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import cudf

from cugraph.utilities.utils import import_optional

# networkx is an optional (test) dependency of cugraph; only the code paths
# that use it require it to be installed.
nx = import_optional("networkx")


def count_unique(edge_list):
    """Count the number of unique vertices in an edge list."""
    return len(set().union(*edge_list))


@dataclass
class MotifData:
    """A small building-block graph ("motif") used to decompose the pattern
    graph during motif-based subgraph isomorphism.

    Parameters
    ----------
    name : str
        Identifier for the motif, used in the solver's decomposition report.
    motif : list of (int, int)
        Edge list over vertices ``0..k-1`` defining the motif graph.
    """

    name: str
    motif: list
    graph: object = field(init=False, default=None, repr=False)
    size: int = field(init=False, default=0)
    isomorphisms: object = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.size = count_unique(self.motif) if self.motif else 1
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.size))
        self.graph.add_edges_from(self.motif)

    def copy(self):
        """Copy this MotifData; the isomorphisms DataFrame is deep-copied."""
        new_instance = self.__class__(motif=self.motif, name=self.name)
        new_instance.graph = self.graph
        new_instance.size = self.size
        if self.isomorphisms is not None:
            new_instance.isomorphisms = self.isomorphisms.copy()
        return new_instance


def data_to_dataframe(data, num_vertices):
    # Down-cast by vertex-id range; every motif table uses the same rule so
    # cudf merge keys stay dtype-consistent across tables.
    if num_vertices <= 256:
        dtype = "uint8"
    elif num_vertices <= 65536:
        dtype = "uint16"
    else:
        dtype = "uint32"
    return cudf.DataFrame(data, dtype=dtype)


def make_m2_motif(edge_df, num_vertices):
    """Build the base single-edge ("M2") motif whose isomorphisms table is
    the bidirectional, de-duplicated edge list of the target graph.

    The concat + drop_duplicates normalizes the input regardless of whether
    ``edge_df`` is already symmetrized or holds each edge in one direction.

    Parameters
    ----------
    edge_df : cudf.DataFrame
        Two columns (source, destination) of target edges in the compact
        ``0..num_vertices-1`` vertex space, with self-loops already removed.
    num_vertices : int
        Number of vertices in the target graph.
    """
    m2_motif = MotifData(name="M2", motif=[(0, 1)])
    df = data_to_dataframe(edge_df.to_cupy(), num_vertices)
    df_rev = df[[1, 0]]
    df_rev.columns = [0, 1]
    m2_motif.isomorphisms = cudf.concat(
        [df, df_rev], ignore_index=True
    ).drop_duplicates(ignore_index=True, keep="first")
    return m2_motif


def default_motif_library():
    """Return a small library of 3-vertex motifs usable as building blocks.

    Passing these to ``subgraph_isomorphism`` makes the solver precompute
    their embeddings in the target graph (a full solve per motif), which can
    speed up large patterns at the cost of upfront work and memory.
    """
    return [
        MotifData(name="M3-path", motif=[(0, 1), (1, 2)]),
        MotifData(name="M3-triangle", motif=[(0, 1), (1, 2), (0, 2)]),
    ]
