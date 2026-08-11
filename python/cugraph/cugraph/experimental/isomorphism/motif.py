# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import cudf

from cugraph.utilities.utils import import_optional

# networkx is an optional dependency of cugraph (declared for tests only);
# import_optional defers the failure to first use, and only the private
# pattern-decomposition code paths use it — never MotifData instances.
nx = import_optional("networkx")


@dataclass
class EXPERIMENTAL__MotifData:
    """A small building-block graph ("motif") used to decompose the pattern
    graph during motif-based subgraph monomorphism. Holds only plain Python
    data (an edge list) plus, after precomputation, a cuDF table of the
    motif's embeddings in the target graph.

    Parameters
    ----------
    name : str
        Identifier for the motif, used in the solver's decomposition report.
    motif : list of (int, int)
        Edge list over vertices ``0..k-1`` defining the motif graph.
    """

    name: str
    motif: list
    size: int = field(init=False, default=0)
    embeddings: object = field(init=False, default=None, repr=False)

    def __post_init__(self):
        # Materialize first so a generator input can't be silently
        # exhausted between validation and use.
        self.motif = list(self.motif)
        vertices = set().union(*self.motif) if self.motif else set()
        if not self.motif or vertices != set(range(len(vertices))):
            raise ValueError(
                "motif must be a non-empty edge list over contiguous vertices 0..k-1"
            )
        self.size = len(vertices)

    def _to_nx(self):
        """Build a networkx.Graph of this motif (private; used by the CPU
        pattern-decomposition step only)."""
        graph = nx.Graph()
        graph.add_nodes_from(range(self.size))
        graph.add_edges_from(self.motif)
        return graph

    def copy(self):
        """Copy this MotifData. The embeddings table is copied shallowly
        (shared data buffers, independent column metadata): the solver only
        ever renames the copy's columns, so slices of the same motif can
        share one embeddings table instead of duplicating it on the GPU."""
        # Explicit class (not self.__class__) so copies of instances created
        # through the experimental warning wrapper don't re-warn.
        new_instance = EXPERIMENTAL__MotifData(motif=self.motif, name=self.name)
        new_instance.size = self.size
        if self.embeddings is not None:
            new_instance.embeddings = self.embeddings.copy(deep=False)
        return new_instance


# Internal alias: intra-package code (and type hints) use the plain name;
# the public export in cugraph.experimental applies the warning wrapper to
# the EXPERIMENTAL__-prefixed name above. Do not import this alias from
# outside the package — it bypasses the experimental warning. Note that
# instances made internally (default_motif_library, copy()) are of the raw
# class, so isinstance/== checks against the wrapped public class will not
# match them; experimental users should not rely on either.
MotifData = EXPERIMENTAL__MotifData


def _data_to_dataframe(data, num_vertices):
    # Down-cast by vertex-id range; every motif table uses the same rule so
    # cudf merge keys stay dtype-consistent across tables.
    if num_vertices <= 256:
        dtype = "uint8"
    elif num_vertices <= 65536:
        dtype = "uint16"
    elif num_vertices <= 2**32:
        dtype = "uint32"
    else:
        dtype = "uint64"
    return cudf.DataFrame(data, dtype=dtype)


def _make_m2_motif(edge_df, num_vertices):
    """Build the base single-edge ("M2") motif whose embeddings table is
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
    if len(edge_df) >= 2**30:
        # The concat below materializes 2 * len(edge_df) rows BEFORE the
        # dedup, so at 2**30 input rows it exceeds cuDF's 2**31 - 1 row
        # limit regardless of how many duplicates the dedup would remove;
        # fail clearly rather than inside the concat. Support for larger
        # targets would need a partitioned M2 table.
        raise ValueError(
            f"Target graph has {len(edge_df)} edges; the bidirectional "
            "M2 motif table would exceed cuDF's 2**31 - 1 row limit."
        )
    m2_motif = MotifData(name="M2", motif=[(0, 1)])
    df = _data_to_dataframe(edge_df.to_cupy(), num_vertices)
    df_rev = df[[1, 0]]
    df_rev.columns = [0, 1]
    m2_motif.embeddings = cudf.concat([df, df_rev], ignore_index=True).drop_duplicates(
        ignore_index=True, keep="first"
    )
    return m2_motif


def EXPERIMENTAL__default_motif_library():
    """Return a small library of 3-vertex motifs usable as building blocks.

    Passing these to ``subgraph_monomorphism`` makes the solver precompute
    their embeddings in the target graph (a full solve per motif), which can
    speed up large patterns at the cost of upfront work and memory.
    """
    return [
        MotifData(name="M3-path", motif=[(0, 1), (1, 2)]),
        MotifData(name="M3-triangle", motif=[(0, 1), (1, 2), (0, 2)]),
    ]


default_motif_library = EXPERIMENTAL__default_motif_library
