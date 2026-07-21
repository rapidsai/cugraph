# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import time
from typing import Optional

import cudf
import cupy as cp
import numpy as np

from cugraph.experimental.isomorphism.motif import (
    MotifData,
    data_to_dataframe,
    make_m2_motif,
)
from cugraph.experimental.isomorphism.slicing import (
    slice_pattern_graph_using_motifs,
)
from cugraph.utilities.utils import import_optional

nx = import_optional("networkx")


@dataclass
class SolverResult:
    # One row per embedding; columns follow pattern_vertices order.
    mappings: np.ndarray
    # Sorted pattern vertex ids corresponding to the columns of mappings.
    pattern_vertices: np.ndarray
    timings: Optional[dict] = None


class _MotifSubgraphIsomorphismSolver:
    """Solve the subgraph isomorphism (monomorphism) problem using
    motif-based decomposition: slice the pattern graph into small motifs,
    look up each motif's embeddings in the target graph, and assemble full
    embeddings via relational joins with an overlap-consistency filter.

    The target graph is given as a cudf edge list in the compact
    ``0..num_target_vertices-1`` vertex space with self-loops removed.
    """

    def __init__(
        self,
        target_edge_df,
        num_target_vertices,
        motif_metadata=None,
        batch_size=None,
    ):
        self._target_edge_df = target_edge_df
        self._num_target_vertices = num_target_vertices
        # Merge in row batches so no single join input exceeds cuDF's
        # 2**31 - 1 column size limit; <= 0 disables batching.
        self._batch_size = int(batch_size) if batch_size else 0
        self.motif_metadata = motif_metadata if motif_metadata is not None else []
        self.decomposition = None

    def generate_graph_motif_data(self, motifs=None):
        """Precompute embeddings for the given motifs in the target graph.

        The base single-edge M2 motif is always generated first; each further
        motif is solved recursively using the motifs generated before it.
        Sets and returns ``self.motif_metadata``.
        """
        if motifs is None:
            motifs = []

        m2_motif = make_m2_motif(self._target_edge_df, self._num_target_vertices)
        motif_data_list = [m2_motif]
        for motif_data in motifs:
            solver = _MotifSubgraphIsomorphismSolver(
                self._target_edge_df,
                self._num_target_vertices,
                motif_metadata=motif_data_list,
                batch_size=self._batch_size,
            )
            result = solver.solve(motif_data.graph)
            if result is None:
                continue

            motif_data.isomorphisms = data_to_dataframe(
                result.mappings, self._num_target_vertices
            )
            motif_data_list.append(motif_data)

        self.motif_metadata = motif_data_list
        return motif_data_list

    def solve(self, pattern_graph: nx.Graph) -> SolverResult | None:
        """Find all embeddings of pattern_graph in the target graph."""
        self._validate_input(pattern_graph)
        self._times = defaultdict(float)

        # Step 1: Decompose the pattern graph into motif slices
        start = time()
        slicing_res = slice_pattern_graph_using_motifs(
            pattern_graph, self.motif_metadata
        )
        self.decomposition = [motif.name for motif in slicing_res.chosen_motifs]
        self._times["step 1 pattern decomposition"] = time() - start

        if not slicing_res.chosen_motifs:
            return None

        # Step 2: Seed with the first motif's embeddings
        initial_motif = slicing_res.chosen_motifs.pop(0)
        slicing_res.boundaries.pop(0)
        slicing_res.intersections.pop(0)

        if len(initial_motif.isomorphisms) == 0:
            return None

        current_mappings_df = initial_motif.isomorphisms
        # Rename columns: m4_v5 means 4th motif, 5th motif vertex
        current_mappings_df.columns = [
            f"m0_v{col}" for col in current_mappings_df.columns
        ]

        count = 0
        while slicing_res.chosen_motifs:
            next_motif = slicing_res.chosen_motifs.pop(0)
            next_boundary = slicing_res.boundaries.pop(0)
            intersection = slicing_res.intersections.pop(0)
            count += 1

            processed_batches = []
            for batch_df in self._create_batches(
                current_mappings_df, self._batch_size
            ):
                # Step 2-1: Match the two tables on the boundary conditions
                start = time()
                batch_df = self._merging_with_new_motif(
                    batch_df, count, next_motif, next_boundary
                )
                self._times["step 2-1 join"] += time() - start

                # Step 2-2: Keep only rows with the correct number of
                # overlapped vertices
                start = time()
                batch_df = self._filtering_invalid_mappings(
                    batch_df, slicing_res.slices, intersection, count
                )
                self._times["step 2-2 filter"] += time() - start
                processed_batches.append(batch_df)

            start = time()
            if len(processed_batches) == 1:
                current_mappings_df = processed_batches[0]
            else:
                current_mappings_df = cudf.concat(
                    processed_batches, ignore_index=True
                )
            del processed_batches
            self._times["step 2-3 concat batches"] += time() - start

        # Step 3: Format the output
        start = time()
        mappings, pattern_vertices = self._format_output(
            slicing_res.slices, current_mappings_df
        )
        self._times["step 3 format output"] = time() - start

        del current_mappings_df

        return SolverResult(mappings, pattern_vertices, dict(self._times))

    def _merging_with_new_motif(
        self,
        current_mappings_df,
        count,
        next_motif: MotifData,
        next_boundary,
    ):
        next_motifs_df = next_motif.isomorphisms
        original_column_names = next_motifs_df.columns
        next_motifs_df.columns = [
            f"m{count}_v{col}" for col in next_motifs_df.columns
        ]

        current_boundary_columns, next_boundary_columns = [], []
        for (group, node_idx), next_node_idx in next_boundary:
            current_boundary_columns.append(f"m{group}_v{node_idx}")
            next_boundary_columns.append(f"m{count}_v{next_node_idx}")

        # The input ordering is not preserved with cuDF
        current_mappings_df = current_mappings_df.merge(
            next_motifs_df,
            how="inner",
            left_on=current_boundary_columns,
            right_on=next_boundary_columns,
        )
        # Restore names so this motif can be merged again (next batch or
        # reuse)
        next_motifs_df.columns = original_column_names
        return current_mappings_df

    @staticmethod
    def _create_batches(df, batch_size):
        if batch_size <= 0 or len(df) == 0:
            yield df
            return
        for i in range(0, len(df), batch_size):
            yield df.iloc[i : i + batch_size]

    def _filtering_invalid_mappings(
        self,
        current_mappings_df,
        slices,
        intersection,
        count,
    ):
        num_overlapped_nodes = sum(
            node in intersection
            for each_slice in slices[:count]
            for node in each_slice
        )

        prev_cols = [
            col
            for col in current_mappings_df.columns
            if not col.startswith(f"m{count}_")
        ]
        new_cols = [
            col for col in current_mappings_df.columns if col.startswith(f"m{count}_")
        ]

        prev_values = cp.from_dlpack(current_mappings_df[prev_cols].to_dlpack())
        new_values = cp.from_dlpack(current_mappings_df[new_cols].to_dlpack())
        del prev_cols, new_cols

        # Use broadcasting to compare every row's previous values with its
        # new values, summing in chunks so no chunk exceeds 2**32 elements
        output_shape = cp.broadcast(
            prev_values[..., None], new_values[:, None, :]
        ).shape
        match_counts = cp.empty(output_shape[0], dtype=cp.int64)
        max_size = 2**32
        chunk_size = max(1, int(max_size / np.prod(output_shape[1:])))
        for chunk_start in range(0, output_shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, output_shape[0])
            match_counts[chunk_start:chunk_end] = (
                prev_values[chunk_start:chunk_end, ..., None]
                == new_values[chunk_start:chunk_end, None, :]
            ).sum(axis=(1, 2))
        del prev_values, new_values

        # Keep only rows where the exact overlap count occurs
        current_mappings_df = current_mappings_df[
            match_counts == num_overlapped_nodes
        ]
        del match_counts
        return current_mappings_df

    def _format_output(self, slices, mappings_df):
        """Reorder columns from motif-slice order to sorted pattern-vertex
        order and return (mappings ndarray, sorted pattern vertices)."""
        flat_vertices = np.array([node for slice_ in slices for node in slice_])
        pattern_vertices, idx = np.unique(flat_vertices, return_index=True)
        mappings = cp.asnumpy(cp.from_dlpack(mappings_df.to_dlpack())[:, idx])
        return mappings, pattern_vertices

    def _validate_input(self, pattern_graph: nx.Graph) -> None:
        """Validate the input pattern graph against the target graph."""
        if pattern_graph.number_of_nodes() == 0:
            raise ValueError("Validation failed: Pattern graph is empty.")

        if not nx.is_connected(pattern_graph):
            raise ValueError("Validation failed: Pattern graph must be connected.")

        if not self.motif_metadata:
            raise ValueError(
                "Motif metadata has not been generated; call "
                "generate_graph_motif_data() before solve()."
            )

        if pattern_graph.number_of_nodes() > self._num_target_vertices:
            raise ValueError(
                "Validation failed: Pattern graph exceeds target graph capacity."
            )

        # motif_metadata[0] is always M2, whose isomorphisms table is the
        # bidirectional target edge list.
        num_target_edges = len(self.motif_metadata[0].isomorphisms)
        if 2 * pattern_graph.number_of_edges() > num_target_edges:
            raise ValueError(
                "Validation failed: Pattern graph requires more connectivity "
                "than available on the target graph."
            )
