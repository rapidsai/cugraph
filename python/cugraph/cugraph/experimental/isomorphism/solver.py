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
    # One row per embedding; columns follow pattern_vertices order. A device
    # (cupy) array when the result fit in a single partition (the common
    # case, avoiding a device->host round trip), otherwise a host (numpy,
    # 64-bit indexed) array assembled from multiple partitions.
    mappings: object
    # Sorted pattern vertex ids corresponding to the columns of mappings.
    pattern_vertices: np.ndarray
    timings: Optional[dict] = None


class _PartitionWriter:
    """Accumulate filtered chunks into partitions each below a row limit.

    cuDF columns cannot exceed ~2**31 rows. This writer buffers small
    filtered chunks and emits partitions just under ``row_limit`` so that
    arbitrarily large solution sets stay representable; they are combined
    only as NumPy arrays (64-bit indexed) at the very end.
    """

    def __init__(self, row_limit):
        self._row_limit = row_limit
        self._done = []
        self._buffer = []
        self._buffered_rows = 0

    def add(self, df):
        n = len(df)
        if n == 0:
            return
        if self._buffer and self._buffered_rows + n > self._row_limit:
            self._flush()
        self._buffer.append(df)
        self._buffered_rows += n
        if self._buffered_rows >= self._row_limit:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        if len(self._buffer) == 1:
            part = self._buffer[0]
        else:
            part = cudf.concat(self._buffer, ignore_index=True)
        self._done.append(part)
        self._buffer = []
        self._buffered_rows = 0

    def finish(self):
        self._flush()
        return self._done


class _MotifSubgraphIsomorphismSolver:
    """Solve the subgraph isomorphism (monomorphism) problem using
    motif-based decomposition: slice the pattern graph into small motifs,
    look up each motif's embeddings in the target graph, and assemble full
    embeddings via relational joins with an overlap-consistency filter.

    The join-and-filter step is streamed so the intermediate produced by
    each motif merge stays within a memory budget rather than being
    materialized in full; the running table is held as a list of partitions
    (each below cuDF's ~2**31 row limit) and combined only as NumPy at the
    end. Join-key columns contributed by each new motif duplicate the
    columns they were joined on and are dropped after every merge, which
    nearly halves the table width for path-like decompositions.

    The target graph is given as a cudf edge list in the compact
    ``0..num_target_vertices-1`` vertex space with self-loops removed.
    """

    #: Fraction of currently-free device memory targeted for a single
    #: streamed join's intermediate result when no explicit batch_size is
    #: given. The filter step allocates transients of a few times the merged
    #: table's size, so this stays well below 1.
    _JOIN_MEM_BUDGET_FRACTION = 1 / 8
    #: Floor (GiB) for the join memory budget, used when free-memory
    #: information is unavailable or very low.
    _JOIN_MEM_BUDGET_MIN_GIB = 0.5
    #: Number of left rows sampled to estimate join fan-out before batching.
    _FANOUT_SAMPLE_ROWS = 8192
    #: Max rows per cuDF partition (kept below the ~2**31 column row limit).
    _ROW_LIMIT = 1_800_000_000

    def __init__(
        self,
        target_edge_df,
        num_target_vertices,
        motif_metadata=None,
        batch_size=None,
    ):
        self._target_edge_df = target_edge_df
        self._num_target_vertices = num_target_vertices
        # Explicit rows-per-batch override; <= 0 selects adaptive batching
        # that targets _JOIN_MEM_BUDGET_GIB per join intermediate.
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

        initial_df = initial_motif.isomorphisms
        # Positional renaming: m4_v5 means 4th motif, 5th motif vertex.
        initial_df.columns = [f"m0_v{i}" for i in range(len(initial_df.columns))]
        partitions = [initial_df]

        count = 0
        while slicing_res.chosen_motifs:
            next_motif = slicing_res.chosen_motifs.pop(0)
            next_boundary = slicing_res.boundaries.pop(0)
            intersection = slicing_res.intersections.pop(0)
            count += 1
            partitions = self._merge_and_filter_streamed(
                partitions, count, next_motif, next_boundary, intersection
            )

        # Step 3: Format the output
        start = time()
        mappings, pattern_vertices = self._format_output_parts(
            slicing_res.slices, partitions
        )
        self._times["step 3 format output"] = time() - start

        del partitions

        return SolverResult(mappings, pattern_vertices, dict(self._times))

    def _merge_and_filter_streamed(
        self,
        partitions,
        count,
        next_motif: MotifData,
        next_boundary,
        intersection,
    ):
        """Merge each partition with the next motif and filter, batch by
        batch.

        The join-key columns contributed by the new motif duplicate the
        columns they were joined on, so they are dropped after filtering.
        Output is re-partitioned to keep every cuDF table below the ~2**31
        row limit.
        """
        next_df = next_motif.isomorphisms
        next_df.columns = [f"m{count}_v{i}" for i in range(len(next_df.columns))]

        left_keys, right_keys = [], []
        for (group, node_idx), next_node_idx in next_boundary:
            left_keys.append(f"m{group}_v{node_idx}")
            right_keys.append(f"m{count}_v{next_node_idx}")

        # With duplicate columns dropped, each retained vertex is unique, so
        # a valid row shares exactly ``len(intersection)`` values between the
        # previous block and the new motif's block.
        num_overlapped_nodes = len(intersection)

        writer = _PartitionWriter(self._ROW_LIMIT)
        budget_bytes = self._join_mem_budget_bytes()
        total_batches = 0
        for part in partitions:
            n_rows = len(part)
            if n_rows == 0:
                continue
            if self._batch_size > 0:
                batch_rows = self._batch_size
            else:
                batch_rows = self._choose_batch_rows(
                    part, next_df, left_keys, right_keys, budget_bytes
                )
            for chunk_start in range(0, n_rows, batch_rows):
                left_batch = part.iloc[chunk_start : chunk_start + batch_rows]
                writer.add(
                    self._merge_filter_one(
                        left_batch,
                        next_df,
                        left_keys,
                        right_keys,
                        count,
                        num_overlapped_nodes,
                    )
                )
                del left_batch
                total_batches += 1
        self._times["step 2 batches"] += total_batches

        result = writer.finish()
        return result if result else [partitions[0].iloc[:0]]

    def _merge_filter_one(
        self,
        left_df,
        next_df,
        left_keys,
        right_keys,
        count,
        num_overlapped_nodes,
    ):
        """Merge one left batch with the motif table, filter, and drop the
        duplicate join-key columns."""
        start = time()
        merged = left_df.merge(
            next_df, how="inner", left_on=left_keys, right_on=right_keys
        )
        self._times["step 2-1 join"] += time() - start
        if len(merged) == 0:
            return merged.drop(columns=right_keys)
        start = time()
        filtered = self._filter_merged(merged, count, num_overlapped_nodes)
        filtered = filtered.drop(columns=right_keys)
        self._times["step 2-2 filter"] += time() - start
        return filtered

    def _join_mem_budget_bytes(self):
        """Memory budget for one streamed join's intermediate result.

        Scales with the free device memory reported by the driver (a
        read-only query; allocator state is never modified) so larger GPUs
        run fewer, bigger batches. Falls back to the floor if the query
        fails or memory is tight. Note: under a pooling allocator the
        driver's free-memory figure can undercount what is actually
        available; the floor keeps progress possible in that case.
        """
        floor = int(self._JOIN_MEM_BUDGET_MIN_GIB * (1024**3))
        try:
            free_bytes, _total = cp.cuda.runtime.memGetInfo()
        except Exception:
            return floor
        return max(floor, int(free_bytes * self._JOIN_MEM_BUDGET_FRACTION))

    def _choose_batch_rows(self, left_df, next_df, left_keys, right_keys, budget_bytes):
        """Pick a left-batch size so the merged intermediate stays within
        the memory budget.

        The join fan-out is estimated from a small sample of the left table;
        the batch size is then the memory budget divided by
        ``fan-out * row_width``.
        """
        n_rows = len(left_df)
        if n_rows <= self._FANOUT_SAMPLE_ROWS:
            return n_rows

        dtype_bytes = int(np.dtype(left_df[left_df.columns[0]].dtype).itemsize)
        row_width = len(left_df.columns) + len(next_df.columns)
        bytes_per_row = max(row_width * dtype_bytes, 1)

        sample = left_df.iloc[: self._FANOUT_SAMPLE_ROWS]
        sample_merged = sample.merge(
            next_df, how="inner", left_on=left_keys, right_on=right_keys
        )
        fanout = len(sample_merged) / self._FANOUT_SAMPLE_ROWS
        del sample, sample_merged
        if fanout <= 0:
            return n_rows

        budget_rows = budget_bytes / bytes_per_row
        batch_rows = int(budget_rows / fanout)
        # Independent of the byte budget, never let a single merge output
        # exceed the cuDF row limit: with small vertex dtypes on large GPUs
        # the byte budget alone can project past 2**31 rows. fanout is a
        # sampled estimate; _ROW_LIMIT's ~15% margin under 2**31 absorbs
        # estimation error.
        batch_rows = min(batch_rows, int(self._ROW_LIMIT / fanout))
        return max(1, min(batch_rows, n_rows))

    def _filter_merged(self, merged_df, count, num_overlapped_nodes):
        prev_cols = [
            col for col in merged_df.columns if not col.startswith(f"m{count}_")
        ]
        new_cols = [
            col for col in merged_df.columns if col.startswith(f"m{count}_")
        ]
        match_counts = self._count_cross_matches(merged_df, prev_cols, new_cols)
        return merged_df[match_counts == num_overlapped_nodes]

    @staticmethod
    def _count_cross_matches(df, prev_cols, new_cols):
        """Count, per row, shared vertex assignments between two column
        blocks.

        Accumulating one new column at a time uses ``rows x |prev|`` memory
        instead of the ``rows x |prev| x |new|`` boolean tensor, with the
        same result.
        """
        prev_values = cp.from_dlpack(df[prev_cols].to_dlpack())
        new_values = cp.from_dlpack(df[new_cols].to_dlpack())
        match_counts = cp.zeros(prev_values.shape[0], dtype=cp.int64)
        for j in range(new_values.shape[1]):
            match_counts += (prev_values == new_values[:, j : j + 1]).sum(axis=1)
        return match_counts

    def _format_output_parts(self, slices, partitions):
        """Format result partitions into one ``(n_solutions, n_vertices)``
        array plus the sorted pattern-vertex order of its columns.

        Each retained column ``m{g}_v{i}`` holds the target vertex matched
        to pattern vertex ``slices[g][i]``; duplicate columns were dropped
        during the joins, so every pattern vertex maps to exactly one
        retained column. Partitions are formatted independently and the
        (NumPy, 64-bit indexed) arrays concatenated, so the combined
        solution set can exceed cuDF's ~2**31 row limit.
        """
        available = set(partitions[0].columns)
        col_for_vertex = {}
        for group, slice_ in enumerate(slices):
            for node_idx, vertex in enumerate(slice_):
                col = f"m{group}_v{node_idx}"
                if col in available:
                    col_for_vertex.setdefault(vertex, col)

        pattern_vertices = np.array(sorted(col_for_vertex))
        ordered_cols = [col_for_vertex[v] for v in pattern_vertices]

        parts = [part for part in partitions if len(part)]
        if not parts:
            mappings = np.empty((0, len(ordered_cols)), dtype=np.int64)
        elif len(parts) == 1:
            # Common case: the whole result fits in one partition; keep it
            # on device (to_dlpack yields a self-owned contiguous copy).
            mappings = cp.from_dlpack(parts[0][ordered_cols].to_dlpack())
        else:
            # Multi-partition results are near or beyond cuDF's 2**31 row
            # limit; assemble on host with 64-bit indexing. An on-device
            # concat is deliberately avoided: it would transiently need
            # ~2x the (tens-of-GiB) result resident at once.
            mappings = np.concatenate(
                [
                    cp.asnumpy(cp.from_dlpack(part[ordered_cols].to_dlpack()))
                    for part in parts
                ],
                axis=0,
            )
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
