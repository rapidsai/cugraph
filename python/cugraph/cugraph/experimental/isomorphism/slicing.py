# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from cugraph.experimental.isomorphism.motif import MotifData
from cugraph.utilities.utils import import_optional

nx = import_optional("networkx")


@dataclass
class SlicingResults:
    """
    Stores the results from pattern graph slicing into motifs.

    Attributes:
        chosen_motifs: A list of MotifData objects selected during slicing.
        slices: Each sublist contains pattern vertex ids for a slice.
        boundaries: A list where each element represents the boundary
                    conditions for a slice as tuples of
                    ([slice_index, vertex_index], vertex_index).
        intersections: Each set contains pattern vertices at the intersection
                       of a slice and its boundary.
    """

    chosen_motifs: List[MotifData] = field(default_factory=list)
    slices: List[List[int]] = field(default_factory=list)
    boundaries: List[List[Tuple[List[int], int]]] = field(default_factory=list)
    intersections: List[set] = field(default_factory=list)

    def extend(self, other: "SlicingResults") -> None:
        """Appends the content of another SlicingResults instance to the
        current one."""
        self.chosen_motifs.extend(other.chosen_motifs)
        self.slices.extend(other.slices)
        self.boundaries.extend(other.boundaries)
        self.intersections.extend(other.intersections)


def slice_pattern_graph_using_motifs(
    pattern_graph: nx.Graph,
    motif_metadata: List[MotifData],
) -> SlicingResults:
    """Slice the pattern graph into subgraphs based on predefined motifs."""
    graph = nx.Graph()
    graph.add_nodes_from(pattern_graph.nodes)
    graph.add_edges_from(pattern_graph.edges)
    results = SlicingResults()
    boundary_nodes: set = set()

    # Precompute adjacency list for fast lookups
    adjacency_map = build_adjacency_map(graph)

    def _extract_next_slice(
        graph: nx.Graph,
        motif_metadata: List[MotifData],
        adjacency_map: dict,
        boundary_nodes: set,
        existing_slices: List[List[int]],
    ) -> SlicingResults | None:
        """Attempt to extract the next valid motif slice."""
        remaining_nodes = set(graph.nodes)
        existing_slices_node = set(
            node for slice_ in existing_slices for node in slice_
        )

        # Start from the largest and most complex motifs
        for motif_data in motif_metadata[::-1]:
            if len(remaining_nodes) < motif_data.size:
                continue

            # Node-induced subgraph matches of the motif in the residual
            # pattern graph (same semantics as rustworkx vf2_mapping with
            # subgraph=True).
            matcher = nx.algorithms.isomorphism.GraphMatcher(graph, motif_data.graph)

            for mapping in matcher.subgraph_isomorphisms_iter():
                # mapping is {pattern_node: motif_node}; invert it so that
                # slice_nodes[i] is the pattern vertex matched to motif
                # vertex i. Dict iteration order carries no such guarantee.
                inv = {motif_node: pat_node for pat_node, motif_node in mapping.items()}
                slice_nodes = [inv[i] for i in range(motif_data.size)]
                slice_set = set(slice_nodes)

                # The found slice should cover at least one existing
                # boundary node
                if boundary_nodes and not (slice_set & boundary_nodes):
                    continue

                combined_nodes = slice_set | existing_slices_node
                slice_edges = [
                    (slice_nodes[u], slice_nodes[v]) for (u, v) in motif_data.motif
                ]
                updated_boundary = find_boundary_nodes(
                    combined_nodes, slice_edges, adjacency_map
                )

                nodes_to_remove = slice_set - updated_boundary
                # The found slice should contribute to the removal of pattern
                # graph nodes. If no nodes are removed, it indicates a
                # potential infinite loop.
                if not nodes_to_remove:
                    # If the slice fully covers existing slices, skip it
                    if any(not (slice_set - set(slice_)) for slice_ in existing_slices):
                        continue
                graph.remove_nodes_from(list(nodes_to_remove))

                intersection = slice_set & boundary_nodes
                boundary_nodes.clear()
                boundary_nodes.update(updated_boundary)
                boundary_condition = [
                    (
                        find_element_index(existing_slices, node),
                        slice_nodes.index(node),
                    )
                    for node in intersection
                ]
                return SlicingResults(
                    [motif_data.copy()],
                    [slice_nodes],
                    [boundary_condition],
                    [intersection],
                )
        return None

    while graph.number_of_nodes():
        _step_res = _extract_next_slice(
            graph,
            motif_metadata,
            adjacency_map,
            boundary_nodes,
            results.slices,
        )

        if not _step_res:
            raise ValueError("Cannot slice the Pattern Graph.")

        results.extend(_step_res)

    return results


def build_adjacency_map(graph: nx.Graph) -> dict:
    """Build adjacency map from graph edges."""
    adjacency_map = {u: set() for u in graph.nodes}
    for u, v in graph.edges:
        adjacency_map[u].add(v)
        adjacency_map[v].add(u)
    return adjacency_map


def find_element_index(slices: List[List[int]], target: int) -> List[int]:
    """Find the index of an element in a ragged 2D list.
    Return [slice_index, vertex_index] if found, else []."""
    for i, group in enumerate(slices):
        if target in group:
            return [i, group.index(target)]
    return []


def find_boundary_nodes(
    combined_nodes: set,
    current_slice_edges: list,
    adjacency_map: dict,
) -> set:
    """
    Find updated boundary nodes after extracting a motif slice.
    A node is a boundary if it connects to any untouched node outside the
    current slice.
    """
    # Remove the adjacent node if the edge is covered by the current slice
    for node1, node2 in current_slice_edges:
        if node1 in adjacency_map and node2 in adjacency_map[node1]:
            adjacency_map[node1].remove(node2)
        if node2 in adjacency_map and node1 in adjacency_map[node2]:
            adjacency_map[node2].remove(node1)

    # Identify boundary nodes
    return {node for node in combined_nodes if adjacency_map.get(node, set())}
