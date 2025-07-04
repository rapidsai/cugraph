#!/usr/bin/env python3

import time
import numpy as np
import cupy as cp
import cudf
import rmm

# Configure RMM BEFORE importing pylibcugraph to avoid memory issues
print("Configuring RMM memory pool...")
rmm.reinitialize(
    pool_allocator=True,
    initial_pool_size=2**30,  # 1GB initial pool
    maximum_pool_size=2**32   # 4GB max pool
)
print("RMM configured successfully")

# Now import pylibcugraph after RMM configuration
from pylibcugraph import (
    SGGraph, 
    ResourceHandle, 
    GraphProperties,
    betweenness_centrality
)

def create_test_graph():
    """Load the Manhattan road network from GraphML file"""
    print("Loading Manhattan road network from GraphML...")
    
    import networkx as nx
    
    # Load the GraphML file
    G_nx = nx.readwrite.graphml.read_graphml("/datasets/rratzel/OSM/manhatten.graphml")
    print(f"Loaded NetworkX graph: {G_nx.number_of_nodes()} nodes, {G_nx.number_of_edges()} edges")
    
    # Convert to edge list
    edges = []
    for u, v in G_nx.edges():
        edges.append((int(u), int(v)))
    
    # Create cuDF DataFrame
    edges_df = cudf.DataFrame(edges, columns=["src", "dst"])
    
    # Convert to cupy arrays
    src_array = edges_df['src'].values.astype(cp.int64)
    dst_array = edges_df['dst'].values.astype(cp.int64)
    weight_array = cp.ones(len(edges), dtype=cp.float32)
    
    print(f"Converted to edge list: {len(edges)} edges")
    
    # Create graph
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
    
    G = SGGraph(
        resource_handle=resource_handle,
        graph_properties=graph_props,
        src_or_offset_array=src_array,
        dst_or_index_array=dst_array,
        weight_array=weight_array,
        store_transposed=False,
        renumber=True,
        do_expensive_check=False,
    )
    
    print(f"Graph created successfully")
    return G, resource_handle, G_nx

def test_betweenness_centrality(G, resource_handle, G_nx):
    """Test exact betweenness centrality using all vertices as sources"""
    print(f"\n=== Testing Exact Betweenness Centrality (All Vertices as Sources) ===")
    
    try:
        print("Running betweenness centrality...")
        start_time = time.time()
        
        result = betweenness_centrality(
            resource_handle=resource_handle,
            graph=G,
            k=None,
            random_state=None,
            normalized=True,
            include_endpoints=False,
            do_expensive_check=False
        )
        
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.2f} seconds")
        
        # Extract results - pylibcugraph returns a tuple (vertices, centralities)
        vertices, centralities = result
        
        print(f"Result vertices shape: {vertices.shape}")
        print(f"Result centralities shape: {centralities.shape}")
        
        if centralities.size > 0:
            print(f"Centrality range: [{cp.min(centralities):.6f}, {cp.max(centralities):.6f}]")
            print(f"First 10 centralities: {centralities[:10]}")
            
                    # Find vertex with maximum centrality
        max_idx = cp.argmax(centralities)
        max_vertex = vertices[max_idx] if vertices.size > 0 else max_idx
        max_centrality = centralities[max_idx]
        print(f"Max centrality: vertex {max_vertex} = {max_centrality:.6f}")
        
        # Get coordinates of the vertex with maximum centrality
        # Convert back to original vertex ID if renumbering was used
        if hasattr(G, 'renumbered') and G.renumbered:
            # For pylibcugraph, we need to handle renumbering differently
            # The vertices array contains the renumbered IDs
            original_vertex_id = max_vertex
        else:
            original_vertex_id = max_vertex
            
        # Get coordinates from the original NetworkX graph
        try:
            # Convert to string since NetworkX node IDs are strings
            node_id_str = str(original_vertex_id)
            if node_id_str in G_nx.nodes:
                y = G_nx.nodes[node_id_str]["y"]
                x = G_nx.nodes[node_id_str]["x"]
                print(f"Max centrality intersection coordinates: ({y}, {x})")
            else:
                print(f"Warning: Could not find coordinates for vertex {original_vertex_id}")
        except Exception as e:
            print(f"Warning: Could not retrieve coordinates: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error with {num_seeds} seeds: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Testing Optimized Betweenness Centrality with pylibcugraph ===")
    
    # Create test graph
    G, resource_handle, G_nx = create_test_graph()
    
    # Test exact betweenness centrality using all vertices as sources
    print("\n" + "="*60)
    print("TESTING EXACT BETWEENNESS CENTRALITY (ALL VERTICES AS SOURCES)")
    print("="*60)
    
    success = test_betweenness_centrality(G, resource_handle, G_nx)
    if not success:
        print("Failed with all vertices, stopping tests")
        return
    
    print("\n=== Test Summary ===")
    print("This test used your optimized betweenness centrality implementation")
    print("with vertex list optimization in the backward pass.")
    print("The implementation is accessed through pylibcugraph which calls")
    print("the same C++ libcugraph code as the direct C++ tests.")

if __name__ == "__main__":
    main()
