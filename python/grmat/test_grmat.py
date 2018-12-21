import cuGraph
vertices, edges, sources, destinations = cuGraph.grmat_gen('grmat --rmat_scale=2 --rmat_edgefactor=2 --device=0 --normalized --quiet')

