import cugraph
vertices, edges, sources, destinations = cugraph.grmat_gen('grmat --rmat_scale=2 --rmat_edgefactor=2 --device=0 --normalized --quiet')

