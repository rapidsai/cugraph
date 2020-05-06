from cugraph.structure.graph_new cimport *

cdef extern from "algorithms.hpp" namespace "cugraph":

    cdef void mg_pagerank_temp[VT,ET,WT](
        const GraphCSC[VT,ET,WT] &graph,
        WT *pagerank) except +
