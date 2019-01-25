#pragma once

#include <nvgraph/nvgraph.h>
#include <cugraph.h>

gdf_error gdf_createGraph_nvgraph(nvgraphHandle_t nvg_handle, gdf_graph* gdf_G, nvgraphGraphDescr_t * nvgraph_G, bool use_transposed = false);
gdf_error gdf_sssp_nvgraph(gdf_graph *gdf_G, const int *source_vert, gdf_column *sssp_distances);