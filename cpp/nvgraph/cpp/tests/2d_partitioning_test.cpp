#include "gtest/gtest.h"
#include "nvgraph.h"
#include <iostream>

TEST(SimpleBFS2D, DummyTest) {
	nvgraphHandle_t handle;
	int* devices = (int*) malloc(sizeof(int) * 2);
	devices[0] = 0;
	devices[1] = 1;
	nvgraphCreateMulti(&handle, 2, devices);
	nvgraphGraphDescr_t graph;
	nvgraphCreateGraphDescr(handle, &graph);
	int rowIds[38] = { 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5,
			5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8 };
	int colIds[38] = { 1, 2, 7, 8, 0, 2, 4, 7, 8, 0, 1, 3, 6, 8, 2, 4, 5, 6, 8, 1, 3, 5, 8, 3, 4, 6,
			7, 2, 3, 5, 0, 1, 5, 0, 1, 2, 3, 4 };
	nvgraph2dCOOTopology32I_st topo;
	topo.nvertices = 9;
	topo.nedges = 38;
	topo.source_indices = rowIds;
	topo.destination_indices = colIds;
	topo.valueType = CUDA_R_32I;
	topo.values = NULL;
	topo.numDevices = 2;
	topo.devices = devices;
	topo.blockN = 2;
	topo.tag = NVGRAPH_DEFAULT;
	nvgraphSetGraphStructure(handle, graph, &topo, NVGRAPH_2D_32I_32I);
	int* distances = (int*) malloc(sizeof(int) * 9);
	int* predecessors = (int*) malloc(sizeof(int) * 9);
	int sourceId = 0;
	std::cout << "Source ID: " << sourceId << "\n";
	nvgraph2dBfs(handle, graph, sourceId, distances, predecessors);
	std::cout << "Distances:\n";
	for (int i = 0; i < 9; i++)
		std::cout << i << ":" << distances[i] << "  ";
	std::cout << "\nPredecessors:\n";
	for (int i = 0; i < 9; i++)
		std::cout << i << ":" << predecessors[i] << "  ";
	std::cout << "\n";
	int exp_pred[9] = {-1,0,0,2,1,7,2,0,0};
	int exp_dist[9] = {0,1,1,2,2,2,2,1,1};
	for (int i = 0; i < 9; i++){
		ASSERT_EQ(exp_pred[i], predecessors[i]);
		ASSERT_EQ(exp_dist[i], distances[i]);
	}
	std::cout << "Test run!\n";
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
