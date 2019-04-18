#include <boost/config.hpp>
#include <iostream>
#include <fstream> //file output
#include <cfloat>
#include <omp.h>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/rmat_graph_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/graph_traits.hpp>
 
void printUsageAndExit()
{
  printf("%s", "Usage:./rmatg x y\n");
  printf("%s", "x is the size of the graph, x>32 (Boost generator hang if x<32)\n");
  printf("%s", "y is the source of sssp\n");
  exit(0);
}

int main(int argc, char *argv[])
{
  // read size
  if (argc < 3) printUsageAndExit();
  int size = atoi (argv[1]);
  if (size<32) printUsageAndExit();
  int source_sssp =atoi (argv[2]);
  assert (size > 1 && size < INT_MAX);
  assert (source_sssp >= 0 && source_sssp < size);
  const unsigned num_edges = 15 * size;
  
  // Some boost types
  typedef boost::no_property VertexProperty;
  typedef boost::property<boost::edge_weight_t, float> EdgeProperty;
  typedef boost::adjacency_list<boost::mapS, boost::vecS, boost::directedS, VertexProperty, EdgeProperty> Graph;
  typedef boost::unique_rmat_iterator<boost::minstd_rand, Graph> RMATGen;
  typedef boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
  boost::minstd_rand gen;
  boost::graph_traits<Graph>::edge_iterator edge, edge_end;

  /************************
   * Random weights
   ************************/
  // !!! WARNING !!!
  // watch the stack
  float* weight = new float[num_edges]; 
  int count = 0;
  for( int i = 0; i < num_edges;  ++i)
    weight[i] = (rand()%10)+(rand()%100)*(1.2e-2f);

  /************************
   * RMAT Gen
   ************************/
  Graph g(RMATGen(gen, size, num_edges, 0.57, 0.19, 0.19, 0.05,true),RMATGen(),weight, size);
  std::cout << "Generator : done. Edges = "<<boost::num_edges(g)<<std::endl; 
  assert (num_edges == boost::num_edges(g));
  // debug print after gen
  //for( boost::tie(edge, edge_end) = boost::edges(g); edge != edge_end; ++edge)
  //  std::cout << boost::source(*edge, g) << ' ' << boost::target(*edge, g)<< ' '<<  boost::get(boost::get(boost::edge_weight, g),*edge) << '\n';
  
  /************************
   * Dijkstra
   ************************/
  std::vector<vertex_descriptor> p(num_vertices(g));
  std::vector<float> d(num_vertices(g));
  vertex_descriptor s = vertex(source_sssp, g); //define soruce node
  
  double start = omp_get_wtime();
  dijkstra_shortest_paths(g, s,
                          predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, g))).
                          distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));

  double stop = omp_get_wtime();
  std::cout << "Time = " << stop-start << "s"<< std::endl;

  /************************
   * Print
   ************************/
  /*
  boost::graph_traits<Graph>::vertex_iterator vi, vend;
  std::cout << "SOURCE = "<< source_sssp << std::endl; 
  for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) 
  {
    if (d[*vi] != FLT_MAX) 
    {
      std::cout << "d(" << *vi << ") = " << d[*vi] << ", ";
      std::cout << "parent = " << p[*vi] << std::endl; 
    }
    else
      std::cout << "d(" << *vi << ") = INF"<< std::endl;
  }
  */
  return 0;
                
}

