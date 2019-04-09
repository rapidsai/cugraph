#include <fstream>
#include <assert.h> 
#include <stdlib.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/rmat_graph_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/graph_traits.hpp>


void printUsageAndExit()
{
  printf("%s", "Usage:./rmatg x\n");
  printf("%s", "x is the size of the graph, x>32 (Boost generator hang if x<32)\n");
  exit(0);
}

int main(int argc, char *argv[])
{
  
  // RMAT paper http://snap.stanford.edu/class/cs224w-readings/chakrabarti04rmat.pdf
  // Boost doc on RMAT http://www.boost.org/doc/libs/1_49_0/libs/graph_parallel/doc/html/rmat_generator.html
  
  typedef boost::adjacency_list<boost::mapS, boost::vecS, boost::directedS> Graph;
  typedef boost::unique_rmat_iterator<boost::minstd_rand, Graph> RMATGen;

  if (argc < 2) printUsageAndExit();
  int size = atoi (argv[1]);
  if (size<32) printUsageAndExit();
  assert (size > 31 && size < INT_MAX);
  const unsigned num_edges = 16 * size;
  /************************
   * RMAT Gen
   ************************/
  std::cout << "generating ... "<<'\n';
  // values of a,b,c,d are from the graph500.
  boost::minstd_rand gen;
  Graph g(RMATGen(gen, size, num_edges, 0.57, 0.19, 0.19, 0.05, true), RMATGen(), size);
  assert (num_edges == boost::num_edges(g));
  
  /************************
   * Print
   ************************/
  boost::graph_traits<Graph>::edge_iterator edge, edge_end;
  std::cout << "vertices : "      << boost::num_vertices(g) <<'\n';
  std::cout << "edges : "         << boost::num_edges(g) <<'\n';
  std::cout << "average degree : "<< static_cast<float>(boost::num_edges(g))/boost::num_vertices(g)<< '\n';
  
  // Print in matrix coordinate real general format
  std::cout << "writing ... "<<'\n';
  std::stringstream tmp;
  tmp <<"local_test_data/rmat_graph_" << size << ".mtx";
  const std::string filename = tmp.str();
  std::ofstream fout(tmp.str().c_str()) ;
  if (argv[2]==NULL)
  {
    // Power law out degree with random weights
    fout << "%%MatrixMarket matrix coordinate real general\n";
    fout << boost::num_vertices(g) <<' '<< boost::num_vertices(g)  <<' '<< boost::num_edges(g) << '\n';
    float val;
    for( boost::tie(edge, edge_end) = boost::edges(g); edge != edge_end; ++edge)
    {
      val = (rand()%10)+(rand()%100)*(1e-2f);
      fout << boost::source(*edge, g) << ' ' << boost::target(*edge, g)<< ' ' << val << '\n';
    }
  }
  else if (argv[2][0]=='i')
  {
    // Power law in degree (ie the transpose will have a power law)
    // -- Edges only --
    // * Wraning * edges will be unsorted, use sort_edges.cpp to sort the dataset.
    fout << boost::num_vertices(g) <<' '<< boost::num_edges(g) << '\n';
    for( boost::tie(edge, edge_end) = boost::edges(g); edge != edge_end; ++edge)
      fout <<boost::target(*edge, g)<< ' ' << boost::source(*edge, g) << '\n';
  }
  else if (argv[2][0]=='o')
  {
    // Power law out degree
    // -- Edges only --
    fout << boost::num_vertices(g) <<' '<< boost::num_edges(g) << '\n';
    for( boost::tie(edge, edge_end) = boost::edges(g); edge != edge_end; ++edge)
      fout << boost::source(*edge, g) << ' ' << boost::target(*edge, g)<< '\n';
  }
  else printUsageAndExit();
  fout.close();
  std::cout << "done"<<'\n';
  return 0;
}

