#include <fstream>
#include <assert.h> 
#include <stdlib.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/plod_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/graph_traits.hpp>


void printUsageAndExit()
{
  printf("%s", "Usage:./plodg x\n");
  printf("%s", "x is the size of the graph\n");
  exit(0);
}

int main(int argc, char *argv[])
{
  
  /* " The Power Law Out Degree (PLOD) algorithm generates a scale-free graph from three parameters, n, alpha, and beta.
  [...] The value of beta controls the y-intercept of the curve, so that increasing beta increases the average degree of vertices (credit = beta*x^-alpha). 
  [...] The value of alpha controls how steeply the curve drops off, with larger values indicating a steeper curve. */
  // From Boost documentation http://www.boost.org/doc/libs/1_47_0/libs/graph/doc/plod_generator.html
  
  // we use setS aka std::set for edges storage
  // so we have at most one edges between 2 vertices
  // the extra cost is O(log(E/V)).
  typedef boost::adjacency_list<boost::setS> Graph;
  typedef boost::plod_iterator<boost::minstd_rand, Graph> SFGen;

  if (argc < 2) printUsageAndExit();
  int size = atoi (argv[1]);
  assert (size > 1 && size < INT_MAX);
  double alpha = 2.57; // It is known that web graphs have alpha ~ 2.72.
  double beta = size*512+1024; // This will give an average degree ~ 15

  // generation
  std::cout << "generating ... "<<'\n';
  boost::minstd_rand gen;
  Graph g(SFGen(gen, size, alpha, beta, false), SFGen(), size);
  boost::graph_traits<Graph>::edge_iterator edge, edge_end;
  
  std::cout << "vertices : "      << num_vertices(g) <<'\n';
  std::cout << "edges : "         << num_edges(g) <<'\n';
  std::cout << "average degree : "<< static_cast<float>(num_edges(g))/num_vertices(g)<< '\n';
  // Print in matrix coordinate real general format
  std::cout << "writing ... "<<'\n';
  std::stringstream tmp;
  tmp <<"local_test_data/plod_graph_" << size << ".mtx";
  const std::string filename = tmp.str();
  std::ofstream fout(tmp.str().c_str()) ;
  
  if (argv[2]==NULL)
  {
    // Power law out degree with random weights
    fout << "%%MatrixMarket matrix coordinate real general\n";
    fout << num_vertices(g) <<' '<< num_vertices(g)  <<' '<< num_edges(g) << '\n';
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
    fout << num_vertices(g) <<' '<< num_edges(g) << '\n';
    for( boost::tie(edge, edge_end) = boost::edges(g); edge != edge_end; ++edge)
      fout <<boost::target(*edge, g)<< ' ' << boost::source(*edge, g) << '\n';
  }
  else if (argv[2][0]=='o')
  {
    // Power law out degree
    // -- Edges only --
    fout << num_vertices(g) <<' '<< num_edges(g) << '\n';
    for( boost::tie(edge, edge_end) = boost::edges(g); edge != edge_end; ++edge)
      fout << boost::source(*edge, g) << ' ' << boost::target(*edge, g)<< '\n';
  }
  else printUsageAndExit();

  fout.close();
  std::cout << "done!"<<'\n';
  return 0;
}

