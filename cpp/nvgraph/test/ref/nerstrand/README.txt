This is stand alone host app that reads an undirected graph in matrix market format, convert it into CSR, call Nerstrand with default parameters and returns the modularity score of the clustering.   

Make sure you have downloaded and installed nerstrand : http://www-users.cs.umn.edu/~lasalle/nerstrand/
You should have libnerstrand.a in <nerstrand_directory>/build/Linux-x86_64/lib, move it to the directory containing this README or adjust the Makefile.

Type "make" to compile the small benchmarking app and "./nerstrand_bench <graph> <number of clusters>" to execute.
For convenience there is also a benchmarking script that calls the benchmarking app (please adjust paths to binary and data sets).

Use the following reference: 
@article{lasalle2014nerstrand,
  title={Multi-threaded Modularity Based Graph Clustering using the Multilevel Paradigm},
  journal = "Journal of Parallel and Distributed Computing ",
  year = "2014",
  issn = "0743-7315",
  doi = "http://dx.doi.org/10.1016/j.jpdc.2014.09.012",
  url = "http://www.sciencedirect.com/science/article/pii/S0743731514001750",
  author = "Dominique LaSalle and George Karypis"
}​
