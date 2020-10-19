
# cuGraph Introduction


## Terminology

cuGraph is a collection of GPU accelerated graph algorithms and graph utility
functions. The application of graph analysis covers a lot of areas.
For Example:
* [Network Science](https://en.wikipedia.org/wiki/Network_science)
* [Complex Network](https://en.wikipedia.org/wiki/Complex_network)
* [Graph Theory](https://en.wikipedia.org/wiki/Graph_theory)
* [Social Network Analysis](https://en.wikipedia.org/wiki/Social_network_analysis)

cuGraph does not favor one field over another.  Our developers span the
breadth of fields with the focus being to produce the best graph library
possible.  However, each field has its own argot (jargon) for describing the
graph (or network).  In our documentation, we try to be consistent.  In Python
documentation we will mostly use the terms __Node__ and __Edge__ to better
match NetworkX preferred term use, as well as other Python-based tools.  At
the CUDA/C layer, we favor the mathematical terms of __Vertex__ and __Edge__.  

