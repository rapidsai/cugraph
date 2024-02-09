#########################################
nx_cugraph as a backend for NetworkX code
#########################################

Wheras previous versions of cuGraph have included mechanisms to make it
trivial to plug in cuGraph algorithm calls. Beginning with version 24.02, nx-cuGraph 
is now a `networkX backend <https://networkx.org/documentation/stable/reference/utils.html#backends>`.
The user now need only `install nx-cugraph <https://github.com/rapidsai/cugraph/blob/branch-24.04/python/nx-cugraph/README.md#install>`
to experience GPU speedups.

The following algorithms are suppored and automatically dispatched to nx-cuGraph for acceleration.
 



