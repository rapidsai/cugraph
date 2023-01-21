# External Data Types
cuGraph Python strives to make getting data into and out of cuGraph simple.  To that end, the Python interface accepts 



## Supported Data Types
cuGraph supports graph creation with Source and Destination being expressed as:
* cuDF DataFrame
* Pandas DataFrame
* NetworkX graph classes
* Numpy arrays
* CuPy sparse matrix
* SciPy sparse matrix

cuGraph tries to match the return type based on the input type.  So a NetworkX input will return the same data type that NetworkX would have.

## cuDF
The preferred data type is a cuDF object since it is already in the GPU.  For loading data from disk into cuDF please see the cuDF documentation. 

__Loading data__
  * Graph.from_cudf_adjlist
  * Graph.from_cudf_edgelist


__Results__<br>
Results which are not simple types (ints, floats) are typically cuDF Dataframes. 



## Pandas
The RAPIDS cuDF library can be thought of as accelerated Pandas 


## NetworkX Graph Objects


## 






</br></br>

---
