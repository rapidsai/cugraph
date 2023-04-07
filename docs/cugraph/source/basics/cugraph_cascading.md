
# Method Cascading and cuGraph

BLUF: cuGraph does not support method cascading

[Method Cascading](https://en.wikipedia.org/wiki/Method_cascading) is a popular, and useful, functional programming concept and is a great way to make code more readable.  Python supports method cascading ... _for the most part_.  There are a number of Python built-in classes that do not support cascading. 

An example, from cuDF, is a sequence of method calls for  loading data and then finding the largest values from a subset of the data (yes there are other ways this could be done):

```
gdf = cudf.from_pandas(df).query(‘val > 200’).nlargest(‘va’3)
```

cuGraph does not support method cascading for two main reasons: (1) the object-oriented nature of the Graph data object leverages in-place methods, and (2) the fact that algorithms operate on graphs rather than graphs running algorithms.  

## Graph Data Objects
cuGraph follows an object-oriented design for the Graph objects.  Users create a Graph and can then add data to object, but every add method call returns `None`.  

_Why Inplace methods?_ <br>
cuGraph focuses on the big graph problems where there are 10s of millions to trillions of edges (Giga bytes to Terabytes of data). At that scale, creating a copy of the data becomes memory inefficient.  

_Why not return `self` rather than `None`?_<br>
It would be simple to modify the methods to return `self` rather than `None`, however it opens the methods to misinterpretation.  Consider the following code:

```
# cascade flow - makes sense
G = cugraph.Graph().from_cudf_edgelist(df)

# non-cascaded code can be confusing
G = cugraph.Graph()
G2 = G.from_cudf_edgelist(df)
G3 = G.from_cudf_edgelist(df2)
```
The confusion with the non-cascade code is that G, G1, and G3 are all the same object with the same data.   Users could be confused since it is not obvious that changing G3 would also change both G2 and G.  To prevent confusion, cuGraph has opted to not return `self`.

_Why not add a flag "return_self" to the methods?_<br>
```
# cascade flow - makes sense
G = cugraph.Graph().from_cudf_edgelist(df, return_self=True)
```
The fact that a developer would explicitly add a "return_self" flag to the method indicates that the developer is aware that the method returns None. It is just as easy for the developer to use a non-cascading workflow.

### Algorithms
Algorithms operate on graph objects.
```
cugraph.pagerank(G) and not G.pagerank()
```
This pattern allows cuGraph to maintain a particular object-oriented model, where Graph objects simply maintain graph data, and algorithm functions operate independently on Graph objects. While this model has benefits that simplify the overall design and its usability in the majority of use cases, it does mean that the developer cannot cascade graph creation into an algorithm call.

```
# will not work
G = cugraph.Graph().from_cudf_edgelist(df).pagerank()
```
