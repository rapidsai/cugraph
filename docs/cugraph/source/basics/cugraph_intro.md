
# cuGraph Introduction
The Data Scientist has a collection of techniques within their 
proverbial toolbox. Data engineering, statistical analysis, and 
machine learning are among the most commonly known. However, there 
are numerous cases where the focus of the analysis is on the 
relationship between data elements. In those cases, the data is best 
represented as a graph. Graph analysis, also called network analysis, 
is a collection of algorithms for answering questions posed against 
graph data. Graph analysis is not new.

The first graph problem was posed by Euler in 1736, the [Seven Bridges of 
Konigsberg](https://en.wikipedia.org/wiki/Seven_Bridges_of_K%C3%B6nigsberg), 
and laid the foundation for the mathematical field of graph theory. 
The application of graph analysis covers a wide variety of fields, including 
marketing, biology, physics, computer science, sociology, and cyber to name a few.

RAPIDS cuGraph is a library of graph algorithms that seamlessly integrates 
into the RAPIDS data science ecosystem and allows the data scientist to easily 
call graph algorithms using data stored in a GPU DataFrame, NetworkX Graphs, or even 
CuPy or SciPy sparse Matrix.  


# Vision
The vision of RAPIDS cuGraph is to ___make graph analysis ubiquitous to the 
point that users just think in terms of analysis and not technologies or 
frameworks___. This is a goal that many of us on the cuGraph team have been 
working on for almost twenty years. Many of the early attempts focused on 
solving one problem or using one technique. Those early attempts worked for 
the initial goal but tended to break as the scope changed (e.g., shifting 
to solving a dynamic graph problem with a static graph solution). The limiting 
factors usually came down to compute power, ease-of-use, or choosing a data 
structure that was not suited for all problems. NVIDIA GPUs, CUDA, and RAPIDS 
have totally changed the paradigm and the goal of an accelerated unified graph 
analytic library is now possible.

The compute power of the latest NVIDIA GPUs (RAPIDS supports Pascal and later 
GPU architectures) make graph analytics 1000x faster on average over NetworkX. 
Moreover, the internal memory speed within a GPU allows cuGraph to rapidly 
switch the data structure to best suit the needs of the analytic rather than 
being restricted to a single data structure. cuGraph is working with several 
frameworks for both static and dynamic graph data structures so that we always 
have a solution to any graph problem. Since Python has emerged as the de facto 
language for data science, allowing interactivity and the ability to run graph 
analytics in Python makes cuGraph familiar and approachable. RAPIDS wraps all 
the graph analytic goodness mentioned above with the ability to perform 
high-speed ETL, statistics, and machine learning. To make things even better, 
RAPIDS and DASK allows cuGraph to scale to multiple GPUs to support 
multi-billion edge graphs.


# Terminology

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

# Roadmap
GitHub does not provide a robust project management interface, and so a roadmap turns into simply a projection of when work will be completed and not a complete picture of everything that needs to be done.  To capture the work that requires multiple steps, issues are labels as “EPIC” and include multiple subtasks that could span multiple releases.   The EPIC will be in the release where work in expected to be completed. A better roadmap is being worked an image of the roadmap will be posted when ready.

 * GitHub Project Board:  https://github.com/rapidsai/cugraph/projects/28
 