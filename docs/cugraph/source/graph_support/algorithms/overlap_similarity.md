# Overlap Similarity

The Overlap Coefficient between two sets is defined as the ratio of the volume of their intersection divided by the volume of the smaller set.
The Overlap Coefficient can be defined as

<a href="https://www.codecogs.com/eqnedit.php?latex=oc(A,B)&space;=&space;\frac{|A|&space;\cap&space;|B|}{min(|A|,&space;|B|)&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?oc(A,B)&space;=&space;\frac{|A&space;\cap&space;B|}{min(|A|,&space;|B|)&space;}" title="oc(A,B) = \frac{|A \cap B|}{min(|A|, |B|) }" /></a>


[Learn more about Overlap Similarity](https://en.wikipedia.org/wiki/Overlap_coefficient)

## When to use Overlap Similarity
* You want to find similarty based on shared neighbors instead of the sets as a whole.
* You want to partition a graph into non-overlapping clusters.
* You want to compare subgraphs within a graph

## When not to use Overlap Similarity
* You are trying to compare graphs of extremely different sizes
* In overly sparse or dense graph can overlap similarity can miss relationships or give fals positives respectively.
* In directed graphs, there are better algorithms to use. 


## How computationally expensive is it?
While cuGraph's parallelism migigates run time, [Big O notation](https://en.wikipedia.org/wiki/Big_O_notation) is still the standard to compare algorithm costs.

The cost to compute overlap similarity is O(n*d) where n is the number of nodes and d is the average degree of the nodes.