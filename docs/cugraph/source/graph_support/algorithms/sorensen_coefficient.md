# Sørensen Coefficient

The Sørensen Coefficient, also called the Sørensen-Dice similarity coefficient, quantifies the similarity and overlap between two samples.

It is defined as two times the size of the set intersection divided by the sum of the size of the two sets. The value ranges from 0 to 1.

Sørensen coefficient = $\left(2 * |A \cap B| \right) \over \left(|A| + |B| \right)$


In graphs, the sets refer to the set of connected nodes or neighborhood of nodes A and B.

[Learn more about Sørensen Coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)

## When to use Sørensen Coefficient
* When you want to compare nodes with vastly different sized neighborhoods.
* When the intersection of the node neigborhoods is more important than the overall similarity


## When not to use Sørensen Coefficient
* In directed graphs
* Comparing graphs with different underlying data relationships.
* In weighted graphs, while cuGraph does have a weighted Sørensen implementation, the algorithm did not originally use weights.

## How computationally expensive is it?
While cuGraph's parallelism mitigates run cost, [Big O notation](https://en.wikipedia.org/wiki/Big_O_notation) is still the standard to compare algorithm execution time.
The cost to run O(n * m) where n is the number of nodes in the graph and m is the number of groups to test. 
