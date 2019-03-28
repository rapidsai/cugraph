#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/graph/hilbert_curve.h>

#include <thrust/functional.h>

int main(int argc, char*argv[])
{
    // Use 1M points
    size_t num_points = 1<<20;

    // create an empty dense matrix in column major format
    cusp::array2d<double,cusp::device_memory,cusp::column_major> coords(num_points, 2);

    // initialize x,y coordinates with random values
    cusp::copy(cusp::random_array<double>(2*num_points), coords.values);

    // create vector to contain partition labels of each point
    cusp::array1d<int,cusp::device_memory> parts(num_points);

    // execute hilbert curve partitioning on device
    cusp::graph::hilbert_curve(coords, 2, parts);

    return 0;
}

