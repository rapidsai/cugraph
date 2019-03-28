#include <cusp/array1d.h>
#include <cusp/gallery/poisson.h>
#include <cusp/graph/hilbert_curve.h>

#include <thrust/functional.h>

#include "../timer.h"

int main(int argc, char*argv[])
{
    srand(time(NULL));

    typedef int   IndexType;
    typedef double ValueType;
    typedef cusp::host_memory MemorySpace;

    size_t num_points = 1<<20; // 1M points

    for( int k = 0; k < 4; k++ ) {
        for( int i = 2; i < 4; i++ )
        {
            cusp::array2d<ValueType,MemorySpace,cusp::column_major> coords(num_points, i);
            cusp::copy(cusp::random_array<ValueType>(i*num_points), coords.values);

            cusp::array1d<IndexType,MemorySpace> parts(num_points);
            timer t;
            cusp::graph::hilbert_curve(coords, i, parts);
            std::cout << "Number of points : " << num_points << std::endl;
            std::cout << " hsfc(" << i << "D) : " << t.milliseconds_elapsed() << " (ms)\n" << std::endl;
        }
        num_points <<= 1;
    }

    return EXIT_SUCCESS;
}

