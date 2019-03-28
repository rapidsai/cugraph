#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>

// This example shows how to solve a linear system A * x = b
// where the matrix A and vectors x and b are stored in
// "raw" memory on the device.  The matrix A will be wrapped
// with a coo_matrix_view while the vectors x and b will be
// wrapped with array1d_views.
//
// Views allow you to interface Cusp's solvers with
// data that is managed externally without needing to
// copy the data into a Cusp matrix container.
//
//  Example Matrix:
//   [ 2 -1  0  0]
//   [-1  2 -1  0]
//   [ 0 -1  2 -1]
//   [ 0  0 -1  2]


int main(void)
{
    // COO format in host memory
    int   host_I[10] = { 0, 0, 1, 1, 1, 2, 2, 2, 3, 3}; // COO row indices
    int   host_J[10] = { 0, 1, 0, 1, 2, 1, 2, 3, 2, 3}; // COO column indices
    float host_V[10] = { 2,-1,-1, 2,-1,-1, 2,-1,-1, 2}; // COO values

    // x and b arrays in host memory
    float host_x[4] = {0,0,0,0};
    float host_b[4] = {1,2,2,1};

    // allocate device memory for CSR format
    int   * device_I;
    cudaMalloc(&device_I, 10 * sizeof(int));
    int   * device_J;
    cudaMalloc(&device_J, 10 * sizeof(int));
    float * device_V;
    cudaMalloc(&device_V, 10 * sizeof(float));

    // allocate device memory for x and y arrays
    float * device_x;
    cudaMalloc(&device_x, 4 * sizeof(float));
    float * device_b;
    cudaMalloc(&device_b, 4 * sizeof(float));

    // copy raw data from host to device
    cudaMemcpy(device_I, host_I, 10 * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(device_J, host_J, 10 * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(device_V, host_V, 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, host_x,  4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b,  4 * sizeof(float), cudaMemcpyHostToDevice);

    // matrices and vectors now reside on the device

    // *NOTE* raw pointers must be wrapped with thrust::device_ptr!
    thrust::device_ptr<int>   wrapped_device_I(device_I);
    thrust::device_ptr<int>   wrapped_device_J(device_J);
    thrust::device_ptr<float> wrapped_device_V(device_V);
    thrust::device_ptr<float> wrapped_device_x(device_x);
    thrust::device_ptr<float> wrapped_device_b(device_b);

    // use array1d_view to wrap the individual arrays
    typedef typename cusp::array1d_view< thrust::device_ptr<int>   > DeviceIndexArrayView;
    typedef typename cusp::array1d_view< thrust::device_ptr<float> > DeviceValueArrayView;

    DeviceIndexArrayView row_indices   (wrapped_device_I, wrapped_device_I + 10);
    DeviceIndexArrayView column_indices(wrapped_device_J, wrapped_device_J + 10);
    DeviceValueArrayView values        (wrapped_device_V, wrapped_device_V + 10);
    DeviceValueArrayView x             (wrapped_device_x, wrapped_device_x + 4);
    DeviceValueArrayView b             (wrapped_device_b, wrapped_device_b + 4);

    // combine the three array1d_views into a coo_matrix_view
    typedef cusp::coo_matrix_view<DeviceIndexArrayView,
            DeviceIndexArrayView,
            DeviceValueArrayView> DeviceView;

    // construct a coo_matrix_view from the array1d_views
    DeviceView A(4, 4, 10, row_indices, column_indices, values);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-5
    //  absolute_tolerance = 0
    //  verbose            = true
    cusp::monitor<float> monitor(b, 100, 1e-5, 0, true);

    // solve the linear system A * x = b with the Conjugate Gradient method
    cusp::krylov::cg(A, x, b, monitor);

    // copy the solution back to the host
    cudaMemcpy(host_x, device_x,  4 * sizeof(float), cudaMemcpyDeviceToHost);

    // free device arrays
    cudaFree(device_I);
    cudaFree(device_J);
    cudaFree(device_V);
    cudaFree(device_x);
    cudaFree(device_b);

    return 0;
}

