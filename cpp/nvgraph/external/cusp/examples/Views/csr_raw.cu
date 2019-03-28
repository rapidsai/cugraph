#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/print.h>

// This example shows how to wrap "raw" host and device memory
// with a csr_matrix_view.  This situation arises when interfacing
// Cusp with data that is managed externally.  Once raw data has
// been appropriately wrapped the csr_matrix_view can be used
// in Cusp algorithms like cusp::copy() and cusp::multiply()
// just like a csr_matrix container.
//
//  Example Matrix:
//   [10  0 20]
//   [ 0  0  0]
//   [ 0  0 30]
//   [40 50 60]

int main(void)
{
    // CSR format in raw host memory
    int   host_Ap[5] = {0,2,2,3,6};            // CSR row pointer
    int   host_Aj[6] = {0,2,2,0,1,2};          // CSR column indices
    float host_Ax[6] = {10,20,30,40,50,60};    // CSR values

    // x and y arrays in host memory
    float host_x[3] = {1,1,1};
    float host_y[4] = {0,0,0,0};

    // allocate device memory for CSR format
    int   * device_Ap;
    cudaMalloc(&device_Ap, 5 * sizeof(int));
    int   * device_Aj;
    cudaMalloc(&device_Aj, 6 * sizeof(int));
    float * device_Ax;
    cudaMalloc(&device_Ax, 6 * sizeof(float));

    // allocate device memory for x and y arrays
    float * device_x;
    cudaMalloc(&device_x, 3 * sizeof(float));
    float * device_y;
    cudaMalloc(&device_y, 4 * sizeof(float));

    // copy raw data from host to device
    cudaMemcpy(device_Ap, host_Ap, 5 * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(device_Aj, host_Aj, 6 * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(device_Ax, host_Ax, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_x,  host_x,  3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y,  host_y,  4 * sizeof(float), cudaMemcpyHostToDevice);

    // wrap the host memory with a csr_matrix_view
    {
        // use array1d_view to wrap the individual arrays
        typedef typename cusp::array1d_view<int   *> HostIndexArrayView;
        typedef typename cusp::array1d_view<float *> HostValueArrayView;

        HostIndexArrayView row_offsets   (host_Ap, host_Ap + 5);
        HostIndexArrayView column_indices(host_Aj, host_Aj + 6);
        HostValueArrayView values        (host_Ax, host_Ax + 6);

        HostValueArrayView x (host_x, host_x + 3);
        HostValueArrayView y (host_y, host_y + 4);

        // combine the three array1d_views into a csr_matrix_view
        typedef cusp::csr_matrix_view<HostIndexArrayView,
                HostIndexArrayView,
                HostValueArrayView> HostView;

        HostView A(4, 3, 6, row_offsets, column_indices, values);

        // print view
        std::cout << "\nhost csr_matrix_view" << std::endl;
        cusp::print(A);

        // compute y = A* x
        cusp::multiply(A, x, y);

        // print x
        std::cout << "\nx array" << std::endl;
        cusp::print(x);

        // print y
        std::cout << "\n y array, y = A * x" << std::endl;
        cusp::print(y);
    }

    // wrap the device memory with a csr_matrix_view
    {
        // *NOTE* raw pointers must be wrapped with thrust::device_ptr!
        thrust::device_ptr<int>   wrapped_device_Ap(device_Ap);
        thrust::device_ptr<int>   wrapped_device_Aj(device_Aj);
        thrust::device_ptr<float> wrapped_device_Ax(device_Ax);
        thrust::device_ptr<float> wrapped_device_x(device_x);
        thrust::device_ptr<float> wrapped_device_y(device_y);

        // use array1d_view to wrap the individual arrays
        typedef typename cusp::array1d_view< thrust::device_ptr<int>   > DeviceIndexArrayView;
        typedef typename cusp::array1d_view< thrust::device_ptr<float> > DeviceValueArrayView;

        DeviceIndexArrayView row_offsets   (wrapped_device_Ap, wrapped_device_Ap + 5);
        DeviceIndexArrayView column_indices(wrapped_device_Aj, wrapped_device_Aj + 6);
        DeviceValueArrayView values        (wrapped_device_Ax, wrapped_device_Ax + 6);
        DeviceValueArrayView x (wrapped_device_x, wrapped_device_x + 3);
        DeviceValueArrayView y (wrapped_device_y, wrapped_device_y + 4);

        // combine the three array1d_views into a csr_matrix_view
        typedef cusp::csr_matrix_view<DeviceIndexArrayView,
                DeviceIndexArrayView,
                DeviceValueArrayView> DeviceView;

        DeviceView A(4, 3, 6, row_offsets, column_indices, values);

        // print view
        std::cout << "\ndevice csr_matrix_view" << std::endl;
        cusp::print(A);

        // compute y = A* x
        cusp::multiply(A, x, y);

        // print x
        std::cout << "\nx array" << std::endl;
        cusp::print(x);

        // print y
        std::cout << "\n y array, y = A * x" << std::endl;
        cusp::print(y);
    }

    // free device arrays
    cudaFree(device_Ap);
    cudaFree(device_Aj);
    cudaFree(device_Ax);
    cudaFree(device_x);
    cudaFree(device_y);

    return 0;
}

