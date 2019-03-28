#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>

#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>

#include "benchmark.h"
#include "bytes_per_spmv.h"
#include "utility.h"

#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <limits>

typedef std::map<std::string, std::string> ArgumentMap;
ArgumentMap args;

std::string process_args(int argc, char ** argv)
{
    std::string filename;

    for(int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);

        if (arg.substr(0,2) == "--")
        {
            std::string::size_type n = arg.find('=',2);

            if (n == std::string::npos)
                args[arg.substr(2)] = std::string();              // (key)
            else
                args[arg.substr(2, n - 2)] = arg.substr(n + 1);   // (key,value)
        }
        else
        {
            filename = arg;
        }
    }

    return filename;
}

void usage(int argc, char** argv)
{
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << "\n";
    std::cout << "\t" << argv[0] << " my_matrix.mtx\n";
    std::cout << "\t" << argv[0] << " my_matrix.mtx --device=1\n";
    std::cout << "\t" << argv[0] << " my_matrix.mtx --value_type=double\n\n";
    std::cout << "Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n";
    std::cout << "      If no matrix file is provided then a simple example is created.\n";
}


template <typename IndexType, typename ValueType>
void test_all_formats(std::string& filename)
{
    int device_id  = args.count("device") ? atoi(args["device"].c_str()) :  0;
    set_device(device_id);
    list_devices();

    std::cout << "Running on Device " << device_id << "\n\n";

    // load a matrix stored in MatrixMarket format
    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> host_matrix;

    if (filename == "")
    {
        std::cout << "Generated matrix (poisson5pt) ";
        cusp::gallery::poisson5pt(host_matrix, 512, 512);
    }
    else
    {
        cusp::io::read_matrix_market_file(host_matrix, filename);
        std::cout << "Read matrix (" << filename << ") ";
    }

    std::cout << "with shape ("  << host_matrix.num_rows << "," << host_matrix.num_cols << ") and "
              << host_matrix.num_entries << " entries" << "\n\n";

    FILE * fid = fopen(BENCHMARK_OUTPUT_FILE_NAME, "a");
    fprintf(fid, "file=%s rows=%d cols=%d nonzeros=%d\n", filename.c_str(),
            (int) host_matrix.num_rows, (int) host_matrix.num_cols, (int) host_matrix.num_entries);
    fclose(fid);

    test_coo(host_matrix);
    test_csr(host_matrix);
    test_dia(host_matrix);
    test_ell(host_matrix);
    test_hyb(host_matrix);
}

int main(int argc, char** argv)
{
    std::string filename = process_args(argc, argv);

    if (args.count("help"))
    {
        usage(argc, argv);
        return 0;
    }

    // select ValueType
    std::string value_type = args.count("value_type") ? args["value_type"] : "float";
    std::cout << "\nComputing SpMV with \'" << value_type << "\' values.\n\n";

    if (value_type == "float")
    {
        test_all_formats<int,float>(filename);
    }
    else if (value_type == "double")
    {
        int current_device = -1;
        cudaDeviceProp properties;
        cudaGetDevice(&current_device);
        cudaGetDeviceProperties(&properties, current_device);
        if (properties.major == 1 && properties.minor < 3)
            std::cerr << "ERROR: Support for \'double\' requires Compute Capability 1.3 or greater\n\n";
        else
            test_all_formats<int,double>(filename);
    }
    else
    {
        std::cerr << "ERROR: Unsupported type \'" << value_type << "\'\n\n";
    }

    return 0;
}

