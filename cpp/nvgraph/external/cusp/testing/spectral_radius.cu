#include <unittest/unittest.h>

#include <cusp/ell_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/hyb_matrix.h>

#include <cusp/array2d.h>

#include <cusp/eigen/spectral_radius.h>
#include <cusp/eigen/arnoldi.h>

#include <cusp/gallery/poisson.h>

template <class SparseMatrix>
void TestEstimateSpectralRadius(void)
{
    typedef typename SparseMatrix::value_type   ValueType;

    cusp::array2d<ValueType, cusp::host_memory> A(2,2,0);
    A(0,0) = -5.0;
    A(1,1) =  2.0;

    cusp::array2d<ValueType, cusp::host_memory> B;
    cusp::gallery::poisson5pt(B, 2, 2);

    cusp::array2d<ValueType, cusp::host_memory> C;
    cusp::gallery::poisson5pt(C, 4, 4);

    std::vector< cusp::array2d<ValueType, cusp::host_memory> > matrices;
    matrices.push_back(A);
    matrices.push_back(B);
    matrices.push_back(C);

    std::vector<float> rhos(3);
    rhos[0] = 5.0;
    rhos[1] = 6.0;
    rhos[2] = 7.2360679774997871;

    // test matrix multiply for every pair of compatible matrices
    for(size_t i = 0; i < matrices.size(); i++)
    {
        SparseMatrix A(matrices[i]);
        float rho = rhos[i];

        ASSERT_EQUAL((std::abs(cusp::eigen::estimate_spectral_radius(A) - rho) / rho) < 0.1f, true);
        ASSERT_EQUAL((std::abs(cusp::eigen::ritz_spectral_radius(A) - rho) / rho) < 0.1f, true);
        ASSERT_EQUAL((std::abs(cusp::eigen::ritz_spectral_radius(A,10,true) - rho) / rho) < 0.1f, true);
        ASSERT_EQUAL((std::abs(cusp::eigen::disks_spectral_radius(A) - rho) / rho) < 0.11f, true);
    }
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestEstimateSpectralRadius);

template <class MemorySpace>
void TestEstimateRhoDinvA(void)
{
    // 2x2 diagonal matrix
    {
        cusp::csr_matrix<int, float, MemorySpace> A(2,2,2);
        A.row_offsets[0] = 0;
        A.row_offsets[1] = 1;
        A.row_offsets[2] = 2;
        A.column_indices[0] = 0;
        A.column_indices[1] = 1;
        A.values[0] = -5;
        A.values[1] =  2;
        float rho = 1.0;
        ASSERT_EQUAL((std::abs(cusp::eigen::estimate_rho_Dinv_A(A) - rho) / rho) < 0.1f, true);
    }

    // 2x2 Poisson problem
    {
        cusp::csr_matrix<int, float, MemorySpace> A;
        cusp::gallery::poisson5pt(A, 2, 2);
        float rho = 1.5;
        ASSERT_EQUAL((std::abs(cusp::eigen::estimate_rho_Dinv_A(A) - rho) / rho) < 0.1f, true);
    }

    // 4x4 Poisson problem
    {
        cusp::csr_matrix<int, float, MemorySpace> A;
        cusp::gallery::poisson5pt(A, 4, 4);
        float rho = 1.8090169943749468;
        ASSERT_EQUAL((std::abs(cusp::eigen::estimate_rho_Dinv_A(A) - rho) / rho) < 0.1f, true);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestEstimateRhoDinvA);

