#pragma once

// TODO create double versions that inspect the device capabilities

// Macro to create host and device versions of a unit test
#define DECLARE_HOST_DEVICE_UNITTEST(VTEST)                    \
void VTEST##Host(void)   {  VTEST< cusp::host_memory   >(); }  \
void VTEST##Device(void) {  VTEST< cusp::device_memory >(); }  \
DECLARE_UNITTEST(VTEST##Host);                                 \
DECLARE_UNITTEST(VTEST##Device);

////////////////
// Containers //
////////////////

// Dense Matrix Containers
#define DECLARE_DENSE_MATRIX_UNITTEST(VTEST)                                                                            \
void VTEST##Array2dRowMajorHost(void)      { VTEST< cusp::array2d<float,cusp::host_memory  ,cusp::row_major   > >(); }  \
void VTEST##Array2dRowMajorDevice(void)    { VTEST< cusp::array2d<float,cusp::device_memory,cusp::row_major   > >(); }  \
void VTEST##Array2dColumnMajorHost(void)   { VTEST< cusp::array2d<float,cusp::host_memory  ,cusp::column_major> >(); }  \
void VTEST##Array2dColumnMajorDevice(void) { VTEST< cusp::array2d<float,cusp::device_memory,cusp::column_major> >(); }  \
DECLARE_UNITTEST(VTEST##Array2dRowMajorHost);                                                                           \
DECLARE_UNITTEST(VTEST##Array2dRowMajorDevice);                                                                         \
DECLARE_UNITTEST(VTEST##Array2dColumnMajorHost);                                                                        \
DECLARE_UNITTEST(VTEST##Array2dColumnMajorDevice);

// Sparse Matrix Containers
#if THRUST_VERSION >= 100800

#ifdef LONG_TESTS
#define DECLARE_SPARSE_FORMAT_UNITTEST(VTEST,Fmt,fmt)                                                               \
void VTEST##Fmt##MatrixHost(void)   { VTEST< cusp::fmt##_matrix<int,float,cusp::host_memory> >();                   \
                                      VTEST< cusp::fmt##_matrix<long long,float,cusp::host_memory> >();             \
                                      VTEST< cusp::fmt##_matrix<int,cusp::complex<float>,cusp::host_memory> >(); }  \
void VTEST##Fmt##MatrixDevice(void) { VTEST< cusp::fmt##_matrix<int,float,cusp::device_memory> >();                 \
                                      VTEST< cusp::fmt##_matrix<long long,float,cusp::device_memory> >();           \
                                      VTEST< cusp::fmt##_matrix<int,cusp::complex<float>,cusp::device_memory> >();} \
DECLARE_UNITTEST(VTEST##Fmt##MatrixHost);                                                                           \
DECLARE_UNITTEST(VTEST##Fmt##MatrixDevice);
#else
#define DECLARE_SPARSE_FORMAT_UNITTEST(VTEST,Fmt,fmt)                                                               \
void VTEST##Fmt##MatrixHost(void)   { VTEST< cusp::fmt##_matrix<int,float,cusp::host_memory> >();}                  \
void VTEST##Fmt##MatrixDevice(void) { VTEST< cusp::fmt##_matrix<int,float,cusp::device_memory> >();}                \
DECLARE_UNITTEST(VTEST##Fmt##MatrixHost);                                                                           \
DECLARE_UNITTEST(VTEST##Fmt##MatrixDevice);
#endif

#else

#define DECLARE_SPARSE_FORMAT_UNITTEST(VTEST,Fmt,fmt)                                                                                                                       \
void VTEST##Fmt##MatrixHost(void)   { VTEST< cusp::fmt##_matrix<int,float,cusp::host_memory> >();    VTEST< cusp::fmt##_matrix<long long,float,cusp::host_memory> >();   }  \
void VTEST##Fmt##MatrixDevice(void) { VTEST< cusp::fmt##_matrix<int,float,cusp::device_memory> >();  VTEST< cusp::fmt##_matrix<long long,float,cusp::device_memory> >(); }  \
DECLARE_UNITTEST(VTEST##Fmt##MatrixHost);                                                                                                                                   \
DECLARE_UNITTEST(VTEST##Fmt##MatrixDevice);

#endif

#define DECLARE_SPARSE_MATRIX_UNITTEST(VTEST) \
DECLARE_SPARSE_FORMAT_UNITTEST(VTEST,Coo,coo) \
DECLARE_SPARSE_FORMAT_UNITTEST(VTEST,Csr,csr) \
DECLARE_SPARSE_FORMAT_UNITTEST(VTEST,Dia,dia) \
DECLARE_SPARSE_FORMAT_UNITTEST(VTEST,Ell,ell) \
DECLARE_SPARSE_FORMAT_UNITTEST(VTEST,Hyb,hyb)

// All Matrix Containers
#define DECLARE_MATRIX_UNITTEST(VTEST) \
DECLARE_DENSE_MATRIX_UNITTEST(VTEST);  \
DECLARE_SPARSE_MATRIX_UNITTEST(VTEST);

///////////
// Views //
///////////

// Sparse Matrix Views
#define DECLARE_SPARSE_FORMAT_VIEW_UNITTEST(VTEST,Fmt,fmt)                                                                                                                                      \
void VTEST##Fmt##MatrixViewHost(void)   { VTEST< cusp::fmt##_matrix<int,float,cusp::host_memory>::view   >(); VTEST< cusp::fmt##_matrix<long long,float,cusp::host_memory  >::view >(); }  \
void VTEST##Fmt##MatrixViewDevice(void) { VTEST< cusp::fmt##_matrix<int,float,cusp::device_memory>::view >(); VTEST< cusp::fmt##_matrix<long long,float,cusp::device_memory>::view >(); }  \
DECLARE_UNITTEST(VTEST##Fmt##MatrixViewHost);                                                                                                                                                  \
DECLARE_UNITTEST(VTEST##Fmt##MatrixViewDevice);

