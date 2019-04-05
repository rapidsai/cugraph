/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GRAPH_CONTRACTING_STRUCTS_HXX
#define GRAPH_CONTRACTING_STRUCTS_HXX

#include <nvgraph_error.hxx>
#include <multi_valued_csr_graph.hxx> //which includes all other headers... 
#include <range_view.hxx> // TODO: to be changed to thrust/range_view.h, when toolkit gets in sync with Thrust

#include <thrust_traits.hxx>

//from amgx/amg/base/include/sm_utils.inl
//{
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __PTR   "l"
#else
#define __PTR   "r"
#endif
//}

namespace nvgraph
{
  //from amgx/amg/base/include/sm_utils.inl
  //{
  namespace utils
  {


	// ====================================================================================================================
	// Warp tools.
	// ====================================================================================================================

	static __device__ __forceinline__ int lane_id() 
	{
	  int id;
	  asm( "mov.u32 %0, %%laneid;" : "=r"(id) );
	  return id;
	}

	static __device__ __forceinline__ int lane_mask_lt() 
	{
	  int mask;
	  asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(mask) );
	  return mask;
	}

	static __device__ __forceinline__ int warp_id() 
	{
	  return threadIdx.x >> 5;
	}


	// ====================================================================================================================
	// Atomics.
	// ====================================================================================================================
	static __device__ __forceinline__ void atomic_add( float *address, float value )
	{
	  atomicAdd( address, value );
	}

	static __device__ __forceinline__ void atomic_add( double *address, double value )
	{
	  unsigned long long *address_as_ull = (unsigned long long *) address; 
	  unsigned long long old = __double_as_longlong( address[0] ), assumed; 
	  do { 
		assumed = old; 
		old = atomicCAS( address_as_ull, assumed, __double_as_longlong( value + __longlong_as_double( assumed ) ) ); 
	  } 
	  while( assumed != old ); 
	}


	// ====================================================================================================================
	// Bit tools.
	// ====================================================================================================================

	static __device__ __forceinline__ int bfe( int src, int num_bits ) 
	{
	  unsigned mask;
	  asm( "bfe.u32 %0, %1, 0, %2;" : "=r"(mask) : "r"(src), "r"(num_bits) );
	  return mask;
	}

	static __device__ __forceinline__ int bfind( int src ) 
	{
	  int msb;
	  asm( "bfind.u32 %0, %1;" : "=r"(msb) : "r"(src) );
	  return msb;
	}

	static __device__ __forceinline__ int bfind( unsigned long long src ) 
	{
	  int msb;
	  asm( "bfind.u64 %0, %1;" : "=r"(msb) : "l"(src) );
	  return msb;
	}



	// ====================================================================================================================
	// Shuffle.
	// ====================================================================================================================
	static __device__ __forceinline__ float shfl( float r, int lane, int bound = 32)
	{
#if __CUDA_ARCH__ >= 300
	  return __shfl( r, lane, bound );
#else
	  return 0.0f;
#endif
	}

	static __device__ __forceinline__ double shfl( double r, int lane, int bound=32 )
	{
#if __CUDA_ARCH__ >= 300
	  int hi = __shfl( __double2hiint(r), lane, bound );
	  int lo = __shfl( __double2loint(r), lane, bound );
	  return __hiloint2double( hi, lo );
#else
	  return 0.0;
#endif
	}

        static __device__ __forceinline__ float shfl_xor( float r, int mask, int bound=32 )
	{
#if __CUDA_ARCH__ >= 300
	  return __shfl_xor( r, mask, bound );
#else
	  return 0.0f;
#endif
	}

        static __device__ __forceinline__ double shfl_xor( double r, int mask, int bound=32 )
	{
#if __CUDA_ARCH__ >= 300
	  int hi = __shfl_xor( __double2hiint(r), mask, bound );
	  int lo = __shfl_xor( __double2loint(r), mask, bound );
	  return __hiloint2double( hi, lo );
#else
	  return 0.0;
#endif
	}



	// ====================================================================================================================
	// Loads.
	// ====================================================================================================================

	enum Ld_mode { LD_AUTO = 0, LD_CA, LD_CG, LD_TEX, LD_NC };

	template< Ld_mode Mode >
	struct Ld {};

	template<>
	struct Ld<LD_AUTO> 
	{ 
	  template< typename T >
	  static __device__ __forceinline__ T load( const T *ptr ) { return *ptr; }
	};

	template<>
	struct Ld<LD_CG> 
	{ 
	  static __device__ __forceinline__ int load( const int *ptr ) 
	  { 
		int ret; 
		asm volatile ( "ld.global.cg.s32 %0, [%1];"  : "=r"(ret) : __PTR(ptr) ); 
		return ret; 
	  }
  
	  static __device__ __forceinline__ float load( const float *ptr ) 
	  { 
		float ret; 
		asm volatile ( "ld.global.cg.f32 %0, [%1];"  : "=f"(ret) : __PTR(ptr) ); 
		return ret; 
	  }
  
	  static __device__ __forceinline__ double load( const double *ptr ) 
	  { 
		double ret; 
		asm volatile ( "ld.global.cg.f64 %0, [%1];"  : "=d"(ret) : __PTR(ptr) ); 
		return ret; 
	  }

	};

	template<>
	struct Ld<LD_CA> 
	{ 
	  static __device__ __forceinline__ int load( const int *ptr ) 
	  { 
		int ret; 
		asm volatile ( "ld.global.ca.s32 %0, [%1];"  : "=r"(ret) : __PTR(ptr) ); 
		return ret; 
	  }
  
	  static __device__ __forceinline__ float load( const float *ptr ) 
	  { 
		float ret; 
		asm volatile ( "ld.global.ca.f32 %0, [%1];"  : "=f"(ret) : __PTR(ptr) ); 
		return ret; 
	  }
  
	  static __device__ __forceinline__ double load( const double *ptr ) 
	  { 
		double ret; 
		asm volatile ( "ld.global.ca.f64 %0, [%1];"  : "=d"(ret) : __PTR(ptr) ); 
		return ret; 
	  }
	};

	template<>
	struct Ld<LD_NC> 
	{ 
	  template< typename T >
	  static __device__ __forceinline__ T load( const T *ptr ) { return __ldg( ptr ); }
	};


	template < typename T, typename POD_TYPE = T >
    struct util;
   
    template <>
    struct util <float,  float >
    {
        typedef double uptype;
        typedef float downtype;

        static const bool is_real = true;
        static const bool is_complex = false;

        static __host__ __device__ __inline__ float get_zero(){ return 0.f; }
        static __host__ __device__ __inline__ float get_one(){ return 1.f; }
        static __host__ __device__ __inline__ float get_minus_one(){ return -1.f; }
        // exact comaprison, which might result wrong answer in a lot of cases
        static __host__ __device__ __inline__ bool is_zero(const float& val){ return val == get_zero(); }
        static __host__ __device__ __inline__ bool is_equal(const float& val1, const float& val2) { return val1 == val2;} ;
        
        static __host__ __device__ __inline__ float invert(const float& val) {return -val;}
        static __host__ __device__ __inline__ float conjugate(const float& val) {return val;}
        static __host__ __device__ __inline__ void  invert_inplace(float& val) {val = -val;}
        static __host__ __device__ __inline__ void  conjugate_inplace(float& val) {}

        static __host__ __device__ __inline__ float abs (const float& val)
        {
            return fabs(val);
        }

        template <typename V>
        static __host__ __device__ __inline__ void to_uptype (const float& src, V& dst)
        {
            dst = (V)(src);
        }

        static __host__ __device__ __inline__ float to_downtype (const float& src)
        {
            return src;
        }

        static __host__ __device__ __inline__ float volcast (const volatile    float& val) {return val;}
        static __host__ __device__ __inline__ void  volcast (const float& val, volatile float* ret) {*ret = val;}

        /*template <typename M>
        static __host__ __device__ __inline__ float mul(const float& val, const M& mult)
        { 
            static_assert(util<M>::is_real(), "Multiply is supported for real constant only"); 
            return val*mult;
        }*/
        
        static void printf(const char* fmt, const float& val) { ::printf(fmt, val); }
        static void fprintf(FILE* f, const char* fmt, const float& val) { ::fprintf(f, fmt, val); }
    };

    template <>
    struct util <double, double>
    {
        typedef double uptype;
        typedef float downtype;

        static const bool is_real = true;
        static const bool is_complex = false;
        
        static __host__ __device__ __inline__ double get_zero(){ return 0.; }
        static __host__ __device__ __inline__ double get_one(){ return 1.; }
        static __host__ __device__ __inline__ double get_minus_one(){ return -1.; }
        
        static __host__ __device__ __inline__ bool is_zero(const double& val){ return val == get_zero(); }
        static __host__ __device__ __inline__ bool is_equal(const double& val1, double& val2) { return val1 == val2;} ;

        static __host__ __device__ __inline__ double invert(const double& val) {return -val;}
        static __host__ __device__ __inline__ double conjugate(const double& val) {return val;}
        static __host__ __device__ __inline__ void invert_inplace(double& val) {val = -val;}
        static __host__ __device__ __inline__ void conjugate_inplace(double& val) {}

        static __host__ __device__ __inline__ double abs (const double& val)
        {
            return fabs(val);
        }

        template <typename V>
        static __host__ __device__ __inline__ void to_uptype (const float& src, V& dst)
        {
            dst = (V)(src);
        }

        static __host__ __device__ __inline__ float to_downtype (const float& src)
        {
            return (float)src;
        }

        static __host__ __device__ __inline__ double volcast (const volatile   double& val) {return val;}
        static __host__ __device__ __inline__ void   volcast (const double& val, volatile double* ret) {*ret = val;}

        /*
        template <typename M>
        static __host__ __device__ __inline__ double mulf(const double& val, const M& mult) 
        { 
            static_assert(util<M>::is_real(), "Multiply is supported for real constant only"); 
            return val*mult;
        }*/
        
        static void printf(const char* fmt, const double& val) { ::printf(fmt, val); }
        static void fprintf(FILE* f, const char* fmt,const double& val) { ::fprintf(f, fmt, val); }
    };


	// ====================================================================================================================
	// Warp-level reductions.
	// ====================================================================================================================

	struct Add
	{
	  template< typename Value_type >
	  static __device__ __forceinline__ Value_type eval( Value_type x, Value_type y ) { return x+y; }
	};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300

	template< int NUM_THREADS_PER_ITEM, int WARP_SIZE >
	struct Warp_reduce_pow2
	{
	  template< typename Operator, typename Value_type >
	  static __device__ __inline__ Value_type execute( Value_type x )
	  {
#pragma unroll
		for( int mask = WARP_SIZE / 2 ; mask >= NUM_THREADS_PER_ITEM ; mask >>= 1 )
		  x = Operator::eval( x, shfl_xor(x, mask) );
		return x;
	  }
	};

	template< int NUM_THREADS_PER_ITEM, int WARP_SIZE >
	struct Warp_reduce_linear
	{
	  template< typename Operator, typename Value_type >
	  static __device__ __inline__ Value_type execute( Value_type x )
	  {
		const int NUM_STEPS = WARP_SIZE / NUM_THREADS_PER_ITEM;
		int my_lane_id = utils::lane_id();
#pragma unroll
		for( int i = 1 ; i < NUM_STEPS ; ++i )
		  {
			Value_type y = shfl_down( x, i*NUM_THREADS_PER_ITEM );
			if( my_lane_id < NUM_THREADS_PER_ITEM )
			  x = Operator::eval( x, y );
		  }
		return x;
	  }
	};

#else

	template< int NUM_THREADS_PER_ITEM, int WARP_SIZE >
	struct Warp_reduce_pow2
	{
	  template< typename Operator, typename Value_type >
	  static __device__ __inline__ Value_type execute( volatile Value_type *smem, Value_type x )
	  {
		int my_lane_id = utils::lane_id();
#pragma unroll
		for( int offset = WARP_SIZE / 2 ; offset >= NUM_THREADS_PER_ITEM ; offset >>= 1 )
		  if( my_lane_id < offset )
			{
			  x = Operator::eval( x, smem[threadIdx.x+offset] );
			  util<Value_type>::volcast(x, smem + threadIdx.x);
			}
		return x;
	  }
	};

	template< int NUM_THREADS_PER_ITEM, int WARP_SIZE >
	struct Warp_reduce_linear
	{
	  template< typename Operator, typename Value_type >
	  static __device__ __inline__ Value_type execute( volatile Value_type *smem, Value_type x )
	  {
		const int NUM_STEPS = WARP_SIZE / NUM_THREADS_PER_ITEM;
		int my_lane_id = utils::lane_id();
#pragma unroll
		for( int i = 1 ; i < NUM_STEPS ; ++i )
		  if( my_lane_id < NUM_THREADS_PER_ITEM )
			{
			  x = Operator::eval( x, smem[threadIdx.x+i*NUM_THREADS_PER_ITEM] );
			  util<Value_type>::volcast(x, smem + threadIdx.x);
			}
		return x;
	  }
	};

#endif

	// ====================================================================================================================

	template< int NUM_THREADS_PER_ITEM, int WARP_SIZE = 32 >
	struct Warp_reduce : public Warp_reduce_pow2<NUM_THREADS_PER_ITEM, WARP_SIZE> {};

	template< int WARP_SIZE >
	struct Warp_reduce< 3, WARP_SIZE> : public Warp_reduce_linear< 3, WARP_SIZE> {};

	template< int WARP_SIZE >
	struct Warp_reduce< 4, WARP_SIZE> : public Warp_reduce_linear< 4, WARP_SIZE> {};

	template< int WARP_SIZE >
	struct Warp_reduce< 5, WARP_SIZE> : public Warp_reduce_linear< 5, WARP_SIZE> {};

	template< int WARP_SIZE >
	struct Warp_reduce< 6, WARP_SIZE> : public Warp_reduce_linear< 6, WARP_SIZE> {};

	template< int WARP_SIZE >
	struct Warp_reduce< 7, WARP_SIZE> : public Warp_reduce_linear< 7, WARP_SIZE> {};

	template< int WARP_SIZE >
	struct Warp_reduce< 9, WARP_SIZE> : public Warp_reduce_linear< 9, WARP_SIZE> {};

	template< int WARP_SIZE >
	struct Warp_reduce<10, WARP_SIZE> : public Warp_reduce_linear<10, WARP_SIZE> {};

	template< int WARP_SIZE >
	struct Warp_reduce<11, WARP_SIZE> : public Warp_reduce_linear<11, WARP_SIZE> {};

	template< int WARP_SIZE >
	struct Warp_reduce<12, WARP_SIZE> : public Warp_reduce_linear<12, WARP_SIZE> {};

	template< int WARP_SIZE >
	struct Warp_reduce<13, WARP_SIZE> : public Warp_reduce_linear<13, WARP_SIZE> {};

	template< int WARP_SIZE >
	struct Warp_reduce<14, WARP_SIZE> : public Warp_reduce_linear<14, WARP_SIZE> {};

	template< int WARP_SIZE >
	struct Warp_reduce<15, WARP_SIZE> : public Warp_reduce_linear<15, WARP_SIZE> {};

	// ====================================================================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300

	template< int NUM_THREADS_PER_ITEM, typename Operator, typename Value_type >
	static __device__ __forceinline__ Value_type warp_reduce( Value_type x )
	{
	  return Warp_reduce<NUM_THREADS_PER_ITEM>::template execute<Operator>( x );
	}

#else

	template< int NUM_THREADS_PER_ITEM, typename Operator, typename Value_type >
	static __device__ __forceinline__ Value_type warp_reduce( volatile Value_type *smem, Value_type x )
	{
	  return Warp_reduce<NUM_THREADS_PER_ITEM>::template execute<Operator>( smem, x );
	}

	template< int NUM_THREADS_PER_ITEM, typename Value_type, int WARP_SIZE >
	static __device__ __forceinline__ Value_type warp_reduce_sum(volatile Value_type *smem, Value_type x)
	{
	  const int NUM_STEPS = WARP_SIZE / NUM_THREADS_PER_ITEM;
	  int my_lane_id = utils::lane_id();
#pragma unroll
	  for (int i = 1; i < NUM_STEPS; ++i)
		if (my_lane_id < NUM_THREADS_PER_ITEM)
		  {
			x = x + util<Value_type>::volcast(smem[threadIdx.x + i*NUM_THREADS_PER_ITEM]);
			util<Value_type>::volcast(x, smem + threadIdx.x);
		  }
	  return x;
	}

#endif


	
  }//namespace utils
  //}


  template< typename Key_type, int SMEM_SIZE=128, int WARP_SIZE=32 >
  class Hash_index
  {
  public:
    // The number of registers needed to store the index. 
    enum { REGS_SIZE = SMEM_SIZE / WARP_SIZE };

    //private:
    // The partial sums of the index (stored in registers).
    int m_partial[REGS_SIZE];
    // The index in GMEM.
    int *m_gmem;

  public:
    // Create an index (to be associated with a hash set).
    __device__ __forceinline__ Hash_index( int *gmem ) : m_gmem(gmem) {}

    // Build the index from a SMEM buffer of size SMEM_SIZE.
    __device__ __forceinline__ void build_smem_index( const volatile Key_type *s_buffer );
    // Given an offset in SMEM, it finds the index.
    __device__ __forceinline__ int find_smem( int offset ) const;
    // Given an offset in GMEM, it finds the index.
    __device__ __forceinline__ int find_gmem( int offset ) const;
    // Set an indexed item in GMEM.
    __device__ __forceinline__ void set_gmem_index( int offset, int val ) { m_gmem[offset] = val; }
  };

  // ====================================================================================================================

  template< typename Key_type, int SMEM_SIZE, int WARP_SIZE >
  __device__ __forceinline__ 
  void 
  Hash_index<Key_type, SMEM_SIZE, WARP_SIZE>::build_smem_index( const volatile Key_type *s_buffer )
  {
    const int lane_id = utils::lane_id();
#pragma unroll
    for( int i = 0, offset = lane_id ; i < REGS_SIZE ; ++i, offset += WARP_SIZE )
      m_partial[i] = __ballot( s_buffer[offset] != -1 );
  }

  // ====================================================================================================================

  template< typename Key_type, int SMEM_SIZE, int WARP_SIZE >
  __device__ __forceinline__ 
  int 
  Hash_index<Key_type, SMEM_SIZE, WARP_SIZE>::find_smem( int offset ) const
  {
    const int offset_div_warp_size = offset / WARP_SIZE;
    const int offset_mod_warp_size = offset % WARP_SIZE;

    int result = 0;
#pragma unroll
    for( int i = 0 ; i < REGS_SIZE ; ++i )
      {
	int mask = 0xffffffff;
	if( i == offset_div_warp_size )
	  mask = (1 << offset_mod_warp_size) - 1;
	if( i <= offset_div_warp_size )
	  result += __popc( m_partial[i] & mask );
      }
    return result;
  }

  template< typename Key_type, int SMEM_SIZE, int WARP_SIZE >
  __device__ __forceinline__ 
  int 
  Hash_index<Key_type, SMEM_SIZE, WARP_SIZE>::find_gmem( int offset ) const
  {
    return m_gmem[offset];
  }


  
  static __constant__ unsigned c_hash_keys[] = 
    { 
      3499211612,  581869302, 3890346734, 3586334585,  
      545404204,  4161255391, 3922919429,  949333985,
      2715962298, 1323567403,  418932835, 2350294565, 
      1196140740,  809094426, 2348838239, 4264392720 
    };

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  template< typename Key_type, int SMEM_SIZE=128, int NUM_HASH_FCTS=4, int WARP_SIZE=32 >
  class Hash_set
  {
    // Associated index.
    typedef Hash_index<Key_type, SMEM_SIZE, WARP_SIZE> Index;

  protected:
    // The size of the table (occupancy).
    int m_smem_count, m_gmem_count;
    // The keys stored in the hash table.
    volatile Key_type *m_smem_keys, *m_gmem_keys; 
    // The size of the global memory buffer.
    const int m_gmem_size;
    // Is it ok?
    bool m_fail;

    // DEBUG
    // bool m_print;
    // END OF DEBUG.
  
  public:
    // Constructor.
    __device__ __forceinline__ Hash_set( volatile Key_type *smem_keys, volatile Key_type *gmem_keys, int gmem_size ) :
      m_smem_count(0),
      m_gmem_count(1),
      m_smem_keys (smem_keys),
      m_gmem_keys (gmem_keys),
      m_gmem_size (gmem_size),
      m_fail      (false)

      // DEBUG
      // , m_print(true)
      // END OF DEBUG
    {}
  
    // Clear the table.
    __device__ __forceinline__ void clear( bool skip_gmem = false );
    // Compute the size of the table. Only thread with lane_id==0 gives the correct result (no broadcast of the value).
    __device__ __forceinline__ int compute_size();
    // Compute the size of the table. Only thread with lane_id==0 gives the correct result (no broadcast of the value).
    __device__ __forceinline__ int compute_size_with_duplicates();
    // Does the set contain those values?
    __device__ __forceinline__ bool contains( Key_type key ) const;
    // Find an index.
    __device__ __forceinline__ int find_index( Key_type key, const Index &index, bool print_debug ) const;
    // Has the process failed.
    __device__ __forceinline__ bool has_failed() const { return m_fail; }
    // Insert a key inside the set. If status is NULL, ignore failure.
    __device__ __forceinline__ void insert( Key_type key, int *status );
    // Load a set.
    __device__ __forceinline__ void load( int count, const Key_type *keys, const int *pos );
    // Load a set and use it as an index. 
    __device__ __forceinline__ void load_index( int count, const Key_type *keys, const int *pos, Index &index, bool print_debug );
    // Store a set.
    __device__ __forceinline__ void store( int count, Key_type *keys );
    // Store a set.
    __device__ __forceinline__ int  store_with_positions( Key_type *keys, int *pos );
    // Store a set.
    __device__ __forceinline__ int  store( Key_type *keys );
  };

  // ====================================================================================================================

  template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE>
  __device__ __forceinline__ 
  void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::clear( bool skip_gmem )
  {
    int lane_id = utils::lane_id();
  
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll
    for( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
      m_smem_keys[i_step*WARP_SIZE + lane_id] = -1;
    m_smem_count = 0;
  
    if( skip_gmem || m_gmem_count == 0 )
      {
	m_gmem_count = 0;
	return;
      }
    
#pragma unroll 4
    for( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
      m_gmem_keys[offset] = -1;
    m_gmem_count = 0;
  }

  // ====================================================================================================================

  template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE>
  __device__ __forceinline__ 
  int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::compute_size()
  {
    m_smem_count += m_gmem_count;
#pragma unroll
    for( int offset = WARP_SIZE/2 ; offset > 0 ; offset >>= 1 )
      m_smem_count += __shfl_xor( m_smem_count, offset );
    m_gmem_count = __any( m_gmem_count > 0 );
    return m_smem_count;
  }

  // ====================================================================================================================

  template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE>
  __device__ __forceinline__ 
  int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::compute_size_with_duplicates()
  {
    int lane_id = utils::lane_id();

    // Count the number of keys in SMEM.
    int sum = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll
    for( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
      {
	const int offset = i_step*WARP_SIZE + lane_id;
	Key_type key = m_smem_keys[offset];
	sum += __popc( __ballot( key != -1 ) );
      }

    // Is there any key in GMEM. If not, just quit.
    m_gmem_count = __any(m_gmem_count > 0);
    if( !m_gmem_count )
      return sum;

    // Count the number of keys in GMEM.
#pragma unroll 4
    for( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
      {
	Key_type key = m_gmem_keys[offset];
	sum += __popc( __ballot( key != -1 ) );
      }
    return sum;
  }

  // ====================================================================================================================

  template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE>
  __device__ __forceinline__ 
  bool Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::contains( Key_type key ) const
  {
    bool done = key == -1, found = false;
#pragma unroll
    for( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
      {
	if( __all(done) )
	  return found;
	unsigned ukey = reinterpret_cast<unsigned&>( key );
	int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE-1);
	if( !done )
	  {
	    Key_type stored_key = m_smem_keys[hash];
	    if( stored_key == key )
	      found = true;
	    if( found || stored_key == -1 )
	      done = true;
	  }
      }

    const int num_bits = utils::bfind( m_gmem_size ); // TODO: move it outside ::insert.
#pragma unroll
    for( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
      {
	if( __all(done) )
	  return found;
	unsigned ukey = reinterpret_cast<unsigned&>( key );
	int hash = utils::bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );
	if( !done )
	  {
	    Key_type stored_key = m_gmem_keys[hash];
	    if( stored_key == key )
	      found = true;
	    if( found || stored_key == -1 )
	      done = true;
	  }
      }
    return found;
  }

  // ====================================================================================================================

  template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::find_index( Key_type key, const Index &index, bool print_debug ) const
  {
    int idx = -1;
    bool done = key == -1;
#pragma unroll
    for( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
      {
	if( __all(done) )
	  return idx;
	unsigned ukey = reinterpret_cast<unsigned&>( key );
	int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE-1);
	int result = index.find_smem(hash);
	if( !done )
	  {
	    Key_type stored_key = m_smem_keys[hash];
	    if( stored_key == key )
	      {
		idx = result;
		done = true;
	      }
	  }
      }

    const int num_bits = utils::bfind( m_gmem_size ); // TODO: move it outside ::insert.
#pragma unroll
    for( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
      {
	if( __all(done) )
	  return idx;
	unsigned ukey = reinterpret_cast<unsigned&>( key );
	int hash = utils::bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );
	if( !done )
	  {
	    Key_type stored_key = m_gmem_keys[hash];
	    if( stored_key == key )
	      {
		idx = index.find_gmem(hash);
		done = true;
	      }
	  }
      }

    // if( key != -1 && idx == -1 )
    //   printf( "ERROR: Couldn't find the index!!!!\n");
    return idx;
  }

  // ====================================================================================================================

  template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::insert( Key_type key, int *status )
  {
    bool done = key == -1;
#pragma unroll
    for( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
      {
	if( __all(done) )
	  return;
	bool candidate = false;
	unsigned ukey = reinterpret_cast<unsigned&>( key );
	int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE-1);
	if( !done )
	  {
	    Key_type stored_key = m_smem_keys[hash];
	    if( stored_key == key )
	      done = true;
	    candidate = stored_key == -1;
	    if( candidate )
	      m_smem_keys[hash] = key;
	    if( candidate && key == m_smem_keys[hash] ) // More than one candidate may have written to that slot.
	      {
		m_smem_count++;
		done = true;
	      }
	  }
      }

    const int num_bits = utils::bfind( m_gmem_size ); // TODO: move it outside ::insert.
#pragma unroll
    for( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
      {
	if( __all(done) )
	  return;
	bool candidate = false;
	unsigned ukey = reinterpret_cast<unsigned&>( key );
	int hash = utils::bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );
	if( !done )
	  {
	    Key_type stored_key = m_gmem_keys[hash];
	    if( stored_key == key )
	      done = true;
	    candidate = stored_key == -1;
	    if( candidate )
	      m_gmem_keys[hash] = key;
	    if( candidate && key == m_gmem_keys[hash] ) // More than one candidate may have written to that slot.
	      {
		m_gmem_count++;
		done = true;
	      }
	  }
      }

    if( __all(done) )
      return;
    assert( status != NULL );
    if( utils::lane_id() == 0 )
      *status = 1;
    m_fail = true;
  }

  // ====================================================================================================================

  template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::load( int count, const Key_type *keys, const int *pos )
  {
    int lane_id = utils::lane_id();

#pragma unroll 4
    for( int offset = lane_id ; offset < count ; offset += WARP_SIZE )
      {
	Key_type key = keys[offset];
	int idx = pos [offset];

	// Where to store the item.
	volatile Key_type *ptr = m_smem_keys;
	if( idx >= SMEM_SIZE )
	  {
	    ptr = m_gmem_keys;
	    m_gmem_count = 1;
	    idx -= SMEM_SIZE;
	  }

	// Store the item.
	ptr[idx] = key;
      }
    m_gmem_count = __any( m_gmem_count );
  }

  // ====================================================================================================================

  template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::load_index( int count, const Key_type *keys, const int *pos, Index &index, bool print_debug )
  {
#pragma unroll 4
    for( int offset = utils::lane_id() ; offset < count ; offset += WARP_SIZE  )
      {
	Key_type key = keys[offset];
	int idx = pos [offset];

	// Store the item.
	volatile Key_type *ptr = m_smem_keys;
	if( idx >= SMEM_SIZE )
	  {
	    ptr = m_gmem_keys;
	    m_gmem_count = 1;
	    idx -= SMEM_SIZE;
	    index.set_gmem_index( idx, offset );
	  }

	// Store the item.
	ptr[idx] = key;
      }

    // Build the local index.
    index.build_smem_index( m_smem_keys );
    m_gmem_count = __any( m_gmem_count );
  }

  // ====================================================================================================================

  template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( int count, Key_type *keys )
  {
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();

    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll
    for( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
      {
	const int offset = i_step*WARP_SIZE + lane_id;
	Key_type key = m_smem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  keys[dst_offset] = key;
	warp_offset += __popc( poll );
      }

    m_gmem_count = __any( m_gmem_count > 0 );
    if( !m_gmem_count )
      return;

#pragma unroll 4
    for( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
      {
	Key_type key = m_gmem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  keys[dst_offset] = key;
	warp_offset += __popc( poll );
      }
  }

  // ====================================================================================================================

  template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store_with_positions( Key_type *keys, int *pos )
  {
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();

    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll
    for( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
      {
	const int offset = i_step*WARP_SIZE + lane_id;
	Key_type key = m_smem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  {
	    keys[dst_offset] = key;
	    pos [dst_offset] = offset;
	  }
	warp_offset += __popc( poll );
      }

    m_gmem_count = __any( m_gmem_count > 0 );
    if( !m_gmem_count )
      return warp_offset;

#pragma unroll 4
    for( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
      {
	Key_type key = m_gmem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  {
	    keys[dst_offset] = key;
	    pos [dst_offset] = SMEM_SIZE + offset;
	  }
	warp_offset += __popc( poll );
      }
    return warp_offset;
  }


  template< typename Key_type, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  int Hash_set<Key_type, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( Key_type *keys )
  {
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();

    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll
    for( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
      {
	const int offset = i_step*WARP_SIZE + lane_id;
	Key_type key = m_smem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  {
	    keys[dst_offset] = key;
	  }
	warp_offset += __popc( poll );
      }

    m_gmem_count = __any( m_gmem_count > 0 );
    if( !m_gmem_count )
      return warp_offset;

#pragma unroll 4
    for( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
      {
	Key_type key = m_gmem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  {
	    keys[dst_offset] = key;
	  }
	warp_offset += __popc( poll );
      }
    return warp_offset;
  }


  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  union Word { char b8[4]; int b32; };

  // ====================================================================================================================

  template< typename Key_type, typename T, int SMEM_SIZE=128, int NUM_HASH_FCTS=4, int WARP_SIZE=32 >
  class Hash_map
  {
  protected:
    // The keys stored in the map.
    volatile Key_type *m_smem_keys, *m_gmem_keys; 
    // Vote buffer for values.
    volatile Word *m_smem_vote;
    // Registers to store values.
    T m_regs_vals[4];
    // The values stored in the map.
    T *m_gmem_vals;
    // The size of the global memory buffer.
    const int m_gmem_size;
    // Is there any value in GMEM.
    bool m_any_gmem;
  
  public:
    // Constructor.
    __device__ __forceinline__ 
    Hash_map( volatile Key_type *smem_keys, volatile Key_type *gmem_keys, volatile Word *smem_vote, T *gmem_vals, int gmem_size ) :
      m_smem_keys(smem_keys),
      m_gmem_keys(gmem_keys),
      m_smem_vote(smem_vote),
      m_gmem_vals(gmem_vals),
      m_gmem_size(gmem_size),
      m_any_gmem (true)
    {}
  
    // Clear the table. It doesn't clear GMEM values.
    __device__ __forceinline__ void clear();
    // Clear the table. It also clears GMEM values (set them to 0).
    __device__ __forceinline__ void clear_all();
    // Insert a key/value inside the hash table.
    __device__ __forceinline__ void insert( Key_type key, T a_value, T b_value, int *status );
    // Insert a key/value inside the hash table.
    __device__ __forceinline__ void insert_with_duplicates( Key_type key, T val, int *status );
    // Load a set.
    __device__ __forceinline__ void load( int count, const Key_type *keys, const int *pos );
    // Store the map.
    __device__ __forceinline__ void store( int count, T *vals );
    // Store the map.
    __device__ __forceinline__ void store( int count, Key_type *keys, T *vals );
    // Store the map.
    __device__ __forceinline__ void store_map_keys_scale_values( int count, const int *map, Key_type *keys, T alpha, T *vals );
    // Store the map.
    __device__ __forceinline__ void store_keys_scale_values( int count, Key_type *keys, T alpha, T *vals );
    // Update a value in the table but do not insert if it doesn't exist.
    __device__ __forceinline__ bool update( Key_type key, T value );

  protected:
    // Get the selected item in the register buffer.
    __device__ __forceinline__ int get_selected( int hash ) const 
    { 
      return static_cast<int>(m_smem_vote[hash%WARP_SIZE].b8[hash/WARP_SIZE]); 
    }

    // Is it the selected item in the register buffer.
    __device__ __forceinline__ bool is_selected( int hash, int lane_id ) const 
    { 
      return m_smem_vote[hash%WARP_SIZE].b8[hash/WARP_SIZE] == reinterpret_cast<char&>(lane_id); 
    }

    // Push my ID in the register buffer.
    __device__ __forceinline__ void try_selection( int hash, int lane_id ) 
    { 
      m_smem_vote[hash%WARP_SIZE].b8[hash/WARP_SIZE] = reinterpret_cast<char&>(lane_id); 
    }
  };

  // ====================================================================================================================

  template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::clear()
  {
    int lane_id = utils::lane_id();

    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll
    for( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
      m_smem_keys[i_step*WARP_SIZE + lane_id] = -1;

#pragma unroll
    for( int i_regs = 0 ; i_regs < 4 ; ++i_regs )
      m_regs_vals[i_regs] = T(0);

    if( !m_any_gmem )
      return;

#pragma unroll 4
    for( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
      m_gmem_keys[offset] = -1;
    m_any_gmem = false;
  }

  // ====================================================================================================================

  template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::clear_all()
  {
    int lane_id = utils::lane_id();

    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll
    for( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
      m_smem_keys[i_step*WARP_SIZE + lane_id] = -1;

#pragma unroll
    for( int i_regs = 0 ; i_regs < 4 ; ++i_regs )
      m_regs_vals[i_regs] = T(0);

    if( !m_any_gmem )
      return;

#pragma unroll 4
    for( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
      {
		m_gmem_keys[offset] =   -1;
		m_gmem_vals[offset] = T(0);
      }
    m_any_gmem = false;
  }

  // ====================================================================================================================

  template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::insert( Key_type key, T a_value, T b_value, int *status )
  {
    const int lane_id = utils::lane_id();
    bool done = key == -1;

    m_smem_vote[lane_id].b32 = 0x20202020;
#pragma unroll
    for( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
      {
	if( i_hash > 0 && __all(done) )
	  break;
	bool candidate = false;
	unsigned ukey = reinterpret_cast<unsigned&>( key );
	int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE-1);
	if( !done )
	  {
	    Key_type stored_key = m_smem_keys[hash];
	    if( stored_key == key )
	      {
		this->try_selection( hash, lane_id );
		done = true;
	      }
	    candidate = stored_key == -1;
	    if( candidate )
	      m_smem_keys[hash] = key;
	    if( candidate && key == m_smem_keys[hash] )
	      {
		this->try_selection( hash, lane_id );
		done = true;
	      }
	  }
      }

    Word my_vote;
    my_vote.b32 = m_smem_vote[lane_id].b32;
#pragma unroll
    for( int i_regs = 0 ; i_regs < 4 ; ++i_regs )
      {
	int my_src = my_vote.b8[i_regs];
	T other_val = utils::shfl( b_value, my_src );
	if( my_src != WARP_SIZE ) 
	  m_regs_vals[i_regs] = m_regs_vals[i_regs] + a_value * other_val;
      }

    const int num_bits = utils::bfind( m_gmem_size ); // TODO: move it outside ::insert.
#pragma unroll
    for( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
      {
	if( __all(done) )
	  return;
	m_any_gmem = true;
	bool candidate = false;
	unsigned ukey = reinterpret_cast<unsigned&>( key );
	int hash = utils::bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );
	if( !done )
	  {
	    Key_type stored_key = m_gmem_keys[hash];
	    if( stored_key == key )
	      {
		m_gmem_vals[hash] = m_gmem_vals[hash] + a_value * b_value;
		done = true;
	      }
	    candidate = stored_key == -1;
	    if( candidate )
	      m_gmem_keys[hash] = key;
	    if( candidate && key == m_gmem_keys[hash] ) // More than one candidate may have written to that slot.
	      {
		m_gmem_vals[hash] = a_value * b_value;
		done = true;
	      }
	  }
      }
    if( status == NULL || __all(done) )
      return;
    if( lane_id == 0 )
      status[0] = 1;
  }

  // ====================================================================================================================

  template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::insert_with_duplicates( Key_type key, T val, int *status )
  {
    const int lane_id = utils::lane_id();
    bool done = key == -1;

    m_smem_vote[lane_id].b32 = 0x20202020;
#pragma unroll
    for( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
      {
	if( __all(done) )
	  break;
	bool candidate = false;
	bool maybe_in_conflict = false;
	unsigned ukey = reinterpret_cast<unsigned&>( key );
	int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE-1);
	if( !done )
	  {
	    Key_type stored_key = m_smem_keys[hash];
	    if( stored_key == key )
	      {
		this->try_selection( hash, lane_id );
		maybe_in_conflict = true;
		done = true; // Is it really done???
	      }
	    candidate = stored_key == -1;
	    if( candidate )
	      m_smem_keys[hash] = key;
	    if( candidate && key == m_smem_keys[hash] )
	      {
		this->try_selection( hash, lane_id );
		maybe_in_conflict = true;
		done = true;
	      }
	  }

	// Fix conflicts.
	bool in_conflict = maybe_in_conflict && !this->is_selected(hash, lane_id);
	while( __any( in_conflict ) )
	  {
	    int winner = in_conflict ? this->get_selected(hash) : WARP_SIZE;
	    T other_val = utils::shfl( val, winner );
	    if( in_conflict )
	      this->try_selection(hash, lane_id);
	    if( in_conflict && this->is_selected(hash, lane_id) )
	      {
		val = val + other_val;
		in_conflict = false;
	      }
	  }
      }

    Word my_vote;
    my_vote.b32 = m_smem_vote[lane_id].b32;
#pragma unroll
    for( int i_regs = 0 ; i_regs < 4 ; ++i_regs )
      {
	int my_src = my_vote.b8[i_regs];
	T other_val = utils::shfl( val, my_src );
	if( my_src != WARP_SIZE ) 
	  m_regs_vals[i_regs] = m_regs_vals[i_regs] + other_val;
      }

    const int num_bits = utils::bfind( m_gmem_size ); // TODO: move it outside ::insert.
#pragma unroll
    for( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
      {
	if( __all(done) )
	  return;
	m_any_gmem = true;
	bool candidate = false;
	unsigned ukey = reinterpret_cast<unsigned&>( key );
	int hash = utils::bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );
	if( !done )
	  {
	    Key_type stored_key = m_gmem_keys[hash];
	    if( stored_key == key )
	      {
		utils::atomic_add( &m_gmem_vals[hash], val );
		done = true;
	      }
	    candidate = stored_key == -1;
	    if( candidate )
	      m_gmem_keys[hash] = key;
	    if( candidate && key == m_gmem_keys[hash] ) // More than one candidate may have written to that slot.
	      {
		utils::atomic_add( &m_gmem_vals[hash], val );
		done = true;
	      }
	  }
      }
    if( status == NULL || __all(done) )
      return;
    if( lane_id == 0 )
      status[0] = 1;
  }

  // ====================================================================================================================

  template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::load( int count, const Key_type *keys, const int *pos )
  {
    int lane_id = utils::lane_id();

#pragma unroll 4
    for( int offset = lane_id ; offset < count ; offset += WARP_SIZE )
      {
	Key_type key = keys[offset];
	int idx = pos [offset];

	// Where to store the item.
	volatile Key_type *ptr = m_smem_keys;
	if( idx >= SMEM_SIZE )
	  {
	    ptr = m_gmem_keys;
	    m_any_gmem = 1;
	    idx -= SMEM_SIZE;
	    m_gmem_vals[idx] = T(0);
	  }

	// Store the item.
	ptr[idx] = key;
      }
    m_any_gmem = __any( m_any_gmem );
  }

  // ====================================================================================================================

  template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( int count, T *vals )
  {
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();

    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll
    for( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
      {
	const int offset = i_step*WARP_SIZE + lane_id;
	Key_type key = m_smem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  vals[dst_offset] = m_regs_vals[i_step];
	warp_offset += __popc( poll );
      }

    if( !m_any_gmem )
      return;

#pragma unroll 4
    for( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
      {
	Key_type key = m_gmem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  vals[dst_offset] = m_gmem_vals[offset];
	warp_offset += __popc( poll );
      }
  }

  // ====================================================================================================================

  template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store( int count, Key_type *keys, T *vals )
  {
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();

    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll
    for( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
      {
	const int offset = i_step*WARP_SIZE + lane_id;
	Key_type key = m_smem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  {
	    keys[dst_offset] = key;
	    vals[dst_offset] = m_regs_vals[i_step];
	  }
	warp_offset += __popc( poll );
      }

    if( !m_any_gmem )
      return;

#pragma unroll 4
    for( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
      {
	Key_type key = m_gmem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  {
	    keys[dst_offset] = key;
	    vals[dst_offset] = m_gmem_vals[offset];
	  }
	warp_offset += __popc( poll );
      }
  }

  // ====================================================================================================================

  template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store_map_keys_scale_values( int count, const int *map, Key_type *keys, T alpha, T *vals )
  {
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();

    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll
    for( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
      {
	const int offset = i_step*WARP_SIZE + lane_id;
	Key_type key = m_smem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  {
	    keys[dst_offset] = map[key];
	    vals[dst_offset] = alpha*m_regs_vals[i_step];
	  }
	warp_offset += __popc( poll );
      }

    if( !m_any_gmem )
      return;

#pragma unroll 4
    for( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
      {
	Key_type key = m_gmem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  {
	    keys[dst_offset] = map[key];
	    vals[dst_offset] = alpha*m_gmem_vals[offset];
	  }
	warp_offset += __popc( poll );
      }
  }

  template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  void Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::store_keys_scale_values( int count, Key_type *keys, T alpha, T *vals )
  {
    int lane_id = utils::lane_id();
    int lane_mask_lt = utils::lane_mask_lt();

    int warp_offset = 0;
    const int NUM_STEPS = SMEM_SIZE / WARP_SIZE;
#pragma unroll
    for( int i_step = 0 ; i_step < NUM_STEPS ; ++i_step )
      {
	const int offset = i_step*WARP_SIZE + lane_id;
	Key_type key = m_smem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  {
	    keys[dst_offset] = key;
	    vals[dst_offset] = alpha*m_regs_vals[i_step];
	  }
	warp_offset += __popc( poll );
      }

    if( !m_any_gmem )
      return;

#pragma unroll 4
    for( int offset = lane_id ; offset < m_gmem_size ; offset += WARP_SIZE )
      {
	Key_type key = m_gmem_keys[offset];
	int poll = __ballot( key != -1 );
	if( poll == 0 )
	  continue;
	int dst_offset = warp_offset + __popc( poll & lane_mask_lt );
	if( key != -1 )
	  {
	    keys[dst_offset] = key;
	    vals[dst_offset] = alpha*m_gmem_vals[offset];
	  }
	warp_offset += __popc( poll );
      }
  }



  // ====================================================================================================================

  template< typename Key_type, typename T, int SMEM_SIZE, int NUM_HASH_FCTS, int WARP_SIZE >
  __device__ __forceinline__ 
  bool Hash_map<Key_type, T, SMEM_SIZE, NUM_HASH_FCTS, WARP_SIZE>::update( Key_type key, T val )
  {
    const int lane_id = utils::lane_id();
    bool done = key == -1, found = false;

    m_smem_vote[lane_id].b32 = 0x20202020;
#pragma unroll
    for( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
      {
	if( i_hash > 0 && __all(done) )
	  break;
	unsigned ukey = reinterpret_cast<unsigned&>( key );
	int hash = ( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash] ) & (SMEM_SIZE-1);
	if( !done )
	  {
	    Key_type stored_key = m_smem_keys[hash];
	    if( stored_key == key )
	      {
		this->try_selection( hash, lane_id );
		found = true;
	      }
	    done = found || stored_key == -1;
	  }
      }

    Word my_vote;
    my_vote.b32 = m_smem_vote[lane_id].b32;
#pragma unroll
    for( int i_regs = 0 ; i_regs < 4 ; ++i_regs )
      {
	int my_src = my_vote.b8[i_regs];
	T other_val = utils::shfl( val, my_src );
	if( my_src != WARP_SIZE ) 
	  m_regs_vals[i_regs] += other_val;
      }

    const int num_bits = utils::bfind( m_gmem_size ); // TODO: move it outside ::insert.
#pragma unroll
    for( int i_hash = 0 ; i_hash < NUM_HASH_FCTS ; ++i_hash )
      {
	if( __all(done) )
	  return found;
	unsigned ukey = reinterpret_cast<unsigned&>( key );
	int hash = utils::bfe( (ukey ^ c_hash_keys[i_hash]) + c_hash_keys[NUM_HASH_FCTS + i_hash], num_bits );
	if( !done )
	  {
	    Key_type stored_key = m_gmem_keys[hash];
	    if( stored_key == key )
	      {
		m_gmem_vals[hash] += val;
		found = true;
	      }
	    done = found || stored_key == -1;
	  }
      }
    return found;
  }



  
  template<typename IndexT,
	   typename Value_type, 
	   typename Key_type=IndexT>
  class Hash_Workspace
  {
  private:
    // Do we need values on the GPU?
    bool m_allocate_vals;
    // Constant parameters.
    const size_t m_grid_size, m_max_warp_count;
    // The number of threads per row of B.
    size_t m_num_threads_per_row_count, m_num_threads_per_row_compute;
    // The size of the GMEM buffers (number of elements).
    size_t m_gmem_size;
    // The status: OK if count_non_zeroes succeeded, FAILED otherwise.
    SHARED_PREFIX::shared_ptr<IndexT> m_status;
    // The work queue for dynamic load balancing in the kernels.
    SHARED_PREFIX::shared_ptr<IndexT> m_work_queue;
    // The buffer to store keys in GMEM.
    SHARED_PREFIX::shared_ptr<Key_type> m_keys;
    // The buffer to store values in GMEM.
    SHARED_PREFIX::shared_ptr<Value_type> m_vals;

  public:
    // Create a workspace.
    Hash_Workspace( bool allocate_vals = true, 
		    size_t grid_size = 128, 
		    size_t max_warp_count = 8, 
		    size_t gmem_size = 2048 ): 
      m_allocate_vals(allocate_vals),
      m_grid_size(grid_size), 
      m_max_warp_count(max_warp_count), 
      m_num_threads_per_row_count(32),
      m_num_threads_per_row_compute(32),
      m_gmem_size(gmem_size), 
      m_status(allocateDevice<IndexT>(1, NULL)),
      m_work_queue(allocateDevice<IndexT>(1, NULL))
    {
      allocate_workspace();
    }

    // Release memory used by the workspace.
    virtual ~Hash_Workspace()
    {
      //purposely empty...
    }

    // Get the size of GMEM.
    size_t get_gmem_size() const { return m_gmem_size; }
    // Get the status flag.
    IndexT* get_status() const { return m_status.get(); }
    // Get the work queue.
    IndexT* get_work_queue() const { return m_work_queue.get(); }
    // Get the keys.
    Key_type* get_keys() const { return m_keys.get(); }
    // Get the values.
    Value_type* get_vals() const { return m_vals.get(); }

    // Expand the workspace.
    void expand() { m_gmem_size *= 2; allocate_workspace(); }

    // Define the number of threads per row of B.
    void set_num_threads_per_row_count( size_t val ) { m_num_threads_per_row_count = val; }
    // Define the number of threads per row of B.
    void set_num_threads_per_row_compute( size_t val ) { m_num_threads_per_row_compute = val; }

  protected:
    // Allocate memory to store keys/vals in GMEM.
    virtual void allocate_workspace(void)
    {
      const size_t NUM_WARPS_IN_GRID = m_grid_size * m_max_warp_count;
      size_t sz = NUM_WARPS_IN_GRID*m_gmem_size*sizeof(Key_type);

      m_keys = allocateDevice<Key_type>(sz, NULL);

      if( m_allocate_vals )
	{
	  sz = NUM_WARPS_IN_GRID*m_gmem_size*sizeof(Value_type);
	  m_vals = allocateDevice<Value_type>(sz, NULL);
	}
    }
  };

  namespace{ //unnamed...

    static __device__ __forceinline__ int get_work( int *queue, int warp_id, int count = 1 )
    {
#if __CUDA_ARCH__ >= __CUDA_ARCH_THRESHOLD__
      int offset = -1;
      if( utils::lane_id() == 0 )
        offset = atomicAdd( queue, count );
      return __shfl( offset, 0 );
#else
      return 0;
#endif
    }

    enum { WARP_SIZE = 32, GRID_SIZE = 128, SMEM_SIZE = 128 };

    template<size_t NUM_THREADS_PER_ROW,
             size_t CTA_SIZE,
             size_t SMEM_SIZE,
             size_t WARP_SIZE,
             bool HAS_DIAG,
             typename IndexT,
             typename Value_type>
      __global__ 
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= __CUDA_ARCH_THRESHOLD__  
      __launch_bounds__( CTA_SIZE, 8 )
#elif defined(__CUDA_ARCH__)
      __launch_bounds__( CTA_SIZE, 6 )
#endif
      void fill_A_kernel_1x1( const size_t  R_num_rows,
			      const IndexT *R_rows, 
			      const IndexT *R_cols, 
			      const IndexT *A_rows, 
			      const IndexT *A_cols, 
			      const IndexT *A_diag, 
			      const Value_type *A_vals, 
			      const IndexT *aggregates, 
			      const IndexT *Ac_rows, 
			      const IndexT *Ac_cols, 
			      const IndexT *Ac_pos, 
			      const IndexT *Ac_diag, 
			      Value_type *Ac_vals, 
			      size_t gmem_size, 
			      IndexT *g_keys, 
			      Value_type *g_vals, 
			      IndexT *wk_work_queue )
    {
      const size_t NUM_WARPS = CTA_SIZE / WARP_SIZE;
      const size_t NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;

      // The hash keys stored in shared memory.
      __shared__ volatile IndexT s_keys[NUM_WARPS*SMEM_SIZE]; 

#if __CUDA_ARCH__ >= __CUDA_ARCH_THRESHOLD__
      // The hash values stored in shared memory.
      __shared__ volatile Word s_vote[NUM_WARPS*SMEM_SIZE/4]; 
#else
      // Shared memory to vote.
      __shared__ volatile IndexT s_bcast_row[CTA_SIZE];
      // The hash keys stored in shared memory.
      __shared__ Value_type s_vals[NUM_WARPS*SMEM_SIZE]; 
      // Shared memory to acquire work.
      __shared__ volatile IndexT s_offsets[NUM_WARPS];
      // Shared memory to reduce the diagonal.
      __shared__ volatile Value_type s_diag[CTA_SIZE];
#endif
  
      // The coordinates of the thread inside the CTA/warp.
      const IndexT warp_id = utils::warp_id(); 
      const IndexT lane_id = utils::lane_id();

      // Constants.
      const size_t lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
      const size_t lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;

      // First threads load the row IDs of A needed by the CTA...
      IndexT r_row_id = blockIdx.x*NUM_WARPS + warp_id;
  
      // Create local storage for the set.
#if __CUDA_ARCH__ >= __CUDA_ARCH_THRESHOLD__
      Hash_map<IndexT, Value_type, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id*SMEM_SIZE  ], 
							      &g_keys[r_row_id*gmem_size ], 
							      &s_vote[warp_id*SMEM_SIZE/4], 
							      &g_vals[r_row_id*gmem_size ], gmem_size );
#else
      Hash_map<IndexT, Value_type, SMEM_SIZE, 4, WARP_SIZE> map( &s_keys[warp_id*SMEM_SIZE ], 
							      &g_keys[r_row_id*gmem_size], 
							      &s_vals[warp_id*SMEM_SIZE ], 
							      &g_vals[r_row_id*gmem_size], gmem_size );
#endif
  
    // Loop over rows of A.
#if __CUDA_ARCH__ >= __CUDA_ARCH_THRESHOLD__
      for( ; r_row_id < R_num_rows ; r_row_id = get_work( wk_work_queue, warp_id ) )
#else
      for( ; r_row_id < R_num_rows ; r_row_id = get_work( s_offsets, wk_work_queue, warp_id ) )
#endif
	{
	  // The indices of the output row.
	  IndexT ac_col_it  = Ac_rows[r_row_id+0];
	  IndexT ac_col_end = Ac_rows[r_row_id+1];

	  // Clear the set first. TODO: Make sure it's needed. I don't think it is!!!!
	  map.clear();
	  // Populate the map.
	  map.load( ac_col_end-ac_col_it, &Ac_cols[ac_col_it], &Ac_pos[ac_col_it] );

	  // Load the range of the row. TODO: Make sure it helps.
	  IndexT r_col_it  = R_rows[r_row_id + 0];
	  IndexT r_col_end = R_rows[r_row_id + 1];

	  // The diagonal.
	  Value_type r_diag(0);

	  // _iterate over the columns of A to build C_hat.
	  for( r_col_it += lane_id ; __any(r_col_it < r_col_end) ; r_col_it += WARP_SIZE )
	    {
	      // Is it an active thread.
	      const bool is_active = r_col_it < r_col_end;
    
	      // Columns of A maps to rows of B. Each thread of the warp loads its A-col/B-row ID.
	      IndexT a_row_id = -1; 
	      if( is_active )
		a_row_id = R_cols[r_col_it];
#if __CUDA_ARCH__ < __CUDA_ARCH_THRESHOLD__
	      s_bcast_row[threadIdx.x] = a_row_id;
#endif

	      // Update the diagonal (if needed). 
	      if( HAS_DIAG && is_active )
			r_diag = r_diag + A_vals[A_diag[a_row_id]];
	      
	      const size_t num_rows = __popc( __ballot(is_active) );

	      // Uniform loop: threads collaborate to load other elements.  
	      for( IndexT k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
			{
			  IndexT local_k = k+lane_id_div_num_threads;

		  // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
#if __CUDA_ARCH__ >= __CUDA_ARCH_THRESHOLD__
			  const IndexT uniform_a_row_id = __shfl( a_row_id, local_k );
#else
			  IndexT uniform_a_row_id = -1;
			  if( local_k < num_rows )
				uniform_a_row_id = s_bcast_row[warp_id*WARP_SIZE + local_k];
#endif

			  // The range of the row of B.
			  IndexT a_col_it = 0, a_col_end = 0;
			  if( local_k < num_rows )
				{
				  a_col_it  = utils::Ld<utils::LD_CG>::load( &A_rows[uniform_a_row_id + 0] );
				  a_col_end = utils::Ld<utils::LD_CG>::load( &A_rows[uniform_a_row_id + 1] );
				}
        
			  // Iterate over the range of columns of B.
			  for( a_col_it += lane_id_mod_num_threads ; __any(a_col_it < a_col_end) ; a_col_it += NUM_THREADS_PER_ROW )
				{
				  // Load columns and values.
				  IndexT a_col_id = -1; Value_type a_value(Value_type(0));
				  if( a_col_it < a_col_end )
					{
					  a_col_id = A_cols[a_col_it];
					  a_value  = A_vals[a_col_it];
					}

				  // Find the aggregate.
				  IndexT a_agg_id = -1;
				  if( a_col_it < a_col_end )
					a_agg_id = aggregates[a_col_id];


				  // Update the diag/hash map.
				  if( HAS_DIAG && a_agg_id == r_row_id )
					{
					  r_diag = r_diag + a_value;
					  a_agg_id = -1;
					}

				  map.insert_with_duplicates( a_agg_id, a_value, NULL );  // It won't insert. Only update.
				}
			}
	    }

	  // Update the diagonal.
	  if( HAS_DIAG )
	    {
#if __CUDA_ARCH__ >= __CUDA_ARCH_THRESHOLD__
	      r_diag = utils::warp_reduce<1, utils::Add>( r_diag );
#else
		  utils::util<Value_type>::volcast(r_diag, s_diag + threadIdx.x);
#ifdef _MSC_VER
	      r_diag = utils::warp_reduce_sum<1, Value_type, 32>(s_diag, r_diag);
#else
	      r_diag = utils::warp_reduce<1, utils::Add>(s_diag, r_diag);
#endif
#endif
	      if( lane_id == 0 )
			Ac_vals[Ac_diag[r_row_id]] = r_diag;
	    }

	  // Store the results.
	  IndexT count = ac_col_end - ac_col_it;
	  if( count == 0 )
	    continue;
	  map.store( count, &Ac_vals[ac_col_it] );
	}
    }

    template< size_t CTA_SIZE,
              typename Workspace,
              typename IndexT,
              typename Value_type>
      void fill_A_dispatch( Workspace &hash_wk, 
                            const size_t  R_num_rows, // same as num_aggregates.
							const IndexT *R_rows,
							const IndexT *R_cols,
							const IndexT *A_rows,
							const IndexT *A_cols,
							const Value_type *A_vals,
							const IndexT *aggregates,
							const IndexT *Ac_rows, 
							const IndexT *Ac_cols, 
							const IndexT *Ac_pos,
							Value_type *Ac_vals )
    {
      const size_t NUM_WARPS = CTA_SIZE / WARP_SIZE;
      cudaStream_t stream = 0; // for now...

      size_t work_offset = GRID_SIZE*NUM_WARPS;
      cudaMemcpyAsync( hash_wk.get_work_queue(), &work_offset, sizeof(IndexT), cudaMemcpyHostToDevice, stream );
      cudaCheckError();

      fill_A_kernel_1x1<8, CTA_SIZE, SMEM_SIZE, 32, false><<<GRID_SIZE, CTA_SIZE>>>( 
												       R_num_rows, 
												       R_rows, 
												       R_cols, 
												       A_rows, 
												       A_cols, 
												       static_cast<IndexT*>(0), 
												       A_vals, 
												       aggregates, 
												       Ac_rows, 
												       Ac_cols, 
												       Ac_pos,
												       static_cast<IndexT*>(0), 
												       Ac_vals, 
												       hash_wk.get_gmem_size(),
												       hash_wk.get_keys(),
												       hash_wk.get_vals(),
												       hash_wk.get_work_queue() );

     
      cudaCheckError();
    }

    template<size_t NUM_THREADS_PER_ROW,
             size_t CTA_SIZE,
             size_t SMEM_SIZE,
             size_t WARP_SIZE,
             bool HAS_DIAG,
             bool COUNT_ONLY,
             typename IndexT>
    __global__ __launch_bounds__( CTA_SIZE )
    void compute_sparsity_kernel( const size_t  R_num_rows, // same as num_aggregates.
                                  const IndexT *R_rows,
                                  const IndexT *R_cols,
                                  const IndexT *A_rows,
                                  const IndexT *A_cols,
                                  const IndexT *aggregates,
                                  IndexT *Ac_rows, 
                                  IndexT *Ac_cols,
                                  IndexT *Ac_pos,
                                  const size_t gmem_size,
                                  IndexT *g_keys, 
                                  IndexT *wk_work_queue, 
                                  IndexT *wk_status )
    {
      const size_t NUM_WARPS       = CTA_SIZE  / WARP_SIZE;
      const size_t NUM_LOADED_ROWS = WARP_SIZE / NUM_THREADS_PER_ROW;

      // The hash keys stored in shared memory.
      __shared__ IndexT s_keys[NUM_WARPS*SMEM_SIZE]; 

#if __CUDA_ARCH__ < __CUDA_ARCH_THRESHOLD__
      // Shared memory to acquire work.
      __shared__ volatile IndexT s_offsets[NUM_WARPS];
      // Shared memory to vote.
      __shared__ volatile IndexT s_bcast_cols[CTA_SIZE];
#endif

      // The coordinates of the thread inside the CTA/warp.
      const IndexT warp_id = utils::warp_id(); 
      const IndexT lane_id = utils::lane_id();

      printf("###### milestone 1\n");

      // Constants.
      const IndexT lane_id_div_num_threads = lane_id / NUM_THREADS_PER_ROW;
      const IndexT lane_id_mod_num_threads = lane_id % NUM_THREADS_PER_ROW;

      // First threads load the row IDs of A needed by the CTA...
      IndexT r_row_id = blockIdx.x*NUM_WARPS + warp_id;
  
      // Create local storage for the set.
#if __CUDA_ARCH__ >= __CUDA_ARCH_THRESHOLD__
      Hash_set<IndexT, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id*SMEM_SIZE], &g_keys[r_row_id*gmem_size], gmem_size );
#else
      Hash_set<IndexT, SMEM_SIZE, 4, WARP_SIZE> set( &s_keys[warp_id*SMEM_SIZE], &g_keys[r_row_id*gmem_size], gmem_size );
#endif

      printf("###### milestone 2\n");
  
      // Loop over rows of R.
// #if __CUDA_ARCH__ >= __CUDA_ARCH_THRESHOLD__
      for( ; r_row_id < R_num_rows ; r_row_id = get_work( wk_work_queue, warp_id ) )
// #else
// 	for( ; r_row_id < R_num_rows ; r_row_id = get_work( s_offsets, wk_work_queue, warp_id ) )
// #endif
	  {
	    // Make sure we have to proceed.
	    if( COUNT_ONLY )
	      {
		volatile IndexT *status = reinterpret_cast<volatile IndexT*>( wk_status );
		if( set.has_failed() || *status != 0 )
		  return;
	      }
    
	    // Clear the set.
	    set.clear();

	    // Load the range of the row.
	    IndexT r_col_it  = R_rows[r_row_id + 0];
	    IndexT r_col_end = R_rows[r_row_id + 1];

        printf("###### milestone 3\n");
    
	    // Iterate over the columns of R.
	    for( r_col_it += lane_id ; __any(r_col_it < r_col_end) ; r_col_it += WARP_SIZE )
	      {
		// Is it an active thread.
		const bool is_active = r_col_it < r_col_end;
    
		// Columns of R map to rows of A. Each thread of the warp loads its R-col/A-row ID.
		IndexT a_row_id = -1;
		if( is_active ) 
		  a_row_id = R_cols[r_col_it];
#if __CUDA_ARCH__ < __CUDA_ARCH_THRESHOLD__
		s_bcast_cols[threadIdx.x] = a_row_id;
#endif
		const size_t num_rows = __popc( __ballot(is_active) );

         printf("###### milestone 4\n");

		// Uniform loop: threads collaborate to load other elements.  
		for( IndexT k = 0 ; k < num_rows ; k += NUM_LOADED_ROWS )
		  {
		    IndexT local_k = k+lane_id_div_num_threads;
		    // Is it an active thread.
		    bool is_active_k = local_k < num_rows;

		    // Threads in the warp proceeds columns of B in the range [bColIt, bColEnd).
#if __CUDA_ARCH__ >= __CUDA_ARCH_THRESHOLD__
		    const IndexT uniform_a_row_id = __shfl( a_row_id, local_k );
#else
		    IndexT uniform_a_row_id = -1;
		    if( is_active_k )
		      uniform_a_row_id = s_bcast_cols[warp_id*WARP_SIZE + local_k];
#endif

            printf("###### milestone 5\n");
             
		    // Load the range of the row of B.
		    IndexT a_col_it = 0, a_col_end = 0;
		    if( is_active_k )
		      {
			a_col_it  = A_rows[uniform_a_row_id + 0];
			a_col_end = A_rows[uniform_a_row_id + 1];
		      }
        
		    // Iterate over the range of columns of B.
		    for( a_col_it += lane_id_mod_num_threads ; __any(a_col_it < a_col_end) ; a_col_it += NUM_THREADS_PER_ROW )
		      {
			IndexT a_col_id = -1, a_agg_id = -1;
			if( a_col_it < a_col_end )
			  {
			    a_col_id = A_cols[a_col_it];
			    a_agg_id = aggregates[a_col_id];
			  }
			//if( a_agg_id >= R_num_rows )
			//  printf( "Out of range aggregate!!!\n" );
			if( HAS_DIAG && a_agg_id == r_row_id )
			  a_agg_id = -1;
			set.insert( a_agg_id, COUNT_ONLY ? wk_status : NULL );
		      }
		  }
	      }

        printf("###### milestone 6\n");

	    // Store the results.
	    if( COUNT_ONLY )
	      {
		IndexT count = set.compute_size_with_duplicates();
		if( lane_id == 0 ) 
		  Ac_rows[r_row_id] = count;
	      }
	    else
	      {
		IndexT ac_col_it = Ac_rows[r_row_id];
		set.store_with_positions( &Ac_cols[ac_col_it], &Ac_pos[ac_col_it] );
	      }
	  }
    }

   

    template< size_t CTA_SIZE, 
			  bool HAS_DIAG, 
			  bool COUNT_ONLY,
              typename Workspace,
			  typename IndexT>
	  void compute_sparsity_dispatch( Workspace &hash_wk, 
									  const size_t  R_num_rows, 
									  const IndexT *R_rows, 
									  const IndexT *R_cols, 
									  const IndexT *A_rows, 
									  const IndexT *A_cols,
									  const IndexT *aggregates, 
									  IndexT *Ac_rows, 
									  IndexT *Ac_cols, 
									  IndexT *Ac_pos )
    {
      const size_t NUM_WARPS = CTA_SIZE / WARP_SIZE;

      //AMGX uses pool allocator thrust::global_thread_handle::cudaMallocHost(), here...
      //
      SHARED_PREFIX::shared_ptr<IndexT> h_status(new IndexT);
      SHARED_PREFIX::shared_ptr<IndexT> h_work_offset(new IndexT);

      cudaStream_t stream = 0; // for now...

      int attempt = 0;
      for( bool done = false ; !done && attempt < 10 ; ++attempt )
		{
		  // Double the amount of GMEM (if needed).
		  if( attempt > 0 )
			{
			  std::cerr << "LOW_DEG: Requires " << hash_wk.get_gmem_size() << " items per warp!!!" << std::endl;
			  hash_wk.expand();
			}

		  // Reset the status.
		  IndexT *p_status = h_status.get();
		  *p_status = 0;
		  cudaMemcpyAsync( hash_wk.get_status(), p_status, sizeof(IndexT), cudaMemcpyHostToDevice, stream );
		  cudaCheckError();

		  // Reset the work queue.
		  IndexT *p_work_offset = h_work_offset.get();
		  *p_work_offset = GRID_SIZE*NUM_WARPS;
		  cudaMemcpyAsync( hash_wk.get_work_queue(), p_work_offset, sizeof(IndexT), cudaMemcpyHostToDevice, stream );
		  cudaCheckError();

		  // Launch the kernel.
		  compute_sparsity_kernel<8, CTA_SIZE, SMEM_SIZE, WARP_SIZE, HAS_DIAG, COUNT_ONLY><<<GRID_SIZE, CTA_SIZE,0,stream>>>(R_num_rows, R_rows, R_cols, A_rows, A_cols, aggregates, Ac_rows, Ac_cols, Ac_pos, hash_wk.get_gmem_size(), hash_wk.get_keys(), hash_wk.get_work_queue(), hash_wk.get_status() );

		  cudaCheckError();
  
		  // Read the result from count_non_zeroes.
		  cudaMemcpyAsync( p_status, hash_wk.get_status(), sizeof(IndexT), cudaMemcpyDeviceToHost, stream ); 
		  cudaStreamSynchronize(stream); 
		  done = (*p_status == 0);

		  cudaCheckError();
		}
    }
  }//end unnamed namespace

}//nvgraph namespace

#endif
