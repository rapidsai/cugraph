#include "Host/Basic.hpp"               //xlib::is_aligned
#include "Host/Numeric.hpp"             //xlib::lower_approx
#include "Device/Util/Definition.cuh"   //xlib::WARP_SIZE
#include <cassert>                      //assert

namespace device {
namespace detail {

template<unsigned BLOCK_SIZE, typename T, typename VType,
         unsigned UNROLL_STEPS, unsigned LDG_FACTOR>
__device__ __forceinline__
TileT::Tile(int num_items) :
    _index(blockIdx.x * BLOCK_SIZE + threadIdx.x),
    _stride(gridDim.x * BLOCK_SIZE),
    _full_stride(_stride * THREAD_ITEMS),
    _size(xlib::lower_approx(num_items / THREAD_ITEMS, _stride * UNROLL_STEPS))
    {}

template<unsigned BLOCK_SIZE, typename T, typename VType,
         unsigned UNROLL_STEPS, unsigned LDG_FACTOR>
__device__ __forceinline__
bool TileT::is_valid() const {
    return _index < _size;
}

template<unsigned BLOCK_SIZE, typename T, typename VType,
         unsigned UNROLL_STEPS, unsigned LDG_FACTOR>
__device__ __forceinline__
int TileT::last_index() const {
    return _size * THREAD_ITEMS;
}

template<unsigned BLOCK_SIZE, typename T, typename VType,
         unsigned UNROLL_STEPS, unsigned LDG_FACTOR>
__device__ __forceinline__
int TileT::stride() const {
    return _stride;
}

} // namespace detail

//==============================================================================

template<unsigned BLOCK_SIZE, typename T, typename VType, unsigned UNROLL_STEPS>
__device__ __forceinline__
StoreTileT::StoreTile(T* ptr, int num_items) :
                                    TileT1(num_items),
                                    _ptr(ptr) {
                                    //_ptr(ptr + _index * TileT1::RATIO) {
    assert(xlib::is_aligned<VType>(ptr) && "ptr not aligned to VType");
}

template<unsigned BLOCK_SIZE, typename T, typename VType, unsigned UNROLL_STEPS>
__device__ __forceinline__
void StoreTileT::store(T (&array)[THREAD_ITEMS]) {
    auto d_out = reinterpret_cast<VType*>(_ptr);
    auto l_int = reinterpret_cast<VType*>(array);

    #pragma unroll
    for (int J = 0; J < UNROLL_STEPS; J++)
        d_out[_index + _stride * J] = l_int[J];
    _index += _stride;

    /*#pragma unroll
    for (int J = 0; J < UNROLL_STEPS; J++)
        d_out[_stride * J] = l_int[J];
    _index += _stride;
    _ptr   += _full_stride;*/
}

//==============================================================================

template<unsigned BLOCK_SIZE, typename T, typename VType, unsigned UNROLL_STEPS>
__device__ __forceinline__
LoadTileT::LoadTile(const T* ptr, int num_items) :
                                    TileT1(num_items),
                                    //_ptr(ptr) {
                                    //_ptr(reinterpret_cast<const VType*>(ptr)) {
                                    _ptr(ptr + _index * TileT1::RATIO) {
    assert(xlib::is_aligned<VType>(ptr) && "ptr not aligned to VType");
}

template<unsigned BLOCK_SIZE, typename T, typename VType, unsigned UNROLL_STEPS>
__device__ __forceinline__
void LoadTileT::load(T (&array)[THREAD_ITEMS]) {
    auto  d_in = reinterpret_cast<const VType*>(_ptr);
    //auto d_in = _ptr;
    auto l_out = reinterpret_cast<VType*>(array);

    #pragma unroll
    for (int J = 0; J < UNROLL_STEPS; J++)
        l_out[J] = d_in[_index + _stride * J];
    _index += _stride;

    /*#pragma unroll
    for (int J = 0; J < UNROLL_STEPS; J++)
        l_out[J] = d_in[_stride * J];
    _index += _stride;
    _ptr   += _full_stride;*/
}

template<unsigned BLOCK_SIZE, typename T, typename VType, unsigned UNROLL_STEPS>
__device__ __forceinline__
void LoadTileT::load(T (&array)[THREAD_ITEMS],
                     int (&indices)[THREAD_ITEMS]) {
    auto  d_in = reinterpret_cast<const VType*>(_ptr);
    auto l_out = reinterpret_cast<VType*>(array);

    #pragma unroll
    for (int J = 0; J < UNROLL_STEPS; J++) {
        #pragma unroll
        for (int K = 0; K < RATIO; K++)
            indices[RATIO * J + K] = RATIO * (_index + _stride * J) + K;
        l_out[J] = d_in[_stride * J];
    }
    _index += _stride;
    _ptr   += _full_stride;
}

//==============================================================================

template<unsigned BLOCK_SIZE, typename T, typename VType, unsigned UNROLL_STEPS>
__device__ __forceinline__
IlLoadTileT::IlLoadTileT(const T* ptr, int num_items) :
                                        TileT2(num_items),
                                        _ptr(ptr + _index * TileT2::RATIO) {}

template<unsigned BLOCK_SIZE, typename T, typename VType, unsigned UNROLL_STEPS>
__device__ __forceinline__
void IlLoadTileT::load(T (&array)[THREAD_ITEMS]) {
    auto  d_in = reinterpret_cast<const VType*>(_ptr);
    auto l_out = reinterpret_cast<VType*>(array);

    #pragma unroll
    for (int J = 0; J < UNROLL_STEPS; J++) {
        l_out[J * 2]     = d_in[_stride * J * 2];
        l_out[J * 2 + 1] = __ldg(&d_in[_stride * (J * 2 + 1)]);
    }
    _index += _stride;
    _ptr   += _full_stride;
}

template<unsigned BLOCK_SIZE, typename T, typename VType, unsigned UNROLL_STEPS>
__device__ __forceinline__
void IlLoadTileT::load(T (&array)[THREAD_ITEMS],
                       int (&indices)[THREAD_ITEMS]) {
    auto  d_in = reinterpret_cast<const VType*>(_ptr);
    auto l_out = reinterpret_cast<VType*>(array);

    #pragma unroll
    for (int J = 0; J < UNROLL_STEPS; J++) {
        #pragma unroll
        for (int K = 0; K < RATIO; K++) {
            indices[RATIO * J * 2 + K] = RATIO * (_index + _stride * J * 2) + K;
            indices[RATIO * (J * 2 + 1) + K] =
                                  RATIO * (_index + _stride * (J * 2 + 1) ) + K;
        }
        l_out[J * 2]     = d_in[_stride * J * 2];
        l_out[J * 2 + 1] = __ldg(&d_in[_stride * (J * 2 + 1)]);
    }
    _index += _stride;
    _ptr   += _full_stride;
}

} // namespace device
