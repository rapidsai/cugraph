#pragma once

#define TileT1      detail::Tile<BLOCK_SIZE,T,VType,UNROLL_STEPS>
#define TileT2      detail::Tile<BLOCK_SIZE,T,VType,UNROLL_STEPS,2>
#define TileT       detail::Tile<BLOCK_SIZE,T,VType,UNROLL_STEPS,LDG_FACTOR>
#define LoadTileT   LoadTile<BLOCK_SIZE,T,VType,UNROLL_STEPS>
#define IlLoadTileT IlLoadTile<BLOCK_SIZE,T,VType,UNROLL_STEPS>
#define StoreTileT  StoreTile<BLOCK_SIZE,T,VType,UNROLL_STEPS>

namespace device {
namespace detail {

template<unsigned BLOCK_SIZE, typename T, typename VType,
         unsigned UNROLL_STEPS, unsigned LDG_FACTOR = 1>
class Tile {
    static_assert(sizeof(VType) % sizeof(T) == 0,
                  "sizeof(VType) must be a multiple of sizeof(T)");
protected:
    static const unsigned        RATIO = sizeof(VType) / sizeof(T);
public:
    static const unsigned THREAD_ITEMS = UNROLL_STEPS * RATIO * LDG_FACTOR;

    __device__ __forceinline__
    Tile(int num_items);

    __device__ __forceinline__
    bool is_valid() const;

    __device__ __forceinline__
    int last_index() const;

    __device__ __forceinline__
    int stride() const;

protected:
    int   _index;
    int   _stride;
    int   _full_stride;

private:
    int   _size;
};

} // namespace detail

//==============================================================================

template<unsigned BLOCK_SIZE, typename T, typename VType = T,
         unsigned UNROLL_STEPS = 1>
class StoreTile : public TileT1 {
public:
    using TileT1::THREAD_ITEMS;

    __device__ __forceinline__
    StoreTile(T* ptr, int num_items);

    __device__ __forceinline__
    void store(T (&array)[THREAD_ITEMS]);

    __device__ __forceinline__
    void store(T value);

private:
    T* _ptr;
    using TileT1::_index;
    using TileT1::_stride;
    using TileT1::_full_stride;
};

//==============================================================================

template<unsigned BLOCK_SIZE, typename T, typename VType = T,
         unsigned UNROLL_STEPS = 1>
class LoadTile : public TileT1 {
public:
    using TileT1::THREAD_ITEMS;

    __device__ __forceinline__
    LoadTile(const T* ptr, int num_items);

    __device__ __forceinline__
    void load(T (&array)[THREAD_ITEMS]);

    __device__ __forceinline__
    void load(T (&array)[THREAD_ITEMS],
              int (&indices)[THREAD_ITEMS]);
private:
    const T* _ptr;
    using TileT1::_index;
    using TileT1::_stride;
    using TileT1::_full_stride;
    using TileT1::RATIO;
};

//==============================================================================

template<unsigned BLOCK_SIZE, typename T, typename VType,
         unsigned UNROLL_STEPS = 1>
class IlLoadTile : public TileT2 {
public:
    using TileT2::THREAD_ITEMS;

    __device__ __forceinline__
    IlLoadTile(const T* ptr, int num_items);

    __device__ __forceinline__
    void load(T (&array)[THREAD_ITEMS]);

    __device__ __forceinline__
    void load(T (&array)[THREAD_ITEMS], int (&indices)[THREAD_ITEMS]);
private:
    const T* _ptr;
    using TileT2::_index;
    using TileT2::_stride;
    using TileT2::_full_stride;
    using TileT2::RATIO;
};

} // namespace device

#include "impl/Device.i.cuh"

#undef TileT
#undef TileT1
#undef TileT2
#undef IlLoadTileT
#undef LoadTileT
#undef StoreTileT
