#pragma once

#include <iostream>
#include <thrust/execution_policy.h>

template <typename T, unsigned int N>
struct FixedVector
{
    T data[N];

    __host__ __device__
    FixedVector()
    {
        for(unsigned int i = 0; i < N; i++)
            data[i] = T();
    }

    __host__ __device__
    FixedVector(T init)
    {
        for(unsigned int i = 0; i < N; i++)
            data[i] = init;
    }

    __host__ __device__
    FixedVector operator+(const FixedVector& bs) const
    {
        FixedVector output;
        for(unsigned int i = 0; i < N; i++)
            output.data[i] = data[i] + bs.data[i];
        return output;
    }

    __host__ __device__
    bool operator<(const FixedVector& bs) const
    {
        for(unsigned int i = 0; i < N; i++)
        {
            if(data[i] < bs.data[i])
                return true;
            else if(bs.data[i] < data[i])
                return false;
        }
        return false;
    }

    __host__ __device__
    bool operator==(const FixedVector& bs) const
    {
        for(unsigned int i = 0; i < N; i++)
        {
            if(!(data[i] == bs.data[i]))
                return false;
        }
        return true;
    }
};

template<typename Key, typename Value>
struct key_value
{
    typedef Key   key_type;
    typedef Value value_type;

    __host__ __device__
    key_value(void)
        : key(), value()
    {}

    __host__ __device__
    key_value(key_type k, value_type v)
        : key(k), value(v)
    {}

    __host__ __device__
    bool operator<(const key_value &rhs) const
    {
        return key < rhs.key;
    }

    __host__ __device__
    bool operator>(const key_value &rhs) const
    {
        return key > rhs.key;
    }

    __host__ __device__
    bool operator==(const key_value &rhs) const
    {
        return key == rhs.key && value == rhs.value;
    }

    __host__ __device__
    bool operator!=(const key_value &rhs) const
    {
        return !operator==(rhs);
    }

    friend std::ostream &operator<<(std::ostream &os, const key_value &kv)
    {
        return os << "(" << kv.key << ", " << kv.value << ")";
    }

    key_type key;
    value_type value;
};

class my_system : public thrust::device_execution_policy<my_system>
{
public:
    my_system(int)
        : correctly_dispatched(false),
          num_copies(0)
    {}

    my_system(const my_system &other)
        : correctly_dispatched(false),
          num_copies(other.num_copies + 1)
    {}

    void validate_dispatch()
    {
        correctly_dispatched = (num_copies == 0);
    }

    bool is_valid()
    {
        return correctly_dispatched;
    }

private:
    bool correctly_dispatched;

    // count the number of copies so that we can validate
    // that dispatch does not introduce any
    unsigned int num_copies;


    // disallow default construction
    my_system();
};

struct my_tag : thrust::device_execution_policy<my_tag> {};

