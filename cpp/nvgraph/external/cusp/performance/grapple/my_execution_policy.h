#include <cusp/exception.h>
#include <cusp/system/cuda/detail/par.h>

#include <thrust/execution_policy.h>

#include <iostream>
#include <iomanip>
#include <stack>
#include <string>
#include <typeinfo>
#include <vector>

#include "my_policy_map.h"

std::string demangle(const char* name);

template <class T>
std::string type(const T& t) {

      return demangle(typeid(t).name());
}

#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>

struct handle {
    char* p;
    handle(char* ptr) : p(ptr) { }
    ~handle() {
        std::free(p);
    }
};

std::string demangle(const char* name) {

    int status = -4; // some arbitrary value to eliminate the compiler warning

    handle result( abi::__cxa_demangle(name, NULL, NULL, &status) );

    return (status==0) ? result.p : name ;
}

#else

// does nothing if not g++
std::string demangle(const char* name) {
    return name;
}
#endif

class timer
{
private:
    cudaEvent_t start;
    cudaEvent_t end;
    cudaError_t error;

public:
    timer(void)
    {
        if((error = cudaEventCreate(&start)) != cudaSuccess)
            throw cusp::runtime_exception("cudaEventCreate failed (start)");
        if((error = cudaEventCreate(&end)) != cudaSuccess)
            throw cusp::runtime_exception("cudaEventCreate failed (end)");
    }

    ~timer(void)
    {
        if((error = cudaEventDestroy(start)) != cudaSuccess)
            throw cusp::runtime_exception("cudaEventDestroy failed (start)");
        if((error = cudaEventDestroy(end)) != cudaSuccess)
            throw cusp::runtime_exception("cudaEventDestroy failed (end)");
    }

    void Start(void)
    {
        if((error = cudaEventRecord(start, 0)) != cudaSuccess)
        {
            std::cout << error << std::endl;
            throw cusp::runtime_exception("cudaEventRecord failed (start)");
        }
    }

    void Stop(void)
    {
        if((error = cudaEventRecord(end, 0)) != cudaSuccess)
            throw cusp::runtime_exception("cudaEventRecord failed (end)");
        if((error = cudaEventSynchronize(end)) != cudaSuccess)
            throw cusp::runtime_exception("cudaEventSynchronize failed");
    }

    float milliseconds_elapsed(void)
    {
        float elapsed_time;
        if((error = cudaEventElapsedTime(&elapsed_time, start, end)) != cudaSuccess)
            throw cusp::runtime_exception("cudaEventSynchronize failed");

        // if(stat != cudaSuccess) {
        //     switch(stat) {
        //     case(cudaErrorInvalidValue) :
        //         printf("cudaErrorInvalidValue\n");
        //         break;
        //     case(cudaErrorInitializationError) :
        //         printf("cudaErrorInitializationError");
        //         break;
        //     case(cudaErrorInvalidResourceHandle) :
        //         printf("cudaErrorInvalidResourceHandle");
        //         break;
        //     case(cudaErrorLaunchFailure) :
        //         printf("cudaErrorLaunchFailure");
        //         break;
        //     default :
        //         printf("Unknown error");
        //     }
        //     throw cusp::runtime_exception("timer failed");
        // }

        return elapsed_time;
    }

    float seconds_elapsed()
    {
        return milliseconds_elapsed() / 1000.0;
    }
};

class grapple_data
{
private:

    int   func_id;
    int   stack_frame;
    float func_ms;

public:

    int mem_size;

    grapple_data(void)
        : func_id(-1), stack_frame(0), func_ms(0.0), mem_size(0)
    {}

    void set_data(int stack_index, int func_index, float elapsed_time)
    {
        stack_frame = stack_index;
        func_id = func_index;
        func_ms = elapsed_time;
    }

    friend std::ostream &operator<<( std::ostream &output,
                                     const grapple_data &data )
    {
        output << std::string(data.stack_frame, '\t')
               << std::setw(23) << ARR_NAMES[data.func_id] << " : "
               << std::setw( 8) << data.func_ms   << " (ms), allocated : "
               << std::setw(10) << data.mem_size  << " bytes";

        return output;
    }
};

struct my_policy : public cusp::cuda::execution_policy<my_policy>
{
private:
    typedef thrust::execution_policy<my_policy> UpCastPolicy;

    const static size_t STACK_SIZE = 100;

    timer timer_list[STACK_SIZE];
    int func_index[STACK_SIZE];
    int stack_frame;
    int abs_index;

    std::stack<int> func_stack;
public:

    std::vector<grapple_data> data;

    my_policy(void) : stack_frame(0), abs_index(0)
    {
        data.reserve(STACK_SIZE);
    }

    ~my_policy(void)
    {
        print();
    }

    UpCastPolicy& get(void)
    {
        return *this;
    }

    cusp::system::cuda::execution_policy<my_policy>& base(void)
    {
        return *this;
    }

    void start(const size_t id)
    {
        data.push_back(grapple_data());
        func_stack.push(abs_index++);

        func_index[stack_frame] = id;
        timer_list[stack_frame++].Start();
    }

    void stop(void)
    {
        timer_list[--stack_frame].Stop();
        float elapsed_time = timer_list[stack_frame].milliseconds_elapsed();

        data[func_stack.top()].set_data(stack_frame, func_index[stack_frame], elapsed_time);
        func_stack.pop();
    }

    void print(void)
    {
        for(size_t i = 0; i < data.size(); i++)
            std::cout << std::right << "[" << std::setw(2) << i << "]"
                      << std::left << data[i] << std::endl;
    }
};

#include "my_thrust_func.h"
#include "my_cusp_func.h"

