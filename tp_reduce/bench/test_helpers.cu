#include "test_helpers.cuh"
#include "cuda_tools/cuda_error_checking.cuh"

#include <benchmark/benchmark.h>

#include <iostream>

#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

template <typename T>
void check_buffer(const rmm::device_scalar<T>& scalar,
                  T expected,
                  benchmark::State& st)
{
    const T value = scalar.value(scalar.stream());

    if (value != expected)
    {
        std::cout << "Expected " << expected << ", got " << value << std::endl;
        st.SkipWithError("Failed test");
    }
}

template <typename T>
void fill_buffer(const raft::handle_t& handle,
                 rmm::device_uvector<T>& buffer,
                 T val)
{
    thrust::uninitialized_fill(handle.get_thrust_policy(),
                               buffer.begin(),
                               buffer.end(),
                               val);
}

template void check_buffer(const rmm::device_scalar<int>& scalar,
                           int expected,
                           benchmark::State& st);

template void fill_buffer(const raft::handle_t& handle,
                          rmm::device_uvector<int>& buffer,
                          int val);