#pragma once

#include <benchmark/benchmark.h>
#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include "test_helpers.cuh"

class Fixture
{
  public:
    static bool no_check;

    template <typename FUNC, typename... Args>
    void
    bench_reduce(benchmark::State& st, FUNC callback, int size, Args&&... args)
    {
        constexpr int val = 1;
        constexpr int zero = 0;
        const raft::handle_t handle{};
        rmm::device_uvector<int> buffer(size, handle.get_stream());
        rmm::device_scalar<int> total(0, handle.get_stream());
        fill_buffer(handle, buffer, val);

        for (auto _ : st)
        {
            st.PauseTiming();
            total.set_value_async(zero, handle.get_stream());
            st.ResumeTiming();
            callback(buffer, total);
        }

        st.SetBytesProcessed(int64_t(st.iterations()) *
                             int64_t(size * sizeof(int)));

        if (!no_check)
            check_buffer(total, size, st);
    }

    template <typename FUNC>
    void register_reduce(benchmark::State& st, FUNC func)
    {
        int size = st.range(0);
        this->bench_reduce(st, func, size);
    }
};

bool Fixture::no_check = false;