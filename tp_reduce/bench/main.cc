#include "benchmark_registerer.hh"
#include "fixture.hh"
#include "main_helper.hh"
#include "to_bench.cuh"

#include <benchmark/benchmark.h>

#include <rmm/mr/device/pool_memory_resource.hpp>

int main(int argc, char** argv)
{
    // Google bench setup
    using benchmark_t = benchmark::internal::Benchmark;
    ::benchmark::Initialize(&argc, argv);

    // RMM Setup
    auto memory_resource = make_pool();
    rmm::mr::set_current_device_resource(memory_resource.get());

    bool bench_nsight = parse_arguments(argc, argv);

    // Benchmarks registration
    Fixture fx;
    {
        // Add the sizes to benchmark here
        // Start with 1 block of 64 (block reduce)
        // Then 2 blocks of 64 each (grid reduce)
        // Then and odd size
        // Finally the true sizes
        // TODO
        constexpr std::array sizes = {
            64
            // 128,
            // 129,
            // 524288,
            // 1048576
        };

        // Add the name and function to benchmark here
        // TODO
        constexpr std::tuple reduce_to_bench{
            "Baseline_reduce",
            &baseline_reduce,
            "Your_reduce",
            &your_reduce,
        };

        //  / 2 because we store name + function pointer
        benchmark_t* b[tuple_length(reduce_to_bench) / 2];
        int function_index = 0;

        // Call to registerer
        registerer_reduce(&fx,
                          b,
                          function_index,
                          sizes,
                          bench_nsight,
                          reduce_to_bench);
    }
    ::benchmark::RunSpecifiedBenchmarks();
}
