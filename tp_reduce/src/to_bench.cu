%%file /content/tp_reduce_colab/src/to_bench.cu

#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>


template <typename T>
__global__
void kernel_reduce_baseline(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < buffer.size())
        atomicAdd(total.data(), buffer[id]);
}

void baseline_reduce(rmm::device_uvector<int>& buffer,
                     rmm::device_scalar<int>& total)
{
    constexpr int blocksize = 64;
    const int gridsize = (buffer.size() + blocksize - 1) / blocksize;

	  kernel_reduce_baseline<int><<<gridsize, blocksize, 0, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()),
        raft::device_span<int>(total.data(), 1));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

template <typename T>
__global__
void kernel_your_reduce(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    // TODO
    // ...
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < buffer.size()) {
        sdata[tid] = buffer[i];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();
    for (int s=blockDim.x / 2; s > 0; s /= 2){
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }



    if (tid == 0) total[blockIdx.x] = sdata[0];
}

/*
__inline__ __device__
int warp_reduce(int val)
{
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, i);
    }
    return val;
}

template <typename T>
__global__
void kernel_your_reduce_grid_stride_loop(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int grid_size = blockDim.x * gridDim.x;

    // Somme des éléments avec une grid-stride loop
    int sum = 0;
    for (int idx = i; idx < buffer.size(); idx += grid_size) {
        sum += buffer[idx];
    }

    // Réduction intra-warp
    sum = warp_reduce(sum);

    // Réduction inter-warp au niveau du bloc
    if (tid % 32 == 0) {
        atomicAdd(total.data(), sum); // Atomic addition des résultats des warps dans total
    }
}
*/

template <typename T>
__global__
void kernel_your_reduce_grid_stride_loop(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int grid_size = blockDim.x * gridDim.x;

    // Initialisation locale du tableau partagé
    sdata[tid] = 0;

    // Utilisation de la grid-stride loop pour traiter plusieurs éléments
    for (unsigned int idx = i; idx < buffer.size(); idx += grid_size) {
        sdata[tid] += buffer[idx];
    }
    __syncthreads();

    // Réduction intra-bloc
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Le thread 0 écrit le résultat partiel dans le tableau total
    if (tid == 0) total[blockIdx.x] = sdata[0];
}




void your_reduce(rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total)
{
    constexpr int blocksize = 64;
    int gridsize = (buffer.size() + blocksize - 1) / blocksize;

    rmm::device_uvector<int> partial_sums(gridsize, buffer.stream());

    
    kernel_your_reduce_grid_stride_loop<int><<<gridsize, blocksize, blocksize * sizeof(int), buffer.stream()>>>(
        raft::device_span<const int>(buffer.data(), buffer.size()),
        raft::device_span<int>(partial_sums.data(), partial_sums.size()));
    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));

    while (gridsize > 1) {
    
        int new_gridsize = (gridsize + blocksize - 1) / blocksize;

        rmm::device_uvector<int> new_partial_sums(new_gridsize, buffer.stream());

        kernel_your_reduce_grid_stride_loop<int><<<new_gridsize, blocksize, blocksize * sizeof(int), buffer.stream()>>>(
            raft::device_span<const int>(partial_sums.data(), partial_sums.size()),
            raft::device_span<int>(new_partial_sums.data(), new_partial_sums.size()));
        CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));


        partial_sums = std::move(new_partial_sums);  
        gridsize = new_gridsize;
    }

    kernel_your_reduce_grid_stride_loop<int><<<1, blocksize, blocksize * sizeof(int), buffer.stream()>>>(
        raft::device_span<int>(partial_sums.data(), partial_sums.size()),
        raft::device_span<int>(total.data(), 1));
    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}
