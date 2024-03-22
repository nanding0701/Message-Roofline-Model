#pragma once

#include "xmr.hpp"
#include "commons/mpi.hpp"
#include "commons/cuda.hpp"
#include "commons/nvshmem.hpp"

// change to reuse the same functions
using data_t = int;

// 64MB in bytes
constexpr long int Bytes = 64*MB;

// max number of data_t sized messages
constexpr int N = Bytes/sizeof(data_t);

// how many loops to do
constexpr int loop_count = 50;
constexpr int warmup_loop_count = 10;

// niterations = log2(1GB) - log2(sizeof(data_t))
long int niters = static_cast<int>(log2(Bytes)) - static_cast<int>(log2(sizeof(data_t)));

#if defined(SET_CLOCK)
    // device clockrate
    __device__ int clockrate;
#endif // SET_CLOCK

// ------------------------------------------------------------------------------------------------------- //

//
// data generation
//
__global__ void data_gen(data_t *src, int size, int mype)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    for (; idx < size; idx += blockDim.x * gridDim.x)
    {
        /* compute */
        src[idx] = idx + mype;
    }
}

// ------------------------------------------------------------------------------------------------------- //

//
// empty kernel to insert launch delay
//
__global__ void empty_kernel()
{
    __syncthreads();
}

// ------------------------------------------------------------------------------------------------------- //

//
// alltoall kernel
//
__global__ void alltoall_kernel(data_t *dest, data_t *src, int size, nvshmem_team_t team)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    // these APIs seem to not be available.
    if ( size >= blockDim.x)
    {
        //select first block
        if ( !bid )
        {
            nvshmemx_int_alltoall_block(team, dest, src, size);
        }
    }
    else if ( size > 1 )
    {
        constexpr static short warp_size = 32;

        //select first warp from first CUDA block
        if ( !bid && threadIdx.x/warp_size == 0)
        {
            nvshmemx_int_alltoall_warp(team, dest, src, size);
        }
    }
    else
    {
        //select first thread
        if ( !tid )
        {
            nvshmem_int_alltoall(team, dest, src, size);
        }
    }
}

// ------------------------------------------------------------------------------------------------------- //

//
// broadcast kernel
//
__global__ void broadcast_kernel(data_t *dest, data_t *src, int size, int root, nvshmem_team_t team)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    // these APIs seem to not be available.
    if ( size >= blockDim.x)
    {
        //select first block
        if ( !bid )
        {
            nvshmemx_int_broadcast_block(team, dest, src, size, root);
        }
    }
    else if ( size > 1 )
    {
        constexpr static short warp_size = 32;

        //select first warp from first CUDA block
        if ( !bid && threadIdx.x/warp_size == 0)
        {
            nvshmemx_int_broadcast_warp(team, dest, src, size, root);
        }
    }
    else
    {
        //select first thread
        if ( !tid )
        {
            nvshmem_int_broadcast(team, dest, src, size, root);
        }
    }
}

// ------------------------------------------------------------------------------------------------------- //

//
// broadcast kernel with puts
//
__global__ void broadcast_kernel_put(data_t *dest, data_t *src, int size, int root, nvshmem_team_t team)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    int mype = nvshmem_team_my_pe(team);
    int npes = nvshmem_team_n_pes(team);

    if (mype == root)
    {
        // these APIs seem to not be available.
        if ( size >= blockDim.x)
        {
            //select first block
            if ( !bid )
            {
                for (int i = 0; i < npes; i++)
                    if (i != mype)
                        nvshmemx_int_put_nbi_block(dest, src, size, i);
            }
        }
        else if ( size > 1 )
        {
            constexpr static short warp_size = 32;

            //select first warp from first CUDA block
            if ( !bid && threadIdx.x/warp_size == 0)
            {
                for (int i = 0; i < npes; i++)
                    if (i != mype)
                        nvshmemx_int_put_nbi_warp(dest, src, size, i);
            }
        }
            else
            {
                //select first thread
                if ( !tid )
                {
                for (int i = 0; i < npes; i++)
                    if (i != mype)
                        nvshmem_int_put_nbi(dest, src, size, i);
                }
        }

        nvshmem_quiet();
    }
}

// ------------------------------------------------------------------------------------------------------- //

//
// broadcast kernel with gets
//
__global__ void broadcast_kernel_get(data_t *dest, data_t *src, int size, int root, nvshmem_team_t team)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    int mype = nvshmem_team_my_pe(team);
    int npes = nvshmem_team_n_pes(team);

    if (mype != root)
    {
        // these APIs seem to not be available.
        if ( size >= blockDim.x)
        {
            //select first block
            if ( !bid )
            {
                nvshmemx_int_get_nbi_block(dest, src, size, root);
            }
        }
        else if ( size > 1 )
        {
            constexpr static short warp_size = 32;

            //select first warp from first CUDA block
            if ( !bid && threadIdx.x/warp_size == 0)
            {
                nvshmemx_int_get_nbi_warp(dest, src, size, root);
            }
        }
            else
            {
                //select first thread
                if ( !tid )
                {
                    nvshmem_int_get_nbi(dest, src, size, root);
                }
        }

        nvshmem_fence();
    }
}

// ------------------------------------------------------------------------------------------------------- //

//
// reduce sum kernel
//
__global__ void reduce_sum_kernel(data_t *dest, data_t *src, int nreduce, nvshmem_team_t team)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    // these APIs seem to not be available.
    if ( nreduce >= blockDim.x)
    {
        //select first block
        if ( !bid )
        {
            nvshmemx_int_sum_reduce_block(team, dest, src, nreduce);
        }
    }
    else if ( nreduce > 1 )
    {
        constexpr static short warp_size = 32;

        //select first warp from first CUDA block
        if ( !bid && threadIdx.x/warp_size == 0)
        {
            nvshmemx_int_sum_reduce_warp(team, dest, src, nreduce);
        }
    }
    else
    {
        //select first thread
        if ( !tid )
        {
            nvshmem_int_sum_reduce(team, dest, src, nreduce);
        }
    }
}

// ------------------------------------------------------------------------------------------------------- //

//
// reduce prod kernel
//
__global__ void reduce_prod_kernel(data_t *dest, data_t *src, int nreduce, nvshmem_team_t team)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    // these APIs seem to not be available.
    if ( nreduce >= blockDim.x)
    {
        //select first block
        if ( !bid )
        {
            nvshmemx_int_prod_reduce_block(team, dest, src, nreduce);
        }
    }
    else if ( nreduce > 1 )
    {
        constexpr static short warp_size = 32;

        //select first warp from first CUDA block
        if ( !bid && threadIdx.x/warp_size == 0)
        {
            nvshmemx_int_prod_reduce_warp(team, dest, src, nreduce);
        }
    }
    else
    {
        //select first thread
        if ( !tid )
        {
            nvshmem_int_prod_reduce(team, dest, src, nreduce);
        }
    }
}

// ------------------------------------------------------------------------------------------------------- //

//
// reduce max kernel
//
__global__ void reduce_max_kernel(data_t *dest, data_t *src, int nreduce, nvshmem_team_t team)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    // these APIs seem to not be available.
    if ( nreduce >= blockDim.x)
    {
        //select first block
        if ( !bid )
        {
            nvshmemx_int_max_reduce_block(team, dest, src, nreduce);
        }
    }
    else if ( nreduce > 1 )
    {
        constexpr static short warp_size = 32;

        //select first warp from first CUDA block
        if ( !bid && threadIdx.x/warp_size == 0)
        {
            nvshmemx_int_max_reduce_warp(team, dest, src, nreduce);
        }
    }
    else
    {
        //select first thread
        if ( !tid )
        {
            nvshmem_int_max_reduce(team, dest, src, nreduce);
        }
    }
}

// ------------------------------------------------------------------------------------------------------- //

//
// reduce and kernel
//
__global__ void reduce_and_kernel(data_t *dest, data_t *src, int nreduce, nvshmem_team_t team)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    // these APIs seem to not be available.
    if ( nreduce >= blockDim.x)
    {
        //select first block
        if ( !bid )
        {
            nvshmemx_int32_and_reduce_block(team, dest, src, nreduce);
        }
    }
    else if ( nreduce > 1 )
    {
        constexpr static short warp_size = 32;

        //select first warp from first CUDA block
        if ( !bid && threadIdx.x/warp_size == 0)
        {
            nvshmemx_int32_and_reduce_warp(team, dest, src, nreduce);
        }
    }
    else
    {
        //select first thread
        if ( !tid )
        {
            nvshmem_int32_and_reduce(team, dest, src, nreduce);
        }
    }
}

// ------------------------------------------------------------------------------------------------------- //

//
// reduce xor kernel
//
__global__ void reduce_xor_kernel(data_t *dest, data_t *src, int nreduce, nvshmem_team_t team)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    // these APIs seem to not be available.
    if ( nreduce >= blockDim.x)
    {
        //select first block
        if ( !bid )
        {
            nvshmemx_int32_xor_reduce_block(team, dest, src, nreduce);
        }
    }
    else if ( nreduce > 1 )
    {
        constexpr static short warp_size = 32;

        //select first warp from first CUDA block
        if ( !bid && threadIdx.x/warp_size == 0)
        {
            nvshmemx_int32_xor_reduce_warp(team, dest, src, nreduce);
        }
    }
    else
    {
        //select first thread
        if ( !tid )
        {
            nvshmem_int32_xor_reduce(team, dest, src, nreduce);
        }
    }
}

// ------------------------------------------------------------------------------------------------------- //

//
// fcollect kernel
//
__global__ void fcollect_kernel(data_t *dest, data_t *src, int size, nvshmem_team_t team)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    // these APIs seem to not be available.
    if ( size >= blockDim.x )
    {
        //select first block
        if ( !bid )
        {
            nvshmemx_int_fcollect_block(team, dest, src, size);
        }
    }
    else if ( size > 1 )
    {
        constexpr static short warp_size = 32;

        //select first warp from first CUDA block
        if ( !bid && threadIdx.x/warp_size == 0)
        {
            nvshmemx_int_fcollect_warp(team, dest, src, size);
        }
    }
    else
    {
        //select first thread
        if ( !tid )
        {
            nvshmem_int_fcollect(team, dest, src, size);
        }
    }
}

// ------------------------------------------------------------------------------------------------------- //

//
// print stats
//
void print_stats(double elapsed_time, int nreduce, int nmessages, int loop_count)
{
        long int msg_size = nreduce * sizeof(data_t);
        double GBs = (double)msg_size / (double)GB;
        double avg_time_per_transfer = elapsed_time / ((double)loop_count * nmessages);

        printf("Msg size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n",
                msg_size, avg_time_per_transfer, GBs/avg_time_per_transfer);
}

// ------------------------------------------------------------------------------------------------------- //
