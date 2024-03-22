/*
 * MIT License
 *
 * Copyright (c) 2023 Muhammad Haseeb, Nan Ding, and The Regents of the
 * University of California, through Lawrence Berkeley National Laboratory
 * (subject to receipt of any required approvals from the U.S. Dept. of
 * Energy). All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#pragma once

#include "xmr.hpp"
#include "commons/mpi.hpp"
#include "commons/cuda.hpp"

// change to reuse the same functions
using data_t =            int;
#define MPI_TYPE          MPI_INT

// ------------------------------------------------------------------------------------------------------- //

//
// constants
//

// 64MB in bytes
constexpr long int Bytes = 64*MB;

// max number of data_t sized messages
constexpr int N = Bytes/sizeof(data_t);

// how many loops to do
constexpr int loop_count = 50;
constexpr int warmup_loop_count = 10;

// compile time log2 function
constexpr int ilog2(int n) { return ( (n < 2) ? 1 : 1 + ilog2(n/2)); }

// niterations = log2(1GB) - log2(sizeof(data_t))
constexpr long int niters = static_cast<int>(ilog2(Bytes)) - static_cast<int>(ilog2(sizeof(data_t)));

// ------------------------------------------------------------------------------------------------------- //

//
// empty kernel launch stats
//
int dimBlock = 256;
int dimGrid = 1;

// ------------------------------------------------------------------------------------------------------- //

//
// optional: set gpu clock
//
#if defined(SET_CLOCK)
    // device clockrate
    __device__ int clockrate;
#endif // SET_CLOCK

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

// ------------------------------------------------------------------------------------------------------- //

#define print_operation(x)                if (!rank) std::cout << "\nLaunching Collective: "    \
                                                               << x << std::endl << std::endl;
