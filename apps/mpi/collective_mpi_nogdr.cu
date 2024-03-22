
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


#include "collective_mpi.cuh"

// ------------------------------------------------------------------------------------------------------- //

//
// host pointers needed if not using GPUDirectRDMA
//
data_t *h_src;
data_t *h_dst;

// ------------------------------------------------------------------------------------------------------- //

//
// reduce kernel benchmark
//
void benchmark_reduce_op(void (*reduce_fn)(), data_t *src,
                         data_t *dst, int rank, MPI_Op op, std::string &&name, MPI_Comm &comm)
{
    // print operation
    print_operation(name);

    // reduce increasing number of data elements
    for (int iter = 0; iter <= niters; iter++)
    {
        double start_time, stop_time, elapsed_time;
        int nreduce = pow(2, iter);

        for (int loop = 0; loop < warmup_loop_count; loop++)
        {
            // launch reduce kernel
            (*reduce_fn)<<<dimGrid, dimBlock>>>();
            // wait to sync
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_src, src, nreduce * sizeof(data_t), cudaMemcpyDeviceToHost));
            MPI_Allreduce(h_src, h_dst, nreduce, MPI_TYPE, op, comm);
            CUDA_CHECK(cudaMemcpy(dst, h_dst, nreduce * sizeof(data_t), cudaMemcpyHostToDevice));
        }

        // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);

        start_time = MPI_Wtime();

        for (int loop = 0; loop < loop_count; loop++)
        {
            // launch reduce kernel
            (*reduce_fn)<<<dimGrid, dimBlock>>>();
            // wait to sync
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_src, src, nreduce * sizeof(data_t), cudaMemcpyDeviceToHost));
            MPI_Allreduce(h_src, h_dst, nreduce, MPI_TYPE, op, comm);
            CUDA_CHECK(cudaMemcpy(dst, h_dst, nreduce * sizeof(data_t), cudaMemcpyHostToDevice));
        }

        // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        if (!rank)
            print_stats(elapsed_time, nreduce, 1, loop_count);

    }

    // wait to sync
    MPI_Barrier(comm);
}

// ------------------------------------------------------------------------------------------------------- //

//
// broadcast kernel benchmark
//
void benchmark_bcast_op(void (*bcast_fn)(), data_t *src, data_t *dst, int rank,
                        int root, std::string &&name, MPI_Comm &comm)
{
    // print operation
    print_operation(name);

    // reduce increasing number of data elements
    for (int iter = 0; iter <= niters; iter++)
    {
        double start_time, stop_time, elapsed_time;
        int nelems = pow(2, iter);

        for (int loop = 0; loop < warmup_loop_count; loop++)
        {
            // launch reduce kernel
            (*bcast_fn)<<<dimGrid, dimBlock>>>();
            // wait to sync
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_src, src, nelems * sizeof(data_t), cudaMemcpyDeviceToHost));
            MPI_Bcast(h_src, nelems, MPI_TYPE, root, comm);
            CUDA_CHECK(cudaMemcpy(dst, h_src, nelems * sizeof(data_t), cudaMemcpyHostToDevice));
        }

        // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);

        start_time = MPI_Wtime();

        for (int loop = 0; loop < loop_count; loop++)
        {
            // launch reduce kernel
            (*bcast_fn)<<<dimGrid, dimBlock>>>();
            // wait to sync
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_src, src, nelems * sizeof(data_t), cudaMemcpyDeviceToHost));
            MPI_Bcast(h_src, nelems, MPI_TYPE, root, comm);
            CUDA_CHECK(cudaMemcpy(dst, h_src, nelems * sizeof(data_t), cudaMemcpyHostToDevice));
        }

        // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        // print stats
        if(!rank)
            print_stats(elapsed_time, nelems, 1, loop_count);
    }

    // wait to sync
    MPI_Barrier(comm);
}

// ------------------------------------------------------------------------------------------------------- //

//
// reduce kernel benchmark
//

void benchmark_alltoall_op(void (*alltoall_fn)(), data_t *src, data_t *dst, int rank,
                           int world, std::string &&name, MPI_Comm &comm)
{
    // print operation
    print_operation(name);

    // all to all increasing number of data elements
    int lniters = niters - ceil(log2(world));

    // reduce increasing number of data elements
    for (int iter = 0; iter <= lniters; iter++)
    {
        double start_time, stop_time, elapsed_time;
        int nelems = pow(2, iter);

        for (int loop = 0; loop < warmup_loop_count; loop++)
        {
            // launch reduce kernel
            (*alltoall_fn)<<<dimGrid, dimBlock>>>();
            // wait to sync
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_src, src, nelems * sizeof(data_t), cudaMemcpyDeviceToHost));
            MPI_Alltoall(h_src, nelems, MPI_TYPE, h_dst, nelems, MPI_TYPE, comm);
            CUDA_CHECK(cudaMemcpy(dst, h_dst, world * nelems * sizeof(data_t), cudaMemcpyDeviceToHost));
        }

        // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);

        start_time = MPI_Wtime();

        for (int loop = 0; loop < loop_count; loop++)
        {
            // launch reduce kernel
            (*alltoall_fn)<<<dimGrid, dimBlock>>>();
            // wait to sync
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_src, src, nelems * sizeof(data_t), cudaMemcpyDeviceToHost));
            MPI_Alltoall(src, nelems, MPI_TYPE, dst, nelems, MPI_TYPE, comm);
            CUDA_CHECK(cudaMemcpy(dst, h_dst, world * nelems * sizeof(data_t), cudaMemcpyDeviceToHost));
        }

        // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        if (!rank)
            print_stats(elapsed_time, nelems, 1, loop_count);
    }

    // wait to sync
    MPI_Barrier(comm);
}

// ------------------------------------------------------------------------------------------------------- //

void benchmark_fcollect_op(void (*fcollect_fn)(), data_t *src, data_t *dst, int rank,
                           int world, std::string &&name, MPI_Comm &comm)
{
    // print operation
    print_operation(name);

    // reduce increasing number of data elements
    int lniters = niters - ceil(log2(world));

    for (int iter = 0; iter <= lniters; iter++)
    {
        double start_time, stop_time, elapsed_time;
        int nelems = pow(2, iter);

        for (int loop = 0; loop < warmup_loop_count; loop++)
        {
            // launch reduce kernel
            (*fcollect_fn)<<<dimGrid, dimBlock>>>();
            // wait to sync
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_src, src, nelems * sizeof(data_t), cudaMemcpyDeviceToHost));
            MPI_Allgather(h_src, nelems, MPI_TYPE, h_dst, nelems, MPI_TYPE, comm);
            CUDA_CHECK(cudaMemcpy(dst, h_dst, nelems * world * sizeof(data_t), cudaMemcpyHostToDevice));
        }

        // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);

        start_time = MPI_Wtime();

        for (int loop = 0; loop < loop_count; loop++)
        {
            // launch reduce kernel
            (*fcollect_fn)<<<dimGrid, dimBlock>>>();
            // wait to sync
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_src, src, nelems * sizeof(data_t), cudaMemcpyDeviceToHost));
            MPI_Allgather(h_src, nelems, MPI_TYPE, h_dst, nelems, MPI_TYPE, comm);
            CUDA_CHECK(cudaMemcpy(dst, h_dst, nelems * world * sizeof(data_t), cudaMemcpyHostToDevice));
        }

        // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_Barrier(comm);

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        if (!rank)
            print_stats(elapsed_time, nelems, 1, loop_count);
    }

    // wait to sync
    MPI_Barrier(comm);
}

// ------------------------------------------------------------------------------------------------------- //

//
// main function
//
int main(int argc, char *argv[])
{
    // local variables
    int ngpus = 0;
    int rank = 0;
    int world = 1;

    // CUDA grid dimensions
    dimBlock = 256;
    dimGrid = 1;

    // initialize MPI
    auto mpi_comm = MPI_COMM_WORLD;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(mpi_comm, &rank));
    MPI_CHECK(MPI_Comm_size(mpi_comm, &world));

    // check if we have at least 2 ranks
    if (world < 2)
    {
        std::cout << "FATAL: Need at least 2 ranks to run the mpi_collective benchmark\n";
        return -1;
    }
    else if (!rank)
        std::cout << "Running mpi_collective benchmark with " << world << " ranks\n\n"
                  << std::flush;


    // application picks the device each PE will use
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    CUDA_CHECK(cudaSetDevice(rank%ngpus));

    // print status
    printf("[rank: %d] setting device: %d of %d\n", rank, rank%ngpus, ngpus);
    std::cout << std::flush;

    MPI_Barrier(MPI_COMM_WORLD);

#if defined(SET_CLOCK)
    // optional: set cuda device clock
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *) &prop.clockRate,
                                  sizeof(int), 0, cudaMemcpyHostToDevice));
#endif // SET_CLOCK

    // allocate GPU memory via cudaMalloc
    data_t *src = nullptr;
    data_t *dst = nullptr;
    cudaMalloc(&src, Bytes);
    cudaMalloc(&dst, Bytes);

    // allocate pinned CPU memory via cudaHostMalloc
    cudaMallocHost(&h_src, Bytes);
    cudaMallocHost(&h_dst, Bytes);

    //
    // initialize the src
    //
    data_gen<<<N/256, 256>>>(src, N, rank);

    // wait to sync
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(mpi_comm);

    //
    // Reduce
    //
    benchmark_reduce_op(empty_kernel, src, dst, rank, MPI_SUM, "Reduce (sum)", mpi_comm);
    benchmark_reduce_op(empty_kernel, src, dst, rank, MPI_PROD, "Reduce (prod)", mpi_comm);
    benchmark_reduce_op(empty_kernel, src, dst, rank, MPI_MAX, "Reduce (max)", mpi_comm);
    benchmark_reduce_op(empty_kernel, src, dst, rank, MPI_BAND, "Reduce (and)", mpi_comm);
    //benchmark_reduce_op(reduce_and_kernel, src, dst, "Reduce (xor)", mpi_comm);

    //
    // Broadcast
    //
    benchmark_bcast_op(empty_kernel, src, dst, rank, 0, "Broadcast", mpi_comm);

    //
    // alltoall
    //
    benchmark_alltoall_op(empty_kernel, src, dst, rank, world, "AlltoAll", mpi_comm);

    //
    // fcollect
    //
    benchmark_fcollect_op(empty_kernel, src, dst, rank, world, "Fcollect", mpi_comm);

    // synchronize
    MPI_Barrier(mpi_comm);

    // print status
    printf("[%d of %d] run complete \n", rank, world);

    // cleanup
    cudaFree(dst);
    cudaFree(src);

    cudaFreeHost(h_src);
    cudaFreeHost(h_dst);

    // finalize MPI
    MPI_Finalize();

    return 0;
}

// ------------------------------------------------------------------------------------------------------- //
