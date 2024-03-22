
#include "collective_nvshmem.cuh"

//
// kernel launch stats
//
dim3 *dimBlock = nullptr;
dim3 *dimGrid = nullptr;

// ------------------------------------------------------------------------------------------------------- //

#define print_operation(x)                if (!mype) std::cout << "\nLaunching Collective: "    \
                                                               << x << std::endl << std::endl;

// ------------------------------------------------------------------------------------------------------- //

//
// reduce kernel benchmark
//
void benchmark_reduce_op(void (*reduce_fn)(data_t *, data_t *, int, nvshmem_team_t), data_t *src,
                         data_t *dst, int mype, std::string &&name, nvshmem_team_t &nvshmem_team)
{
    // print operation
    print_operation(name);

    // reduce increasing number of data elements
    for (int iter = 0; iter <= niters; iter++)
    {
        double start_time, stop_time, elapsed_time;
        int nreduce = pow(2, iter);
        void *args[] = {&dst, &src, &nreduce, &nvshmem_team};

        for (int loop = 0; loop < warmup_loop_count; loop++)
        {
            // launch reduce kernel
            NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)reduce_fn, *dimGrid, *dimBlock, args, 0, 0));
        }
            // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());

        start_time = MPI_Wtime();

        for (int loop = 0; loop < loop_count; loop++)
        {
            // launch reduce kernel
            NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)reduce_fn, *dimGrid, *dimBlock, args, 0, 0));
        }

        // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        if (!mype)
            print_stats(elapsed_time, nreduce, 1, loop_count);

    }

    nvshmem_barrier(nvshmem_team);
}

// ------------------------------------------------------------------------------------------------------- //

//
// broadcast kernel benchmark
//
void benchmark_bcast_op(void (*bcast_fn)(data_t *, data_t *, int, int, nvshmem_team_t), data_t *src, data_t *dst,
                        int mype, std::string &&name, nvshmem_team_t &nvshmem_team)
{
    // print operation
    print_operation(name);

    // reduce increasing number of data elements
    for (int iter = 0; iter <= niters; iter++)
    {
        double start_time, stop_time, elapsed_time;
        int nreduce = pow(2, iter);
        int root = 0;
        void *args[] = {&dst, &src, &nreduce, &root, &nvshmem_team};

        for (int loop = 0; loop < warmup_loop_count; loop++)
        {
            // launch reduce kernel
            NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)bcast_fn, *dimGrid, *dimBlock, args, 0, 0));
        }

        // wait to sync
        NVSHMEM_CHECK(nvshmem_barrier(nvshmem_team));
        CUDA_CHECK(cudaDeviceSynchronize());

        start_time = MPI_Wtime();

        for (int loop = 0; loop < loop_count; loop++)
        {
            // launch reduce kernel
            NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)bcast_fn, *dimGrid, *dimBlock, args, 0, 0));
        }

        // wait to sync
        NVSHMEM_CHECK(nvshmem_barrier(nvshmem_team));
        CUDA_CHECK(cudaDeviceSynchronize());

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        // print stats
        if(!mype)
            print_stats(elapsed_time, nreduce, 1, loop_count);
    }

    nvshmem_barrier(nvshmem_team);
}

// ------------------------------------------------------------------------------------------------------- //

//
// reduce kernel benchmark
//
void benchmark_alltoall_op(void (*alltoall_fn)(data_t *, data_t *, int, nvshmem_team_t), data_t *src, data_t *dst,
                           int mype, int npes, std::string &&name, nvshmem_team_t &nvshmem_team)
{
    // print operation
    print_operation(name);

    // all to all increasing number of data elements
    int lniters = niters - ceil(log2(npes));

    // reduce increasing number of data elements
    for (int iter = 0; iter <= lniters; iter++)
    {
        double start_time, stop_time, elapsed_time;
        int nelems = pow(2, iter);
        void *args[] = {&dst, &src, &nelems, &nvshmem_team};

        for (int loop = 0; loop < warmup_loop_count; loop++)
        {
            // launch reduce kernel
            NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)alltoall_fn, *dimGrid, *dimBlock, args, 0, 0));
        }
            // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());

        start_time = MPI_Wtime();

        for (int loop = 0; loop < loop_count; loop++)
        {
            // launch reduce kernel
            NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)alltoall_fn, *dimGrid, *dimBlock, args, 0, 0));
        }

        // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        if (!mype)
            print_stats(elapsed_time, nelems, 1, loop_count);
    }

    nvshmem_barrier(nvshmem_team);
}

// ------------------------------------------------------------------------------------------------------- //

void benchmark_fcollect_op(void (*fcollect_fn)(data_t *, data_t *, int, nvshmem_team_t), data_t *src, data_t *dst,
                           int mype, int npes, std::string &&name, nvshmem_team_t &nvshmem_team)
{
    // print operation
    print_operation(name);

    // reduce increasing number of data elements
    int lniters = niters - ceil(log2(npes));

    for (int iter = 0; iter <= lniters; iter++)
    {
        double start_time, stop_time, elapsed_time;
        int nelems = pow(2, iter);
        void *args[] = {&dst, &src, &nelems, &nvshmem_team};

        for (int loop = 0; loop < warmup_loop_count; loop++)
        {
            // launch reduce kernel
            NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)fcollect_fn, *dimGrid, *dimBlock, args, 0, 0));
        }
            // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());

        start_time = MPI_Wtime();

        for (int loop = 0; loop < loop_count; loop++)
        {
            // launch reduce kernel
            NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)fcollect_fn, *dimGrid, *dimBlock, args, 0, 0));
        }

        // wait to sync
        CUDA_CHECK(cudaDeviceSynchronize());

        stop_time = MPI_Wtime();
        elapsed_time = stop_time - start_time;

        if (!mype)
            print_stats(elapsed_time, nelems, 1, loop_count);
    }

    nvshmem_barrier(nvshmem_team);
}

// ------------------------------------------------------------------------------------------------------- //

//
// main function
//
int main(int argc, char *argv[])
{
    // local variables
    int mype, npes, ngpus = 0;
    int rank = 0;
    int world = 1;

    // CUDA grid dimensions
    dimBlock = new dim3(256);
    dimGrid = new dim3(1);

    // nvshmem attributes
    nvshmemx_init_attr_t attr;

    // initialize MPI
    auto mpi_comm = MPI_COMM_WORLD;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(mpi_comm, &rank));
    MPI_CHECK(MPI_Comm_size(mpi_comm, &world));

    // check if we have at least 2 ranks
    if (world < 2)
    {
        std::cout << "FATAL: Need at least 2 ranks to run the nvshmem_collective benchmark\n";
        return -1;
    }
    else if (!rank)
        std::cout << "Running nvshmem_collective benchmark with " << world << " ranks\n\n"
                  << std::flush;

    // initialize nvshmem
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    nvshmem_team_t nvshmem_team = NVSHMEM_TEAM_WORLD;

    // application picks the device each PE will use
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    CUDA_CHECK(cudaSetDevice(rank%ngpus));

    // print status
    printf("[pe: %d] setting device: %d of %d\n", mype, rank%ngpus, ngpus);
    std::cout << std::flush;

    MPI_Barrier(MPI_COMM_WORLD);

#if defined(SET_CLOCK)
    // optional: set cuda device clock
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *) &prop.clockRate,
                                  sizeof(int), 0, cudaMemcpyHostToDevice));
#endif // SET_CLOCK

    // allocate nvshmem symmetric memory
    data_t *src = (data_t *)nvshmem_malloc(Bytes);
    data_t *dst = (data_t *)nvshmem_malloc(Bytes);

    //
    // initialize the src
    //
    data_gen<<<N/256, 256>>>(src, N, mype);

    // wait to sync
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    //
    // Reduce
    //
    benchmark_reduce_op(reduce_sum_kernel, src, dst, mype, "Reduce (sum)", nvshmem_team);
    benchmark_reduce_op(reduce_prod_kernel, src, dst, mype, "Reduce (prod)", nvshmem_team);
    benchmark_reduce_op(reduce_max_kernel, src, dst, mype, "Reduce (max)", nvshmem_team);
    benchmark_reduce_op(reduce_and_kernel, src, dst, mype, "Reduce (and)", nvshmem_team);
    benchmark_reduce_op(reduce_and_kernel, src, dst, mype, "Reduce (xor)", nvshmem_team);

    //
    // Broadcast
    //
    benchmark_bcast_op(broadcast_kernel, src, dst, mype, "Broadcast", nvshmem_team);
    benchmark_bcast_op(broadcast_kernel_put, src, dst, mype, "Custom Broadcast (Put)", nvshmem_team);
    benchmark_bcast_op(broadcast_kernel_get, src, dst, mype, "Custom Broadcast (Get)", nvshmem_team);
    //
    // alltoall
    //
    benchmark_alltoall_op(alltoall_kernel, src, dst, mype, npes, "AlltoAll", nvshmem_team);

    //
    // fcollect
    //
    benchmark_fcollect_op(fcollect_kernel, src, dst, mype, npes, "Fcollect", nvshmem_team);

    // print status
    printf("[%d of %d] run complete \n", mype, npes);

    // cleanup
    nvshmem_free(dst);
    nvshmem_free(src);

    // finalize nvshmem
    nvshmem_finalize();

    // finalize MPI
    MPI_Finalize();

    // delete dim pointers
    delete dimBlock;
    delete dimGrid;

    return 0;
}

// ------------------------------------------------------------------------------------------------------- //
