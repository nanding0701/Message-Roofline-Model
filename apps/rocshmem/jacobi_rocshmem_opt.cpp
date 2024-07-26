#include "xmr.hpp"
#include "commons/mpi.hpp"
#include "commons/rocshmem.hpp"

#define HAVE_ROCPRIM

#ifdef HAVE_ROCPRIM
#include <rocprim/block/block_reduce.hpp>
#endif

// convert NVSHMEM_SYMMETRIC_SIZE string to long long unsigned int
long long unsigned int parse_nvshmem_symmetric_size(char *value) {
    long long unsigned int units, size;

    assert(value != NULL);

    if (strchr(value, 'G') != NULL) {
        units=1e9;
    } else if (strchr(value, 'M') != NULL) {
        units=1e6;
    } else if (strchr(value, 'K') != NULL) {
        units=1e3;
    } else {
        units=1;
    }

    assert(atof(value) >= 0);
    size = (long long unsigned int) atof(value) * units;

    return size;
}

typedef float real;
constexpr real tol = 1.0e-8;
const real PI = 2.0 * std::asin(1.0);

/* This kernel implements neighborhood synchronization for Jacobi. It updates
   the neighbor PEs about its arrival and waits for notification from them. */
__global__ void syncneighborhood_kernel(int my_pe, int num_pes, int* sync_arr,
                                        int counter) {
    int tid = hipThreadIdx_x;
    int status[2];
    status[0]=0;
    status[1]=0;                                   
    int next_rank = (my_pe + 1) % num_pes;
    int prev_rank = (my_pe == 0) ? num_pes - 1 : my_pe - 1;
    roc_shmem_quiet(); /* To ensure all prior nvshmem operations have been completed */

    /* Notify neighbors about arrival */

    // MH: @Nan: since signaling not available in roch_shmemx so using atomic_set
    // unavailable: roc_shmemx_signal_op(sync_arr, counter, ROC_SHMEM_SIGNAL_SET, next_rank);
    // unavailable: roc_shmemx_signal_op(sync_arr + 1, counter, ROC_SHMEM_SIGNAL_SET, prev_rank);
    //roc_shmem_uint64_atomic_set(sync_arr, counter, next_rank);
    //roc_shmem_uint64_atomic_set(sync_arr + 1, counter, prev_rank);
    roc_shmem_int_p(sync_arr, counter, next_rank);
    roc_shmem_int_p(sync_arr+1, counter, prev_rank);
    
    
    /* Wait for neighbors notification */
    // MH: @Nan: this is not available: roc_shmem_uint64_wait_until_all(sync_arr, 2, NULL, ROC_SHMEM_CMP_GE, counter);
    // roc_shmem_wait_until_all is not available but since the size is 2, we will simply call 2 functions with addresses sync_arr and sync_arr+1
    

    //roc_shmem_int_wait_until(sync_arr, ROC_SHMEM_CMP_GE, counter);
    //roc_shmem_int_wait_until(sync_arr+1, ROC_SHMEM_CMP_GE, counter);
    //__syncthreads();
    roc_shmem_int_wait_until_all(sync_arr, 2, status, ROC_SHMEM_CMP_GE, counter);
}

__global__ void initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                      const real pi, const int offset, const int nx,
                                      const int my_ny, int ny) {
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x) {
        const real y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        a[(iy + 1) * nx + 0] = y0;
        a[(iy + 1) * nx + (nx - 1)] = y0;
        a_new[(iy + 1) * nx + 0] = y0;
        a_new[(iy + 1) * nx + (nx - 1)] = y0;
    }
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                              real* __restrict__ const l2_norm, const int iy_start,
                              const int iy_end, const int nx, const int top_pe, const int top_iy,
                              const int bottom_pe, const int bottom_iy) {

#ifdef HAVE_ROCPRIM
    using BlockReduce = rocprim::block_reduce<real, BLOCK_DIM_X,
            rocprim::block_reduce_algorithm::using_warp_reduce, BLOCK_DIM_Y>;
    __shared__ typename BlockReduce::storage_type temp_storage;
#endif  // HAVE_ROCPRIM

    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    real local_l2_norm = 0.0;

    if (iy < iy_end && ix < (nx - 1)) {
        const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                     a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;
        real residue = new_val - a[iy * nx + ix];
        local_l2_norm += residue * residue;
    }

    /* starting (x, y) coordinate of the block */
    int block_iy =
        iy - threadIdx.y; /* Alternatively, block_iy = blockIdx.y * blockDim.y + iy_start */
    int block_ix = ix - threadIdx.x; /* Alternatively, block_ix = blockIdx.x * blockDim.x + 1 */

    /* Communicate the boundaries */
    if ((block_iy <= iy_start) && (iy_start < block_iy + blockDim.y)) {
        roc_shmemx_float_put_nbi_wg(a_new + top_iy * nx + block_ix, a_new + iy_start * nx + block_ix,
                                   min(blockDim.x, nx - 1 - block_ix), top_pe);
    }
    if ((block_iy < iy_end) && (iy_end <= block_iy + blockDim.y)) {
        roc_shmemx_float_put_nbi_wg(a_new + bottom_iy * nx + block_ix,
                                   a_new + (iy_end - 1) * nx + block_ix,
                                   min(blockDim.x, nx - 1 - block_ix), bottom_pe);
    }

#ifdef HAVE_ROCPRIM
    real block_l2_norm;
    BlockReduce().reduce(local_l2_norm, block_l2_norm, temp_storage);
    if (0 == threadIdx.y && 0 == threadIdx.x) atomicAdd(l2_norm, block_l2_norm);
#else
    atomicAdd(l2_norm, local_l2_norm);
#endif  // HAVE_ROCPRIM

}

double single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref_h,
                  const int nccheck, const bool print, int mype);

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

struct l2_norm_buf {
    hipEvent_t copy_done;
    real* d;
    real* h;
};

int main(int argc, char* argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const bool csv = get_arg(argv, argv + argc, "-csv");

    if (nccheck != 1) {
        fprintf(stderr, "Only nccheck=1 is supported\n");
        return -1;
    }

    real* a_new;

    real* a_ref_h;
    real* a_h;
    double runtime_serial = 0.0;

    real l2_norms[2];

    int rank = 0, size = 1;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    int num_devices;
    auto status = hipGetDeviceCount(&num_devices);

    int local_rank = -1;
    int local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &local_comm));

        MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CHECK(MPI_Comm_size(local_comm, &local_size));

        MPI_CHECK(MPI_Comm_free(&local_comm));
    }
    if ( 1 < num_devices && num_devices < local_size )
    {
        fprintf(stderr,"ERROR Number of visible devices (%d) is less than number of ranks on the node (%d)!\n", num_devices, local_size);
        MPI_CHECK(MPI_Finalize());
        return 1;
    }
    if ( 1 == num_devices ) {
        // Only 1 device visible, assuming GPU affinity is handled via status = hip_VISIBLE_DEVICES
        status = hipSetDevice(0);
    } else {
        status = hipSetDevice(local_rank);
    }
    status = hipFree(0);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    long long unsigned int mesh_size_per_rank = nx * (((ny - 2) + size - 1) / size + 2);
    long long unsigned int required_symmetric_heap_size =
        2 * mesh_size_per_rank * sizeof(real) *
        1.1;  // Factor 2 is because 2 arrays are allocated - a and a_new
              // 1.1 factor is just for alignment or other usage

    char * value = getenv("NVSHMEM_SYMMETRIC_SIZE");
    if (value) { /* env variable is set */
        long long unsigned int size_env = parse_nvshmem_symmetric_size(value);
        if (size_env < required_symmetric_heap_size) {
            fprintf(stderr, "ERROR: Minimum NVSHMEM_SYMMETRIC_SIZE = %lluB, Current NVSHMEM_SYMMETRIC_SIZE = %s\n", required_symmetric_heap_size, value);
            MPI_CHECK(MPI_Finalize());
            return -1;
        }
    } else {
        char symmetric_heap_size_str[100];
        sprintf(symmetric_heap_size_str, "%llu", required_symmetric_heap_size);
        if (!rank && !csv)
            printf("Setting environment variable NVSHMEM_SYMMETRIC_SIZE = %llu\n", required_symmetric_heap_size);
        setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);
    }
    roc_shmem_init(); //_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int npes = roc_shmem_n_pes();
    int mype = roc_shmem_my_pe();

    roc_shmem_barrier_all();

    bool result_correct = true;
    real* a;

    hipStream_t compute_stream;
    hipStream_t reset_l2_norm_stream;
    hipEvent_t compute_done[2];
    hipEvent_t reset_l2_norm_done[2];
    hipEvent_t communication_event_timers[2];
    l2_norm_buf l2_norm_bufs[2];

    float communication_milliseconds{0.0f};

    status = hipHostMalloc(&a_ref_h, nx * ny * sizeof(real));
    status = hipHostMalloc(&a_h, nx * ny * sizeof(real));
    runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h, nccheck, !csv && (0 == mype), mype);

    roc_shmem_barrier_all();
    // ny - 2 rows are distributed amongst `size` ranks in such a way
    // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
    // This optimizes load balancing when (ny - 2) % size != 0
    int chunk_size;
    int chunk_size_low = (ny - 2) / npes;
    int chunk_size_high = chunk_size_low + 1;
    // To calculate the number of ranks that need to compute an extra row,
    // the following formula is derived from this equation:
    // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
    int num_ranks_low = npes * chunk_size_low + npes -
                        (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
    if (mype < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    a = (real*)roc_shmem_malloc(
        nx * (chunk_size_high + 2) *
        sizeof(real));  // Using chunk_size_high so that it is same across all PEs
    a_new = (real*)roc_shmem_malloc(nx * (chunk_size_high + 2) * sizeof(real));

    status = hipMemset(a, 0, nx * (chunk_size + 2) * sizeof(real));
    status = hipMemset(a_new, 0, nx * (chunk_size + 2) * sizeof(real));

    // Calculate local domain boundaries
    int iy_start_global;  // My start index in the global array
    if (mype < num_ranks_low) {
        iy_start_global = mype * chunk_size_low + 1;
    } else {
        iy_start_global =
            num_ranks_low * chunk_size_low + (mype - num_ranks_low) * chunk_size_high + 1;
    }
    int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array
    // do not process boundaries
    iy_end_global = std::min(iy_end_global, ny - 4);

    int iy_start = 1;
    int iy_end = (iy_end_global - iy_start_global + 1) + iy_start;

    // calculate boundary indices for top and bottom boundaries
    int top_pe = mype > 0 ? mype - 1 : (npes - 1);
    int bottom_pe = (mype + 1) % npes;

    int iy_end_top = (top_pe < num_ranks_low) ? chunk_size_low + 1 : chunk_size_high + 1;
    int iy_start_bottom = 0;

    // Set diriclet boundary conditions on left and right boundary
    initialize_boundaries<<<(ny / npes) / 128 + 1, 128>>>(a, a_new, PI, iy_start_global - 1, nx,
                                                          chunk_size, ny - 2);
    status = hipGetLastError();
    status = hipDeviceSynchronize();

    status = hipStreamCreateWithFlags(&compute_stream, hipStreamNonBlocking);
    status = hipStreamCreate(&reset_l2_norm_stream);
    status = hipEventCreateWithFlags(&compute_done[0], hipEventDisableTiming);
    status = hipEventCreateWithFlags(&compute_done[1], hipEventDisableTiming);
    status = hipEventCreateWithFlags(&reset_l2_norm_done[0], hipEventDisableTiming);
    status = hipEventCreateWithFlags(&reset_l2_norm_done[1], hipEventDisableTiming);
    
    // TODO: Error checking
    status = hipEventCreate(&communication_event_timers[0]);
    status = hipEventCreate(&communication_event_timers[1]);

    for (int i = 0; i < 2; ++i) {
        status = hipEventCreateWithFlags(&l2_norm_bufs[i].copy_done, hipEventDisableTiming);
        status = hipMalloc(&l2_norm_bufs[i].d, sizeof(real));
        status = hipMemset(l2_norm_bufs[i].d, 0, sizeof(real));
        status = hipHostMalloc(&l2_norm_bufs[i].h, sizeof(real));
        *(l2_norm_bufs[i].h) = 1.0;
    }

    roc_shmem_barrier_all(); // fixme: @nan: may introduce some overhead compared to non-existent: roc_shmemx_barrier_all_on_stream(compute_stream);
    MPI_CHECK(MPI_Allreduce(l2_norm_bufs[0].h, &l2_norms[0], 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(l2_norm_bufs[1].h, &l2_norms[1], 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
    status = hipDeviceSynchronize();

    if (!mype) {
        if (!csv) printf("Jacobi relaxation: %d iterations on %d x %d mesh\n", iter_max, ny, nx);
    }

    constexpr int dim_block_x = 1024;
    constexpr int dim_block_y = 1;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x,
                  (chunk_size + dim_block_y - 1) / dim_block_y, 1);

    std::cout << mype << ",dim_grid=" << dim_grid.x << "," << dim_grid.y << "," << dim_grid.z << std::endl;
    int iter = 0;
    if (!mype) {
        for (int i = 0; i < 2; ++i) {
            l2_norms[i] = 1.0;
        }
    }

    roc_shmem_barrier_all();

    double start = MPI_Wtime();
    //// PUSH_RANGE("Jacobi solve", 0)
    bool l2_norm_greater_than_tol = true;

    /* Used by syncneighborhood kernel */
    int* sync_arr = NULL;
    sync_arr = (int*)roc_shmem_malloc(2 * sizeof(int));
    status = hipMemsetAsync(sync_arr, 0, 2 * sizeof(int), compute_stream);
    status = hipStreamSynchronize(compute_stream);
    int synccounter = 1;

    while (l2_norm_greater_than_tol && iter < iter_max) {
        // on new iteration: old current vars are now previous vars, old
        // previous vars are no longer needed
        int prev = iter % 2;
        int curr = (iter + 1) % 2;

        status = hipStreamWaitEvent(compute_stream, reset_l2_norm_done[curr], 0);

        dim3 dim_block = {dim_block_x, dim_block_y, 1};
        jacobi_kernel<dim_block_x, dim_block_y>
            <<<dim_grid, dim_block, 0, compute_stream>>>(
                a_new, a, l2_norm_bufs[curr].d, iy_start, iy_end, nx, top_pe, iy_end_top, bottom_pe,
                iy_start_bottom);
        status = hipGetLastError();

        /* Instead of using nvshmemx_barrier_all_on_stream, we are using a custom implementation
           of barrier that just synchronizes with the neighbor PEs that is the PEs with whom a PE
           communicates. This will perform faster than a global barrier that would do redundant-nccheck
           synchronization for this application. */
        hipEventRecord(communication_event_timers[0], compute_stream);
        syncneighborhood_kernel<<<1, 4, 0, compute_stream>>>(mype, npes, sync_arr, synccounter);
        hipEventRecord(communication_event_timers[1], compute_stream);
        synccounter++;

        // perform L2 norm calculation
        if ((iter % nccheck) == 0 || (!csv && (iter % 100) == 0)) {
            // as soon as computation is complete -> D2H-copy L2 norm
            status = hipMemcpyAsync(l2_norm_bufs[curr].h, l2_norm_bufs[curr].d, sizeof(real),
                                         hipMemcpyDeviceToHost, compute_stream);
            status = hipEventRecord(l2_norm_bufs[curr].copy_done, compute_stream);

            // ensure previous D2H-copy is completed before using the data for
            // calculation
            status = hipEventSynchronize(l2_norm_bufs[prev].copy_done);

            MPI_CHECK(MPI_Allreduce(l2_norm_bufs[prev].h, &l2_norms[prev], 1, MPI_FLOAT, MPI_SUM,
                                   MPI_COMM_WORLD));

            l2_norms[prev] = std::sqrt(l2_norms[prev]);
            l2_norm_greater_than_tol = (l2_norms[prev] > tol);

            if (!csv && (iter % 100) == 0) {
                if (!mype) printf("%5d, %0.6f\n", iter, l2_norms[prev]);
            }

            // reset everything for next iteration
            l2_norms[prev] = 0.0;
            *(l2_norm_bufs[prev].h) = 0.0;
            status = hipMemcpyAsync(l2_norm_bufs[prev].d, l2_norm_bufs[prev].h, sizeof(real),
                                         hipMemcpyHostToDevice, reset_l2_norm_stream);
            status = hipEventRecord(reset_l2_norm_done[prev], reset_l2_norm_stream);
        }

        std::swap(a_new, a);
        iter++;

        float iter_communication_milliseconds{0.0f};

        hipEventSynchronize(communication_event_timers[1]);
        hipEventElapsedTime(&iter_communication_milliseconds,
                            communication_event_timers[0],
                            communication_event_timers[1]);

        communication_milliseconds += iter_communication_milliseconds;
    }

    status = hipDeviceSynchronize();
    roc_shmem_barrier_all();
    double stop = MPI_Wtime();
    // POP_RANGE

    roc_shmem_barrier_all();

    status = hipMemcpy(a_h + iy_start_global * nx, a + nx,
                            std::min(ny - 2 - iy_start_global, chunk_size) * nx * sizeof(real),
                            hipMemcpyDeviceToHost);

    result_correct = true;
    for (int iy = iy_start_global; result_correct && (iy < iy_end_global); ++iy) {
        for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
            if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                fprintf(stderr,
                        "ERROR on rank %d: a[%d * %d + %d] = %f does not match %f "
                        "(reference)\n",
                        rank, iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                result_correct = false;
            }
        }
    }

    int global_result_correct = 1;
    MPI_CHECK(MPI_Allreduce(&result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN,
                           MPI_COMM_WORLD));
    result_correct = global_result_correct;

    if (!mype && result_correct) {
        if (csv) {
            printf("rocshmem_opt, %d, %d, %d, %d, %d, 1, %f, %f, %f\n", nx, ny, iter_max, nccheck, npes,
                   (stop - start), runtime_serial, communication_milliseconds * 0.001f);
        } else {
            printf("Num GPUs: %d.\n", npes);
            printf(
                "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, "
                "efficiency: %8.2f \n",
                ny, nx, runtime_serial, npes, (stop - start), runtime_serial / (stop - start),
                runtime_serial / (npes * (stop - start)) * 100);
        }
    }

    for (int i = 0; i < 2; ++i) {
        status = hipHostFree(l2_norm_bufs[i].h);
        status = hipFree(l2_norm_bufs[i].d);
        status = hipEventDestroy(l2_norm_bufs[i].copy_done);
    }

    roc_shmem_free(a);
    roc_shmem_free(a_new);
    roc_shmem_free(sync_arr);

    status = hipEventDestroy(reset_l2_norm_done[1]);
    status = hipEventDestroy(reset_l2_norm_done[0]);
    status = hipEventDestroy(compute_done[1]);
    status = hipEventDestroy(compute_done[0]);
    status = hipStreamDestroy(reset_l2_norm_stream);
    status = hipStreamDestroy(compute_stream);

    status = hipHostFree(a_h);
    status = hipHostFree(a_ref_h);

    roc_shmem_finalize();
    MPI_CHECK(MPI_Finalize());

    return (result_correct == 1) ? 0 : 1;
}

double single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref_h,
                  const int nccheck, const bool print, int mype) {
    real* a;
    real* a_new;

    hipStream_t compute_stream;

    real* l2_norm_d;
    real* l2_norm_h;

    int iy_start = 1;
    int iy_end = ny - 3;

    auto status = hipMalloc((void**)&a, nx * ny * sizeof(real));
    status = hipMalloc((void**)&a_new, nx * ny * sizeof(real));    

    status = hipMemset(a, 0, nx * ny * sizeof(real));
    status = hipMemset(a_new, 0, nx * ny * sizeof(real));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, 0, nx, ny - 2, ny - 2);

    status = hipGetLastError();
    status = hipDeviceSynchronize();

    status = hipStreamCreate(&compute_stream);

    status = hipMalloc(&l2_norm_d, sizeof(real));
    status = hipHostMalloc(&l2_norm_h, sizeof(real));

    status = hipDeviceSynchronize();

    if (print)
        printf(
            "Single GPU jacobi relaxation: %d iterations on %d x %d mesh with "
            "norm "
            "check every %d iterations\n",
            iter_max, ny, nx, nccheck);

    constexpr int dim_block_x = 1024;
    constexpr int dim_block_y = 1;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, ((ny - 2) + dim_block_y - 1) / dim_block_y,
                  1);

    int iter = 0;
    real l2_norm = 1.0;

    status = hipDeviceSynchronize();
    double start = MPI_Wtime();
    // PUSH_RANGE("Jacobi solve", 0)

    while (l2_norm > tol && iter < iter_max) {
        status = hipMemsetAsync(l2_norm_d, 0, sizeof(real), compute_stream);

        dim3 dim_block = {dim_block_x, dim_block_y, 1};
        jacobi_kernel<dim_block_x, dim_block_y>
            <<<dim_grid, dim_block, 0, compute_stream>>>(
                a_new, a, l2_norm_d, iy_start, iy_end, nx, mype, iy_end + 1, mype, (iy_start - 1));
        status = hipGetLastError();

        if ((iter % nccheck) == 0 || (print && ((iter % 100) == 0))) {
            status = hipMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real), hipMemcpyDeviceToHost,
                                         compute_stream);
            status = hipStreamSynchronize(compute_stream);
            l2_norm = *l2_norm_h;
            l2_norm = std::sqrt(l2_norm);
            if (print && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }

        std::swap(a_new, a);
        iter++;
    }
    status = hipDeviceSynchronize();
    // POP_RANGE
    double stop = MPI_Wtime();

    status = hipMemcpy(a_ref_h, a, nx * ny * sizeof(real), hipMemcpyDeviceToHost);

    status = hipStreamDestroy(compute_stream);

    status = hipHostFree(l2_norm_h);
    status = hipFree(l2_norm_d);

    status = hipFree(a_new);
    status = hipFree(a);
    return (stop - start);
}
