#include "xmr.hpp"
#include "commons/mpi.hpp"
#include "commons/hip.hpp"
#include "commons/rccl.hpp"

// #define HAVE_ROCPRIM

#ifdef HAVE_ROCPRIM
#include <rocprim/block/block_reduce.hpp>
#endif

#ifdef USE_DOUBLE
typedef double real;
#define MPI_REAL_TYPE MPI_DOUBLE
#define NCCL_REAL_TYPE ncclDouble
#else
typedef float real;
#define MPI_REAL_TYPE MPI_FLOAT
#define NCCL_REAL_TYPE ncclFloat
#endif

constexpr real tol = 1.0e-8;
const real PI = 2.0 * std::asin(1.0);

__global__ void initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                      const real pi, const int offset, const int nx,
                                      const int my_ny, const int ny) {
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x) {
        const real y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        a[iy * nx + 0] = y0;
        a[iy * nx + (nx - 1)] = y0;
        a_new[iy * nx + 0] = y0;
        a_new[iy * nx + (nx - 1)] = y0;
    }
}

void launch_initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                  const real pi, const int offset, const int nx, const int my_ny,
                                  const int ny) {
    initialize_boundaries<<<my_ny / 128 + 1, 128>>>(a_new, a, pi, offset, nx, my_ny, ny);
    CHECK_HIP(hipGetLastError());
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                              real* __restrict__ const l2_norm, const int iy_start,
                              const int iy_end, const int nx, const bool calculate_norm) {
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
        if (calculate_norm) {
            real residue = new_val - a[iy * nx + ix];
            local_l2_norm += residue * residue;
        }
    }
    if (calculate_norm) {
#ifdef HAVE_ROCPRIM
    real block_l2_norm;
    BlockReduce().reduce(local_l2_norm, block_l2_norm, temp_storage);
    if (0 == threadIdx.y && 0 == threadIdx.x) atomicAdd(l2_norm, block_l2_norm);
#else
    atomicAdd(l2_norm, local_l2_norm);
#endif  // HAVE_ROCPRIM
    }
}

void launch_jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                          real* __restrict__ const l2_norm, const int iy_start, const int iy_end,
                          const int nx, const bool calculate_norm, hipStream_t stream) {
    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x,
                  ((iy_end - iy_start) + dim_block_y - 1) / dim_block_y, 1);
    dim3 dim_block = {dim_block_x, dim_block_y, 1};
    jacobi_kernel<dim_block_x, dim_block_y><<<dim_grid, dim_block, 0, stream>>>(
        a_new, a, l2_norm, iy_start, iy_end, nx, calculate_norm);
    CHECK_HIP(hipGetLastError());
}


double single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref_h,
                  const int nccheck, const bool print);

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

int main(int argc, char* argv[]) {

    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int size;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    int num_devices = 0;
    CHECK_HIP(hipGetDeviceCount(&num_devices));

    ncclUniqueId nccl_uid;
    if (rank == 0) NCCL_CALL(ncclGetUniqueId(&nccl_uid));
    MPI_CHECK(MPI_Bcast(&nccl_uid, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
    // MPI_Barrier ensures that all processs have completed the MPI_Bcast.
    // This can be required when combining MPI with other communication libraries like NCCL.
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool csv = get_arg(argv, argv + argc, "-csv");

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
        // Only 1 device visbile assuming GPU affinity is handled via CUDA_VISIBLE_DEVICES
        CHECK_HIP(hipSetDevice(0));
    } else {
        CHECK_HIP(hipSetDevice(local_rank));
    }
    CHECK_HIP(hipFree(0));
    std::cout << rank << ": " << ": my device is " << local_rank << std::endl;

    ncclComm_t nccl_comm;
    NCCL_CALL(ncclCommInitRank(&nccl_comm, size, nccl_uid, rank));

    real* a_ref_h;
    CHECK_HIP(hipHostMalloc(&a_ref_h, nx * ny * sizeof(real)));
    real* a_h;
    CHECK_HIP(hipHostMalloc(&a_h, nx * ny * sizeof(real)));
    double runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h, nccheck, !csv && (0 == rank));

    // ny - 2 rows are distributed amongst `size` ranks in such a way
    // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
    // This optimizes load balancing when (ny - 2) % size != 0
    int chunk_size;
    int chunk_size_low = (ny - 2) / size;
    int chunk_size_high = chunk_size_low + 1;
    // To calculate the number of ranks that need to compute an extra row,
    // the following formula is derived from this equation:
    // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
    int num_ranks_low = size * chunk_size_low + size -
                        (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
    if (rank < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    real* a;
    CHECK_HIP(hipMalloc(&a, nx * (chunk_size + 2) * sizeof(real)));
    real* a_new;
    CHECK_HIP(hipMalloc(&a_new, nx * (chunk_size + 2) * sizeof(real)));

    CHECK_HIP(hipMemset(a, 0, nx * (chunk_size + 2) * sizeof(real)));
    CHECK_HIP(hipMemset(a_new, 0, nx * (chunk_size + 2) * sizeof(real)));

    // Calculate local domain boundaries
    int iy_start_global;  // My start index in the global array
    if (rank < num_ranks_low) {
        iy_start_global = rank * chunk_size_low + 1;
    } else {
        iy_start_global =
            num_ranks_low * chunk_size_low + (rank - num_ranks_low) * chunk_size_high + 1;
    }
    int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array

    int iy_start = 1;
    int iy_end = iy_start + chunk_size;

    // Set diriclet boundary conditions on left and right boarder
    launch_initialize_boundaries(a, a_new, PI, iy_start_global - 1, nx, (chunk_size + 2), ny);
    CHECK_HIP(hipDeviceSynchronize());

    int leastPriority = 0;
    int greatestPriority = leastPriority;
    CHECK_HIP(hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    hipStream_t compute_stream;
    CHECK_HIP(hipStreamCreateWithPriority(&compute_stream, hipStreamDefault, leastPriority));
    hipStream_t push_stream;
    CHECK_HIP(
        hipStreamCreateWithPriority(&push_stream, hipStreamDefault, greatestPriority));

    hipEvent_t push_prep_done;
    CHECK_HIP(hipEventCreateWithFlags(&push_prep_done, hipEventDisableTiming));
    hipEvent_t push_done;
    CHECK_HIP(hipEventCreateWithFlags(&push_done, hipEventDisableTiming));
    hipEvent_t reset_l2norm_done;
    CHECK_HIP(hipEventCreateWithFlags(&reset_l2norm_done, hipEventDisableTiming));

    real* l2_norm_d;
    CHECK_HIP(hipMalloc(&l2_norm_d, sizeof(real)));
    real* l2_norm_h;
    CHECK_HIP(hipHostMalloc(&l2_norm_h, sizeof(real)));

    // PUSH_RANGE("NCCL_Warmup", 5)
    for (int i = 0; i < 10; ++i) {
        const int top = rank > 0 ? rank - 1 : (size - 1);
        const int bottom = (rank + 1) % size;
        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclRecv(a_new,                     nx, NCCL_REAL_TYPE, top,    nccl_comm, compute_stream));
        NCCL_CALL(ncclSend(a_new + (iy_end - 1) * nx, nx, NCCL_REAL_TYPE, bottom, nccl_comm, compute_stream));
        NCCL_CALL(ncclRecv(a_new + (iy_end * nx),     nx, NCCL_REAL_TYPE, bottom, nccl_comm, compute_stream));
        NCCL_CALL(ncclSend(a_new + iy_start * nx,     nx, NCCL_REAL_TYPE, top,    nccl_comm, compute_stream));
        NCCL_CALL(ncclGroupEnd());
        CHECK_HIP(hipStreamSynchronize(compute_stream));
        std::swap(a_new, a);
    }
    // POP_RANGE

    CHECK_HIP(hipDeviceSynchronize());

    if (!csv && 0 == rank) {
        printf(
            "Jacobi relaxation: %d iterations on %d x %d mesh with norm check "
            "every %d iterations\n",
            iter_max, ny, nx, nccheck);
    }

    int iter = 0;
    bool calculate_norm = true;
    real l2_norm = 1.0;

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    // PUSH_RANGE("Jacobi solve", 0)
    while (l2_norm > tol && iter < iter_max) {
        CHECK_HIP(hipMemsetAsync(l2_norm_d, 0, sizeof(real), compute_stream));
        CHECK_HIP(hipEventRecord(reset_l2norm_done, compute_stream));

        CHECK_HIP(hipStreamWaitEvent(push_stream, reset_l2norm_done, 0));
        calculate_norm = (iter % nccheck) == 0 || (!csv && (iter % 100) == 0);
        launch_jacobi_kernel(a_new, a, l2_norm_d, iy_start, (iy_start + 1), nx, calculate_norm,
                             push_stream);

        launch_jacobi_kernel(a_new, a, l2_norm_d, (iy_end - 1), iy_end, nx, calculate_norm,
                             push_stream);
        CHECK_HIP(hipEventRecord(push_prep_done, push_stream));

        launch_jacobi_kernel(a_new, a, l2_norm_d, (iy_start + 1), (iy_end - 1), nx, calculate_norm,
                             compute_stream);

        if (calculate_norm) {
            CHECK_HIP(hipStreamWaitEvent(compute_stream, push_prep_done, 0));
            CHECK_HIP(hipMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real), hipMemcpyDeviceToHost,
                                         compute_stream));
        }

        const int top = rank > 0 ? rank - 1 : (size - 1);
        const int bottom = (rank + 1) % size;

        // Apply periodic boundary conditions
        // PUSH_RANGE("NCCL_LAUNCH", 5)
        NCCL_CALL(ncclGroupStart());
        NCCL_CALL(ncclRecv(a_new,                     nx, NCCL_REAL_TYPE, top,    nccl_comm, push_stream));
        NCCL_CALL(ncclSend(a_new + (iy_end - 1) * nx, nx, NCCL_REAL_TYPE, bottom, nccl_comm, push_stream));
        NCCL_CALL(ncclRecv(a_new + (iy_end * nx),     nx, NCCL_REAL_TYPE, bottom, nccl_comm, push_stream));
        NCCL_CALL(ncclSend(a_new + iy_start * nx,     nx, NCCL_REAL_TYPE, top,    nccl_comm, push_stream));
        NCCL_CALL(ncclGroupEnd());
        CHECK_HIP(hipEventRecord(push_done, push_stream));
        // POP_RANGE

        if (calculate_norm) {
            CHECK_HIP(hipStreamSynchronize(compute_stream));
            MPI_CHECK(MPI_Allreduce(l2_norm_h, &l2_norm, 1, MPI_REAL_TYPE, MPI_SUM, MPI_COMM_WORLD));
            l2_norm = std::sqrt(l2_norm);

            if (!csv && 0 == rank && (iter % 100) == 0) {
                printf("%5d, %0.6f\n", iter, l2_norm);
            }
        }
        CHECK_HIP(hipStreamWaitEvent(compute_stream, push_done, 0));

        std::swap(a_new, a);
        iter++;
    }
    CHECK_HIP(hipDeviceSynchronize());
    double stop = MPI_Wtime();
    // POP_RANGE

    CHECK_HIP(hipMemcpy(a_h + iy_start_global * nx, a + nx,
                            std::min((ny - iy_start_global) * nx, chunk_size * nx) * sizeof(real),
                            hipMemcpyDeviceToHost));


    int result_correct = 1;
    for (int iy = iy_start_global; result_correct && (iy < iy_end_global); ++iy) {
        for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
            if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                fprintf(stderr,
                        "ERROR on rank %d: a[%d * %d + %d] = %f does not match %f "
                        "(reference)\n",
                        rank, iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                result_correct = 0;
            }
        }
    }

    int global_result_correct = 1;
    MPI_CHECK(MPI_Allreduce(&result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN,
                           MPI_COMM_WORLD));
    result_correct = global_result_correct;

    if (rank == 0 && result_correct) {
        if (csv) {
            printf("nccl_overlap, %d, %d, %d, %d, %d, 1, %f, %f\n", nx, ny, iter_max, nccheck, size,
                   (stop - start), runtime_serial);
        } else {
            printf("Num GPUs: %d.\n", size);
            printf(
                "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, "
                "efficiency: %8.2f \n",
                ny, nx, runtime_serial, size, (stop - start), runtime_serial / (stop - start),
                runtime_serial / (size * (stop - start)) * 100);
        }
    }
    CHECK_HIP(hipEventDestroy(reset_l2norm_done));
    CHECK_HIP(hipEventDestroy(push_done));
    CHECK_HIP(hipEventDestroy(push_prep_done));
    CHECK_HIP(hipStreamDestroy(push_stream));
    CHECK_HIP(hipStreamDestroy(compute_stream));

    CHECK_HIP(hipHostFree(l2_norm_h));
    CHECK_HIP(hipFree(l2_norm_d));

    CHECK_HIP(hipFree(a_new));
    CHECK_HIP(hipFree(a));

    CHECK_HIP(hipHostFree(a_h));
    CHECK_HIP(hipHostFree(a_ref_h));

    NCCL_CALL(ncclCommDestroy(nccl_comm));

    MPI_CHECK(MPI_Finalize());
    return (result_correct == 1) ? 0 : 1;
}

double single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref_h,
                  const int nccheck, const bool print) {
    real* a;
    real* a_new;

    hipStream_t compute_stream;
    hipStream_t push_top_stream;
    hipStream_t push_bottom_stream;
    hipEvent_t compute_done;
    hipEvent_t push_top_done;
    hipEvent_t push_bottom_done;

    real* l2_norm_d;
    real* l2_norm_h;

    int iy_start = 1;
    int iy_end = (ny - 1);

    CHECK_HIP(hipMalloc(&a, nx * ny * sizeof(real)));
    CHECK_HIP(hipMalloc(&a_new, nx * ny * sizeof(real)));

    CHECK_HIP(hipMemset(a, 0, nx * ny * sizeof(real)));
    CHECK_HIP(hipMemset(a_new, 0, nx * ny * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    launch_initialize_boundaries(a, a_new, PI, 0, nx, ny, ny);
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipStreamCreate(&compute_stream));
    CHECK_HIP(hipStreamCreate(&push_top_stream));
    CHECK_HIP(hipStreamCreate(&push_bottom_stream));
    CHECK_HIP(hipEventCreateWithFlags(&compute_done, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&push_top_done, hipEventDisableTiming));
    CHECK_HIP(hipEventCreateWithFlags(&push_bottom_done, hipEventDisableTiming));

    CHECK_HIP(hipMalloc(&l2_norm_d, sizeof(real)));
    CHECK_HIP(hipHostMalloc(&l2_norm_h, sizeof(real)));

    CHECK_HIP(hipDeviceSynchronize());

    if (print)
        printf(
            "Single GPU jacobi relaxation: %d iterations on %d x %d mesh with "
            "norm "
            "check every %d iterations\n",
            iter_max, ny, nx, nccheck);

    int iter = 0;
    bool calculate_norm = true;
    real l2_norm = 1.0;

    double start = MPI_Wtime();
    // PUSH_RANGE("Jacobi solve", 0)
    while (l2_norm > tol && iter < iter_max) {
        CHECK_HIP(hipMemsetAsync(l2_norm_d, 0, sizeof(real), compute_stream));

        CHECK_HIP(hipStreamWaitEvent(compute_stream, push_top_done, 0));
        CHECK_HIP(hipStreamWaitEvent(compute_stream, push_bottom_done, 0));

        calculate_norm = (iter % nccheck) == 0 || (iter % 100) == 0;
        launch_jacobi_kernel(a_new, a, l2_norm_d, iy_start, iy_end, nx, calculate_norm,
                             compute_stream);
        CHECK_HIP(hipEventRecord(compute_done, compute_stream));

        if (calculate_norm) {
            CHECK_HIP(hipMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real), hipMemcpyDeviceToHost,
                                         compute_stream));
        }

        // Apply periodic boundary conditions

        CHECK_HIP(hipStreamWaitEvent(push_top_stream, compute_done, 0));
        CHECK_HIP(hipMemcpyAsync(a_new, a_new + (iy_end - 1) * nx, nx * sizeof(real),
                                     hipMemcpyDeviceToDevice, push_top_stream));
        CHECK_HIP(hipEventRecord(push_top_done, push_top_stream));

        CHECK_HIP(hipStreamWaitEvent(push_bottom_stream, compute_done, 0));
        CHECK_HIP(hipMemcpyAsync(a_new + iy_end * nx, a_new + iy_start * nx, nx * sizeof(real),
                                     hipMemcpyDeviceToDevice, compute_stream));
        CHECK_HIP(hipEventRecord(push_bottom_done, push_bottom_stream));

        if (calculate_norm) {
            CHECK_HIP(hipStreamSynchronize(compute_stream));
            l2_norm = *l2_norm_h;
            l2_norm = std::sqrt(l2_norm);
            if (print && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }

        std::swap(a_new, a);
        iter++;
    }
    // POP_RANGE
    double stop = MPI_Wtime();

    CHECK_HIP(hipMemcpy(a_ref_h, a, nx * ny * sizeof(real), hipMemcpyDeviceToHost));

    CHECK_HIP(hipEventDestroy(push_bottom_done));
    CHECK_HIP(hipEventDestroy(push_top_done));
    CHECK_HIP(hipEventDestroy(compute_done));
    CHECK_HIP(hipStreamDestroy(push_bottom_stream));
    CHECK_HIP(hipStreamDestroy(push_top_stream));
    CHECK_HIP(hipStreamDestroy(compute_stream));

    CHECK_HIP(hipHostFree(l2_norm_h));
    CHECK_HIP(hipFree(l2_norm_d));

    CHECK_HIP(hipFree(a_new));
    CHECK_HIP(hipFree(a));
    return (stop - start);
}