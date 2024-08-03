#include "xmr.hpp"
#include "commons/mpi.hpp"
#include "commons/hip.hpp"
#include "commons/rccl.hpp"

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

    int result_correct = 1;

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