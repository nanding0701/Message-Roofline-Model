#include "xmr.hpp"
#include "commons/mpi.hpp"
#include "commons/hip.hpp"
#include "commons/rocshmem.hpp"

#define B_TO_GB (1 << 30)
#define MS_TO_S 1000

#define MAX_ITERS 10
#define MAX_SKIP 10
#define THREADS 1024
#define BLOCKS 4
#define MAX_MSG_SIZE (16 * 1024 * 1024)

__global__ void atomic_compare_swap_bw(
    uint64_t *data_d,
    volatile unsigned int *counter_d,
    int len,
    int pe,
    int iter,
    int peer) {

  int i, j, tid, slice;
  unsigned int counter;
  int threads = gridDim.x * blockDim.x;
  tid = blockIdx.x * blockDim.x + threadIdx.x;

  //peer = !pe;
  slice = threads;

  for (i = 0; i < iter; i++) {
    for (j = 0; j < len - slice; j += slice) {
      int idx = j + tid;
      roc_shmem_uint64_atomic_compare_swap(data_d + idx, i, i + 1, peer);
      __syncthreads();
    }

    int idx = j + tid;
    if (idx < len) {
      roc_shmem_uint64_atomic_compare_swap(data_d + idx, i, i + 1, peer);
    }

    /* synchronizing across blocks */
    __syncthreads();

    if (!threadIdx.x) {
      __threadfence();
      counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
      if (counter == (gridDim.x * (i + 1) - 1)) {
        *(counter_d + 1) += 1;
      }
      while (*(counter_d + 1) != i + 1)
        ;
    }

    __syncthreads();
  }

  /* synchronizing across blocks */
  __syncthreads();

  if (!threadIdx.x) {
    __threadfence();
    counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
    if (counter == (gridDim.x * (i + 1) - 1)) {
      roc_shmem_quiet();
      *(counter_d + 1) += 1;
    }
    while (*(counter_d + 1) != i + 1)
      ;
  }
}

int main(int argc, char *argv[]) {
  int mype, npes;
  int mypeer;
  uint64_t *data_d = NULL;
  uint64_t set_value;
  unsigned int *counter_d;
  int max_blocks = BLOCKS, max_threads = THREADS;
  int array_size, i;
  void **h_tables;
  uint64_t *h_size_arr;
  double *h_bw;

  int iter = MAX_ITERS;
  int skip = MAX_SKIP;
  int max_msg_size = MAX_MSG_SIZE;

  float milliseconds;
  hipEvent_t start, stop;

  int rank, nranks;
  MPI_CHECK(MPI_Init(&argc, &argv));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
  //printf("Rank %d, MPI \n", rank);

  // Set the device before calling `roc_shmem_init`
  int ndevices, get_cur_dev;
  CHECK_HIP(hipGetDeviceCount(&ndevices));
  CHECK_HIP(hipSetDevice(rank % ndevices));
  CHECK_HIP(hipGetDevice(&get_cur_dev));

  printf("Device ID %d, HIP \n", get_cur_dev);

  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));

  roc_shmem_init();
  mype = roc_shmem_my_pe();
  npes = roc_shmem_n_pes();
  printf("Rank %d, ROC_SHMEM \n", mype);
  fflush(stdout);

  const char *gpu_id_list;
  const char *rocr_visible_devices = getenv("ROCR_VISIBLE_DEVICES");
  if (rocr_visible_devices == NULL) {
    gpu_id_list = "N/A";
  } else {
    gpu_id_list = rocr_visible_devices;
  }

  printf("roc_shmem %d/%d , ndevices=%d,cur=%d, GPU_ID=%s\n", mype, npes,
         ndevices, get_cur_dev, gpu_id_list);
  fflush(stdout);

  if (argc > 1) {
    // mypeer is unused, keeping it to match other tests
    // device 1 is used as the peer
    mypeer = atoi(argv[1]);
    if (atoi(argv[2]) > 0) max_blocks = atoi(argv[2]);
    if (atoi(argv[3]) > 0) max_threads = atoi(argv[3]);
    if (atoi(argv[4]) > 0) iter = atoi(argv[4]);
  }
  if (!mype) {
    printf("max_blocks=%d, max_threads=%d, iter=%d\n",
            max_blocks, max_threads, iter);
    fflush(stdout);
  }

  data_d = (uint64_t *)roc_shmem_malloc(max_msg_size);
  CHECK_HIP(hipMemset(data_d, 0, max_msg_size));

  CHECK_HIP(hipMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
  CHECK_HIP(hipMemset(counter_d, 0, sizeof(unsigned int) * 2));

  CHECK_HIP(hipDeviceSynchronize());

  int size;
  i = 0;
  if (mype == 0) {
    for (size = 8; size <= MAX_MSG_SIZE; size *= 2) {

      int blocks = max_blocks, threads = max_threads;

      CHECK_HIP(hipMemset(counter_d, 0, sizeof(unsigned int) * 2));
      atomic_compare_swap_bw<<<blocks, threads>>>(
        data_d, counter_d, size / sizeof(uint64_t), mype, skip, mypeer);

      CHECK_HIP(hipGetLastError());
      CHECK_HIP(hipDeviceSynchronize());
      roc_shmem_barrier_all();

      // reset values in code
      CHECK_HIP(hipMemset(counter_d, 0, sizeof(unsigned int) * 2));
      CHECK_HIP(hipGetLastError());
      CHECK_HIP(hipDeviceSynchronize());
      roc_shmem_barrier_all();

      CHECK_HIP(hipEventRecord(start));
      atomic_compare_swap_bw<<<blocks, threads>>>(
        data_d, counter_d, size / sizeof(uint64_t), mype, iter, mypeer);
      CHECK_HIP(hipEventRecord(stop));
      CHECK_HIP(hipGetLastError());
      CHECK_HIP(hipEventSynchronize(stop));
      CHECK_HIP(hipEventElapsedTime(&milliseconds, start, stop));

      printf("peer,%d,size,%d,iter,%d,bw,%f\n", mypeer, size, iter,
              size / (milliseconds * (B_TO_GB / (iter * MS_TO_S))));
      fflush(stdout);
      roc_shmem_barrier_all();
    }
  }  else {
    for (size = 8; size <= MAX_MSG_SIZE; size *= 2) {
      roc_shmem_barrier_all();
      roc_shmem_barrier_all();
      roc_shmem_barrier_all();
    }
  }

  if (data_d) roc_shmem_free(data_d);
  if (counter_d) CHECK_HIP(hipFree(counter_d));

  roc_shmem_finalize();
  MPI_Finalize();
  return 0;
}