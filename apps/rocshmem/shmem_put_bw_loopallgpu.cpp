#include <assert.h>
#include <getopt.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <roc_shmem.hpp>

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#define B_TO_GB (1 << 30)
#define MS_TO_S 1000

#define MAX_MSG_SIZE (32 * 1024 * 1024)
#define MAX_ITERS 200
#define MAX_SKIP 20
#define BLOCKS 4
#define THREADS_PER_BLOCK 1024
using namespace rocshmem;

#define CHECK_HIP(cmd)                                                        \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), \
              error, __FILE__, __LINE__);                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

__global__ void bw(double *data_d, int *flag_d,
                   volatile unsigned int *counter_d, int len, int pe, int iter,
                   int peer) {
  int i;
  unsigned int counter;
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  int sig = 1;
  roc_shmem_wg_init();
  for (i = 0; i < iter; i++) {
    roc_shmemx_double_put_nbi_wg(data_d + (bid * (len / nblocks)),
                                 data_d + (bid * (len / nblocks)),
                                 len / nblocks, peer);

    roc_shmem_fence();
    //  synchronizing across blocks
    __syncthreads();
    if (!tid) {
      __threadfence();
      counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
      if (counter == (hipGridDim_x * (i + 1) - 1)) {
        *(counter_d + 1) += 1;
      }
      while (*(counter_d + 1) != i + 1);
    }
    __syncthreads();
  }

  // synchronize and call roc_shmem_quiet
  __syncthreads();
  if (!tid) {
    __threadfence();
    counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
    if (counter == (hipGridDim_x * (i + 1) - 1)) {
      roc_shmem_quiet();
      *(counter_d + 1) += 1;
    }
    while (*(counter_d + 1) != i + 1);
  }
  __syncthreads();
  roc_shmem_wg_finalize();
}

int main(int argc, char *argv[]) {
  int mype, npes;
  double *data_d = NULL;
  int *flag_d = NULL;
  unsigned int *counter_d;
  int max_blocks = BLOCKS, max_threads = THREADS_PER_BLOCK;

  int i;

  int iter = MAX_ITERS;
  int skip = MAX_SKIP;

  float milliseconds;
  hipEvent_t start, stop;

  int rank, nranks;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
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

  data_d = (double *)roc_shmem_malloc(MAX_MSG_SIZE);
  flag_d = (int *)roc_shmem_malloc((iter + skip) * sizeof(int));
  CHECK_HIP(hipMemset(data_d, 0, MAX_MSG_SIZE));
  CHECK_HIP(hipMemset(flag_d, 0, (iter + skip) * sizeof(int)));
  CHECK_HIP(hipMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
  CHECK_HIP(hipMemset(counter_d, 0, sizeof(unsigned int) * 2));
  CHECK_HIP(hipDeviceSynchronize());
  int mypeer = atoi(argv[1]);
  if (atoi(argv[2]) > 0) max_blocks = atoi(argv[2]);
  if (atoi(argv[3]) > 0) max_threads = atoi(argv[3]);
  if (atoi(argv[4]) > 0) iter = atoi(argv[4]);
  if (!mype) {
    printf("max_blocks=%d, max_threads=%d, iter=%d\n", max_blocks, max_threads,
           iter);
  }
  fflush(stdout);
  for (int peer = mypeer; peer < 8; peer++) {
    if (mype == 0) {
      i = 0;
      for (int size = 8; size <= MAX_MSG_SIZE; size *= 2) {
        CHECK_HIP(hipMemset(counter_d, 0, sizeof(unsigned int) * 2));
        bw<<<max_blocks, max_threads>>>(
            data_d, flag_d, counter_d, size / sizeof(double), mype, skip, peer);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemset(counter_d, 0, sizeof(unsigned int) * 2));

        CHECK_HIP(hipEventRecord(start));
        bw<<<max_blocks, max_threads>>>(
            data_d, flag_d, counter_d, size / sizeof(double), mype, iter, peer);
        CHECK_HIP(hipEventRecord(stop));

        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipEventSynchronize(stop));

        CHECK_HIP(hipEventElapsedTime(&milliseconds, start, stop));
        printf("peer,%d,size,%d,iter, %d, bw,%f\n", peer, size, iter,
               size / (milliseconds * (B_TO_GB / (iter * MS_TO_S))));
        roc_shmem_barrier_all();
        i++;
      }
    } else {
      for (int size = 8; size <= MAX_MSG_SIZE; size *= 2) {
        roc_shmem_barrier_all();
      }
    }
  }
finalize:

  if (data_d) {
    roc_shmem_free(data_d);
  }
  if (counter_d) {
    CHECK_HIP(hipFree(counter_d));
  }
  roc_shmem_finalize();
  MPI_Finalize();
  return 0;
}
