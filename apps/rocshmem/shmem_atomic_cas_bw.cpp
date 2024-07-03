#include "xmr.hpp"
#include "commons/mpi.hpp"
#include "commons/hip.hpp"
#include "commons/rocshmem.hpp"

int main(int argc, char *argv[]) {
  int mype, npes;

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

  roc_shmem_finalize();
  MPI_Finalize();
  return 0;
}