# Message Roofline Model
(On top of the work from this paper: "Evaluating the Performance of One-sided Communication on CPUs and GPUs." Proceedings of the SC'23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis. 2023.)

# Copyright
The NVSHMEM and NCCL jacobi code is from https://github.com/NVIDIA/multi-gpu-programming-models.git. Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.

See the COPYRIGHT file for more information on any additional dependencies.

# ROC_SHMEM and RCCL
The ROC_SHMEM and RCCL code are modified from NVIDIA versions.

# Intel® SHMEM
To build the Intel SHMEM applications, please disable CUDA and NVSHMEM, then
set `ISHMEM_ROOT` to the installation directory of Intel SHMEM and `SHMEM_ROOT`
to the installation directory of a host OpenSHMEM runtime (as of this writing,
Sandia OpenSHMEM is required).  Please set `CXX` to the `oshc++` compiler
wrapper. The following was tested on Intel® Tiber™ AI Cloud with Intel® Max
Series GPU (PVC) on 4th Gen Intel® Xeon® processors – 1550 series (8x) with
machine image Ubuntu 22.04 LTS (Jammy Jellyfish) v20240522:
```
CXX=oshc++ cmake .. -DUSE_CUDA=OFF -DUSE_NVSHMEM=OFF -DUSE_ISHMEM=ON -DISHMEM_ROOT=$ISHMEM_ROOT -DSHMEM_ROOT=$SHMEM_ROOT
mpirun -n 16 ishmrun ./apps/ishmem/jacobi
```
