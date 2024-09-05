#!/bin/bash -le

ml use /global/cfs/cdirs/m1759/mhaseeb/nvshmem_src_2.9.0-2/modulefiles
ml nvshmem

BUILD=/global/homes/m/mhaseeb/repos/ex-msg-roofline/build/apps/mpi

make -C ${BUILD}

nodes=(1 2 4 8)

for n in "${nodes[@]}"; do

    sbatch ./mpi_collectives_${n}.sh

done
