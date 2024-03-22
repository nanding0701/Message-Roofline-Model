#!/bin/bash -le

#SBATCH -A nstaff_g
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --time=3:00:00
#SBATCH --nodes=4
#SBATCH --gpus=16
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH -o nvshmem_colls.o%j
#SBATCH -e nvshmem_colls.e%j
#SBATCH -J nvshmem_colls

ml use /global/cfs/cdirs/m1759/mhaseeb/nvshmem_src_2.9.0-2/modulefiles
ml nvshmem

export NVSHMEM_SYMMETRIC_SIZE=67108864

BUILD=/global/homes/m/mhaseeb/repos/ex-msg-roofline/build/apps/nvshmem

mkdir -p ${BUILD}/logs

srun --export=ALL -n 4 -G 4 --gpus-per-node=1  --ntasks-per-node=1   ${BUILD}/collective_nvshmem |& tee ${BUILD}/logs/coll_4n_4g.log
srun --export=ALL -n 8 -G 8 --gpus-per-node=2  --ntasks-per-node=2   ${BUILD}/collective_nvshmem |& tee ${BUILD}/logs/coll_4n_8g.log
srun --export=ALL -n 16 -G 16 --gpus-per-node=4  --ntasks-per-node=4 ${BUILD}/collective_nvshmem |& tee ${BUILD}/logs/coll_4n_16g.log
