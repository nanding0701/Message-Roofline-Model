#!/bin/bash -le

#SBATCH -A nstaff_g
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --time=5:00:00
#SBATCH --nodes=4
#SBATCH --gpus=16
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH -o mpi_colls_4.o%j
#SBATCH -e mpi_colls_4.e%j
#SBATCH -J mpi_colls_4

ml PrgEnv-gnu

BUILD=/global/homes/m/mhaseeb/repos/ex-msg-roofline/build/apps/mpi

mkdir -p ${BUILD}/logs

srun --export=ALL -n 4 -G 4 --gpus-per-node=1  --ntasks-per-node=1   ${BUILD}/collective_mpi |& tee ${BUILD}/logs/${1}gdr_coll_4n_4g.log
srun --export=ALL -n 8 -G 8 --gpus-per-node=2  --ntasks-per-node=2   ${BUILD}/collective_mpi |& tee ${BUILD}/logs/${1}gdr_coll_4n_8g.log
srun --export=ALL -n 16 -G 16 --gpus-per-node=4  --ntasks-per-node=4 ${BUILD}/collective_mpi |& tee ${BUILD}/logs/${1}gdr_coll_4n_16g.log

srun --export=ALL -n 4 -G 4 --gpus-per-node=1  --ntasks-per-node=1   ${BUILD}/collective_mpi_nogdr |& tee ${BUILD}/logs/${1}nogdr_coll_4n_4g.log
srun --export=ALL -n 8 -G 8 --gpus-per-node=2  --ntasks-per-node=2   ${BUILD}/collective_mpi_nogdr |& tee ${BUILD}/logs/${1}nogdr_coll_4n_8g.log
srun --export=ALL -n 16 -G 16 --gpus-per-node=4  --ntasks-per-node=4 ${BUILD}/collective_mpi_nogdr |& tee ${BUILD}/logs/${1}nogdr_coll_4n_16g.log

