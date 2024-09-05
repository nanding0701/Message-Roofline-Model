#!/bin/bash -le

#SBATCH -A nstaff_g
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH -o mpi_colls_1.o%j
#SBATCH -e mpi_colls_1.e%j
#SBATCH -J mpi_colls_1

ml PrgEnv-gnu

BUILD=/global/homes/m/mhaseeb/repos/ex-msg-roofline/build/apps/mpi

mkdir -p ${BUILD}/logs

srun --export=ALL -n 2 -G 2 --gpus-per-node=2 --ntasks-per-node=2 ${BUILD}/collective_mpi |& tee ${BUILD}/logs/gdr_coll_1n_2g.log
srun --export=ALL -n 4 -G 4 --gpus-per-node=4 --ntasks-per-node=4 ${BUILD}/collective_mpi |& tee ${BUILD}/logs/gdr_coll_1n_4g.log


srun --export=ALL -n 2 -G 2 --gpus-per-node=2 --ntasks-per-node=2 ${BUILD}/collective_mpi_nogdr |& tee ${BUILD}/logs/nogdr_coll_1n_2g.log
srun --export=ALL -n 4 -G 4 --gpus-per-node=4 --ntasks-per-node=4 ${BUILD}/collective_mpi_nogdr |& tee ${BUILD}/logs/nogdr_coll_1n_4g.log
