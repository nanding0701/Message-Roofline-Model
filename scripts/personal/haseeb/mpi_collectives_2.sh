#!/bin/bash -le

#SBATCH -A nstaff_g
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --time=3:00:00
#SBATCH --nodes=2
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH -o mpi_colls_2.o%j
#SBATCH -e mpi_colls_2.e%j
#SBATCH -J mpi_colls_2

ml PrgEnv-gnu

BUILD=/global/homes/m/mhaseeb/repos/ex-msg-roofline/build/apps/mpi

mkdir -p ${BUILD}/logs

srun --export=ALL -n 2 -G 2 --gpus-per-node=1 --ntasks-per-node=1 ${BUILD}/collective_mpi |& tee ${BUILD}/logs/gdr_coll_2n_2g.log
srun --export=ALL -n 4 -G 4 --gpus-per-node=2  --ntasks-per-node=2 ${BUILD}/collective_mpi |& tee ${BUILD}/logs/gdr_coll_2n_4g.log
srun --export=ALL -n 8 -G 8 --gpus-per-node=4  --ntasks-per-node=4 ${BUILD}/collective_mpi |& tee ${BUILD}/logs/gdr_coll_2n_8g.log


srun --export=ALL -n 2 -G 2 --gpus-per-node=1 --ntasks-per-node=1 ${BUILD}/collective_mpi_nogdr |& tee ${BUILD}/logs/nogdr_coll_2n_2g.log
srun --export=ALL -n 4 -G 4 --gpus-per-node=2  --ntasks-per-node=2 ${BUILD}/collective_mpi_nogdr |& tee ${BUILD}/logs/nogdr_coll_2n_4g.log
srun --export=ALL -n 8 -G 8 --gpus-per-node=4  --ntasks-per-node=4 ${BUILD}/collective_mpi_nogdr |& tee ${BUILD}/logs/nogdr_coll_2n_8g.log
