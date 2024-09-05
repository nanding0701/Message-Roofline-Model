#!/bin/bash -le

#SBATCH -A nstaff_g
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --time=6:00:00
#SBATCH --nodes=8
#SBATCH --gpus=32
#SBATCH --exclusive
#SBATCH --gpu-bind=none
#SBATCH -o mpi_colls_8.o%j
#SBATCH -e mpi_colls_8.e%j
#SBATCH -J mpi_colls_8

ml PrgEnv-gnu

BUILD=/global/homes/m/mhaseeb/repos/ex-msg-roofline/build/apps/mpi

mkdir -p ${BUILD}/logs

srun --export=ALL -n 8 -G 8 --gpus-per-node=1  --ntasks-per-node=1   ${BUILD}/collective_mpi |& tee ${BUILD}/logs/gdr_coll_8n_8g.log
srun --export=ALL -n 16 -G 16 --gpus-per-node=2  --ntasks-per-node=2 ${BUILD}/collective_mpi |& tee ${BUILD}/logs/gdr_coll_8n_16g.log
srun --export=ALL -n 32 -G 32 --gpus-per-node=4  --ntasks-per-node=4 ${BUILD}/collective_mpi |& tee ${BUILD}/logs/gdr_coll_8n_32g.log

srun --export=ALL -n 8 -G 8 --gpus-per-node=1  --ntasks-per-node=1   ${BUILD}/collective_mpi_nogdr |& tee ${BUILD}/logs/nogdr_coll_8n_8g.log
srun --export=ALL -n 16 -G 16 --gpus-per-node=2  --ntasks-per-node=2 ${BUILD}/collective_mpi_nogdr |& tee ${BUILD}/logs/nogdr_coll_8n_16g.log
srun --export=ALL -n 32 -G 32 --gpus-per-node=4  --ntasks-per-node=4 ${BUILD}/collective_mpi_nogdr |& tee ${BUILD}/logs/nogdr_coll_8n_32g.log
