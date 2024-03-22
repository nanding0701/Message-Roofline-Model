#!/bin/bash
#SBATCH -J test               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 2                 # Total number of mpi tasks requested
#SBATCH -t 00:30:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p mi1008x            # Desired partition


#peer, blocks, threads, iter
ROC_SHMEM_NUM_BLOCK=1 srun -N1 -n2 -c4 ./shmem_put_bw_loopallgpu 1 1 512 1
