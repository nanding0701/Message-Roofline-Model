#!/bin/bash
#SBATCH -J test               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 8                # Total number of mpi tasks requested
#SBATCH -t 02:00:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -p mi1008x            # Desired partition


#peer, blocks, threads, iter
module use /share/bpotter/modulefiles/
module load rocshmem/1.6.3
module list

gpus=(1 2 3 4 5 6 7 8)
ins=4000000
for mygpu in ${gpus[@]}
do
	#my_ins_num=`expr $ins / ${mygpu}`
	#echo ${mygpu}, ${my_ins_num}
	#srun -N1 -n${mygpu} -c4 ./hashtable_rocshmem ${my_ins_num} |& tee log_hash_G${mygpu}_N${my_ins_num}
	srun -N1 -n${mygpu} -c4 ./hashtable_rocshmem ${ins} |& tee log_hash_weak_scale_4e6_G${mygpu}
done
