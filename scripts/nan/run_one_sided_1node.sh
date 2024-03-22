#!/bin/bash
#SBATCH --job-name=jacobi
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --time=02:00:00
###SBATCH --mail-user=nanding@lbl.gov
###SBATCH --mail-type=ALL

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

MYDATE=$(date '+%Y-%m-%d-%H-%M')
echo $MYDATE

grid=16384
energy=1
iters=1000
proc=(4 8 16 32 64 128) #49 64 81 84)
px=(2 4 4 8 8 16) #7 8 9 7)
py=(2 2 4 4 8 8) #7 8 9 12)
for i in ${!proc[@]};
do
    myc=`expr 256 / ${proc[$i]}`
	echo $i,${proc[$i]},${px[$i]},${py[$i]},${myc}
	#run the application:
	#srun -n${proc[$i]} -c${myc}  ./stencil_mpi_ddt_rma_2msg $grid $energy $iters ${px[$i]} ${py[$i]} |& tee log_${MYDATE}_stencil_mpi_ddt_rma_2msg_${proc[$i]}
	#srun -n${proc[$i]} -c${myc}  ./stencil_mpi_ddt_2msg $grid $energy $iters ${px[$i]} ${py[$i]} |& tee log_${MYDATE}_stencil_mpi_ddt_2msg_${proc[$i]}
	srun -n${proc[$i]} -c${myc}  ./stencil_mpi_ddt_rma $grid $energy $iters ${px[$i]} ${py[$i]} |& tee log_${MYDATE}_stencil_mpi_ddt_rma__${proc[$i]}
	srun -n${proc[$i]} -c${myc}  ./stencil_mpi_ddt $grid $energy $iters ${px[$i]} ${py[$i]} |& tee log_${MYDATE}_stencil_mpi_ddt_${proc[$i]}
done


