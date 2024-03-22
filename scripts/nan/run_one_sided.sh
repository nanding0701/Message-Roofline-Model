#!/bin/bash
#BSUB -P BIF115
#BSUB -W 02:00
#BSUB -nnodes 1
#BSUB -J bench

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

MYDATE=$(date '+%Y-%m-%d-%H-%M')
echo $MYDATE
grid=16384
energy=1
iters=1000
proc=(4 16 25 36 42) #49 64 81 84)
px=(2 3 4 5 6) #7 8 9 7)
py=(2 3 4 5 7) #7 8 9 12)
for i in ${!proc[@]};
do
	echo $i,${proc[$i]},${px[$i]},${py[$i]}
	#run the application:
	jsrun -n${proc[$i]} -c1 -a1 ./stencil_mpi_ddt_rma_2msg $grid $energy $iters ${px[$i]} ${py[$i]} |& tee log_${MYDATE}_stencil_mpi_ddt_rma_2msg_${proc[$i]}
	jsrun -n${proc[$i]} -c1 -a1 ./stencil_mpi_ddt_2msg $grid $energy $iters ${px[$i]} ${py[$i]} |& tee log_${MYDATE}_stencil_mpi_ddt_2msg_${proc[$i]}
	jsrun -n${proc[$i]} -c1 -a1 ./stencil_mpi_ddt_rma $grid $energy $iters ${px[$i]} ${py[$i]} |& tee log_${MYDATE}_stencil_mpi_ddt_rma__${proc[$i]}
	jsrun -n${proc[$i]} -c1 -a1 ./stencil_mpi_ddt $grid $energy $iters ${px[$i]} ${py[$i]} |& tee log_${MYDATE}_stencil_mpi_ddt_${proc[$i]}
done
#jsrun -n84 -c1 -a1 ./exe_put_flush $iter      21 |& tee log_${MYDATE}_2node_putflush_84_21_${iter}
#jsrun -n84 -c1 -a1 ./exe_two_sided $iter1     21  |& tee log_${MYDATE}_2node_twosided_84_21_${iter}

#jsrun -n42 -c1 -a1 ./exe_put_flush_theo $iter 21 |& tee log_${MYDATE}_1node_putflush_theo_42_21_${iter}
#jsrun -n42 -c1 -a1 ./exe_put_flush $iter      21 |& tee log_${MYDATE}_1node_putflush_42_21_${iter}
#jsrun -n42 -c1 -a1 ./exe_two_sided $iter      21 |& tee log_${MYDATE}_1node_twosided_42_21_${iter}


