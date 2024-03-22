#!/bin/bash

grid=16384
energy=1
iters=1000
proc=(4 8 16 32 64 128 256 512)
px=(2 3 4 5 6 7 8 9 7)
py=(2 3 4 5 7 7 8 9 12)
data='twosided,4msg'
data1='twosided,2msg'
data2='onesided,4msg'
data3='onesided,2msg'
for p in ${proc[@]};
do 
    
	    data+=`grep -ri "last heat" log_2023-04-07-11-57_stencil_mpi_ddt_${p} | awk '{print $6}'`
	    data+=','
	    data1+=`grep -ri "last heat" log_2023-04-07-11-57_stencil_mpi_ddt_2msg_${p} | awk '{print $6}'`
	    data1+=','
	    data2+=`grep -ri "last heat" log_2023-04-07-11-57_stencil_mpi_ddt_rma__${p} | awk '{print $6}'`
	    data2+=','
	    data3+=`grep -ri "last heat" log_2023-04-07-11-57_stencil_mpi_ddt_rma_2msg_${p} | awk '{print $6}'`
	    data3+=','

done
echo $data
echo $data1
echo $data2
echo $data3

unset data, data1,data2,data3

proc=(4 8 16 32 64 128)
data='twosided,4msg,2node,'
data1='twosided,2msg,2node,'
data2='onesided,4msg,2node,'
data3='onesided,2msg,2node,'
for p in ${proc[@]};
do
    data+=`grep -ri "last heat" log_2023-04-07-12-02_2node_stencil_mpi_ddt_${p} | awk '{print $6}'`
    data+=','
    data1+=`grep -ri "last heat" log_2023-04-07-12-02_2node_stencil_mpi_ddt_2msg_${p} | awk '{print $6}'`
    data1+=','
    data2+=`grep -ri "last heat" log_2023-04-07-12-02_2node_stencil_mpi_ddt_rma__${p} | awk '{print $6}'`
    data2+=','
    data3+=`grep -ri "last heat" log_2023-04-07-12-02_2node_stencil_mpi_ddt_rma_2msg_${p} | awk '{print $6}'`
    data3+=','
done
echo $data
echo $data1
echo $data2
echo $data3


#jsrun -n84 -c1 -a1 ./exe_put_flush $iter      21 |& tee log_${MYDATE}_2node_putflush_84_21_${iter}
#jsrun -n84 -c1 -a1 ./exe_two_sided $iter1     21  |& tee log_${MYDATE}_2node_twosided_84_21_${iter}

#jsrun -n42 -c1 -a1 ./exe_put_flush_theo $iter 21 |& tee log_${MYDATE}_1node_putflush_theo_42_21_${iter}
#jsrun -n42 -c1 -a1 ./exe_put_flush $iter      21 |& tee log_${MYDATE}_1node_putflush_42_21_${iter}
#jsrun -n42 -c1 -a1 ./exe_two_sided $iter      21 |& tee log_${MYDATE}_1node_twosided_42_21_${iter}


