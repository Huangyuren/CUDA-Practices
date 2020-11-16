#!/bin/bash
nvcc print_deviceINFO.cu -o info.out
./info.out


nvcc vector_mult_2D.cu -o Mult2D.out
echo "Start normal matrix multiplication process."
reso_base=1024
for multiplier in $(seq 1 5)
do
    ./Mult2D.out $reso_base $reso_base $reso_base
    reso_base=$((2 * $reso_base))
done

nvcc vector_mult_2D_UM.cu -o Mult2D_UM.out
echo "Start unified memory with prefetcher process."
reso_base=1024
for multiplier in $(seq 1 6)
do
    ./Mult2D_UM.out $reso_base $reso_base $reso_base
    reso_base=$((2 * $reso_base))
done
