#!/bin/sh

#SBATCH --account=csmpistud
#SBATCH --cpus-per-task=32
#SBATCH --partition=csmpi_fpga_short
#SBATCH --time=00:10:00
#SBATCH --output=stencil.out

# Compile on the machine, not the head node
make bin/stencil_mpi
make clean -C util
make -C util

printf "P,mean,min,max\n" > results/mpi/stencil.csv

for P in 1 2 4 8 16 32; do
    run=1
    while [ "$run" -le 10 ]; do
        {
            mpirun -np "$P" ./bin/stencil_mpi 33554432 256
            printf "\n"
        } >> results/mpi/stencil_temp.csv
        run=$(( run + 1 ))
    done

    {
        printf "%s," "$P"
        util/stat results/mpi/stencil_temp.csv
        printf "\n"
    } >> results/mpi/stencil.csv

    rm results/mpi/stencil_temp.csv
done