#!/bin/sh

#SBATCH --account=csmpistud
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --threads-per-core=1
#SBATCH --partition=csmpi_short
#SBATCH --time=00:05:00
#SBATCH --output=hello_mpi_8.out

# Compile on the machine, not the head node
make bin/hello_world_mpi

mpirun bin/hello_world_mpi > results/hello_mpi_8.txt
