#!/bin/sh

#!/bin/bash

#SBATCH --account=csmpi
#SBATCH --partition=csmpi_fpga_short
#SBATCH --gres=gpu:1
#SBATCH --time=0:05:00
#SBATCH --output=opencl.out

# Compile on the machine, not the head node
make bin/square_cl

bin/square_cl > results/square.txt
