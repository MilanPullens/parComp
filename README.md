# Installation

It is recommended to use WSL on windows in order to program locally,
as MPI on windows is a pain, and the cluster you will need for experiments
uses Ubuntu. Install either mpich or openmpi dev packages for your distribution.
E.g.

```
sudo apt install libopenmpi-dev
```

in Ubuntu. On the cluster everything is already installed.

# Compilation

Put your sequential programs in src/seq, your OpenMP programs in src/omp,
your MPI programs in src/mpi.
Compile with make and find your programs in bin/ with suffixes _seq, _omp, _mpi

# Experiments on the cluster

To run experiments on the cluster, you will need to login to slurm22, e.g.

```
ssh tkoopman@lilo7.science.ru.nl
ssh slurm22
```

and request resources from there. This is done through SLURM, look up the
documentation. omp_hello.sh is a simple example script. Submit these
using sbatch, so

```
sbatch omp_hello.sh
```

With MPI, you have to be careful how you allocate resources, to make sure
you do not schedule multiple ranks on one physical core (CPUs i and i + 8 are
the same physical core). The script mpi_hello.sh generates SLURM scripts from
mpi_hello.sh.template and submits them.

scp is a useful command to get files to and from the cluster.

# Example solution OpenMP
For the MPI assignment you are asked to compare the performance against the
OpenMP implementation. As you may not have succeeded or parallelised all three
versions, an example solution is provided. In results you can find csvs for
these three versions on input n = 999000000, iterations = 300
P is processors, mean is the average Gflops/s, min is the slowest run,
max is the fastest run.
They were generated using omp_stencil.sh, omp_stencil_opt.sh, omp_stencil_avx2.sh
See graph.tex for how you can generate a graph with
error bars in Latex from these csvs. Using a different tool is of course
also allowed.
