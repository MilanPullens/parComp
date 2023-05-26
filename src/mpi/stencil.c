#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <sched.h>
#include <immintrin.h>

#define REAL double

const REAL a = 0.1;
const REAL b = 0.2;
const REAL c = 0.3;

void Stencil(REAL *in, REAL *out, size_t n, int iterations)
{
    for (int t = 1; t <= iterations; t++) {
        for (int i = 1; i < n - 1; i++) {
            out[i] = a * in[i - 1] +
                     b * in[i] +
                     c * in[i + 1];
        }

        if (t != iterations) {
            REAL *temp = in;
            in = out;
            out = temp;
        }
    }    
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 3) {
        if (world_rank == 0) {
            printf("Please specify 2 arguments (n, iterations).\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    int iterations = atoi(argv[2]);

    size_t local_n = n / world_size;
    if (world_rank < n % world_size)
        local_n++;

    // Allocate memory for local data
    REAL *local_in = malloc(local_n * sizeof(REAL));
    REAL *local_out = malloc(local_n * sizeof(REAL));

    // Allocate memory for the global data on process 0
    REAL *global_in = NULL;
    if (world_rank == 0)
        global_in = malloc(n * sizeof(REAL));

    // Initialize the global data on process 0
    if (world_rank == 0) {
        global_in[0] = 100;
        global_in[n - 1] = 1000;
    }

    // Scatter the global input data to all processes
    MPI_Scatter(global_in, local_n, MPI_DOUBLE, local_in, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform stencil computation
    double start_time = MPI_Wtime();
    Stencil(local_in, local_out, local_n, iterations);
    double end_time = MPI_Wtime();

    double duration = end_time - start_time;   


    REAL *result = NULL;
    if (world_rank == 0)
        result = malloc(n * sizeof(REAL));

    MPI_Gather(local_out, local_n, MPI_DOUBLE, result, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);



    if (world_rank == 0) {
        // Compute the overall performance

        // Compute the overall GFLOPS/s
        double gflopsS = 5.0 * (n - 2) * iterations / 1e9 / duration;

        printf("%lf", gflopsS);

        free(result);
    }

    free(local_in);
    free(local_out);
    free(global_in);

    MPI_Finalize();

    return EXIT_SUCCESS;
}