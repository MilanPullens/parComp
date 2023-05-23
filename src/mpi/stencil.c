#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0)
            printf("Please specify 2 arguments (n, iterations).\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    size_t n = atoll(argv[1]);
    int iterations = atoi(argv[2]);

    // Determine the local size of the data for each process
    size_t local_n = n / size;
    if (rank < n % size)
        local_n++;

    // Allocate memory for the local portion of the data
    REAL *local_in = calloc(local_n, sizeof(REAL));
    REAL *local_out = malloc(local_n * sizeof(REAL));

    // Initialize the local portion of the data
    local_in[0] = 100;
    local_in[local_n - 1] = 1000;

    double start_time = MPI_Wtime();
    Stencil(local_in, local_out, local_n, iterations);
    double end_time = MPI_Wtime();

    // Compute the local duration
    double duration = end_time - start_time;

    // Gather the results from all processes to process 0
    REAL *result = NULL;
    if (rank == 0) {
        result = malloc(n * sizeof(REAL));
    }
    MPI_Gather(local_out, local_n, MPI_DOUBLE, result, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Compute the overall performance
        double total_duration;
        MPI_Reduce(&duration, &total_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Compute the overall GFLOPS/s
        double gflopsS = 5.0 * (n - 2) * iterations / 1e9 / total_duration;

        printf("This took %lf seconds, or %lf GFLOPS/s\n", total_duration, gflopsS);

        free(result);
    }

    free(local_in);
    free(local_out);

    MPI_Finalize();

    return EXIT_SUCCESS;
}