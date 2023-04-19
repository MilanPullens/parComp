#include <stdlib.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#define CHECK

/* You may need a different method of timing if you are not on Linux. */
#define TIME(duration, fncalls)                                        \
    do {                                                               \
        struct timeval tv1, tv2;                                       \
        gettimeofday(&tv1, NULL);                                      \
        fncalls                                                        \
        gettimeofday(&tv2, NULL);                                      \
        duration = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +    \
         (double) (tv2.tv_sec - tv1.tv_sec);                           \
    } while (0)

const double a = 0.3;
const double b = 0.5;
const double c = 0.4;

/* We split up the stencil in smaller stencils, of roughly SPACEBLOCK size,
 * and treat them for TIMEBLOCK iterations. Play around with these. Do the considerations
 * change when parallelising? */
const int SPACEBLOCK = 1250;
const int TIMEBLOCK = 100;

/* Takes buffers *in, *out of size n + iterations.
 * out[0: n - 1] is the first part of the stencil of in[0, n + iterations - 1]. */
void Left(double **in, double **out, size_t n, int iterations)
{
    (*out)[0] = (*in)[0];

    for (int t = 1; t <= iterations; t++) {
        for (size_t i = 1; i < n + iterations - t; i++) {
            (*out)[i] = a * (*in)[i - 1] + b * (*in)[i] + c * (*in)[i + 1];
        }

        if (t != iterations) {
            double *temp = *in;
            *in = *out;
            *out = temp;
        }
    }
}

/* Takes buffers *in, *out of size n + 2 * iterations.
 * out[iterations: n + iterations - 1] is the
 * middle part of the stencil of in[0, n + 2 * iterations - 1]. */
void Middle(double **in, double **out, size_t n, int iterations)
{
    for (int t = 1; t <= iterations; t++) {
        for (size_t i = t; i < n + 2 * iterations - t; i++) {
            (*out)[i] = a * (*in)[i - 1] + b * (*in)[i] + c * (*in)[i + 1];
        }

        if (t != iterations) {
            double *temp = *in;
            *in = *out;
            *out = temp;
        }
    }

}

/* Takes buffers *in, *out of size n + iterations.
 * out[iterations: n + iterations - 1] is the last part of
 * the stencil of in[0, n + iterations - 1]. */
void Right(double **in, double **out, size_t n, int iterations)
{
    (*out)[n + iterations - 1] = (*in)[n + iterations - 1];

    for (int t = 1; t <= iterations; t++) {
        for (size_t i = t; i < n + iterations - 1; i++) {
            (*out)[i] = a * (*in)[i - 1] + b * (*in)[i] + c * (*in)[i + 1];
        }

        if (t != iterations) {
            double *temp = *in;
            *in = *out;
            *out = temp;
        }
    }
}

void StencilBlocked(double **in, double **out, size_t n, int iterations)
{
    double *inBuffer = malloc((SPACEBLOCK + 2 * iterations) * sizeof(double));
    double *outBuffer = malloc((SPACEBLOCK + 2 * iterations) * sizeof(double));

    for (size_t block = 0; block < n / SPACEBLOCK; block++) {
        if (block == 0) {
            memcpy(inBuffer, *in, (SPACEBLOCK + iterations) * sizeof(double));
            Left(&inBuffer, &outBuffer, SPACEBLOCK, iterations);
            memcpy(*out, outBuffer, SPACEBLOCK * sizeof(double));
        } else if (block == n / SPACEBLOCK - 1) {
            memcpy(inBuffer, *in + block * SPACEBLOCK - iterations,
                    (SPACEBLOCK + iterations) * sizeof(double));
            Right(&inBuffer, &outBuffer, SPACEBLOCK, iterations);
            memcpy(*out + block * SPACEBLOCK, outBuffer + iterations, SPACEBLOCK * sizeof(double));
        } else {
            memcpy(inBuffer, *in + block * SPACEBLOCK - iterations,
                    (SPACEBLOCK + 2 * iterations) * sizeof(double));
            Middle(&inBuffer, &outBuffer, SPACEBLOCK, iterations);
            memcpy(*out + block * SPACEBLOCK, outBuffer + iterations, SPACEBLOCK * sizeof(double));
        }
    }

    free(inBuffer);
    free(outBuffer);
}

void Stencil(double **in, double **out, size_t n, int iterations)
{
    int rest_iters = iterations % TIMEBLOCK;
    if (rest_iters != 0) {
        fprintf(stderr, "rest iter\n");
        StencilBlocked(in, out, n, rest_iters);
        double *temp = *out;
        *out = *in;
        *in = temp;
    }

    for (int t = rest_iters; t < iterations; t += TIMEBLOCK) {
        fprintf(stderr, "%d\n", t);
        StencilBlocked(in, out, n, TIMEBLOCK);
        double *temp = *out;
        *out = *in;
        *in = temp;
    }

    double *temp = *out;
    *out = *in;
    *in = temp;
}

void StencilSlow(double **in, double **out, size_t n, int iterations)
{
    (*out)[0] = (*in)[0];
    (*out)[n - 1] = (*in)[n - 1];

    for (int t = 1; t <= iterations; t++) {
        /* Update only the inner values. */
        for (int i = 1; i < n - 1; i++) {
            (*out)[i] = a * (*in)[i - 1] +
                        b * (*in)[i] +
                        c * (*in)[i + 1];
        }

        /* The output of this iteration is the input of the next iteration (if there is one). */
        if (t != iterations) {
            double *temp = *in;
            *in = *out;
            *out = temp;
        }
    }
}

#ifdef CHECK
bool equal(double *x, double *y, size_t n, double error)
{
    for (size_t i = 0; i < n; i++) {
        if (fabs(x[i] - y[i]) > error) {
            printf("Index %zu: %lf != %lf\n", i, x[i], y[i]);
            return false;
        }
    }

    return true;
}
#endif

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Please specify 2 arguments (n, iterations).\n");
        return EXIT_FAILURE;
    }

    size_t n = atoll(argv[1]);
    int iterations = atoi(argv[2]);

    if (n % SPACEBLOCK != 0) {
        printf("I am lazy, so assumed that SPACEBLOCK divides n. Suggestion: n = %ld\n",
                n / SPACEBLOCK * SPACEBLOCK);
        return EXIT_FAILURE;
    }

    double *in = calloc(n, sizeof(double));
    in[0] = 100;
    in[n / 2] = n;
    in[n - 1] = 1000;
    double *out = malloc(n * sizeof(double));

    double duration;
    TIME(duration, Stencil(&in, &out, n, iterations););
    printf("Faster version took %lfs, or ??? Gflops/s\n", duration);

#ifdef CHECK
    double *in2 = calloc(n, sizeof(double));
    in2[0] = 100;
    in2[n / 2] = n;
    in2[n - 1] = 1000;
    double *out2 = malloc(n * sizeof(double));
    TIME(duration, StencilSlow(&in2, &out2, n, iterations););
    printf("Slow version took %lfs, or ??? Gflops/s\n", duration);
    printf("Checking whether they computed the same result with error 0.0000...\n");
    if (equal(out, out2, n, 0.0000)) {
        printf("They are (roughly) equal\n");
    }
    free(in2);
    free(out2);
#endif

    free(in);
    free(out);

    return EXIT_SUCCESS;
}
