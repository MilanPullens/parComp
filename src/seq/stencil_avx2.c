/* NON FUNCTIONAL!
 *
 * The optimised version only achieves 55% of peak performance on my laptop, which annoys me.
 * I did not have time to optimise it further before the 19th of April, so this file is still
 * a work in progress. It will be optimised for processors having avx2 and fma, and I will only
 * change Middle, so this will not interfere with your parallel versions. Just swap in this version
 * of Middle instead. */

#include <stdlib.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

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

const double a = 0.1;
const double b = 0.2;
const double c = 0.3;
const double d = 0.5;
const double e = 0.6;

/* We split up the stencil in smaller stencils, of roughly SPACEBLOCK size,
 * and treat them for TIMEBLOCK iterations. */
const int SPACEBLOCK = 10000;
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

        /* Buffer swap to save memory. */
        if (t != iterations - 1) {
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

        /* Buffer swap to save memory. */
        if (t != iterations - 1) {
            double *temp = *in;
            *in = *out;
            *out = temp;
        }
    }
}

/* This version is written with avx2 in mind. Each vector register can fit 4 doubles.
 * So we need 3 loads (in[i - 1], in[i], in[i + 1]) and one load (out[i]) to calculate
 * out[i] for i, i + 1, i + 2, i + 3. This takes 3 instructions.
 *
 * Instead, we can calculate 3 iterations at a time. That means out[i] is a linear combination
 * of in[i - 3], in[i - 2], in[i - 1], in[i], in[i + 1], in[i + 2], in[i + 3].
 * This fits in the 16 registers as we need 7 loads of in, 7 constants, and one register
 * for the result. The nice thing is that in[i + 1] = in[i + 4 - 3], in[i + 2] = in[i + 4 - 2],
 * in[i + 3] = in[i + 4 - 1]. So we can keep these in the registers for the next iteration.
 * So really, we only have 4 loads, and one store on 7 instructions.
 *
 * */
void Middle_avx2(double **in, double **out, size_t n, int iterations)
{
    const double c1 = 0.1;
    const double c2 = 0.1;
    const double c3 = 0.1;
    const double c4 = 0.1;
    const double c5 = 0.1;
    const double c6 = 0.1;
    const double c7 = 0.1;

    int t = 3;
    for (; t <= iterations - 3; t += 3) {

        for (size_t i = t; i < n + 2 * iterations - t; i++) {
            (*out)[i] = c1 * (*in)[i - 3] +
                        c2 * (*in)[i - 2] +
                        c3 * (*in)[i - 1] +
                        c4 * (*in)[i] +
                        c5 * (*in)[i + 1] +
                        c6 * (*in)[i + 2] +
                        c7 * (*in)[i + 3];
        }

        /* Buffer swap to save memory. */
        if (t != iterations - 1) {
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

        /* Buffer swap to save memory. */
        if (t != iterations - 1) {
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
            Middle_avx2(&inBuffer, &outBuffer, SPACEBLOCK, iterations);
            memcpy(*out + block * SPACEBLOCK, outBuffer + iterations, SPACEBLOCK * sizeof(double));
        }
    }

    free(inBuffer);
    free(outBuffer);
}

void Stencil(double **in, double **out, size_t n, int iterations)
{
    for (int t = TIMEBLOCK; t <= iterations; t += TIMEBLOCK) {
        StencilBlocked(in, out, n, TIMEBLOCK);
        double *temp = *out;
        *out = *in;
        *in = temp;
    }
    if (iterations % TIMEBLOCK != 0) {
        StencilBlocked(in, out, n, iterations % TIMEBLOCK);
    } else {
        /* We did one buffer swap too many */
        double *temp = *out;
        *out = *in;
        *in = temp;
    }
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
        if (fabs(x[i]) < error) {
            if (fabs(x[i]) > error) {
                printf("Index %zu: %lf != %lf\n", i, x[i], y[i]);
                return false;
            } else {
                continue;
            }
        }
        if (fabs((x[i] - y[i]) / x[i]) > error) {
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
    in[n - 1] = 1000;
    double *out = malloc(n * sizeof(double));
    double *out2 = malloc(n * sizeof(double));

    double duration;
    TIME(duration, Stencil(&in, &out, n, iterations););
    printf("Fast version took took %lfs, or ??? Gflops/s\n", duration);

#ifdef CHECK
    TIME(duration, StencilSlow(&in, &out2, n, iterations););
    printf("Slow version took took %lfs, or ??? Gflops/s\n", duration);
    printf("Checking whether they computed the same result with relative error 0.0001...\n");
    if (equal(out, out2, n, 0.0001)) {
        printf("They are (roughly) equal\n");
    }
#endif

    free(in);
    free(out);
    free(out2);

    return EXIT_SUCCESS;
}
