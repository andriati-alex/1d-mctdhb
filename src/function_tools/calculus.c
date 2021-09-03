#include "function_tools/calculus.h"
#include "assistant/arrays_definition.h"
#include "linalg/basic_linalg.h"
#include <math.h>
#include <mkl_dfti.h>
#include <stdio.h>
#include <stdlib.h>

dcomplex
Csimps(uint32_t n, double h, Carray f)
{
    uint32_t i;
    dcomplex sum;
    sum = 0;
    if (n < 3)
    {
        printf("\n\nERROR : less than 3 point to integrate by simps!\n\n");
        exit(EXIT_FAILURE);
    }
    if (n % 2 == 0)
    {
        //  Case the number of points is even integrate the last
        //  chunk using simpson's 3/8 rule to maintain accuracy.
        for (i = 0; i < (n - 4); i = i + 2)
        {
            sum = sum + f[i] + 4 * f[i + 1] + f[i + 2];
        }
        sum = sum * h / 3; // End 3-point simpsons intervals
        sum =
            sum + (f[n - 4] + 3 * (f[n - 3] + f[n - 2]) + f[n - 1]) * 3 * h / 8;
    } else
    {
        for (i = 0; i < n - 2; i = i + 2)
        {
            sum = sum + f[i] + 4 * f[i + 1] + f[i + 2];
        }
        sum = sum * h / 3; // End 3-point simpsons intervals
    }
    return sum;
}

double
Rsimps(uint32_t n, double h, Rarray f)
{
    uint32_t i;
    double   sum;
    sum = 0;
    if (n < 3)
    {
        printf("\n\nERROR : less than 3 point to integrate by simps !\n\n");
        exit(EXIT_FAILURE);
    }
    if (n % 2 == 0)
    {
        //  Case the number of points is even integrate the last
        //  chunk using simpson's 3/8 rule to maintain accuracy.
        for (i = 0; i < (n - 4); i = i + 2)
        {
            sum = sum + f[i] + 4 * f[i + 1] + f[i + 2];
        }
        sum = sum * h / 3; // End 3-point simpsons intervals
        sum =
            sum + (f[n - 4] + 3 * (f[n - 3] + f[n - 2]) + f[n - 1]) * 3 * h / 8;

    } else
    {
        for (i = 0; i < n - 2; i = i + 2)
        {
            sum = sum + f[i] + 4 * f[i + 1] + f[i + 2];
        }
        sum = sum * h / 3; // End 3-point simpsons intervals
    }
    return sum;
}

double
real_border_integral(int grid_size, int chunk, Rarray f, double h)
{
    if (chunk < 3)
    {
        printf("\n\nERROR : chunk size must be greater "
               "than 2 to compute border norm!\n\n");
        exit(EXIT_FAILURE);
    }
    if (chunk > grid_size / 2)
    {
        printf(
            "\n\nERROR : chunk in border integral cannot exceed "
            "1/2 of grid size: %d and %d given respectively.\n\n",
            chunk,
            grid_size);
        exit(EXIT_FAILURE);
    }
    return Rsimps(chunk, h, f) + Rsimps(chunk, h, &f[grid_size - chunk]);
}

double
cplx_border_integral(int grid_size, int chunk, Carray f, double h)
{
    if (chunk < 2)
    {
        printf("\n\nERROR : chunk size must be greater "
               "than 1 to compute border norm!\n\n");
        exit(EXIT_FAILURE);
    }
    if (chunk > grid_size / 2)
    {
        printf(
            "\n\nERROR : chunk in border integral cannot exceed "
            "1/2 of grid size: %d and %d given respectively.\n\n",
            chunk,
            grid_size);
        exit(EXIT_FAILURE);
    }
    return Csimps(chunk, h, f) + Csimps(chunk, h, &f[grid_size - chunk]);
}

void
renormalize(uint32_t n, double dx, double new_norm, Carray f)
{
    uint32_t i;
    double   renorm;
    Rarray   mod_square;

    mod_square = get_double_array(n);
    carrAbs2(n, f, mod_square);
    renorm = new_norm * sqrt(1.0 / Rsimps(n, dx, mod_square));
    for (i = 0; i < n; i++) f[i] = f[i] * renorm;
    free(mod_square);
}

dcomplex
scalar_product(uint32_t n, double h, Carray fstar, Carray f)
{
    uint32_t i;
    dcomplex overlap;
    Carray   integ;

    integ = get_dcomplex_array(n);
    for (i = 0; i < n; i++) integ[i] = conj(fstar[i]) * f[i];
    overlap = Csimps(n, h, integ);
    free(integ);
    return overlap;
}

double
cplx_function_norm(uint32_t n, double h, Carray f)
{
    double norm;
    Rarray mod_square;
    mod_square = get_double_array(n);
    carrAbs2(n, f, mod_square);
    norm = sqrt(Rsimps(n, h, mod_square));
    free(mod_square);
    return norm;
}

double
real_function_norm(uint32_t n, double h, Rarray f)
{
    double norm;
    Rarray mod_square;
    mod_square = get_double_array(n);
    rarrAbs2(n, f, mod_square);
    norm = sqrt(Rsimps(n, h, mod_square));
    free(mod_square);
    return norm;
}

void
orthonormalize(uint32_t Npts, double dx, uint32_t Nfun, Cmatrix F)
{
    uint32_t i, j, k;
    Carray   integ;

    integ = get_dcomplex_array(Npts);
    renormalize(Npts, dx, 1.0, F[0]); // Only normalize first function
    for (i = 1; i < Nfun; i++)
    {
        for (j = 0; j < i; j++)
        {
            // The projection are integrals of the product below
            for (k = 0; k < Npts; k++) integ[k] = conj(F[j][k]) * F[i][k];
            // Iterative Gram-Schmidt (see wikipedia) is
            // different to improve  numerical stability
            for (k = 0; k < Npts; k++)
            {
                F[i][k] = F[i][k] - Csimps(Npts, dx, integ) * F[j][k];
            }
        }
        // normalized to unit the new vector
        renormalize(Npts, dx, 1.0, F[i]);
    }
    free(integ);
}

void
dxFFT(uint32_t Npts, double dx, Carray f, Carray dfdx)
{
    int                    i, N;
    double                 Ndx, freq;
    MKL_LONG               s;
    DFTI_DESCRIPTOR_HANDLE desc;

    N = Npts - 1; // Assume the connection f[n-1] = f[0] at the boundary
    Ndx = N * dx; // total domain length

    carrCopy(N, f, dfdx); // Copy to execute in-place computation.

    s = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, N);
    s = DftiSetValue(desc, DFTI_FORWARD_SCALE, 1.0 / sqrt((double) N));
    s = DftiSetValue(desc, DFTI_BACKWARD_SCALE, 1.0 / sqrt((double) N));
    s = DftiCommitDescriptor(desc);

    s = DftiComputeForward(desc, dfdx);

    for (i = 0; i < N; i++)
    {
        if (i <= (N - 1) / 2)
            freq = (2 * PI * i) / Ndx;
        else
            freq = (2 * PI * (i - N)) / Ndx;
        dfdx[i] = dfdx[i] * freq * I;
    }

    s = DftiComputeBackward(desc, dfdx);
    s = DftiFreeDescriptor(&desc);

    dfdx[N] = dfdx[0]; // boundary point
}

void
d2xFFT(uint32_t Npts, double dx, Carray f, Carray dfdx)
{
    int                    i, N;
    double                 Ndx, freq;
    MKL_LONG               s;
    DFTI_DESCRIPTOR_HANDLE desc;

    N = Npts - 1; // Assumes the connection f[n-1] = f[0] at the boundary
    Ndx = N * dx; // total domain length

    carrCopy(N, f, dfdx); // Copy to execute in-place computation.

    s = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, N);
    s = DftiSetValue(desc, DFTI_FORWARD_SCALE, 1.0 / sqrt((double) N));
    s = DftiSetValue(desc, DFTI_BACKWARD_SCALE, 1.0 / sqrt((double) N));
    s = DftiCommitDescriptor(desc);

    s = DftiComputeForward(desc, dfdx); // FORWARD FFT

    for (i = 0; i < N; i++)
    {
        if (i <= (N - 1) / 2)
            freq = (2 * PI * i) / Ndx;
        else
            freq = (2 * PI * (i - N)) / Ndx;
        dfdx[i] = dfdx[i] * (-freq * freq);
    }

    s = DftiComputeBackward(desc, dfdx); // BACKWARD FFT
    s = DftiFreeDescriptor(&desc);

    dfdx[N] = dfdx[0]; // boundary point
}

void
dxFD(uint32_t n, double dx, Carray f, Carray dfdx)
{
    uint32_t i;
    double   r;

    r = 1.0 / (12 * dx); // ratio for a fourth-order scheme
    dfdx[0] = (f[n - 3] - f[2] + 8 * (f[1] - f[n - 2])) * r;
    dfdx[1] = (f[n - 2] - f[3] + 8 * (f[2] - f[0])) * r;
    dfdx[n - 2] = (f[n - 4] - f[1] + 8 * (f[0] - f[n - 3])) * r;
    dfdx[n - 1] = dfdx[0]; // assume last point IS the boundary

    for (i = 2; i < n - 2; i++)
    {
        dfdx[i] = (f[i - 2] - f[i + 2] + 8 * (f[i + 1] - f[i - 1])) * r;
    }
}

void
d2xFD(uint32_t n, double dx, Carray f, Carray dfdx)
{
    dfdx[0] = (f[1] - 2 * f[0] + f[n - 2]) / dx / dx;
    dfdx[n - 1] = dfdx[0];
    for (uint32_t i = 1; i < n - 1; i++)
    {
        dfdx[i] = (f[i + 1] - 2 * f[i] + f[i - 1]) / dx / dx;
    }
}
