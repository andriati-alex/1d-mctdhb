#include "function_tools/calculus.h"
#include "assistant/arrays_definition.h"
#include "linalg/basic_linalg.h"
#include <math.h>
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
            sum += f[i] + 4 * f[i + 1] + f[i + 2];
        }
        sum *= h / 3; // End 3-point simpsons intervals
        sum += (f[n - 4] + 3 * (f[n - 3] + f[n - 2]) + f[n - 1]) * 3 * h / 8;
        return sum;
    }

    for (i = 0; i < n - 2; i = i + 2)
    {
        sum += f[i] + 4 * f[i + 1] + f[i + 2];
    }
    sum *= h / 3; // End 3-point simpsons intervals
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
            sum += f[i] + 4 * f[i + 1] + f[i + 2];
        }
        sum *= h / 3; // End 3-point simpsons intervals
        sum += (f[n - 4] + 3 * (f[n - 3] + f[n - 2]) + f[n - 1]) * 3 * h / 8;
        return sum;
    }

    for (i = 0; i < n - 2; i = i + 2)
    {
        sum += f[i] + 4 * f[i + 1] + f[i + 2];
    }
    sum *= h / 3; // End 3-point simpsons intervals
    return sum;
}

double
real_border_integral(int grid_size, int chunk, Rarray f, double h)
{
    if (chunk < 3)
    {
        printf("\n\nERROR : chunk size must be larger "
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

dcomplex
cplx_border_integral(int grid_size, int chunk, Carray f, double h)
{
    if (chunk < 3)
    {
        printf("\n\nERROR : chunk size must be larger "
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
orthonormalize(uint32_t npts, double dx, uint32_t nfun, Cmatrix F)
{
    uint32_t i, j, k;

    renormalize(npts, dx, 1.0, F[0]); // First function as lagorithm start

    for (i = 1; i < nfun; i++)
    {
        for (j = 0; j < i; j++)
        {
            // Iterative Gram-Schmidt (see wikipedia) is
            // different to improve  numerical stability
            for (k = 0; k < npts; k++)
            {
                F[i][k] -= scalar_product(npts, dx, F[j], F[i]) * F[j][k];
            }
        }
        // normalized to unit the new vector
        renormalize(npts, dx, 1.0, F[i]);
    }
}

void
dxFFT(uint32_t npts, double dx, Carray f, Carray dfdx)
{
    int      i, nfft;
    double   len, freq;
    MKL_LONG s;

    DFTI_DESCRIPTOR_HANDLE desc;

    nfft = npts - 1; // Assume the connection f[n-1] = f[0] at the boundary
    len = nfft * dx; // total domain length

    carrCopy(nfft, f, dfdx); // Copy to execute in-place computation.

    s = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, nfft);
    s = DftiSetValue(desc, DFTI_FORWARD_SCALE, 1.0 / sqrt((double) nfft));
    s = DftiSetValue(desc, DFTI_BACKWARD_SCALE, 1.0 / sqrt((double) nfft));
    s = DftiCommitDescriptor(desc);

    s = DftiComputeForward(desc, dfdx);

    for (i = 0; i < nfft; i++)
    {
        if (i <= (nfft - 1) / 2)
        {
            freq = (2 * PI * i) / len;
        } else
        {
            freq = (2 * PI * (i - nfft)) / len;
        }
        dfdx[i] = dfdx[i] * freq * I;
    }

    s = DftiComputeBackward(desc, dfdx);
    s = DftiFreeDescriptor(&desc);

    dfdx[nfft] = dfdx[0]; // boundary point
}

void
d2xFFT(uint32_t npts, double dx, Carray f, Carray dfdx)
{
    int      i, nfft;
    double   len, freq;
    MKL_LONG s;

    DFTI_DESCRIPTOR_HANDLE desc;

    nfft = npts - 1; // Assumes the connection f[n-1] = f[0] at the boundary
    len = nfft * dx; // total domain length

    carrCopy(nfft, f, dfdx); // Copy to execute in-place computation.

    s = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, nfft);
    s = DftiSetValue(desc, DFTI_FORWARD_SCALE, 1.0 / sqrt((double) nfft));
    s = DftiSetValue(desc, DFTI_BACKWARD_SCALE, 1.0 / sqrt((double) nfft));
    s = DftiCommitDescriptor(desc);

    s = DftiComputeForward(desc, dfdx); // FORWARD FFT

    for (i = 0; i < nfft; i++)
    {
        if (i <= (nfft - 1) / 2)
        {
            freq = (2 * PI * i) / len;
        } else
        {
            freq = (2 * PI * (i - nfft)) / len;
        }
        dfdx[i] = dfdx[i] * (-freq * freq);
    }

    s = DftiComputeBackward(desc, dfdx); // BACKWARD FFT
    s = DftiFreeDescriptor(&desc);

    dfdx[nfft] = dfdx[0]; // boundary point
}

void
set_fft_freq(int32_t fft_size, double h, Rarray freq)
{
    double length = fft_size * h;
    for (int32_t i = 0; i < fft_size; i++)
    {
        if (i <= (fft_size - 1) / 2)
        {
            freq[i] = (2 * PI * i) / length;
        } else
        {
            freq[i] = (2 * PI * (i - fft_size)) / length;
        }
    }
}

void
dxFD(uint32_t n, double dx, Carray f, Carray dfdx)
{
    uint32_t i;
    double   r;

    r = 1.0 / (12 * dx); // ratio for a fourth-order scheme
    dfdx[0] = (f[n - 3] - f[2] + 8 * (f[1] - f[n - 2])) * r;
    dfdx[1] = (f[n - 2] - f[3] + 8 * (f[2] - f[0])) * r;
    dfdx[n - 3] = (f[n - 5] - f[0] + 8 * (f[n - 2] - f[n - 4])) * r;
    dfdx[n - 2] = (f[n - 4] - f[1] + 8 * (f[0] - f[n - 3])) * r;
    dfdx[n - 1] = dfdx[0]; // assume last point IS the boundary

    for (i = 2; i < n - 3; i++)
    {
        dfdx[i] = (f[i - 2] - f[i + 2] + 8 * (f[i + 1] - f[i - 1])) * r;
    }
}

void
d2xFD(uint32_t n, double dx, Carray f, Carray dfdx)
{
    dfdx[0] = (f[1] - 2 * f[0] + f[n - 2]) / dx / dx;
    dfdx[n - 2] = (f[0] - 2 * f[n - 2] + f[n - 3]) / dx / dx;
    dfdx[n - 1] = dfdx[0];
    for (uint32_t i = 1; i < n - 2; i++)
    {
        dfdx[i] = (f[i + 1] - 2 * f[i] + f[i - 1]) / dx / dx;
    }
}
