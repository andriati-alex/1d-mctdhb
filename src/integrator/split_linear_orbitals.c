#include "integrator/split_linear_orbitals.h"
#include "assistant/arrays_definition.h"
#include "function_tools/calculus.h"
#include "linalg/basic_linalg.h"
#include "linalg/tridiagonal_solver.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double linear_tstep_frac = 0.5;

static void
assert_mkl_descriptor(MKL_LONG status)
{
    if (status != 0)
    {
        printf(
            "\n\nMKL_DFTI ERROR: Reported message:\n%s\n\n",
            DftiErrorMessage(status));
        exit(EXIT_FAILURE);
    }
}

void
set_cn_tridiagonal(
    OrbitalEquation eq_desc, Carray upper, Carray lower, Carray mid)
{
    uint16_t i, npts;
    double   a2, dx;
    dcomplex a1, dt;
    Rarray   pot;

    dt = linear_tstep_frac * eq_desc->prop_dt;
    npts = eq_desc->grid_size;
    a2 = eq_desc->d2coef;
    a1 = eq_desc->d1coef;
    dx = eq_desc->dx;
    pot = eq_desc->pot_grid;

    for (i = 0; i < npts; i++)
    {
        mid[i] = I + a2 * dt / dx / dx - dt * pot[i] / 2;
    }
    for (i = 0; i < npts - 2; i++)
    {
        upper[i] = -a2 * dt / dx / dx / 2 - a1 * dt / dx / 4;
        lower[i] = -a2 * dt / dx / dx / 2 + a1 * dt / dx / 4;
    }
    if (eq_desc->bounds == ZERO_BOUNDS)
    {
        upper[npts - 2] = -a2 * dt / dx / dx / 2 - a1 * dt / dx / 4;
        lower[npts - 2] = -a2 * dt / dx / dx / 2 + a1 * dt / dx / 4;
    } else
    {
        upper[npts - 2] = -a2 * dt / dx / dx / 2 + a1 * dt / dx / 4;
        lower[npts - 2] = -a2 * dt / dx / dx / 2 - a1 * dt / dx / 4;
    }
}

void
set_hder_fftexp(OrbitalEquation eq_desc, Carray hder_exp)
{
    int16_t  i, m;
    double   freq, dx, d2coef;
    dcomplex dt, d1coef;

    dt = linear_tstep_frac * eq_desc->prop_dt;
    dx = eq_desc->dx;
    m = eq_desc->grid_size - 1;
    d1coef = eq_desc->d1coef;
    d2coef = eq_desc->d2coef;
    for (i = 0; i < m; i++)
    {
        if (i <= (m - 1) / 2)
        {
            freq = (2 * PI * i) / (m * dx);
        } else
        {
            freq = (2 * PI * (i - m)) / (m * dx);
        }
        hder_exp[i] =
            cexp(-I * dt * (I * d1coef * freq - d2coef * freq * freq));
    }
}

void
set_expdvr_mat(OrbitalEquation eq_desc, Carray expdvr_mat)
{
    int16_t  i, j, k, kmom, npts;
    double   length, d2coef;
    dcomplex summ, diag, d1coef;
    Carray   d1dvr, d2dvr;

    d2coef = eq_desc->d2coef;
    d1coef = eq_desc->d1coef;
    npts = eq_desc->grid_size - 1;
    length = npts * eq_desc->dx;
    d1dvr = get_dcomplex_array(npts * npts);
    d2dvr = get_dcomplex_array(npts * npts);

    // SETUP FIRST ORDER DERIVATIVE MATRIX IN DVR BASIS
    for (i = 0; i < npts; i++)
    {
        // USE COMPLEX CONJ. TO COMPUTE UPPER TRIANGULAR PART ONLY j > i
        for (j = i + 1; j < npts; j++)
        {
            summ = 0;
            for (k = 0; k < npts; k++)
            {
                kmom = k - npts / 2;
                // NOTE THIS MINUS SIGN - I HAD NO EXPLANATION FOR IT
                diag = -(2 * I * PI * kmom / length);
                summ += diag * cexp(2 * I * kmom * (j - i) * PI / npts) / npts;
            }
            d1dvr[i * npts + j] = summ;
            d1dvr[j * npts + i] = -conj(summ);
        }
        // COMPUTE SEPARATELY THE DIAGONAL
        summ = 0;
        for (k = 0; k < npts; k++)
        {
            kmom = k - npts / 2;
            diag = -(2 * I * PI * kmom / length);
            summ = summ + diag / npts;
        }
        d1dvr[i * npts + i] = summ;
    }

    // SETUP SECOND ORDER DERIVATIVE MATRIX IN DVR BASIS
    // 'mom'entum variables because of exponential basis
    for (i = 0; i < npts; i++)
    {
        // USE COMPLEX CONJ. TO COMPUTE UPPER TRIANGULAR PART ONLY j > i
        for (j = i + 1; j < npts; j++)
        {
            summ = 0;
            for (k = 0; k < npts; k++)
            {
                kmom = k - npts / 2;
                diag = -(2 * PI * kmom / length) * (2 * PI * kmom / length);
                summ += diag * cexp(2 * I * kmom * (j - i) * PI / npts) / npts;
            }
            d2dvr[i * npts + j] = summ;
            d2dvr[j * npts + i] = conj(summ);
        }
        // COMPUTE SEPARATELY THE DIAGONAL
        summ = 0;
        for (k = 0; k < npts; k++)
        {
            kmom = k - npts / 2;
            diag = -(2 * PI * kmom / length) * (2 * PI * kmom / length);
            summ = summ + diag / npts;
        }
        d2dvr[i * npts + i] = summ;
    }

    // SETUP MATRIX CORRESPONDING TO DERIVATIVES ON HAMILTONIAN
    // INCLUDING THE EQUATION COEFFICIENTS IN FRONT OF THEM
    for (i = 0; i < npts; i++)
    {
        for (j = 0; j < npts; j++)
        {
            expdvr_mat[i * npts + j] =
                d2coef * d2dvr[i * npts + j] + d1coef * d1dvr[i * npts + j];
        }
    }
    free(d1dvr);
    free(d2dvr);
}

void
set_sinedvr_mat(OrbitalEquation eq_desc, Carray sinedvr_mat)
{
    uint16_t i, j, k, i1, j1, npts;
    double   length, sum;
    Rarray   d2mat, udvr;

    npts = eq_desc->grid_size;
    length = (npts + 1) * eq_desc->dx;
    d2mat = get_double_array(npts);       // derivative in default basis
    udvr = get_double_array(npts * npts); // unitary transform to DVR basis

    for (i = 0; i < npts; i++)
    {
        i1 = i + 1;
        for (j = 0; j < npts; j++)
        {
            j1 = j + 1;
            udvr[i * npts + j] =
                sqrt(2.0 / (npts + 1)) * sin(i1 * j1 * PI / (npts + 1));
        }
        d2mat[i] = -(i1 * PI / length) * (i1 * PI / length);
    }

    // Transform second order derivative matrix to DVR basis
    // multiplying by the equation's derivative  coefficient
    for (i = 0; i < npts; i++)
    {
        for (j = 0; j < npts; j++)
        {
            sum = 0;
            for (k = 0; k < npts; k++)
            {
                sum += udvr[i * npts + k] * d2mat[k] * udvr[k * npts + j];
            }
            sinedvr_mat[i * npts + j] = eq_desc->d2coef * sum;
        }
    }
    free(d2mat);
    free(udvr);
}

void
linear_horb_fd(OrbitalEquation eq_desc, Carray f, Carray out)
{
    BoundaryCondition bounds;
    int               i, npts;
    double            a2, dx;
    dcomplex          a1, mid, upper, lower;
    Rarray            pot;

    bounds = eq_desc->bounds;
    npts = eq_desc->grid_size;
    a2 = eq_desc->d2coef;
    a1 = eq_desc->d1coef;
    dx = eq_desc->dx;
    pot = eq_desc->pot_grid;

    upper = a2 * dx / dx + a1 * dx / 2;
    lower = a2 * dx / dx - a1 * dx / 2;

    if (bounds)
    {
        mid = -2 * a2 / dx / dx + pot[0];
        out[0] = mid * f[0] + upper * f[1];
    } else
    {
        mid = -2 * a2 * dx / dx + pot[0];
        out[0] = mid * f[0] + upper * f[1] + lower * f[npts - 2];
    }

    for (i = 1; i < npts - 1; i++)
    {
        mid = -2 * a2 * dx / dx + pot[i];
        out[i] = mid * f[i] + upper * f[i + 1] + lower * f[i - 1];
    }

    if (bounds)
    {
        mid = -2 * a2 / dx / dx + pot[npts - 1];
        out[npts - 1] = mid * f[npts - 1] + lower * f[npts - 2];
    } else
    {
        mid = -2 * a2 / dx / dx + pot[npts - 2];
        out[npts - 2] = mid * f[npts - 2] + upper * f[0] + lower * f[npts - 3];
        out[npts - 1] = out[0];
    }
}

void
linear_horb_fft(
    OrbitalEquation eq_desc, OrbitalWorkspace work, Carray f, Carray out)
{
    uint16_t i, n;
    double   d2coef;
    dcomplex d1coef;
    Rarray   pot, freq;
    Carray   der_part;

    n = eq_desc->grid_size - 1;
    d2coef = eq_desc->d2coef;
    d1coef = eq_desc->d1coef;
    pot = eq_desc->pot_grid;
    freq = work->fft_freq;

    der_part = get_dcomplex_array(n);
    carrCopy(n, f, der_part);
    DftiComputeForward(work->fft_desc, der_part);
    for (i = 0; i < n; i++)
    {
        der_part[i] *= (freq[i] * I * d1coef - freq[i] * freq[i] * d2coef);
    }
    DftiComputeBackward(work->fft_desc, der_part);
    for (i = 0; i < n; i++)
    {
        out[i] = der_part[i] + pot[i] * f[i];
    }
    out[n] = out[0];
    free(der_part);
}

void
cn_rhs(OrbitalEquation eq_desc, Carray f, Carray out)
{
    BoundaryCondition bounds;
    int               i, npts;
    double            a2, dx;
    dcomplex          a1, mid, upper, lower, dt;
    Rarray            pot;

    bounds = eq_desc->bounds;
    dt = linear_tstep_frac * eq_desc->prop_dt;
    npts = eq_desc->grid_size;
    a2 = eq_desc->d2coef;
    a1 = eq_desc->d1coef;
    dx = eq_desc->dx;
    pot = eq_desc->pot_grid;

    upper = a2 * dt / dx / dx / 2 + a1 * dt / dx / 4;
    lower = a2 * dt / dx / dx / 2 - a1 * dt / dx / 4;

    if (bounds)
    {
        mid = I - a2 * dt / dx / dx + dt * pot[0] / 2;
        out[0] = mid * f[0] + upper * f[1];
    } else
    {
        mid = I - a2 * dt / dx / dx + dt * pot[0] / 2;
        out[0] = mid * f[0] + upper * f[1] + lower * f[npts - 2];
    }

    for (i = 1; i < npts - 1; i++)
    {
        mid = I - a2 * dt / dx / dx + dt * pot[i] / 2;
        out[i] = mid * f[i] + upper * f[i + 1] + lower * f[i - 1];
    }

    if (bounds)
    {
        mid = I - a2 * dt / dx / dx + dt * pot[npts - 1] / 2;
        out[npts - 1] = mid * f[npts - 1] + lower * f[npts - 2];
    } else
    {
        mid = I - a2 * dt / dx / dx + dt * pot[npts - 2] / 2;
        out[npts - 2] = mid * f[npts - 2] + upper * f[0] + lower * f[npts - 3];
        out[npts - 1] = out[0];
    }
}

void
advance_linear_crank_nicolson(
    OrbitalEquation eq_desc, OrbitalWorkspace work, Cmatrix orb)
{
    BoundaryCondition bounds;
    uint16_t          k, npts;
    Carray            rhs, upper, lower, mid;

    bounds = eq_desc->bounds;
    rhs = get_dcomplex_array(eq_desc->grid_size);
    upper = work->cn_upper;
    lower = work->cn_lower;
    mid = work->cn_mid;

    // Last point is excluded considered as repetition due to periodicity
    if (bounds == ZERO_BOUNDS)
    {
        npts = eq_desc->grid_size;
    } else
    {
        npts = eq_desc->grid_size - 1;
    }

    // for each orbital solve the Crank-Nicolson tridiagonal system
    for (k = 0; k < work->norb; k++)
    {
        cn_rhs(eq_desc, orb[k], rhs);
        if (bounds == ZERO_BOUNDS)
        {
            solve_cplx_tridiag(npts, upper, lower, mid, rhs, orb[k]);
        } else
        {
            solve_cplx_cyclic_tridiag_sm(npts, upper, lower, mid, rhs, orb[k]);
            orb[k][npts] = orb[k][0];
        }
    }
    free(rhs);
}

void
advance_linear_fft(OrbitalWorkspace work, Cmatrix orb)
{

    uint16_t k, fft_size;
    MKL_LONG s;
    Carray   forward_fft, back_fft;

    fft_size = work->grid_size - 1;
    forward_fft = get_dcomplex_array(fft_size);
    back_fft = get_dcomplex_array(fft_size);

    for (k = 0; k < work->norb; k++)
    {
        carrCopy(fft_size, orb[k], forward_fft);
        s = DftiComputeForward(work->fft_desc, forward_fft);
        // Apply Exp. derivative operator in momentum space
        carrMultiply(fft_size, work->fft_hder_exp, forward_fft, back_fft);
        // Go back to position space
        s = DftiComputeBackward(work->fft_desc, back_fft);
        carrCopy(fft_size, back_fft, orb[k]);
        // last point assumed as cyclic boundary
        orb[k][fft_size] = orb[k][0];
        assert_mkl_descriptor(s);
    }
    free(forward_fft);
    free(back_fft);
}
