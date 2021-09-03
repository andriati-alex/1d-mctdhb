#include "integrator/split_linear_orbitals.h"
#include "assistant/arrays_definition.h"
#include "function_tools/calculus.h"
#include "linalg/basic_linalg.h"
#include "linalg/tridiagonal_solver.h"
#include <stdlib.h>

static void
cn_rhs(OrbitalEquation eq_desc, Carray f, Carray out)
{
    Bool     isTrapped;
    int      i, N;
    double   a2, dx;
    dcomplex a1, mid, upper, lower, dt;
    Rarray   V;

    isTrapped = eq_desc->trapped;
    dt = eq_desc->prop_dt;
    N = eq_desc->grid_size;
    a2 = eq_desc->d2coef;
    a1 = eq_desc->d1coef;
    dx = eq_desc->dx;
    V = eq_desc->pot_grid;

    upper = a2 * dt / dx / dx / 2 + a1 * dt / dx / 4;
    lower = a2 * dt / dx / dx / 2 - a1 * dt / dx / 4;

    if (isTrapped)
    {
        mid = I - a2 * dt / dx / dx + dt * V[0] / 2;
        out[0] = mid * f[0] + upper * f[1];
    } else
    {
        mid = I - a2 * dt / dx / dx + dt * V[0] / 2;
        out[0] = mid * f[0] + upper * f[1] + lower * f[N - 2];
    }

    for (i = 1; i < N - 1; i++)
    {
        mid = I - a2 * dt / dx / dx + dt * V[i] / 2;
        out[i] = mid * f[i] + upper * f[i + 1] + lower * f[i - 1];
    }

    if (isTrapped)
    {
        mid = I - a2 * dt / dx / dx + dt * V[N - 1] / 2;
        out[N - 1] = mid * f[N - 1] + lower * f[N - 2];
    } else
    {
        mid = I - a2 * dt / dx / dx + dt * V[N - 2] / 2;
        out[N - 2] = mid * f[N - 2] + upper * f[0] + lower * f[N - 3];
        out[N - 1] = out[0];
    }
}

void
linear_horb_fd(OrbitalEquation eq_desc, Carray f, Carray out)
{
    Bool     isTrapped;
    int      i, N;
    double   a2, dx;
    dcomplex a1, mid, upper, lower;
    Rarray   V;

    isTrapped = eq_desc->trapped;
    N = eq_desc->grid_size;
    a2 = eq_desc->d2coef;
    a1 = eq_desc->d1coef;
    dx = eq_desc->dx;
    V = eq_desc->pot_grid;

    upper = a2 * dx / dx + a1 * dx / 2;
    lower = a2 * dx / dx - a1 * dx / 2;

    if (isTrapped)
    {
        mid = -2 * a2 / dx / dx + V[0];
        out[0] = mid * f[0] + upper * f[1];
    } else
    {
        mid = -2 * a2 * dx / dx + V[0];
        out[0] = mid * f[0] + upper * f[1] + lower * f[N - 2];
    }

    for (i = 1; i < N - 1; i++)
    {
        mid = -2 * a2 * dx / dx + V[i];
        out[i] = mid * f[i] + upper * f[i + 1] + lower * f[i - 1];
    }

    if (isTrapped)
    {
        mid = -2 * a2 / dx / dx + V[N - 1];
        out[N - 1] = mid * f[N - 1] + lower * f[N - 2];
    } else
    {
        mid = -2 * a2 / dx / dx + V[N - 2];
        out[N - 2] = mid * f[N - 2] + upper * f[0] + lower * f[N - 3];
        out[N - 1] = out[0];
    }
}

void
linear_horb_fft(
    DFTI_DESCRIPTOR_HANDLE* desc, OrbitalEquation eq_desc, Carray f, Carray out)
{
    int      i, n;
    double   d2coef, freq, length;
    dcomplex d1coef;
    Rarray   V;
    Carray   der_part;

    n = eq_desc->grid_size - 1;
    length = n * eq_desc->dx;
    d2coef = eq_desc->d2coef;
    d1coef = eq_desc->d1coef;
    V = eq_desc->pot_grid;

    der_part = get_dcomplex_array(n);
    carrCopy(n, f, der_part);
    DftiComputeForward(*desc, der_part);
    for (i = 0; i < n; i++)
    {
        if (i <= (n - 1) / 2)
        {
            freq = (2 * PI * i) / length;
        } else
        {
            freq = (2 * PI * (i - n)) / length;
        }
        der_part[i] = der_part[i] * (freq * I * d1coef - freq * freq * d2coef);
    }
    DftiComputeBackward(*desc, der_part);
    for (i = 0; i < n; i++)
    {
        out[i] = der_part[i] + V[i] * f[i];
    }
    out[n] = out[0];
    free(der_part);
}

void
set_cn_tridiagonal(
    OrbitalEquation eq_desc, Carray upper, Carray lower, Carray mid)
{
    uint16_t i, N;
    double   a2, dx;
    dcomplex a1, dt;
    Rarray   V;

    dt = eq_desc->prop_dt;
    N = eq_desc->grid_size;
    a2 = eq_desc->d2coef;
    a1 = eq_desc->d1coef;
    dx = eq_desc->dx;
    V = eq_desc->pot_grid;

    for (i = 0; i < N; i++)
    {
        mid[i] = I + a2 * dt / dx / dx - dt * V[i] / 2;
    }
    for (i = 0; i < N - 2; i++)
    {
        upper[i] = -a2 * dt / dx / dx / 2 - a1 * dt / dx / 4;
        lower[i] = -a2 * dt / dx / dx / 2 + a1 * dt / dx / 4;
    }
    if (eq_desc->trapped)
    {
        upper[N - 2] = -a2 * dt / dx / dx / 2 - a1 * dt / dx / 4;
        lower[N - 2] = -a2 * dt / dx / dx / 2 + a1 * dt / dx / 4;
    } else
    {
        upper[N - 2] = -a2 * dt / dx / dx / 2 + a1 * dt / dx / 4;
        lower[N - 2] = -a2 * dt / dx / dx / 2 - a1 * dt / dx / 4;
    }
}

void
advance_linear_crank_nicolson(
    OrbitalEquation eq_desc,
    uint16_t        norb,
    Carray          upper,
    Carray          lower,
    Carray          mid,
    Cmatrix         Orb)
{
    Bool     isTrapped;
    uint16_t k, size;
    Carray   rhs;

    isTrapped = eq_desc->trapped;
    rhs = get_dcomplex_array(eq_desc->grid_size);

    if (isTrapped)
    {
        size = eq_desc->grid_size;
    } else
    {
        size = eq_desc->grid_size - 1;
    }
    // for each orbital solve the Crank-Nicolson tridiagonal system
    for (k = 0; k < norb; k++)
    {
        cn_rhs(eq_desc, Orb[k], rhs);
        if (isTrapped)
        {
            solve_cplx_tridiag(size, upper, lower, mid, rhs, Orb[k]);
        } else
        {
            solve_cplx_cyclic_tridiag_sm(size, upper, lower, mid, rhs, Orb[k]);
            Orb[k][size] = Orb[k][0];
        }
    }
    free(rhs);
}

void
advance_linear_fft(
    DFTI_DESCRIPTOR_HANDLE* desc,
    int                     Mpos,
    int                     Morb,
    Carray                  exp_der,
    Cmatrix                 Orb)
{

    int      k;
    MKL_LONG s;
    Carray   forward_fft, back_fft;

    forward_fft = get_dcomplex_array(Mpos - 1);
    back_fft = get_dcomplex_array(Mpos - 1);

    for (k = 0; k < Morb; k++)
    {
        carrCopy(Mpos - 1, Orb[k], forward_fft);
        s = DftiComputeForward((*desc), forward_fft);
        // Apply Exp. derivative operator in momentum space
        carrMultiply(Mpos - 1, exp_der, forward_fft, back_fft);
        // Go back to position space
        s = DftiComputeBackward((*desc), back_fft);
        carrCopy(Mpos - 1, back_fft, Orb[k]);
        // last point assumed as cyclic boundary
        Orb[k][Mpos - 1] = Orb[k][0];
    }
    free(forward_fft);
    free(back_fft);
}
