#include "function_tools/builtin_potential.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** \brief Compute value of linear time dependent paramter */
static double
linear_sweep_param(double t, double init, double final, double period)
{
    if (period <= 0) return final;
    return init + (final - init) * t / period;
}

static double
trapezoid_sweep_param(double t, double top_val, double ramp_t, double static_t)
{
    double curr_val;

    curr_val = 0;
    if (t <= ramp_t)
    {
        curr_val = linear_sweep_param(t, 0, top_val, ramp_t);
    }
    if (t > ramp_t && t < ramp_t + static_t)
    {
        curr_val = top_val;
    }
    if (t >= ramp_t + static_t && t < 2 * ramp_t + static_t)
    {
        curr_val = top_val - (t - ramp_t - static_t) / ramp_t * top_val;
    }
    return curr_val;
}

void
potfunc_harmonic(double t, uint16_t npts, Rarray x, void* params, Rarray V)
{
    double omega, omega0, omegaf, sweep_period;
    omega0 = ((double*) params)[0];
    omegaf = ((double*) params)[1];
    sweep_period = ((double*) params)[2];

    omega = linear_sweep_param(t, omega0, omegaf, sweep_period);
    for (uint16_t i = 0; i < npts; i++)
    {
        V[i] = 0.5 * omega * omega * x[i] * x[i];
    }
}

void
potfunc_doublewell(double t, uint16_t npts, Rarray x, void* params, Rarray V)
{
    double at, a0, af, b, sweep_period, a2, b2, x2;
    b = ((double*) params)[0];
    a0 = ((double*) params)[1];
    af = ((double*) params)[2];
    sweep_period = ((double*) params)[3];
    at = linear_sweep_param(t, a0, af, sweep_period);
    b2 = b * b;
    a2 = at * at;

    for (uint16_t i = 0; i < npts; i++)
    {
        x2 = x[i] * x[i];
        V[i] = b2 * (x2 - a2 / b2) * (x2 - a2 / b2);
    }
}

void
potfunc_harmonicgauss(double t, uint16_t npts, Rarray x, void* params, Rarray V)
{
    double height, x2, omega, h0, hf, w, sweep_period;
    omega = ((double*) params)[0]; // harmonic oscillator frequency
    w = ((double*) params)[1];     // width of gaussian barrier
    h0 = ((double*) params)[2];    // initial barrier height
    hf = ((double*) params)[3];    // final barrier height
    sweep_period = ((double*) params)[4];
    height = linear_sweep_param(t, h0, hf, sweep_period);

    if (w < x[1] - x[0])
    {
        printf("\n\nERROR: linear potential barrier requires "
               "a width greater than spatial grid step size\n\n");
        exit(EXIT_FAILURE);
    }

    for (uint16_t i = 0; i < npts; i++)
    {
        x2 = x[i] * x[i];
        V[i] = omega * omega * x2 / 2 + height * exp(-x2 / (w * w));
    }
}

void
potfunc_barrier(double t, uint16_t M, Rarray x, void* params, Rarray V)
{
    double h0, hf, height, width, sweep_period, background;
    width = ((double*) params)[0];
    h0 = ((double*) params)[1];
    hf = ((double*) params)[2];
    sweep_period = ((double*) params)[3];
    background = ((double*) params)[4];
    height = linear_sweep_param(t, h0, hf, sweep_period);

    if (width < x[1] - x[0])
    {
        printf("\n\nERROR: linear potential barrier requires "
               "a width greater than spatial grid step size\n\n");
        exit(EXIT_FAILURE);
    }

    for (uint16_t i = 0; i < M; i++)
    {
        V[i] = background;
        if (fabs(x[i]) <= width / 2)
        {
            V[i] += height * cos(x[i] * PI / width) * cos(x[i] * PI / width);
        }
    }
}

void
potfunc_time_trapezoid_barrier(
    double t, uint16_t npts, Rarray x, void* params, Rarray pot)
{
    double ht, height, width, ramp_period, static_period;
    width = ((double*) params)[0];
    ht = ((double*) params)[1];
    ramp_period = ((double*) params)[2];
    static_period = ((double*) params)[3];

    height = trapezoid_sweep_param(t, ht, ramp_period, static_period);

    for (uint16_t i = 0; i < npts; i++)
    {
        pot[i] = 0;
        if (fabs(x[i]) <= width / 2)
        {
            pot[i] = height * cos(x[i] * PI / width) * cos(x[i] * PI / width);
        }
    }
}

void
potfunc_opticallattice(
    double t, uint16_t npts, Rarray x, void* params, Rarray pot)
{
    double len, k, height, h0, hf, sweep_period;
    len = x[npts - 1] - x[0];   // get length of domain grid
    k = ((double*) params)[0];  // optical wave number (must be integer)
    h0 = ((double*) params)[1]; // initial height
    hf = ((double*) params)[2]; // final height
    sweep_period = ((double*) params)[3];
    height = linear_sweep_param(t, h0, hf, sweep_period);

    for (uint16_t i = 0; i < npts; i++)
    {
        pot[i] = height * sin(PI * k * x[i] / len) * sin(PI * k * x[i] / len);
    }
}

void
potfunc_time_trapezoid_opticallattice(
    double t, uint16_t npts, Rarray x, void* params, Rarray pot)
{
    double len, k, height, ht, ramp_period, static_period;
    len = x[npts - 1] - x[0];   // get length of domain grid
    k = ((double*) params)[0];  // optical wave number (must be integer)
    ht = ((double*) params)[1]; // final height
    ramp_period = ((double*) params)[2];
    static_period = ((double*) params)[3];

    height = trapezoid_sweep_param(t, ht, ramp_period, static_period);

    for (uint16_t i = 0; i < npts; i++)
    {
        pot[i] = height * sin(PI * k * x[i] / len) * sin(PI * k * x[i] / len);
    }
}

void
potfunc_step(double t, uint16_t M, Rarray x, void* params, Rarray V)
{
    double xs, v0, vf, sweep_period, vt;
    xs = ((double*) params)[0];
    v0 = ((double*) params)[1];
    vf = ((double*) params)[2];
    sweep_period = ((double*) params)[3];
    vt = linear_sweep_param(t, v0, vf, sweep_period);
    for (uint16_t i = 0; i < M; i++)
    {
        V[i] = 0;
        if (x[i] >= xs) V[i] = vt;
    }
}

void
potfunc_square_well(double t, uint16_t M, Rarray x, void* params, Rarray V)
{
    double xi, xf, v_out0, v_outf, sweep_period, vt;
    xi = ((double*) params)[0];
    xf = ((double*) params)[1];
    v_out0 = ((double*) params)[2];
    v_outf = ((double*) params)[3];
    sweep_period = ((double*) params)[4];
    vt = linear_sweep_param(t, v_out0, v_outf, sweep_period);
    for (uint16_t i = 0; i < M; i++)
    {
        V[i] = vt;
        if (x[i] > xi || x[i] < xf) V[i] = 0;
    }
}

single_particle_pot
get_builtin_pot(char pot_name[])
{
    if (strcmp(pot_name, "harmonic") == 0)
    {
        return &potfunc_harmonic;
    }
    if (strcmp(pot_name, "doublewell") == 0)
    {
        return &potfunc_doublewell;
    }
    if (strcmp(pot_name, "harmonicgauss") == 0)
    {
        return &potfunc_harmonicgauss;
    }
    if (strcmp(pot_name, "barrier") == 0)
    {
        return &potfunc_barrier;
    }
    if (strcmp(pot_name, "trapezoid_barrier") == 0)
    {
        return &potfunc_time_trapezoid_barrier;
    }
    if (strcmp(pot_name, "opticallattice") == 0)
    {
        return &potfunc_opticallattice;
    }
    if (strcmp(pot_name, "trapezoid_opticallattice") == 0)
    {
        return &potfunc_time_trapezoid_opticallattice;
    }
    if (strcmp(pot_name, "potfunc_step") == 0)
    {
        return &potfunc_step;
    }
    if (strcmp(pot_name, "potfunc_square_well") == 0)
    {
        return &potfunc_square_well;
    }
    if (strcmp(pot_name, "custom") == 0)
    {
        return NULL;
    }
    printf("\n\nERROR: Potential '%s' not valid\n\n", pot_name);
    exit(EXIT_FAILURE);
}
