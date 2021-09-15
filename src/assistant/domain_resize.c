#include "assistant/domain_resize.h"
#include "assistant/arrays_definition.h"
#include "function_tools/calculus.h"
#include "function_tools/interpolation.h"
#include "linalg/basic_linalg.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double reduce_norm_tolerance = 1E-6;
double enlarge_grid_factor = 2;

static uint16_t
cutoff_border_threshold(uint16_t fsize, Carray f, double dx, double tol)
{
    uint16_t i, chunk_step;
    Rarray   mod_square;

    chunk_step = 5;
    i = chunk_step;
    mod_square = get_double_array(fsize);
    carrAbs2(fsize, f, mod_square);
    while (sqrt(real_border_integral(fsize, i, mod_square, dx)) < tol)
    {
        i = i + chunk_step;
    }
    free(mod_square);
    return i - chunk_step;
}

void
domain_reduction(OrbitalEquation eq_desc, ManyBodyState state)
{
    uint16_t i, j, norb, npts, cut_id;
    double   xi, xf, dx, old_xf, old_xi, old_dx;
    Rarray   real_orb, imag_orb, real_interpol, imag_interpol, old_grid;

    if (eq_desc->bounds == PERIODIC_BOUNDS) return;

    // Unpack some relevant struct parameters(possibly to be updated)
    norb = state->norb;
    npts = state->grid_size;
    old_xi = eq_desc->xi;
    old_xf = eq_desc->xf;
    old_dx = eq_desc->dx;
    old_grid = get_double_array(npts);
    rarrCopy(npts, eq_desc->grid_pts, old_grid);

    cut_id = UINT16_MAX;
    for (i = 0; i < norb; i++)
    {
        j = cutoff_border_threshold(
            npts, state->orbitals[i], old_dx, reduce_norm_tolerance);
        if (j < cut_id) cut_id = j;
    }

    // Proceed only if reduction is at least 10% of current domain
    if (fabs(old_grid[cut_id] - old_xi) / (old_xf - old_xi) < 0.05)
    {
        free(old_grid);
        return;
    }

    real_orb = get_double_array(npts);
    imag_orb = get_double_array(npts);
    real_interpol = get_double_array(npts);
    imag_interpol = get_double_array(npts);

    // New domain values
    xi = old_xi + cut_id * old_dx;
    xf = old_xf - cut_id * old_dx;
    dx = (xf - xi) / (npts - 1);

    // Set new values in equation descriptor
    rarrFillInc(npts, xi, dx, eq_desc->grid_pts);
    eq_desc->xi = xi;
    eq_desc->xf = xf;
    eq_desc->dx = dx;

    printf("\n\nNew spatial domain: [%.2lf,%.2lf]\n\n", xi, xf);

    // setup new one-body potential in discretized positions
    eq_desc->pot_func(
        eq_desc->t,
        npts,
        eq_desc->grid_pts,
        eq_desc->pot_extra_args,
        eq_desc->pot_grid);

    // Interpolate orbitals for the new domain
    for (i = 0; i < norb; i++)
    {
        carrRealPart(npts, state->orbitals[i], real_orb);
        carrImagPart(npts, state->orbitals[i], imag_orb);
        lagrange(
            npts,
            5,
            old_grid,
            real_orb,
            npts,
            eq_desc->grid_pts,
            real_interpol);
        lagrange(
            npts,
            5,
            old_grid,
            imag_orb,
            npts,
            eq_desc->grid_pts,
            imag_interpol);
        for (j = 0; j < npts; j++)
        {
            state->orbitals[i][j] = real_interpol[j] + I * imag_interpol[j];
        }
    }

    free(old_grid);
    free(real_orb);
    free(imag_orb);
    free(real_interpol);
    free(imag_interpol);
}

void
domain_extention(OrbitalEquation eq_desc, ManyBodyState state)
{

    uint16_t i, j, norb, old_size, tail_pts, new_size;
    double   dx;
    Rarray   oldx, x;
    Cmatrix  new_orb;

    // Unpack some relevant struct parameters
    norb = state->norb;
    old_size = state->grid_size;
    dx = eq_desc->dx;

    oldx = get_double_array(old_size);
    rarrCopy(old_size, eq_desc->grid_pts, oldx);

    tail_pts = old_size * (enlarge_grid_factor - 1.0) * 0.5;
    new_size = old_size + 2 * tail_pts;

    // Define new grid points preserving dx
    x = get_double_array(new_size);
    for (i = 0; i < old_size; i++)
    {
        x[i + tail_pts] = oldx[i];
    }
    for (i = 0; i < tail_pts; i++)
    {
        x[i] = oldx[0] - (tail_pts - i) * dx;
        x[i + old_size + tail_pts] = oldx[old_size - 1] + (i + 1) * dx;
    }

    // Update grid data on equation descriptor
    eq_desc->grid_size = new_size;
    eq_desc->xi = x[0];
    eq_desc->xf = x[new_size - 1];
    free(eq_desc->grid_pts);
    eq_desc->grid_pts = x;

    printf("\n\nDomain resized to [%.2lf,%.2lf]\n\n", x[0], x[new_size - 1]);

    // setup new one-body potential in discretized positions
    free(eq_desc->pot_grid);
    eq_desc->pot_grid = get_double_array(new_size);
    eq_desc->pot_func(
        eq_desc->t,
        eq_desc->grid_size,
        eq_desc->grid_pts,
        eq_desc->pot_extra_args,
        eq_desc->pot_grid);

    // Update orbitals in state descriptor
    new_orb = get_dcomplex_matrix(norb, new_size);
    for (j = 0; j < norb; j++)
    {
        for (i = 0; i < old_size; i++)
        {
            new_orb[j][i + tail_pts] = state->orbitals[j][i];
        }
        for (i = 0; i < tail_pts; i++)
        {
            new_orb[j][i] = 0.0;
            new_orb[j][i + old_size + tail_pts] = 0.0;
        }
    }
    destroy_dcomplex_matrix(norb, state->orbitals);
    state->orbitals = new_orb;
    state->grid_size = new_size;

    free(oldx);
}
