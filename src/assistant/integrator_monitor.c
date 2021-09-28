#include "assistant/integrator_monitor.h"
#include "assistant/arrays_definition.h"
#include "configurational/hamiltonian.h"
#include "function_tools/calculus.h"
#include "linalg/basic_linalg.h"
#include <math.h>
#include <stdlib.h>

double
eig_residual(
    MultiConfiguration multiconf, Carray c, Cmatrix hob, Carray hint, double e)
{
    double max_norm;
    Carray hc;

    hc = get_dcomplex_array(multiconf->dim);
    apply_hamiltonian(multiconf, c, hob, hint, hc);
    max_norm = 0;
    for (uint32_t i = 0; i < multiconf->dim; i++)
    {
        if (cabs(e * c[i] - hc[i]) > max_norm)
        {
            max_norm = cabs(e * c[i] - hc[i]);
        }
    }
    free(hc);
    return max_norm / multiconf->npar;
}

double
overlap_residual(uint16_t norb, uint16_t grid_size, double dx, Cmatrix orb)
{
    double summ = 0;
    for (uint16_t k = 0; k < norb; k++)
    {
        for (uint16_t l = k + 1; l < norb; l++)
        {
            summ = summ + cabs(scalar_product(grid_size, dx, orb[k], orb[l]));
        }
    }
    return summ / norb;
}

double
avg_orbitals_norm(uint16_t norb, uint16_t grid_size, double dx, Cmatrix Orb)
{
    double summ = 0;
    for (uint16_t k = 0; k < norb; k++)
    {
        summ = summ + cplx_function_norm(grid_size, dx, Orb[k]);
    }
    return summ / norb;
}

dcomplex
total_energy(ManyBodyState psi)
{
    int      i, j, k, l, s, q;
    uint16_t Morb;
    Cmatrix  rho1, hob;
    Carray   hint, rho2;
    dcomplex z, w;

    Morb = psi->norb;
    rho1 = psi->ob_denmat;
    rho2 = psi->tb_denmat;
    hob = psi->hob;
    hint = psi->hint;
    z = 0;
    w = 0;
    for (k = 0; k < Morb; k++)
    {
        for (l = 0; l < Morb; l++)
        {
            w = w + rho1[k][l] * hob[k][l];
            for (s = 0; s < Morb; s++)
            {
                for (q = 0; q < Morb; q++)
                {
                    i = k + l * Morb + s * Morb * Morb + q * Morb * Morb * Morb;
                    j = k + l * Morb + q * Morb * Morb + s * Morb * Morb * Morb;
                    z = z + rho2[i] * hint[j];
                }
            }
        }
    }
    return (w + z / 2);
}

dcomplex
momentum_per_particle(OrbitalEquation eq_desc, ManyBodyState psi)
{
    uint16_t i, j;
    uint16_t npts, norb;
    double   dx;
    dcomplex overlap, reduc;
    Carray   ddxj;
    Cmatrix  rho, orb;

    npts = eq_desc->grid_size;
    dx = eq_desc->dx;
    norb = psi->norb;
    orb = psi->orbitals;
    rho = psi->ob_denmat;
    ddxj = get_dcomplex_array(npts);

    reduc = 0;
    for (i = 0; i < norb; i++)
    {
        dxFD(npts, dx, orb[i], ddxj);
        overlap = -I * scalar_product(npts, dx, orb[i], ddxj);
        reduc += rho[i][i] * overlap;
        for (j = i + 1; j < norb; j++)
        {
            dxFD(npts, dx, orb[j], ddxj);
            overlap = -I * scalar_product(npts, dx, orb[i], ddxj);
            reduc += rho[i][j] * overlap + conj(rho[i][j] * overlap);
        }
    }

    free(ddxj);
    return reduc / psi->npar;
}

dcomplex
kinect_energy(OrbitalEquation eq_desc, ManyBodyState psi)
{
    int      i, j, k;
    uint16_t npts, norb;
    double   dx, d2coef;
    dcomplex r;
    Carray   ddxi, ddxj, integ;
    Cmatrix  rho, orb;

    npts = eq_desc->grid_size;
    dx = eq_desc->dx;
    d2coef = eq_desc->d2coef;
    norb = psi->norb;
    orb = psi->orbitals;
    rho = psi->ob_denmat;
    ddxi = get_dcomplex_array(npts);
    ddxj = get_dcomplex_array(npts);
    integ = get_dcomplex_array(npts);

    carrFill(npts, 0, integ);

    for (i = 0; i < norb; i++)
    {
        dxFD(npts, dx, orb[i], ddxi);
        for (j = 0; j < norb; j++)
        {
            r = rho[i][j];
            dxFD(npts, dx, orb[j], ddxj);
            for (k = 0; k < npts; k++)
            {
                integ[k] = integ[k] - d2coef * r * conj(ddxi[k]) * ddxj[k];
            }
        }
    }
    r = Csimps(npts, dx, integ);
    free(ddxi);
    free(ddxj);
    free(integ);
    return r;
}

dcomplex
onebody_potential_energy(OrbitalEquation eq_desc, ManyBodyState psi)
{
    int      i, j, k;
    uint16_t norb, npts;
    double   dx;
    dcomplex r;
    Rarray   V;
    Carray   integ;
    Cmatrix  rho, orb;

    npts = eq_desc->grid_size;
    norb = psi->norb;
    dx = eq_desc->dx;
    V = eq_desc->pot_grid;
    orb = psi->orbitals;
    rho = psi->ob_denmat;
    integ = get_dcomplex_array(npts);

    carrFill(npts, 0, integ);

    for (i = 0; i < norb; i++)
    {
        for (j = 0; j < norb; j++)
        {
            r = rho[i][j];
            for (k = 0; k < npts; k++)
                integ[k] = integ[k] + r * V[k] * conj(orb[i][k]) * orb[j][k];
        }
    }
    r = Csimps(npts, dx, integ);
    free(integ);
    return r;
}

dcomplex
interacting_energy(ManyBodyState psi)
{
    uint16_t k, s, q, l, m;
    uint32_t mm, mmm, two_body_id;
    dcomplex reduction;
    Carray   hint, rho;

    m = psi->norb;
    mm = m * m;
    mmm = m * m * m;
    hint = psi->hint;
    rho = psi->tb_denmat;

    reduction = 0;
    for (k = 0; k < m; k++)
    {
        s = k;
        for (q = 0; q < m; q++)
        {
            l = q;
            two_body_id = k + s * m + q * mm + l * mmm;
            reduction += rho[two_body_id] * hint[two_body_id];
            for (l = q + 1; l < m; l++)
            {
                // factor 2 from exchange q <-> l
                two_body_id = k + s * m + q * mm + l * mmm;
                reduction += 2 * rho[two_body_id] * hint[two_body_id];
            }
        }
        for (s = k + 1; s < m; s++)
        {
            for (q = 0; q < m; q++)
            {
                l = q;
                // factor 2 from exchange s <-> k
                two_body_id = k + s * m + q * mm + l * mmm;
                reduction += 2 * rho[two_body_id] * hint[two_body_id];
                for (l = q + 1; l < m; l++)
                {
                    // factor 4 from exchange s <-> k and q <-> l
                    two_body_id = k + s * m + q * mm + l * mmm;
                    reduction += 4 * rho[two_body_id] * hint[two_body_id];
                }
            }
        }
    }
    return reduction;
}

dcomplex
virial_harmonic_residue(OrbitalEquation eq_desc, ManyBodyState psi)
{
    dcomplex kinect, potential, interacting;

    kinect = kinect_energy(eq_desc, psi);
    potential = onebody_potential_energy(eq_desc, psi);
    interacting = interacting_energy(psi);
    return (2 * potential - 2 * kinect - interacting);
}

static dcomplex
square_pos_amplitude(int n, double dx, Rarray x, Carray f, Carray g)
{
    int      i;
    dcomplex r2;
    Carray   integ;

    integ = get_dcomplex_array(n);
    for (i = 0; i < n; i++)
    {
        integ[i] = conj(f[i]) * g[i] * x[i] * x[i];
    }
    r2 = Csimps(n, dx, integ);
    free(integ);
    return r2;
}

double
mean_quadratic_pos(OrbitalEquation eq_desc, ManyBodyState psi)
{
    uint16_t i, j, npar, norb, Ngrid;
    dcomplex amp, xq;
    double   dx;
    Rarray   x;
    Cmatrix  rho, orb;

    Ngrid = eq_desc->grid_size;
    npar = psi->npar;
    norb = psi->norb;
    x = eq_desc->grid_pts;
    dx = eq_desc->dx;
    rho = psi->ob_denmat;
    orb = psi->orbitals;

    xq = 0;
    for (i = 0; i < norb; i++)
    {
        xq =
            xq + rho[i][i] * square_pos_amplitude(Ngrid, dx, x, orb[i], orb[i]);
        for (j = i + 1; j < norb; j++)
        {
            amp =
                rho[i][j] * square_pos_amplitude(Ngrid, dx, x, orb[i], orb[j]);
            xq = xq + amp + conj(amp);
        }
    }
    return sqrt(creal(xq) / npar);
}
