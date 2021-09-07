#include "integrator/split_linear_orbitals.h"

dcomplex
orb_interacting_part(int k, int n, double g, ManyBodyState psi)
{
    int      m, a, s, q, l, mm, mmm;
    dcomplex grho, swap, red;
    Carray   rho;
    Cmatrix  orb, rinv;

    rho = psi->tb_denmat;
    rinv = psi->inv_ob_denmat;
    m = psi->norb;
    orb = psi->orbitals;

    red = 0;
    mm = m * m;
    mmm = m * m * m;

    for (a = 0; a < m; a++)
    {
        for (q = 0; q < m; q++)
        {
            // Two pairs of index equals: a = s and q = l
            grho = g * rinv[k][a] * rho[a + m * a + mm * q + mmm * q];
            red = red + grho * conj(orb[a][n]) * orb[q][n] * orb[q][n];
            for (l = q + 1; l < m; l++)
            {
                // factor 2 from index exchange q <-> l due to l > q
                grho = 2 * g * rinv[k][a] * rho[a + m * a + mm * q + mmm * l];
                red = red + grho * conj(orb[a][n]) * orb[l][n] * orb[q][n];
            }
        }

        // The swap term account for a <-> s due to s > a restriction
        for (s = a + 1; s < m; s++)
        {
            for (q = 0; q < m; q++)
            {
                // Second pair of indexes equal (q == l)
                grho = g * rinv[k][a] * rho[a + m * s + mm * q + mmm * q];
                swap = g * rinv[k][s] * rho[a + m * s + mm * q + mmm * q];
                red = red + (grho * conj(orb[s][n]) + swap * conj(orb[a][n])) *
                                orb[q][n] * orb[q][n];
                for (l = q + 1; l < m; l++)
                {
                    // factor 2 from index exchange q <-> l due to l > q
                    grho =
                        2 * g * rinv[k][a] * rho[a + m * s + mm * q + mmm * l];
                    swap =
                        2 * g * rinv[k][s] * rho[a + m * s + mm * q + mmm * l];
                    red = red +
                          (grho * conj(orb[s][n]) + swap * conj(orb[a][n])) *
                              orb[l][n] * orb[q][n];
                }
            }
        }
    }
    return red;
}

dcomplex
orb_full_nonlinear(int k, int n, double g, ManyBodyState psi)
{
    int      a, m, j, s, q, l, mm, mmm, ind;
    dcomplex rhomult, swap, red;
    Carray   hint, rho;
    Cmatrix  orb, rinv, hob;

    m = psi->norb;
    hint = psi->hint;
    hob = psi->hob;
    rho = psi->tb_denmat;
    rinv = psi->inv_ob_denmat;
    orb = psi->orbitals;

    red = 0;
    mm = m * m;
    mmm = m * m * m;

    for (a = 0; a < m; a++)
    {
        red -= hob[a][k] * orb[a][n]; // subtract one-body projection
        for (q = 0; q < m; q++)
        {
            // both pairs of index equals (a == s and q ==l)
            rhomult = rinv[k][a] * rho[a + m * a + mm * q + mmm * q];
            red += g * rhomult * conj(orb[a][n]) * orb[q][n] * orb[q][n];
            // Subtract interacting projection
            for (j = 0; j < m; j++)
            {
                ind = j + a * m + q * mm + q * mmm;
                red -= rhomult * orb[j][n] * hint[ind];
            }
            for (l = q + 1; l < m; l++)
            {
                // Factor 2 due to l < q (exchange symmetry q<->l)
                rhomult = 2 * rinv[k][a] * rho[a + m * a + mm * q + mmm * l];
                red += g * rhomult * conj(orb[a][n]) * orb[l][n] * orb[q][n];
                // Subtract interacting projection
                for (j = 0; j < m; j++)
                {
                    ind = j + a * m + l * mm + q * mmm;
                    red -= rhomult * orb[j][n] * hint[ind];
                }
            }
        }

        // Case s > a implies a swap factor from a <-> s
        for (s = a + 1; s < m; s++)
        {
            for (q = 0; q < m; q++)
            {
                // Last pair of indexes equal (q == l)
                rhomult = rinv[k][a] * rho[a + m * s + mm * q + mmm * q];
                swap = rinv[k][s] * rho[a + m * s + mm * q + mmm * q];
                red += g *
                       (rhomult * conj(orb[s][n]) + swap * conj(orb[a][n])) *
                       orb[q][n] * orb[q][n];
                // Subtract interacting projection
                for (j = 0; j < m; j++)
                {
                    ind = j + s * m + q * mm + q * mmm;
                    red -= rhomult * orb[j][n] * hint[ind];
                    ind = j + a * m + q * mm + q * mmm;
                    red -= swap * orb[j][n] * hint[ind];
                }
                for (l = q + 1; l < m; l++)
                {
                    // Factor 2 due to l < q (exchange symmetry q<->l)
                    rhomult =
                        2 * rinv[k][a] * rho[a + m * s + mm * q + mmm * l];
                    swap = 2 * rinv[k][s] * rho[a + m * s + mm * q + mmm * l];
                    red +=
                        g *
                        (rhomult * conj(orb[s][n]) + swap * conj(orb[a][n])) *
                        orb[l][n] * orb[q][n];
                    // Subtract interacting projection
                    for (j = 0; j < m; j++)
                    {
                        ind = j + s * m + l * mm + q * mmm;
                        red -= rhomult * orb[j][n] * hint[ind];
                        ind = j + a * m + l * mm + q * mmm;
                        red -= swap * orb[j][n] * hint[ind];
                    }
                }
            }
        }
    }
    return red;
}
