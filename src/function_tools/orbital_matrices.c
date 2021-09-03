#include "function_tools/orbital_matrices.h"
#include "assistant/arrays_definition.h"
#include "function_tools/calculus.h"
#include <stdlib.h>

void
set_overlap_matrix(
    OrbitalEquation eq_desc, uint16_t norb, Cmatrix orb, Cmatrix overlap)
{
    for (uint16_t i = 0; i < norb; i++)
    {
        overlap[i][i] =
            scalar_product(eq_desc->grid_size, eq_desc->dx, orb[i], orb[i]);
        for (uint16_t j = 0; j < norb; j++)
        {
            overlap[i][j] =
                scalar_product(eq_desc->grid_size, eq_desc->dx, orb[i], orb[j]);
            overlap[j][i] = conj(overlap[i][j]);
        }
    }
}

void
set_orbital_hob(
    OrbitalEquation eq_desc, uint16_t norb, Cmatrix orb, Cmatrix hob)
{
    int      i, j, k, npts;
    double   dx, a2;
    dcomplex part, a1;
    Rarray   V;
    Carray   ddxi, ddxj, integ;

    V = eq_desc->pot_grid;
    dx = eq_desc->dx;
    a2 = eq_desc->d2coef;
    a1 = eq_desc->d1coef;
    npts = eq_desc->grid_size;
    ddxi = get_dcomplex_array(npts);
    ddxj = get_dcomplex_array(npts);
    integ = get_dcomplex_array(npts);

    for (i = 0; i < norb; i++)
    {
        dxFD(npts, dx, orb[i], ddxi);
        for (j = i + 1; j < norb; j++)
        {
            dxFD(npts, dx, orb[j], ddxj);
            for (k = 0; k < npts; k++)
            {
                part = -a2 * conj(ddxi[k]) * ddxj[k];
                part = part + a1 * conj(orb[i][k]) * ddxj[k];
                part = part + V[k] * conj(orb[i][k]) * orb[j][k];
                integ[k] = part;
            }
            part = Csimps(npts, dx, integ);
            hob[i][j] = part;
            hob[j][i] = conj(part);
        }
        for (k = 0; k < npts; k++)
        {
            part = -a2 * conj(ddxi[k]) * ddxi[k];
            part = part + a1 * conj(orb[i][k]) * ddxi[k];
            part = part + V[k] * conj(orb[i][k]) * orb[i][k];
            integ[k] = part;
        }
        part = Csimps(npts, dx, integ);
        hob[i][i] = creal(part);
    }
    free(ddxi);
    free(ddxj);
    free(integ);
}

void
set_orbital_hint(
    OrbitalEquation eq_desc, uint16_t norb, Cmatrix orb, Carray hint)
{
    uint16_t i, k, s, q, l;
    uint32_t mm, mmm, Mpos;
    double   g, dx;
    dcomplex common_integ;
    Carray   integ;

    g = eq_desc->g;
    dx = eq_desc->dx;
    Mpos = eq_desc->grid_size;
    mm = norb * norb; // stride for easier index access
    mmm = norb * mm;  // stride for easier index access
    integ = get_dcomplex_array(Mpos);

    for (k = 0; k < norb; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            integ[i] = conj(orb[k][i] * orb[k][i]) * orb[k][i] * orb[k][i];
        }
        hint[k * (1 + norb + mm + mmm)] = g * Csimps(Mpos, dx, integ);

        for (s = k + 1; s < norb; s++)
        {
            for (i = 0; i < Mpos; i++)
            {
                integ[i] = conj(orb[k][i] * orb[s][i]) * orb[k][i] * orb[k][i];
            }
            common_integ = g * Csimps(Mpos, dx, integ);

            hint[k + s * norb + k * mm + k * mmm] = common_integ;
            hint[s + k * norb + k * mm + k * mmm] = common_integ;
            hint[k + k * norb + k * mm + s * mmm] = conj(common_integ);
            hint[k + k * norb + s * mm + k * mmm] = conj(common_integ);

            for (i = 0; i < Mpos; i++)
            {
                integ[i] = conj(orb[s][i] * orb[k][i]) * orb[s][i] * orb[s][i];
            }
            common_integ = g * Csimps(Mpos, dx, integ);

            hint[s + k * norb + s * mm + s * mmm] = common_integ;
            hint[k + s * norb + s * mm + s * mmm] = common_integ;
            hint[s + s * norb + s * mm + k * mmm] = conj(common_integ);
            hint[s + s * norb + k * mm + s * mmm] = conj(common_integ);

            for (i = 0; i < Mpos; i++)
            {
                integ[i] = conj(orb[k][i] * orb[s][i]) * orb[s][i] * orb[k][i];
            }
            common_integ = g * Csimps(Mpos, dx, integ);

            hint[k + s * norb + s * mm + k * mmm] = common_integ;
            hint[s + k * norb + s * mm + k * mmm] = common_integ;
            hint[s + k * norb + k * mm + s * mmm] = common_integ;
            hint[k + s * norb + k * mm + s * mmm] = common_integ;

            for (i = 0; i < Mpos; i++)
            {
                integ[i] = conj(orb[k][i] * orb[k][i]) * orb[s][i] * orb[s][i];
            }
            common_integ = g * Csimps(Mpos, dx, integ);

            hint[k + k * norb + s * mm + s * mmm] = common_integ;
            hint[s + s * norb + k * mm + k * mmm] = conj(common_integ);

            for (q = s + 1; q < norb; q++)
            {
                for (i = 0; i < Mpos; i++)
                {
                    integ[i] =
                        conj(orb[k][i] * orb[s][i]) * orb[q][i] * orb[k][i];
                }
                common_integ = g * Csimps(Mpos, dx, integ);

                hint[k + s * norb + q * mm + k * mmm] = common_integ;
                hint[k + s * norb + k * mm + q * mmm] = common_integ;
                hint[s + k * norb + k * mm + q * mmm] = common_integ;
                hint[s + k * norb + q * mm + k * mmm] = common_integ;

                hint[k + q * norb + s * mm + k * mmm] = conj(common_integ);
                hint[k + q * norb + k * mm + s * mmm] = conj(common_integ);
                hint[q + k * norb + k * mm + s * mmm] = conj(common_integ);
                hint[q + k * norb + s * mm + k * mmm] = conj(common_integ);

                for (i = 0; i < Mpos; i++)
                {
                    integ[i] =
                        conj(orb[s][i] * orb[k][i]) * orb[q][i] * orb[s][i];
                }
                common_integ = g * Csimps(Mpos, dx, integ);

                hint[s + k * norb + q * mm + s * mmm] = common_integ;
                hint[k + s * norb + q * mm + s * mmm] = common_integ;
                hint[k + s * norb + s * mm + q * mmm] = common_integ;
                hint[s + k * norb + s * mm + q * mmm] = common_integ;

                hint[s + q * norb + k * mm + s * mmm] = conj(common_integ);
                hint[s + q * norb + s * mm + k * mmm] = conj(common_integ);
                hint[q + s * norb + s * mm + k * mmm] = conj(common_integ);
                hint[q + s * norb + k * mm + s * mmm] = conj(common_integ);

                for (i = 0; i < Mpos; i++)
                {
                    integ[i] =
                        conj(orb[q][i] * orb[s][i]) * orb[k][i] * orb[q][i];
                }
                common_integ = g * Csimps(Mpos, dx, integ);

                hint[q + s * norb + k * mm + q * mmm] = common_integ;
                hint[q + s * norb + q * mm + k * mmm] = common_integ;
                hint[s + q * norb + q * mm + k * mmm] = common_integ;
                hint[s + q * norb + k * mm + q * mmm] = common_integ;

                hint[k + q * norb + s * mm + q * mmm] = conj(common_integ);
                hint[k + q * norb + q * mm + s * mmm] = conj(common_integ);
                hint[q + k * norb + s * mm + q * mmm] = conj(common_integ);
                hint[q + k * norb + q * mm + s * mmm] = conj(common_integ);

                for (i = 0; i < Mpos; i++)
                {
                    integ[i] =
                        conj(orb[k][i] * orb[k][i]) * orb[q][i] * orb[s][i];
                }
                common_integ = g * Csimps(Mpos, dx, integ);

                hint[k + k * norb + q * mm + s * mmm] = common_integ;
                hint[k + k * norb + s * mm + q * mmm] = common_integ;
                hint[q + s * norb + k * mm + k * mmm] = conj(common_integ);
                hint[s + q * norb + k * mm + k * mmm] = conj(common_integ);

                for (i = 0; i < Mpos; i++)
                {
                    integ[i] =
                        conj(orb[s][i] * orb[s][i]) * orb[k][i] * orb[q][i];
                }
                common_integ = g * Csimps(Mpos, dx, integ);

                hint[s + s * norb + k * mm + q * mmm] = common_integ;
                hint[s + s * norb + q * mm + k * mmm] = common_integ;
                hint[k + q * norb + s * mm + s * mmm] = conj(common_integ);
                hint[q + k * norb + s * mm + s * mmm] = conj(common_integ);

                for (i = 0; i < Mpos; i++)
                {
                    integ[i] =
                        conj(orb[q][i] * orb[q][i]) * orb[k][i] * orb[s][i];
                }
                common_integ = g * Csimps(Mpos, dx, integ);

                hint[q + q * norb + k * mm + s * mmm] = common_integ;
                hint[q + q * norb + s * mm + k * mmm] = common_integ;
                hint[k + s * norb + q * mm + q * mmm] = conj(common_integ);
                hint[s + k * norb + q * mm + q * mmm] = conj(common_integ);

                for (l = q + 1; l < norb; l++)
                {
                    for (i = 0; i < Mpos; i++)
                    {
                        integ[i] =
                            conj(orb[k][i] * orb[s][i]) * orb[q][i] * orb[l][i];
                    }
                    common_integ = g * Csimps(Mpos, dx, integ);

                    hint[k + s * norb + q * mm + l * mmm] = common_integ;
                    hint[k + s * norb + l * mm + q * mmm] = common_integ;
                    hint[s + k * norb + q * mm + l * mmm] = common_integ;
                    hint[s + k * norb + l * mm + q * mmm] = common_integ;

                    hint[q + l * norb + k * mm + s * mmm] = conj(common_integ);
                    hint[l + q * norb + k * mm + s * mmm] = conj(common_integ);
                    hint[l + q * norb + s * mm + k * mmm] = conj(common_integ);
                    hint[q + l * norb + s * mm + k * mmm] = conj(common_integ);

                    for (i = 0; i < Mpos; i++)
                    {
                        integ[i] =
                            conj(orb[k][i] * orb[q][i]) * orb[s][i] * orb[l][i];
                    }
                    common_integ = g * Csimps(Mpos, dx, integ);

                    hint[k + q * norb + s * mm + l * mmm] = common_integ;
                    hint[k + q * norb + l * mm + s * mmm] = common_integ;
                    hint[q + k * norb + s * mm + l * mmm] = common_integ;
                    hint[q + k * norb + l * mm + s * mmm] = common_integ;

                    hint[s + l * norb + k * mm + q * mmm] = conj(common_integ);
                    hint[s + l * norb + q * mm + k * mmm] = conj(common_integ);
                    hint[l + s * norb + q * mm + k * mmm] = conj(common_integ);
                    hint[l + s * norb + k * mm + q * mmm] = conj(common_integ);

                    for (i = 0; i < Mpos; i++)
                    {
                        integ[i] =
                            conj(orb[k][i] * orb[l][i]) * orb[s][i] * orb[q][i];
                    }
                    common_integ = g * Csimps(Mpos, dx, integ);

                    hint[k + l * norb + s * mm + q * mmm] = common_integ;
                    hint[k + l * norb + q * mm + s * mmm] = common_integ;
                    hint[l + k * norb + s * mm + q * mmm] = common_integ;
                    hint[l + k * norb + q * mm + s * mmm] = common_integ;

                    hint[s + q * norb + k * mm + l * mmm] = conj(common_integ);
                    hint[s + q * norb + l * mm + k * mmm] = conj(common_integ);
                    hint[q + s * norb + l * mm + k * mmm] = conj(common_integ);
                    hint[q + s * norb + k * mm + l * mmm] = conj(common_integ);
                }
            }
        }
    }
    free(integ);
}
