#include "configurational/density_matrices.h"
#include "assistant/arrays_definition.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void
set_onebody_dm(MultiConfiguration multiconf, Carray coef, Cmatrix rho)
{
    uint16_t  norb, k, l;
    uint32_t  i, j, occ_k, occ_l, dim;
    uint16_t* ht;
    uint32_t* map;
    double    mod_square;
    dcomplex  rho_sum;

    dim = multiconf->dim;
    norb = multiconf->norb;
    ht = multiconf->hash_table;
    map = multiconf->op_maps->map;

    for (k = 0; k < norb; k++)
    {
        // Diagonal elements
        rho_sum = 0;
        for (i = 0; i < dim; i++)
        {
            mod_square = creal(coef[i]) * creal(coef[i]) +
                         cimag(coef[i]) * cimag(coef[i]);
            rho_sum = rho_sum + mod_square * ht[k + i * norb];
        }
        rho[k][k] = rho_sum;
        // Off-diagonal elements
        for (l = k + 1; l < norb; l++)
        {
            rho_sum = 0;
            for (i = 0; i < dim; i++)
            {
                occ_k = ht[k + norb * i];
                if (occ_k < 1) continue;
                occ_l = ht[l + norb * i];
                j = map[i + k * dim + l * norb * dim];
                rho_sum += conj(coef[i]) * coef[j] *
                           sqrt((double) (occ_l + 1) * occ_k);
            }
            // exploit hermiticity
            rho[k][l] = rho_sum;
            rho[l][k] = conj(rho_sum);
        }
    }
}

void
set_twobody_dm(MultiConfiguration multiconf, Carray coef, Carray rho)
{
    uint16_t  k, s, q, l, h, g, norb, occ_k, occ_s, occ_q, occ_l;
    uint32_t  i, j, dim, norb2, norb3, chunks, strideOrb;
    uint16_t* ht;
    uint32_t *map, *mapot, *maptt, *strides_ot, *strides_tt;
    double    mod_square, bose_fac;
    dcomplex  rho_sum;

    dim = multiconf->dim;
    norb = multiconf->norb;
    ht = multiconf->hash_table;
    map = multiconf->op_maps->map;
    mapot = multiconf->op_maps->mapot;
    maptt = multiconf->op_maps->maptt;
    strides_ot = multiconf->op_maps->strideot;
    strides_tt = multiconf->op_maps->stridett;

    norb2 = norb * norb;
    norb3 = norb * norb * norb;

    // Rule 1: Creation on k k / Annihilation on k k
    for (k = 0; k < norb; k++)
    {
        rho_sum = 0;
#pragma omp parallel for private(i, mod_square, bose_fac) reduction(+ : rho_sum)
        for (i = 0; i < dim; i++)
        {
            if (ht[k + i * norb] < 2) continue;
            mod_square = creal(coef[i]) * creal(coef[i]) +
                         cimag(coef[i]) * cimag(coef[i]);
            bose_fac = ht[k + i * norb] * (ht[k + i * norb] - 1);
            rho_sum += mod_square * bose_fac;
        }
        rho[k + norb * k + norb2 * k + norb3 * k] = rho_sum;
    }

    // Rule 2: Creation on k s / Annihilation on k s
    for (k = 0; k < norb; k++)
    {
        for (s = k + 1; s < norb; s++)
        {
            rho_sum = 0;
#pragma omp parallel for private(i, mod_square, bose_fac) reduction(+ : rho_sum)
            for (i = 0; i < dim; i++)
            {
                mod_square = creal(coef[i]) * creal(coef[i]) +
                             cimag(coef[i]) * cimag(coef[i]);
                bose_fac = ht[k + i * norb] * ht[s + i * norb];
                rho_sum += mod_square * bose_fac;
            }
            // commutation of bosonic operators is used
            // to fill elements by exchange  of indexes
            rho[k + s * norb + k * norb2 + s * norb3] = rho_sum;
            rho[s + k * norb + k * norb2 + s * norb3] = rho_sum;
            rho[s + k * norb + s * norb2 + k * norb3] = rho_sum;
            rho[k + s * norb + s * norb2 + k * norb3] = rho_sum;
        }
    }

    // Rule 3: Creation on k k / Annihilation on q q
    for (k = 0; k < norb; k++)
    {
        for (q = k + 1; q < norb; q++)
        {
            rho_sum = 0;
#pragma omp parallel for private(i, j, h, occ_k, occ_q, chunks, strideOrb, bose_fac) reduction(+ : rho_sum)
            for (i = 0; i < dim; i++)
            {
                h = i * norb; // auxiliar stride for IF
                if (ht[h + k] < 2) continue;
                occ_k = ht[h + k];
                occ_q = ht[h + q];

                chunks = 0;
                for (j = 0; j < k; j++)
                {
                    if (ht[h + j] > 1) chunks++;
                }
                strideOrb = chunks * norb * norb;

                j = mapot[strides_ot[i] + strideOrb + q + q * norb];
                bose_fac = sqrt(
                    (double) (occ_k - 1) * occ_k * (occ_q + 1) * (occ_q + 2));
                rho_sum += conj(coef[i]) * coef[j] * bose_fac;
            }
            // Use 2-index-'hermiticity'
            rho[k + k * norb + q * norb2 + q * norb3] = rho_sum;
            rho[q + q * norb + k * norb2 + k * norb3] = conj(rho_sum);
        }
    }

    // Rule 4: Creation on k k / Annihilation on k l
    for (k = 0; k < norb; k++)
    {
        for (l = k + 1; l < norb; l++)
        {
            rho_sum = 0;
#pragma omp parallel for private(i, j, occ_k, occ_l, bose_fac) reduction(+ : rho_sum)
            for (i = 0; i < dim; i++)
            {
                occ_k = ht[i * norb + k];
                occ_l = ht[i * norb + l];
                if (occ_k < 2) continue;
                j = map[i + k * dim + l * norb * dim];
                bose_fac = (occ_k - 1) * sqrt((double) occ_k * (occ_l + 1));
                rho_sum += conj(coef[i]) * coef[j] * bose_fac;
            }
            rho[k + k * norb + k * norb2 + l * norb3] = rho_sum;
            rho[k + k * norb + l * norb2 + k * norb3] = rho_sum;
            rho[l + k * norb + k * norb2 + k * norb3] = conj(rho_sum);
            rho[k + l * norb + k * norb2 + k * norb3] = conj(rho_sum);
        }
    }

    // Rule 5: Creation on k s / Annihilation on s s
    for (k = 0; k < norb; k++)
    {
        for (s = k + 1; s < norb; s++)
        {
            rho_sum = 0;
#pragma omp parallel for private(i, j, occ_k, occ_s, sqrtOf) reduction(+ : rho_sum)
            for (i = 0; i < dim; i++)
            {
                occ_k = ht[i * norb + k];
                occ_s = ht[i * norb + s];
                if (occ_k < 1 || occ_s < 1) continue;
                j = map[i + k * dim + s * norb * dim];
                bose_fac = occ_s * sqrt((double) occ_k * (occ_s + 1));
                rho_sum += conj(coef[i]) * coef[j] * bose_fac;
            }
            rho[k + s * norb + s * norb2 + s * norb3] = rho_sum;
            rho[s + k * norb + s * norb2 + s * norb3] = rho_sum;
            rho[s + s * norb + s * norb2 + k * norb3] = conj(rho_sum);
            rho[s + s * norb + k * norb2 + s * norb3] = conj(rho_sum);
        }
    }

    // Rule 6.0: Creation on k k / Annihilation on q l (k < q < l)
    for (k = 0; k < norb; k++)
    {
        for (q = k + 1; q < norb; q++)
        {
            for (l = q + 1; l < norb; l++)
            {
                rho_sum = 0;
#pragma omp parallel for private(i, j, h, occ_k, occ_q, occ_l, chunks, strideOrb, bose_fac) reduction(+ : rho_sum)
                for (i = 0; i < dim; i++)
                {
                    h = i * norb;
                    if (ht[h + k] < 2) continue;
                    occ_k = ht[h + k];
                    occ_q = ht[h + q];
                    occ_l = ht[h + l];
                    chunks = 0;
                    for (j = 0; j < k; j++)
                    {
                        if (ht[h + j] > 1) chunks++;
                    }
                    strideOrb = chunks * norb * norb;
                    j = mapot[strides_ot[i] + strideOrb + l + q * norb];
                    bose_fac = sqrt(
                        (double) occ_k * (occ_k - 1) * (occ_q + 1) *
                        (occ_l + 1));
                    rho_sum += conj(coef[i]) * coef[j] * bose_fac;
                }
                rho[k + k * norb + q * norb2 + l * norb3] = rho_sum;
                rho[k + k * norb + l * norb2 + q * norb3] = rho_sum;
                rho[l + q * norb + k * norb2 + k * norb3] = conj(rho_sum);
                rho[q + l * norb + k * norb2 + k * norb3] = conj(rho_sum);
            }
        }
    }

    // Rule 6.1: Creation on k k / Annihilation on q l (q < k < l)
    for (q = 0; q < norb; q++)
    {
        for (k = q + 1; k < norb; k++)
        {
            for (l = k + 1; l < norb; l++)
            {
                rho_sum = 0;
#pragma omp parallel for private(i, j, h, occ_k, occ_q, occ_l, chunks, strideOrb, bose_fac) reduction(+ : rho_sum)
                for (i = 0; i < dim; i++)
                {
                    h = i * norb;
                    if (ht[h + k] < 2) continue;
                    occ_k = ht[h + k];
                    occ_q = ht[h + q];
                    occ_l = ht[h + l];

                    chunks = 0;
                    for (j = 0; j < k; j++)
                    {
                        if (ht[h + j] > 1) chunks++;
                    }
                    strideOrb = chunks * norb * norb;

                    j = mapot[strides_ot[i] + strideOrb + l + q * norb];
                    bose_fac = sqrt(
                        (double) occ_k * (occ_k - 1) * (occ_q + 1) *
                        (occ_l + 1));

                    rho_sum += conj(coef[i]) * coef[j] * bose_fac;
                }
                rho[k + k * norb + q * norb2 + l * norb3] = rho_sum;
                rho[k + k * norb + l * norb2 + q * norb3] = rho_sum;
                rho[l + q * norb + k * norb2 + k * norb3] = conj(rho_sum);
                rho[q + l * norb + k * norb2 + k * norb3] = conj(rho_sum);
            }
        }
    }

    // Rule 6.2: Creation on k k / Annihilation on q l (q < l < k)
    for (q = 0; q < norb; q++)
    {
        for (l = q + 1; l < norb; l++)
        {
            for (k = l + 1; k < norb; k++)
            {
                rho_sum = 0;
#pragma omp parallel for private(i, j, h, occ_k, occ_q, occ_l, chunks, strideOrb, bose_fac) reduction(+ : rho_sum)
                for (i = 0; i < dim; i++)
                {
                    h = i * norb;
                    if (ht[h + k] < 2) continue;
                    occ_k = ht[h + k];
                    occ_q = ht[h + q];
                    occ_l = ht[h + l];

                    chunks = 0;
                    for (j = 0; j < k; j++)
                    {
                        if (ht[h + j] > 1) chunks++;
                    }
                    strideOrb = chunks * norb * norb;

                    j = mapot[strides_ot[i] + strideOrb + l + q * norb];
                    bose_fac = sqrt(
                        (double) occ_k * (occ_k - 1) * (occ_q + 1) *
                        (occ_l + 1));
                    rho_sum += conj(coef[i]) * coef[j] * bose_fac;
                }
                rho[k + k * norb + q * norb2 + l * norb3] = rho_sum;
                rho[k + k * norb + l * norb2 + q * norb3] = rho_sum;
                rho[l + q * norb + k * norb2 + k * norb3] = conj(rho_sum);
                rho[q + l * norb + k * norb2 + k * norb3] = conj(rho_sum);
            }
        }
    }

    // Rule 7.0: Creation on k s / Annihilation on s l (s < k < l)
    for (s = 0; s < norb; s++)
    {
        for (k = s + 1; k < norb; k++)
        {
            for (l = k + 1; l < norb; l++)
            {
                rho_sum = 0;
#pragma omp parallel for private(i, j, occ_k, occ_s, occ_l, bose_fac) reduction(+ : rho_sum)
                for (i = 0; i < dim; i++)
                {
                    occ_s = ht[i * norb + s];
                    occ_k = ht[i * norb + k];
                    occ_l = ht[i * norb + l];
                    if (occ_k < 1 || occ_s < 1) continue;
                    j = map[i + k * dim + l * norb * dim];
                    bose_fac = occ_s * sqrt((double) occ_k * (occ_l + 1));
                    rho_sum += conj(coef[i]) * coef[j] * bose_fac;
                }
                rho[k + s * norb + s * norb2 + l * norb3] = rho_sum;
                rho[s + k * norb + s * norb2 + l * norb3] = rho_sum;
                rho[s + k * norb + l * norb2 + s * norb3] = rho_sum;
                rho[k + s * norb + l * norb2 + s * norb3] = rho_sum;
                rho[l + s * norb + s * norb2 + k * norb3] = conj(rho_sum);
                rho[s + l * norb + s * norb2 + k * norb3] = conj(rho_sum);
                rho[s + l * norb + k * norb2 + s * norb3] = conj(rho_sum);
                rho[l + s * norb + k * norb2 + s * norb3] = conj(rho_sum);
            }
        }
    }

    // Rule 7.1: Creation on k s / Annihilation on s l (k < s < l)
    for (k = 0; k < norb; k++)
    {
        for (s = k + 1; s < norb; s++)
        {
            for (l = s + 1; l < norb; l++)
            {
                rho_sum = 0;
#pragma omp parallel for private(i, j, occ_k, occ_s, occ_l, bose_fac) reduction(+ : rho_sum)
                for (i = 0; i < dim; i++)
                {
                    occ_s = ht[i * norb + s];
                    occ_k = ht[i * norb + k];
                    occ_l = ht[i * norb + l];
                    if (occ_k < 1 || occ_s < 1) continue;
                    j = map[i + k * dim + l * norb * dim];
                    bose_fac = occ_s * sqrt((double) occ_k * (occ_l + 1));
                    rho_sum += conj(coef[i]) * coef[j] * bose_fac;
                }
                rho[k + s * norb + s * norb2 + l * norb3] = rho_sum;
                rho[s + k * norb + s * norb2 + l * norb3] = rho_sum;
                rho[s + k * norb + l * norb2 + s * norb3] = rho_sum;
                rho[k + s * norb + l * norb2 + s * norb3] = rho_sum;
                rho[l + s * norb + s * norb2 + k * norb3] = conj(rho_sum);
                rho[s + l * norb + s * norb2 + k * norb3] = conj(rho_sum);
                rho[s + l * norb + k * norb2 + s * norb3] = conj(rho_sum);
                rho[l + s * norb + k * norb2 + s * norb3] = conj(rho_sum);
            }
        }
    }

    // Rule 7.2: Creation on k s / Annihilation on s l (k < l < s)
    for (k = 0; k < norb; k++)
    {
        for (l = k + 1; l < norb; l++)
        {
            for (s = l + 1; s < norb; s++)
            {
                rho_sum = 0;
#pragma omp parallel for private(i, j, occ_k, occ_s, occ_l, bose_fac) reduction(+ : rho_sum)
                for (i = 0; i < dim; i++)
                {
                    occ_s = ht[i * norb + s];
                    occ_k = ht[i * norb + k];
                    occ_l = ht[i * norb + l];
                    if (occ_k < 1 || occ_s < 1) continue;
                    j = map[i + k * dim + l * norb * dim];
                    bose_fac = occ_s * sqrt((double) occ_k * (occ_l + 1));
                    rho_sum += conj(coef[i]) * coef[j] * bose_fac;
                }
                rho[k + s * norb + s * norb2 + l * norb3] = rho_sum;
                rho[s + k * norb + s * norb2 + l * norb3] = rho_sum;
                rho[s + k * norb + l * norb2 + s * norb3] = rho_sum;
                rho[k + s * norb + l * norb2 + s * norb3] = rho_sum;
                rho[l + s * norb + s * norb2 + k * norb3] = conj(rho_sum);
                rho[s + l * norb + s * norb2 + k * norb3] = conj(rho_sum);
                rho[s + l * norb + k * norb2 + s * norb3] = conj(rho_sum);
                rho[l + s * norb + k * norb2 + s * norb3] = conj(rho_sum);
            }
        }
    }

    // Rule 8: Creation on k s / Annihilation on q l
    for (k = 0; k < norb; k++)
    {
        for (s = k + 1; s < norb; s++)
        {
            for (q = 0; q < norb; q++)
            {
                if (q == s || q == k) continue;
                for (l = q + 1; l < norb; l++)
                {
                    if (l == k || l == s) continue;
                    rho_sum = 0;
#pragma omp parallel for private(i, j, h, g, occ_k, occ_q, occ_l, occ_s, chunks, strideOrb, bose_fac) reduction(+ : rho_sum)
                    for (i = 0; i < dim; i++)
                    {
                        j = i * norb;
                        if (ht[j + k] < 1 || ht[j + s] < 1) continue;
                        occ_k = ht[j + k];
                        occ_l = ht[j + l];
                        occ_q = ht[j + q];
                        occ_s = ht[j + s];

                        chunks = 0;
                        for (h = 0; h < k; h++)
                        {
                            for (g = h + 1; g < norb; g++)
                            {
                                if (ht[h + j] > 0 && ht[g + j] > 0) chunks++;
                            }
                        }
                        for (g = k + 1; g < s; g++)
                        {
                            if (ht[g + j] > 0) chunks++;
                        }
                        strideOrb = chunks * norb * norb;

                        bose_fac = sqrt(
                            (double) occ_k * occ_s * (occ_q + 1) * (occ_l + 1));
                        j = maptt[strides_tt[i] + strideOrb + q + l * norb];
                        rho_sum += conj(coef[i]) * coef[j] * bose_fac;
                    }
                    rho[k + s * norb + q * norb2 + l * norb3] = rho_sum;
                    rho[s + k * norb + q * norb2 + l * norb3] = rho_sum;
                    rho[k + s * norb + l * norb2 + q * norb3] = rho_sum;
                    rho[s + k * norb + l * norb2 + q * norb3] = rho_sum;
                } // Finish l loop
            }     // Finish q loop
        }         // Finish s loop
    }             // Finish k loop
}
