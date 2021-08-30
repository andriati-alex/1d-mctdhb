#include "configurational/hamiltonian.h"
#include "assistant/arrays_definition.h"
#include <math.h>
#include <stdlib.h>

void
apply_hamiltonian(
    MultiConfiguration multiconf,
    Carray             coef,
    Cmatrix            hob,
    Carray             hint,
    Carray             hcoef)
{
    uint16_t  h, g, k, l, s, q, norb;
    uint32_t  i, j, dim, chunks, norb2, norb3, strideOrb;
    uint16_t *ht, *occ;
    uint32_t *map, *mapot, *maptt, *stride_ot, *stride_tt;
    double    bose_fac;
    dcomplex  z, w;

    dim = multiconf->dim;
    norb = multiconf->norb;
    ht = multiconf->hash_table;
    map = multiconf->op_maps->map;
    mapot = multiconf->op_maps->mapot;
    maptt = multiconf->op_maps->maptt;
    stride_ot = multiconf->op_maps->strideot;
    stride_tt = multiconf->op_maps->stridett;

    norb2 = norb * norb;
    norb3 = norb * norb * norb;

#pragma omp parallel private( \
    i, j, k, s, q, l, h, g, chunks, strideOrb, z, w, bose_fac, occ)
    {
        occ = get_uint16_array(norb);
#pragma omp for schedule(static)
        for (i = 0; i < dim; i++)
        {
            w = 0;
            z = 0;
            for (k = 0; k < norb; k++) occ[k] = ht[k + i * norb];

            for (k = 0; k < norb; k++)
            {
                if (occ[k] < 1) continue;
                w = w + hob[k][k] * occ[k] * coef[i];
                for (l = 0; l < norb; l++)
                {
                    if (l == k) continue;
                    bose_fac = sqrt((double) occ[k] * (occ[l] + 1));
                    j = map[i + k * dim + l * norb * dim];
                    w = w + hob[k][l] * bose_fac * coef[j];
                }
            }

            // Rule 1: Creation on k k / Annihilation on k k
            for (k = 0; k < norb; k++)
            {
                bose_fac = occ[k] * (occ[k] - 1);
                z += hint[k + norb * k + norb2 * k + norb3 * k] * coef[i] *
                     bose_fac;
            }

            // Rule 2: Creation on k s / Annihilation on k s
            for (k = 0; k < norb; k++)
            {
                if (occ[k] < 1) continue;
                for (s = k + 1; s < norb; s++)
                {
                    bose_fac = occ[k] * occ[s];
                    z += 4 * hint[k + s * norb + k * norb2 + s * norb3] *
                         bose_fac * coef[i];
                    // FACTOR 4 USED IN THE LINE ABOVE COMES FROM
                    // z += Hint[s + k*M + k*M2 + s*M3] * sqrtOf * C[i];
                    // z += Hint[s + k*M + s*M2 + k*M3] * sqrtOf * C[i];
                    // z += Hint[k + s*M + s*M2 + k*M3] * sqrtOf * C[i];
                }
            }

            // Rule 3: Creation on k k / Annihilation on q q
            for (k = 0; k < norb; k++)
            {
                if (occ[k] < 2) continue;
                for (q = 0; q < norb; q++)
                {
                    if (q == k) continue;
                    bose_fac = sqrt(
                        (double) (occ[k] - 1) * occ[k] * (occ[q] + 1) *
                        (occ[q] + 2));

                    chunks = 0;
                    for (j = 0; j < k; j++)
                    {
                        if (occ[j] > 1) chunks++;
                    }
                    strideOrb = chunks * norb * norb;

                    j = mapot[stride_ot[i] + strideOrb + q + q * norb];
                    z += hint[k + k * norb + q * norb2 + q * norb3] * coef[j] *
                         bose_fac;
                }
            }

            // Rule 4: Creation on k k / Annihilation on k l
            for (k = 0; k < norb; k++)
            {
                if (occ[k] < 2) continue;
                for (l = 0; l < norb; l++)
                {
                    if (l == k) continue;
                    bose_fac =
                        (occ[k] - 1) * sqrt((double) occ[k] * (occ[l] + 1));
                    j = map[i + k * dim + l * norb * dim];
                    z += 2 * hint[k + k * norb + k * norb2 + l * norb3] *
                         coef[j] * bose_fac;
                    // FACTOR 2 IN THE LINE ABOVE COMES FROM
                    // z += Hint[k + k * M + l * M2 + k * M3] * C[j] * sqrtOf;
                }
            }

            // Rule 5: Creation on k s / Annihilation on s s
            for (k = 0; k < norb; k++)
            {
                if (occ[k] < 1) continue;
                for (s = 0; s < norb; s++)
                {
                    if (s == k || occ[s] < 1) continue;
                    bose_fac = occ[s] * sqrt((double) occ[k] * (occ[s] + 1));
                    j = map[i + k * dim + s * norb * dim];
                    z += 2 * hint[k + s * norb + s * norb2 + s * norb3] *
                         coef[j] * bose_fac;
                    // FACTOR 2 IN THE LINE ABOVE COMES FROM
                    // z += Hint[s + k * M + s * M2 + s * M3] * C[j] * sqrtOf;
                }
            }

            // Rule 6.0: Creation on k k / Annihilation on q l (k < q < l)
            for (k = 0; k < norb; k++)
            {
                if (occ[k] < 2) continue;
                for (q = k + 1; q < norb; q++)
                {
                    for (l = q + 1; l < norb; l++)
                    {
                        bose_fac = sqrt(
                            (double) occ[k] * (occ[k] - 1) * (occ[q] + 1) *
                            (occ[l] + 1));

                        chunks = 0;
                        for (j = 0; j < k; j++)
                        {
                            if (occ[j] > 1) chunks++;
                        }
                        strideOrb = chunks * norb * norb;

                        j = mapot[stride_ot[i] + strideOrb + q + l * norb];
                        z += 2 * hint[k + k * norb + q * norb2 + l * norb3] *
                             coef[j] * bose_fac;
                        // FACTOR 2 IN THE LINE ABOVE COMES FROM
                        // z += Hint[k + k*M + l*M2 + q*M3] * C[j] * sqrtOf;
                    }
                }
            }

            // Rule 6.1: Creation on k k / Annihilation on q l (q < k < l)
            for (q = 0; q < norb; q++)
            {
                for (k = q + 1; k < norb; k++)
                {
                    if (occ[k] < 2) continue;
                    for (l = k + 1; l < norb; l++)
                    {
                        bose_fac = sqrt(
                            (double) occ[k] * (occ[k] - 1) * (occ[q] + 1) *
                            (occ[l] + 1));

                        chunks = 0;
                        for (j = 0; j < k; j++)
                        {
                            if (occ[j] > 1) chunks++;
                        }
                        strideOrb = chunks * norb * norb;

                        j = mapot[stride_ot[i] + strideOrb + q + l * norb];
                        z += 2 * hint[k + k * norb + q * norb2 + l * norb3] *
                             coef[j] * bose_fac;
                        // FACTOR 2 IN THE LINE ABOVE COMES FROM
                        // z += Hint[k + k*M + l*M2 + q*M3] * C[j] * sqrtOf;
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
                        if (occ[k] < 2) continue;

                        bose_fac = sqrt(
                            (double) occ[k] * (occ[k] - 1) * (occ[q] + 1) *
                            (occ[l] + 1));

                        chunks = 0;
                        for (j = 0; j < k; j++)
                        {
                            if (occ[j] > 1) chunks++;
                        }
                        strideOrb = chunks * norb * norb;

                        j = mapot[stride_ot[i] + strideOrb + q + l * norb];
                        z += 2 * hint[k + k * norb + q * norb2 + l * norb3] *
                             coef[j] * bose_fac;
                        // FACTOR 2 IN THE LINE ABOVE COMES FROM
                        // z += Hint[k + k*M + l*M2 + q*M3] * C[j] * sqrtOf;
                    }
                }
            }

            // Rule 7.0: Creation on k s / Annihilation on q q (q > k > s)
            for (q = 0; q < norb; q++)
            {
                for (k = q + 1; k < norb; k++)
                {
                    if (occ[k] < 1) continue;
                    for (s = k + 1; s < norb; s++)
                    {
                        if (occ[s] < 1) continue;
                        bose_fac = sqrt(
                            (double) occ[k] * occ[s] * (occ[q] + 1) *
                            (occ[q] + 2));

                        chunks = 0;
                        for (h = 0; h < k; h++)
                        {
                            for (g = h + 1; g < norb; g++)
                            {
                                if (occ[h] > 0 && occ[g] > 0) chunks++;
                            }
                        }

                        for (g = k + 1; g < s; g++)
                        {
                            if (occ[g] > 0) chunks++;
                        }

                        strideOrb = chunks * norb * norb;

                        j = maptt[stride_tt[i] + strideOrb + q + q * norb];
                        z += 2 * hint[k + s * norb + q * norb2 + q * norb3] *
                             coef[j] * bose_fac;
                        // FACTOR 2 IN THE LINE ABOVE COMES FROM
                        // z += Hint[s + k*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    }
                }
            }

            // Rule 7.1: Creation on k s / Annihilation on q q (k > q > s)
            for (k = 0; k < norb; k++)
            {
                if (occ[k] < 1) continue;
                for (q = k + 1; q < norb; q++)
                {
                    for (s = q + 1; s < norb; s++)
                    {
                        if (occ[s] < 1) continue;
                        bose_fac = sqrt(
                            (double) occ[k] * occ[s] * (occ[q] + 1) *
                            (occ[q] + 2));

                        chunks = 0;
                        for (h = 0; h < k; h++)
                        {
                            for (g = h + 1; g < norb; g++)
                            {
                                if (occ[h] > 0 && occ[g] > 0) chunks++;
                            }
                        }

                        for (g = k + 1; g < s; g++)
                        {
                            if (occ[g] > 0) chunks++;
                        }

                        strideOrb = chunks * norb * norb;

                        j = maptt[stride_tt[i] + strideOrb + q + q * norb];
                        z += 2 * hint[k + s * norb + q * norb2 + q * norb3] *
                             coef[j] * bose_fac;
                        // FACTOR 2 IN THE LINE ABOVE COMES FROM
                        // z += Hint[s + k*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    }
                }
            }

            // Rule 7.2: Creation on k s / Annihilation on q q (k > s > q)
            for (k = 0; k < norb; k++)
            {
                if (occ[k] < 1) continue;
                for (s = k + 1; s < norb; s++)
                {
                    if (occ[s] < 1) continue;
                    for (q = s + 1; q < norb; q++)
                    {
                        bose_fac = sqrt(
                            (double) occ[k] * occ[s] * (occ[q] + 1) *
                            (occ[q] + 2));

                        chunks = 0;
                        for (h = 0; h < k; h++)
                        {
                            for (g = h + 1; g < norb; g++)
                            {
                                if (occ[h] > 0 && occ[g] > 0) chunks++;
                            }
                        }

                        for (g = k + 1; g < s; g++)
                        {
                            if (occ[g] > 0) chunks++;
                        }

                        strideOrb = chunks * norb * norb;

                        j = maptt[stride_tt[i] + strideOrb + q + q * norb];
                        z += 2 * hint[k + s * norb + q * norb2 + q * norb3] *
                             coef[j] * bose_fac;
                        // FACTOR 2 IN THE LINE ABOVE COMES FROM
                        // z += Hint[s + k*M + q*M2 + q*M3] * C[j] * sqrtOf;
                    }
                }
            }

            // Rule 8: Creation on k s / Annihilation on s l
            for (s = 0; s < norb; s++)
            {
                if (occ[s] < 1) continue;
                for (k = 0; k < norb; k++)
                {
                    if (occ[k] < 1 || k == s) continue;
                    for (l = 0; l < norb; l++)
                    {
                        if (l == k || l == s) continue;
                        bose_fac =
                            occ[s] * sqrt((double) occ[k] * (occ[l] + 1));

                        j = map[i + k * dim + l * norb * dim];
                        z += 4 * hint[k + s * norb + s * norb2 + l * norb3] *
                             coef[j] * bose_fac;
                        // FACTOR 4 IN THE LINE ABOVE COMES FROM
                        // z += Hint[s + k*M + s*M2 + l*M3] * C[j] * sqrtOf;
                        // z += Hint[s + k*M + l*M2 + s*M3] * C[j] * sqrtOf;
                        // z += Hint[k + s*M + l*M2 + s*M3] * C[j] * sqrtOf;
                    }
                }
            }

            // Rule 9: Creation on k s / Annihilation on q l
            for (k = 0; k < norb; k++)
            {
                if (occ[k] < 1) continue;
                for (s = k + 1; s < norb; s++)
                {
                    if (occ[s] < 1) continue;
                    for (q = 0; q < norb; q++)
                    {
                        if (q == s || q == k) continue;
                        for (l = q + 1; l < norb; l++)
                        {
                            if (l == k || l == s) continue;
                            bose_fac = sqrt(
                                (double) occ[k] * occ[s] * (occ[q] + 1) *
                                (occ[l] + 1));

                            chunks = 0;
                            for (h = 0; h < k; h++)
                            {
                                for (g = h + 1; g < norb; g++)
                                {
                                    if (occ[h] > 0 && occ[g] > 0) chunks++;
                                }
                            }
                            for (g = k + 1; g < s; g++)
                            {
                                if (occ[g] > 0) chunks++;
                            }
                            strideOrb = chunks * norb * norb;

                            j = maptt[stride_tt[i] + strideOrb + q + l * norb];
                            z += 4 *
                                 hint[k + s * norb + q * norb2 + l * norb3] *
                                 coef[j] * bose_fac;
                            // Factor 4 corresponds to s > k and l > q instead
                            // of using s != k and l != q in the loop
                        } // Finish l
                    }     // Finish q
                }         // Finish s
            }             // Finish k
            hcoef[i] = w + 0.5 * z;
        }
        free(occ);
    } // end of parallel region
}
