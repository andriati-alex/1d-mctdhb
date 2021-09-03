#include "mctdhb_types.h"

dcomplex
orb_interacting_part(int k, int n, double g, ManyBodyState psi)
{

    /** SUM OF ALL TERMS CONTRIBUTING TO NONLINEAR PART WITHOUT PROJECTION **/

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

    /** Same stuff from function above without  inverted  overlap matrix for
        imaginary time integration, because the re-orthogonalization is done
        each time by hand. Must do the same  thing  the  function above with
        the overlap matrix considered as the unit                        **/

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

void
derSINEDVR(
    EqDataPkg MC,
    Cmatrix   Orb,
    Cmatrix   dOdt,
    Cmatrix   rho1_inv,
    Carray    rho2,
    Rarray    D2DVR,
    int       impOrtho)
{

    int i, k, s, j, l, M, Mpos;

    double g, a2, dx;

    double complex sumMatMul, interPart, proj;

    Rarray V;

    Carray integ;

    Cmatrix Haction, project, overlap, overlap_inv;

    M = MC->Morb;
    Mpos = MC->Mpos;
    g = MC->g;
    V = MC->V;
    dx = MC->dx;
    a2 = MC->a2;

    integ = carrDef(Mpos);
    Haction = cmatDef(M, Mpos);
    project = cmatDef(M, Mpos);
    overlap = cmatDef(M, M);
    overlap_inv = cmatDef(M, M);

    if (impOrtho)
    {
        for (k = 0; k < M; k++)
        {
            for (l = k; l < M; l++)
            {
                for (s = 0; s < Mpos; s++)
                {
                    integ[s] = conj(Orb[k][s]) * Orb[l][s];
                }
                overlap[k][l] = Csimps(Mpos, integ, dx);
                overlap[l][k] = conj(overlap[k][l]);
            }
        }

        // Invert matrix and check if the operation was successfull
        s = HermitianInv(M, overlap, overlap_inv);
        if (s != 0)
        {
            printf("\n\n\nFailed on Lapack inversion routine ");
            printf("for overlap matrix !\n");
            printf("-----------------------------------");
            printf("--------------------\n\n");

            printf("Matrix given was :\n");
            cmat_print(M, M, overlap);

            if (s > 0)
                printf("\nSingular decomposition : %d\n\n", s);
            else
                printf("\nInvalid argument given : %d\n\n", s);

            exit(EXIT_FAILURE);
        }
    }

// COMPUTE THE ACTION OF FULL NON-LINEAR HAMILTONIAN ACTION ON
// EACH ORBITAL.
#pragma omp parallel for private(k, j, i, sumMatMul, interPart) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            interPart = INTER_CONTRACTION(M, k, j, g, Orb, rho1_inv, rho2);
            sumMatMul = V[j] * Orb[k][j];
            for (i = 0; i < Mpos; i++)
            {
                sumMatMul = sumMatMul + D2DVR[j * Mpos + i] * Orb[k][i];
            }
            Haction[k][j] = interPart + sumMatMul;
        }
    }

    // APPLY PROJECTOR ON ORBITAL SPACE
    if (impOrtho)
    {
#pragma omp parallel for private(k, i, s, l, j, proj) schedule(static)
        for (k = 0; k < M; k++)
        {
            for (i = 0; i < Mpos; i++)
            {
                proj = 0;
                for (s = 0; s < M; s++)
                {
                    for (l = 0; l < M; l++)
                    {
                        proj = proj + Orb[s][i] * overlap_inv[s][l] *
                                          innerL2(Mpos, Orb[l], Haction[k], dx);
                    }
                }
                project[k][i] = proj;
            }
        }
    } else
    {
#pragma omp parallel for private(k, i, s, l, j, proj) schedule(static)
        for (k = 0; k < M; k++)
        {
            for (i = 0; i < Mpos; i++)
            {
                proj = 0;
                for (s = 0; s < M; s++)
                {
                    proj = proj +
                           Orb[s][i] * innerL2(Mpos, Orb[s], Haction[k], dx);
                }
                project[k][i] = proj;
            }
        }
    }

    // SUBTRACT PROJECTION ON ORBITAL SPACE - ORTHOGONAL PROJECTION
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            dOdt[k][j] = -I * (Haction[k][j] - project[k][j]);
        }
    }

    // Release memory
    free(integ);
    cmatFree(M, Haction);
    cmatFree(M, project);
    cmatFree(M, overlap);
    cmatFree(M, overlap_inv);
}

void
derEXPDVR(
    EqDataPkg MC,
    Cmatrix   Orb,
    Cmatrix   dOdt,
    Cmatrix   rho1_inv,
    Carray    rho2,
    Carray    DerDVR,
    int       impOrtho)
{

    int i, k, s, j, l, M, Mpos;

    double g, a2, dx;

    double complex sumMatMul, interPart, proj;

    Rarray V;

    Carray integ;

    Cmatrix Haction, project, overlap, overlap_inv;

    M = MC->Morb;
    Mpos = MC->Mpos;
    g = MC->g;
    V = MC->V;
    dx = MC->dx;
    a2 = MC->a2;

    integ = carrDef(Mpos);
    Haction = cmatDef(M, Mpos);
    project = cmatDef(M, Mpos);
    overlap = cmatDef(M, M);
    overlap_inv = cmatDef(M, M);

    if (impOrtho)
    {
        for (k = 0; k < M; k++)
        {
            for (l = k; l < M; l++)
            {
                for (s = 0; s < Mpos; s++)
                {
                    integ[s] = conj(Orb[k][s]) * Orb[l][s];
                }
                overlap[k][l] = Csimps(Mpos, integ, dx);
                overlap[l][k] = conj(overlap[k][l]);
            }
        }

        // Invert matrix and check if the operation was successfull
        s = HermitianInv(M, overlap, overlap_inv);
        if (s != 0)
        {
            printf("\n\n\nFailed on Lapack inversion routine ");
            printf("for overlap matrix !\n");
            printf("-----------------------------------");
            printf("--------------------\n\n");

            printf("Matrix given was :\n");
            cmat_print(M, M, overlap);

            if (s > 0)
                printf("\nSingular decomposition : %d\n\n", s);
            else
                printf("\nInvalid argument given : %d\n\n", s);

            exit(EXIT_FAILURE);
        }
    }

// COMPUTE THE ACTION OF FULL NON-LINEAR HAMILTONIAN ACTION ON
// EACH ORBITAL.
#pragma omp parallel for private(k, j, i, sumMatMul, interPart) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos - 1; j++)
        {
            interPart = INTER_CONTRACTION(M, k, j, g, Orb, rho1_inv, rho2);
            sumMatMul = V[j] * Orb[k][j];
            for (i = 0; i < Mpos - 1; i++)
            {
                sumMatMul = sumMatMul + DerDVR[j * (Mpos - 1) + i] * Orb[k][i];
            }

            Haction[k][j] = interPart + sumMatMul;
        }
        Haction[k][Mpos - 1] = Haction[k][0];
    }

    // APPLY PROJECTOR ON ORBITAL SPACE
    if (impOrtho)
    {
#pragma omp parallel for private(k, i, s, l, j, proj) schedule(static)
        for (k = 0; k < M; k++)
        {
            for (i = 0; i < Mpos; i++)
            {
                proj = 0;
                for (s = 0; s < M; s++)
                {
                    for (l = 0; l < M; l++)
                    {
                        proj = proj + Orb[s][i] * overlap_inv[s][l] *
                                          innerL2(Mpos, Orb[l], Haction[k], dx);
                    }
                }
                project[k][i] = proj;
            }
        }
    } else
    {
#pragma omp parallel for private(k, i, s, l, j, proj) schedule(static)
        for (k = 0; k < M; k++)
        {
            for (i = 0; i < Mpos; i++)
            {
                proj = 0;
                for (s = 0; s < M; s++)
                {
                    proj = proj +
                           Orb[s][i] * innerL2(Mpos, Orb[s], Haction[k], dx);
                }
                project[k][i] = proj;
            }
        }
    }

    // SUBTRACT PROJECTION ON ORBITAL SPACE - ORTHOGONAL PROJECTION
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            dOdt[k][j] = -I * (Haction[k][j] - project[k][j]);
        }
    }

    // Release memory
    free(integ);
    cmatFree(M, Haction);
    cmatFree(M, project);
    cmatFree(M, overlap);
    cmatFree(M, overlap_inv);
}

void
imagderEXPDVR(
    EqDataPkg MC,
    Cmatrix   Orb,
    Cmatrix   dOdt,
    Cmatrix   rho1_inv,
    Carray    rho2,
    Carray    DerDVR)
{

    int i, k, s, j, M, Mpos;

    double g, a2, dx;

    double complex sumMatMul, interPart, proj;

    Rarray V;

    Carray integ;

    Cmatrix Haction;

    M = MC->Morb;
    Mpos = MC->Mpos;
    g = MC->g;
    V = MC->V;
    dx = MC->dx;
    a2 = MC->a2;

    integ = carrDef(Mpos);
    Haction = cmatDef(M, Mpos);

// COMPUTE THE ACTION OF FULL NON-LINEAR HAMILTONIAN ACTION ON
// EACH ORBITAL.
#pragma omp parallel for private(k, j, i, sumMatMul, interPart) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos - 1; j++)
        {
            interPart = INTER_CONTRACTION(M, k, j, g, Orb, rho1_inv, rho2);
            sumMatMul = V[j] * Orb[k][j];
            for (i = 0; i < Mpos - 1; i++)
            {
                sumMatMul = sumMatMul + DerDVR[j * (Mpos - 1) + i] * Orb[k][i];
            }

            Haction[k][j] = interPart + sumMatMul;
        }
        Haction[k][Mpos - 1] = Haction[k][0];
    }

// APPLY PROJECTOR ON ORBITAL SPACE
#pragma omp parallel for private(k, i, s, proj) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            proj = 0;
            for (s = 0; s < M; s++)
            {
                proj = proj + Orb[s][i] * innerL2(Mpos, Orb[s], Haction[k], dx);
            }
            // SUBTRACT PROJECTION ON ORBITAL SPACE - ORTHOGONAL PROJECTION
            dOdt[k][i] = -I * (Haction[k][i] - proj);
        }
    }

    // Release memory
    free(integ);
    cmatFree(M, Haction);
}
