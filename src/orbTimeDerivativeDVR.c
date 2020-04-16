#include "orbTimeDerivativeDVR.h"

doublec INTER_CONTRACTION(int M, int k, int n, double g, Cmatrix Orb,
        Cmatrix Rinv, Carray R2)
{

/** SUM OF ALL TERMS CONTRIBUTING TO NONLINEAR PART WITHOUT PROJECTION **/

    int a,
        s,
        q,
        l,
        M2,
        M3;

    double complex
        G,
        Ginv,
        X;

    X = 0;
    M2 = M * M;
    M3 = M * M * M;

    for (a = 0; a < M; a++)
    {
        for (q = 0; q < M; q++)
        {
            // Particular case with the two last indices equals
            // to take advantage of the symmetry afterwards
            G = Rinv[k][a] * R2[a + M*a + M2*q + M3*q];
            // Sum interacting part contribution
            X = X + g * G * conj(Orb[a][n]) * Orb[q][n] * Orb[q][n];

            for (l = q + 1; l < M; l++)
            {
                G = 2 * Rinv[k][a] * R2[a + M*a + M2*q + M3*l];
                // Sum interacting part
                X = X + g * G * conj(Orb[a][n]) * Orb[l][n] * Orb[q][n];
            }
        }

        for (s = a + 1; s < M; s++)
        {

            for (q = 0; q < M; q++)
            {
                // Particular case with the two last indices equals
                // to take advantage of the symmetry afterwards

                G = Rinv[k][a] * R2[a + M*s + M2*q + M3*q];
                Ginv = Rinv[k][s] * R2[a + M*s + M2*q + M3*q];

                // Sum interacting part contribution
                X = X + g * (G*conj(Orb[s][n]) + Ginv*conj(Orb[a][n])) * \
                    Orb[q][n]*Orb[q][n];

                for (l = q + 1; l < M; l++)
                {
                    G = 2 * Rinv[k][a] * R2[a + M*s + M2*q + M3*l];
                    Ginv = 2 * Rinv[k][s] * R2[a + M*s + M2*q + M3*l];

                    // Sum interacting part
                    X = X + g * (G*conj(Orb[s][n]) + Ginv*conj(Orb[a][n])) * \
                            Orb[l][n]*Orb[q][n];
                }
            }
        }
    }

    return X;
}



void derSINEDVR(EqDataPkg MC, Cmatrix Orb, Cmatrix dOdt, Cmatrix rho1_inv,
           Carray rho2, Rarray D2DVR)
{

    int
        i,
        k,
        s,
        j,
        l,
        M,
        Mpos;

    double
        g,
        a2,
        dx;

    double complex
        sumMatMul,
        interPart,
        proj;

    Rarray
        V;

    Carray
        integ;

    Cmatrix
        Haction,
        project,
        overlap,
        overlap_inv;

    M = MC->Morb;
    Mpos = MC->Mpos;
    g = MC->g;
    V = MC->V;
    dx = MC->dx;
    a2 = MC->a2;

    integ = carrDef(Mpos);
    Haction = cmatDef(M,Mpos);
    project = cmatDef(M,Mpos);
    overlap = cmatDef(M,M);
    overlap_inv = cmatDef(M,M);

    for (k = 0; k < M; k++)
    {
        for (l = k; l < M; l++)
        {
            for (s = 0; s < Mpos; s++)
            {
                integ[s] = conj(Orb[k][s]) * Orb[l][s];
            }
            overlap[k][l] = Csimps(Mpos,integ,dx);
            overlap[l][k] = conj(overlap[k][l]);
        }
    }

    // Invert matrix and check if the operation was successfull
    s = HermitianInv(M,overlap,overlap_inv);
    if (s != 0)
    {
        printf("\n\n\nFailed on Lapack inversion routine ");
        printf("for overlap matrix !\n");
        printf("-----------------------------------");
        printf("--------------------\n\n");

        printf("Matrix given was :\n");
        cmat_print(M,M,overlap);

        if (s > 0) printf("\nSingular decomposition : %d\n\n",s);
        else       printf("\nInvalid argument given : %d\n\n",s);

        exit(EXIT_FAILURE);
    }

    // COMPUTE THE ACTION OF FULL NON-LINEAR HAMILTONIAN ACTION ON
    // EACH ORBITAL.
    #pragma omp parallel for private(k,j,i,sumMatMul,interPart) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            interPart = INTER_CONTRACTION(M,k,j,g,Orb,rho1_inv,rho2);
            sumMatMul = V[j] * Orb[k][j];
            for (i = 0; i < Mpos; i++)
            {
                sumMatMul = sumMatMul + D2DVR[j*Mpos + i] * Orb[k][i];
            }
            Haction[k][j] = interPart + sumMatMul;
        }
    }

    // APPLY PROJECTOR ON ORBITAL SPACE
    #pragma omp parallel for private(k,i,s,l,j,proj) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            proj = 0;
            for (s = 0; s < M; s++)
            {
                for (l = 0; l < M; l++)
                {
                    proj = proj + Orb[s][i] * overlap_inv[s][l] * \
                           innerL2(Mpos,Orb[l],Haction[k],dx);
                }
            }
            project[k][i] = proj;
        }
    }

    // SUBTRACT PROJECTION ON ORBITAL SPACE - ORTHOGONAL PROJECTION
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            dOdt[k][j] = - I * (Haction[k][j] - project[k][j]);
        }
    }

    // Release memory
    free(integ);
    cmatFree(M,Haction);
    cmatFree(M,project);
    cmatFree(M,overlap);
    cmatFree(M,overlap_inv);
}



void derEXPDVR(EqDataPkg MC, Cmatrix Orb, Cmatrix dOdt, Cmatrix rho1_inv,
           Carray rho2, Carray DerDVR)
{

    int
        i,
        k,
        s,
        j,
        l,
        M,
        Mpos;

    double
        g,
        a2,
        dx;

    double complex
        sumMatMul,
        interPart,
        proj;

    Rarray
        V;

    Carray
        integ;

    Cmatrix
        Haction,
        project,
        overlap,
        overlap_inv;

    M = MC->Morb;
    Mpos = MC->Mpos;
    g = MC->g;
    V = MC->V;
    dx = MC->dx;
    a2 = MC->a2;

    integ = carrDef(Mpos);
    Haction = cmatDef(M,Mpos);
    project = cmatDef(M,Mpos);
    overlap = cmatDef(M,M);
    overlap_inv = cmatDef(M,M);

    for (k = 0; k < M; k++)
    {
        for (l = k; l < M; l++)
        {
            for (s = 0; s < Mpos; s++)
            {
                integ[s] = conj(Orb[k][s]) * Orb[l][s];
            }
            overlap[k][l] = Csimps(Mpos,integ,dx);
            overlap[l][k] = conj(overlap[k][l]);
        }
    }

    // Invert matrix and check if the operation was successfull
    s = HermitianInv(M,overlap,overlap_inv);
    if (s != 0)
    {
        printf("\n\n\nFailed on Lapack inversion routine ");
        printf("for overlap matrix !\n");
        printf("-----------------------------------");
        printf("--------------------\n\n");

        printf("Matrix given was :\n");
        cmat_print(M,M,overlap);

        if (s > 0) printf("\nSingular decomposition : %d\n\n",s);
        else       printf("\nInvalid argument given : %d\n\n",s);

        exit(EXIT_FAILURE);
    }

    // COMPUTE THE ACTION OF FULL NON-LINEAR HAMILTONIAN ACTION ON
    // EACH ORBITAL.
    #pragma omp parallel for private(k,j,i,sumMatMul,interPart) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos-1; j++)
        {
            interPart = INTER_CONTRACTION(M,k,j,g,Orb,rho1_inv,rho2);
            sumMatMul = V[j] * Orb[k][j];
            for (i = 0; i < Mpos-1; i++)
            {
                sumMatMul = sumMatMul + DerDVR[j*(Mpos-1) + i] * Orb[k][i];
            }

            Haction[k][j] = interPart + sumMatMul;
        }
        Haction[k][Mpos-1] = Haction[k][0];
    }

    // APPLY PROJECTOR ON ORBITAL SPACE
    #pragma omp parallel for private(k,i,s,l,j,proj) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            proj = 0;
            for (s = 0; s < M; s++)
            {
                for (l = 0; l < M; l++)
                {
                    proj = proj + Orb[s][i] * overlap_inv[s][l] * \
                           innerL2(Mpos,Orb[l],Haction[k],dx);
                }
            }
            project[k][i] = proj;
        }
    }

    // SUBTRACT PROJECTION ON ORBITAL SPACE - ORTHOGONAL PROJECTION
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            dOdt[k][j] = - I * (Haction[k][j] - project[k][j]);
        }
    }

    // Release memory
    free(integ);
    cmatFree(M,Haction);
    cmatFree(M,project);
    cmatFree(M,overlap);
    cmatFree(M,overlap_inv);
}



void imagderEXPDVR(EqDataPkg MC, Cmatrix Orb, Cmatrix dOdt, Cmatrix rho1_inv,
           Carray rho2, Carray DerDVR)
{

    int
        i,
        k,
        s,
        j,
        M,
        Mpos;

    double
        g,
        a2,
        dx;

    double complex
        sumMatMul,
        interPart,
        proj;

    Rarray
        V;

    Carray
        integ;

    Cmatrix
        Haction;

    M = MC->Morb;
    Mpos = MC->Mpos;
    g = MC->g;
    V = MC->V;
    dx = MC->dx;
    a2 = MC->a2;

    integ = carrDef(Mpos);
    Haction = cmatDef(M,Mpos);

    // COMPUTE THE ACTION OF FULL NON-LINEAR HAMILTONIAN ACTION ON
    // EACH ORBITAL.
    #pragma omp parallel for private(k,j,i,sumMatMul,interPart) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos-1; j++)
        {
            interPart = INTER_CONTRACTION(M,k,j,g,Orb,rho1_inv,rho2);
            sumMatMul = V[j] * Orb[k][j];
            for (i = 0; i < Mpos-1; i++)
            {
                sumMatMul = sumMatMul + DerDVR[j*(Mpos-1) + i] * Orb[k][i];
            }

            Haction[k][j] = interPart + sumMatMul;
        }
        Haction[k][Mpos-1] = Haction[k][0];
    }

    // APPLY PROJECTOR ON ORBITAL SPACE
    #pragma omp parallel for private(k,i,s,proj) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            proj = 0;
            for (s = 0; s < M; s++)
            {
                proj = proj + Orb[s][i] * innerL2(Mpos,Orb[s],Haction[k],dx);
            }
            // SUBTRACT PROJECTION ON ORBITAL SPACE - ORTHOGONAL PROJECTION
            dOdt[k][i] = - I * (Haction[k][i] - proj);
        }
    }

    // Release memory
    free(integ);
    cmatFree(M,Haction);
}
