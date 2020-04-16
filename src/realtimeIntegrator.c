#include "realtimeIntegrator.h"

/*************************************************************************
 *************************************************************************
 ************                                                 ************
 ************      ORBITAL NONLINEAR PART FOR SPLIT-STEP      ************
 ************                                                 ************
 *************************************************************************
 *************************************************************************/

doublec nonlinearOrtho (int M, int k, int n, double g, Cmatrix Orb,
        Cmatrix Rinv, Carray R2, Cmatrix Ho, Carray Hint, Cmatrix Ortho )
{

/** NONLINEAR PART OF SPLIT-STEP - ORBITAL'S EQUATION
    =================================================
    For a orbital 'k' computed at grid point 'n' calculate the
    right-hand-side nonlinear part of orbital's equation.  The
    nonlinearity comes from projections that make the  eq.  an
    integral-differential equation, and other part due to contact
    interactions. Rinv is the one-body density matrix inverted
    and R2 is the two-body  density  matrix  computed from the
    coefficients at the same time step of orbitals. Ho and Hint
    must be set up with the 'Orb' input parameter.

    ADDITIONAL RE-ORTHOGONALIZATION FROM PROJECTIONS is done here
    due to finite precision  avoiding the orbitals to  lose  norm
    specifically for real-time propagation. See the reference :

    "The  multiconfigurational  time-dependent  Hartree  (MCTDH)  method:
    a highly efficient algorithm for propagating wavepackets", M.H. Beck,
    A. Jackle,  G.A. Worth,  Hans-Dieter Meyer,  Physics Reports 324, pp.
    1-105, 2000. DOI 10.1016/S0370-1573(99)00047-2, section 4.8       **/

    int
        a,
        b,
        j,
        s,
        q,
        l,
        M2,
        M3,
        ind;

    double complex
        G,
        Ginv,
        X;

    X = 0;
    M2 = M * M;
    M3 = M * M * M;



    for (a = 0; a < M; a++)
    {
        // Subtract one-body projection
        for (b = 0; b < M; b++)
        {
            X = X - Ortho[a][b] * Ho[b][k] * Orb[a][n];
        }

        for (q = 0; q < M; q++)
        {
            // Particular case with the two last indices equals
            // to take advantage of the symmetry afterwards

            G = Rinv[k][a] * R2[a + M*a + M2*q + M3*q];

            // Sum interacting part contribution
            X = X + g * G * conj(Orb[a][n]) * Orb[q][n] * Orb[q][n];

            // Subtract interacting projection
            for (b = 0; b < M; b++)
            {
                for (j = 0; j < M; j++)
                {
                    ind = b + a * M + q * M2 + q * M3;
                    X = X - G * Ortho[j][b] * Orb[j][n] * Hint[ind];
                }
            }

            for (l = q + 1; l < M; l++)
            {
                G = 2 * Rinv[k][a] * R2[a + M*a + M2*q + M3*l];

                // Sum interacting part
                X = X + g * G * conj(Orb[a][n]) * Orb[l][n] * Orb[q][n];

                // Subtract interacting projection
                for (b = 0; b < M; b++)
                {
                    for (j = 0; j < M; j++)
                    {
                        ind = b + a * M + l * M2 + q * M3;
                        X = X - G * Ortho[j][b] * Orb[j][n] * Hint[ind];
                    }
                }
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

                // Subtract interacting projection
                for (b = 0; b < M; b++)
                {
                    for (j = 0; j < M; j++)
                    {
                        ind = b + s * M + q * M2 + q * M3;
                        X = X - G * Ortho[j][b] * Orb[j][n] * Hint[ind];
                        ind = b + a * M + q * M2 + q * M3;
                        X = X - Ginv * Ortho[j][b] * Orb[j][n] * Hint[ind];
                    }
                }

                for (l = q + 1; l < M; l++)
                {
                    G = 2 * Rinv[k][a] * R2[a + M*s + M2*q + M3*l];
                    Ginv = 2 * Rinv[k][s] * R2[a + M*s + M2*q + M3*l];

                    // Sum interacting part
                    X = X + g * (G*conj(Orb[s][n]) + Ginv*conj(Orb[a][n])) * \
                            Orb[l][n]*Orb[q][n];

                    // Subtract interacting projection
                    for (b = 0; b < M; b++)
                    {
                        for (j = 0; j < M; j++)
                        {
                            ind = b + s * M + l * M2 + q * M3;
                            X = X - G * Ortho[j][b] * Orb[j][n] * Hint[ind];
                            ind = b + a * M + l * M2 + q * M3;
                            X = X - Ginv * Ortho[j][b] * Orb[j][n] * Hint[ind];
                        }
                    }
                }
            }
        }
    }

    return X;
}



void realNL_dOdt (EqDataPkg MC, Cmatrix Orb, Cmatrix dOdt, Cmatrix Ho,
        Carray Hint, Cmatrix rho1, Carray rho2)
{

/** Time derivative for nonlinear part in split-step with additional
    inversion of overlap matrix on projector for real time.  Despite
    the introduction of overlap matrix it is similar to the routines
    presented above **/

    int
        k,
        s,
        j,
        l,
        M = MC->Morb,
        Mpos  = MC->Mpos;



    double
        dx = MC->dx;



    double complex
        Proj,
        g = MC->g;



    Carray
        integ = carrDef(Mpos);



    Cmatrix
        overlap = cmatDef(M,M),
        overlap_inv = cmatDef(M,M),
        rho_inv = cmatDef(M,M);



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



    // Invert matrix and check if the operation was successfull
    s = HermitianInv(M, rho1, rho_inv);
    if (s != 0)
    {
        printf("\n\n\nFailed on Lapack inversion routine!\n");
        printf("-----------------------------------\n\n");

        printf("Matrix given was :\n");
        cmat_print(M, M, rho1);

        if (s > 0) printf("\nSingular decomposition : %d\n\n", s);
        else       printf("\nInvalid argument given : %d\n\n", s);

        exit(EXIT_FAILURE);
    }



    // Update k-th orbital at discretized position j
    #pragma omp parallel for private(k, j) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
            dOdt[k][j] = - I * \
            nonlinearOrtho(M,k,j,g,Orb,rho_inv,rho2,Ho,Hint,overlap_inv);
    }



    // Release memory
    cmatFree(M,rho_inv);
    cmatFree(M,overlap);
    cmatFree(M,overlap_inv);
    free(integ);
}



void realNLTRAP_dOdt (EqDataPkg MC, Cmatrix Orb, Cmatrix dOdt, Cmatrix Ho,
        Carray Hint, Cmatrix rho1, Carray rho2)
{

/** Time derivative for nonlinear part in split-step with  additional
    inversion of overlap matrix on projector for  real time.  Include
    1-body potential to separate derivatives to apply spectral method **/

    int
        k,
        s,
        j,
        l,
        M = MC->Morb,
        Mpos  = MC->Mpos;



    double
        dx = MC->dx,
        * V = MC->V;



    double complex
        Proj,
        g = MC->g;



    Carray
        integ = carrDef(Mpos);



    Cmatrix
        overlap = cmatDef(M,M),
        overlap_inv = cmatDef(M,M),
        rho_inv = cmatDef(M,M);



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



    // Invert matrix and check if the operation was successfull
    s = HermitianInv(M, rho1, rho_inv);
    if (s != 0)
    {
        printf("\n\n\nFailed on Lapack inversion routine!\n");
        printf("-----------------------------------\n\n");

        printf("Matrix given was :\n");
        cmat_print(M, M, rho1);

        if (s > 0) printf("\nSingular decomposition : %d\n\n", s);
        else       printf("\nInvalid argument given : %d\n\n", s);

        exit(EXIT_FAILURE);
    }



    // Update k-th orbital at discretized position j
    #pragma omp parallel for private(k, j) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
            dOdt[k][j] = - I * (V[j] * Orb[k][j] + \
            nonlinearOrtho(M,k,j,g,Orb,rho_inv,rho2,Ho,Hint,overlap_inv));
    }

    cmatFree(M,rho_inv);
    cmatFree(M,overlap);
    cmatFree(M,overlap_inv);
    free(integ);
}



void realNL_RK4(EqDataPkg MC, ManyBodyPkg S, double dt)
{

    int
        k,
        j,
        Morb,
        Mpos;

    double
        g,
        a2,
        dx,
        * V;

    double complex
        a1;

    Cmatrix
        Ok,
        dOdt,
        Oarg;

    Morb = MC->Morb;
    Mpos = MC->Mpos;
    a2 = MC->a2;
    a1 = MC->a1;
    g = MC->g;
    V = MC->V;
    dx = MC->dx;

    Ok = cmatDef(Morb, Mpos);
    dOdt = cmatDef(Morb, Mpos);
    Oarg = cmatDef(Morb, Mpos);



    // COMPUTE K1
    realNL_dOdt(MC,S->Omat,dOdt,S->Ho,S->Hint,S->rho1,S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K1 contribution
            Ok[k][j] = dOdt[k][j];
            // Prepare next argument to compute K2
            Oarg[k][j] = S->Omat[k][j] + dOdt[k][j] * 0.5 * dt;
        }
    }
    SetupHo(Morb,Mpos,Oarg,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,Oarg,dx,g,S->Hint);



    // COMPUTE K2
    realNL_dOdt(MC,Oarg,dOdt,S->Ho,S->Hint,S->rho1,S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K2 contribution
            Ok[k][j] += 2 * dOdt[k][j];
            // Prepare next argument to compute K3
            Oarg[k][j] = S->Omat[k][j] + dOdt[k][j] * 0.5 * dt;
        }
    }
    SetupHo(Morb,Mpos,Oarg,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,Oarg,dx,g,S->Hint);



    // COMPUTE K3
    realNL_dOdt(MC, Oarg, dOdt, S->Ho, S->Hint, S->rho1, S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K3 contribution
            Ok[k][j] += 2 * dOdt[k][j];
            // Prepare next argument to compute K4
            Oarg[k][j] = S->Omat[k][j] + dOdt[k][j] * dt;
        }
    }
    SetupHo(Morb,Mpos,Oarg,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,Oarg,dx,g,S->Hint);



    // COMPUTE K4
    realNL_dOdt(MC, Oarg, dOdt, S->Ho, S->Hint, S->rho1, S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K4 contribution
            Ok[k][j] += dOdt[k][j];
        }
    }



    // Until now Ok holds the sum K1 + 2 * K2 + 2 * K3 + K4
    // from the Fourth order Runge-Kutta algorithm, update:
    for (k = 0; k < Morb; k++)
    {   // Update Orbitals
        for (j = 0; j < Mpos; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt / 6;
        }
    }

    cmatFree(Morb, dOdt);
    cmatFree(Morb, Ok);
    cmatFree(Morb, Oarg);
}



void realNLTRAP_RK4(EqDataPkg MC, ManyBodyPkg S, double dt)
{

    int
        k,
        j,
        Morb,
        Mpos;

    double
        g,
        a2,
        dx,
        * V;

    double complex
        a1;

    Cmatrix
        Ok,
        dOdt,
        Oarg;

    Morb = MC->Morb;
    Mpos = MC->Mpos;
    a2 = MC->a2;
    a1 = MC->a1;
    g = MC->g;
    V = MC->V;
    dx = MC->dx;

    Ok = cmatDef(Morb, Mpos);
    dOdt = cmatDef(Morb, Mpos);
    Oarg = cmatDef(Morb, Mpos);



    // COMPUTE K1
    realNLTRAP_dOdt(MC,S->Omat,dOdt,S->Ho,S->Hint,S->rho1,S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K1 contribution
            Ok[k][j] = dOdt[k][j];
            // Prepare next argument to compute K2
            Oarg[k][j] = S->Omat[k][j] + dOdt[k][j] * 0.5 * dt;
        }
    }
    SetupHo(Morb,Mpos,Oarg,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,Oarg,dx,g,S->Hint);



    // COMPUTE K2
    realNLTRAP_dOdt(MC,Oarg,dOdt,S->Ho,S->Hint,S->rho1,S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K2 contribution
            Ok[k][j] += 2 * dOdt[k][j];
            // Prepare next argument to compute K3
            Oarg[k][j] = S->Omat[k][j] + dOdt[k][j] * 0.5 * dt;
        }
    }
    SetupHo(Morb,Mpos,Oarg,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,Oarg,dx,g,S->Hint);



    // COMPUTE K3
    realNLTRAP_dOdt(MC,Oarg,dOdt,S->Ho,S->Hint,S->rho1,S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K3 contribution
            Ok[k][j] += 2 * dOdt[k][j];
            // Prepare next argument to compute K4
            Oarg[k][j] = S->Omat[k][j] + dOdt[k][j] * dt;
        }
    }
    SetupHo(Morb,Mpos,Oarg,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,Oarg,dx,g,S->Hint);



    // COMPUTE K4
    realNLTRAP_dOdt(MC,Oarg,dOdt,S->Ho,S->Hint,S->rho1,S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K4 contribution
            Ok[k][j] += dOdt[k][j];
        }
    }



    // Until now Ok holds the sum K1 + 2 * K2 + 2 * K3 + K4
    // from the Fourth order Runge-Kutta algorithm, update:
    for (k = 0; k < Morb; k++)
    {   // Update Orbitals
        for (j = 0; j < Mpos; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt / 6;
        }
    }

    cmatFree(Morb, dOdt);
    cmatFree(Morb, Ok);
    cmatFree(Morb, Oarg);
}










/***********************************************************************
 ***********************************************************************
 ******************                                    *****************
 ******************       SPLIT-STEP INTEGRATORS       *****************
 ******************                                    *****************
 ***********************************************************************
 ***********************************************************************/

void realSSFD(EqDataPkg MC, ManyBodyPkg S, double dt, int Nsteps,
     char prefix [], int recInterval)
{

    int l,
        i,
        j,
        k,
        nc,
        Npar,
        Mpos,
        Morb,
        Mrec,
        cyclic,
        isTrapped;

    double
        checkOrbNorm,
        checkOverlap,
        norm,
        R2,
        XI,
        XF,
        dx,
        a2,
        g;

    double complex
        E,
        a1;

    char
        fname[100];

    FILE
        * t_file,
        * rho_file,
        * orb_file;

    Rarray
        V,
        occ;

    Carray
        rho_vec,
        upper,
        lower,
        mid;



    // FROM 1-BODY POTENTIAL DECIDE THE TYPE OF BOUNDARIES
    isTrapped = 0;
    if (strcmp(MC->Vname,"harmonic") == 0)      isTrapped = 1;
    if (strcmp(MC->Vname,"doublewell") == 0)    isTrapped = 1;
    if (strcmp(MC->Vname,"harmonicgauss") == 0) isTrapped = 1;

    // record interval valid values are > 0
    if (recInterval < 1) recInterval = 1;

    // unpack configurational parameters
    nc = MC->nc;
    Npar = MC->Npar;
    Mpos = MC->Mpos;
    Morb = MC->Morb;
    // unpack equation parameters
    dx = MC->dx;
    a2 = MC->a2;
    a1 = MC->a1;
    g = MC->g;
    V = MC->V;

    // natural occupation numbers
    occ = rarrDef(Morb);
    // 1-body density matrix in row-major format
    rho_vec = carrDef(Morb * Morb);
    // Crank-Nicolson tridiagonal system
    upper = carrDef(Mpos - 1);
    lower = carrDef(Mpos - 1);
    mid = carrDef(Mpos);



    // Configure the linear system from Crank-Nicolson scheme with half
    // time step from split-step approach
    setupTriDiagonal(MC,upper,lower,mid,dt/2,isTrapped);



    // RECORD DOMAIN EXTENSION
    if (isTrapped)
    {
        // Define a number of points to the left and to the right in
        // order to do not interact with the boundariers  throughout
        // all time evolved. This is given by 'k'.
        k = Mpos;
        Mrec = Mpos + 2 * k;
        XI = MC->xi - k*dx;
        XF = MC->xf + k*dx;
    }
    else
    {
        Mrec = Mpos;
        XI = MC->xi;
        XF = MC->xf;
    }



    // OPEN FILE TO RECORD 1-BODY DENSITY MATRIX
    strcpy(fname,"output/");
    strcat(fname,prefix);
    strcat(fname,"_domain_realtime.dat");
    t_file = fopen(fname, "w");
    if (t_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(t_file, "# time   XI   XF   Ngrid   fig_xi   fig_xf");



    // OPEN FILE TO RECORD 1-BODY DENSITY MATRIX
    strcpy(fname, "output/");
    strcat(fname, prefix);
    strcat(fname, "_rho_realtime.dat");
    rho_file = fopen(fname, "w");
    if (rho_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(rho_file, "# Row-major vector representatio of rho_1\n");



    // OPEN FILE TO RECORD ORBITALS
    strcpy(fname, "output/");
    strcat(fname, prefix);
    strcat(fname, "_orb_realtime.dat");
    orb_file = fopen(fname, "w");
    if (orb_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(orb_file, "# Row-major vector representatio of Orbitals\n");





    // Setup one/two-body hamiltonian matrix elements
    SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

    // Setup one/two-body density matrix
    OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);



    // Initial observables
    E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
    R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
    hermitianEigvalues(Morb,S->rho1,occ);
    norm = carrMod(nc, S->C);
    checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
    checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

    printf("\n  time         E/Npar      Overlap");
    printf("    Coef-Norm    O-Avg-Norm   sqrt<R^2>   n0");
    sepline();
    printf("%10.6lf  %11.6lf",0.0,creal(E));
    printf("    %7.1E    %9.7lf",checkOverlap,norm);
    printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
    printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

    // Record initial data
    RowMajor(Morb, Morb, S->rho1, rho_vec);
    carr_inline(rho_file, Morb * Morb, rho_vec);
    recorb_inline(orb_file,Morb,Mrec,Mpos,S->Omat);
    fprintf(t_file,"\n%.6lf %.10lf %.10lf %d %.10lf %.10lf",
            0*dt,XI,XF,Mrec,MC->xi,MC->xf);



    // Evolve in fixed basis while the least occupied state is not
    // above a given tolerance
    l = 1;
    j = 0;
    if (((double)occ[0])/occ[Morb-1] < 3E-4)
    {
        printf("\n**    Fixed orbital evolution    **");
    }
    while (((double)occ[0])/occ[Morb-1] < 3E-4)
    {
        LanczosIntegrator(5,MC,S->Ho,S->Hint,dt,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        hermitianEigvalues(Morb,S->rho1,occ);

        // If the number of steps done reach record interval = record data
        // compute some observables to print on screen and record solution
        if (l == recInterval)
        {
            TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,
                  MC->strideTT,MC->IF,S->C,S->rho2);
            E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
            R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
            hermitianEigvalues(Morb,S->rho1,occ);
            norm = carrMod(nc,S->C);
            checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
            checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

            printf("\n%10.6lf  %11.6lf",(j+1)*dt,creal(E));
            printf("    %7.1E    %9.7lf",checkOverlap,norm);
            printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
            printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

            // record 1-body density matrix
            RowMajor(Morb,Morb,S->rho1,rho_vec);
            carr_inline(rho_file,Morb*Morb,rho_vec);
            // record orbitals
            recorb_inline(orb_file,Morb,Mrec,Mpos,S->Omat);
            fprintf(t_file,"\n%.6lf %.10lf %.10lf %d %.10lf %.10lf",
                    (j+1)*dt,XI,XF,Mrec,MC->xi,MC->xf);
            l = 1;
        }
        else { l = l + 1; }

        j = j + 1;
    }
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);
    if (j > 0) printf("\n**    Variational orbital evolution    **");



    // Evolve with variational orbitals due to occupation raising in
    // unoccupied states
    for (i = j; i < Nsteps; i++)
    {

        if (j > 0 && i == j + 100)
        {
            Ortonormalize(Morb,Mpos,dx,S->Omat);
            SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
            SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);
        }

        // HALF STEP THE COEFFICIENTS
        LanczosIntegrator(3,MC,S->Ho,S->Hint,dt/2,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);
        RegularizeMat(Morb,Npar*5E-6,S->rho1);



        // HALF TIME STEP LINEAR PART
        linearCN(MC,upper,lower,mid,S->Omat,isTrapped,dt/2);
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        // FULL TIME STEP NONLINEAR PART
        realNL_RK4(MC,S,dt);
        // ANOTHER HALF TIME STEP LINEAR PART
        linearCN(MC,upper,lower,mid,S->Omat,isTrapped,dt/2);
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);



        // ANOTHER HALF STEP FOR COEFFICIENTS
        LanczosIntegrator(3,MC,S->Ho,S->Hint,dt/2,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);
        RegularizeMat(Morb,Npar*5E-6,S->rho1);



        // Check if the boundaries are still good for trapped systems
        // since the boundaries shall not affect the results
        if ( (i + 1) % 50 == 0 && isTrapped)
        {
            extentDomain(MC,S);

            Mpos = MC->Mpos;
            V = MC->V;

            SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
            SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);

            // Reconfigure Finite differences
            free(upper);
            free(lower);
            free(mid);
            upper = carrDef(Mpos - 1);
            lower = carrDef(Mpos - 1);
            mid = carrDef(Mpos);
            // re-onfigure again the Crank-Nicolson Finite-difference matrix
            setupTriDiagonal(MC,upper,lower,mid,dt/2,isTrapped);

        }



        // Check for consistency in orbitals orthogonality
        if (checkOverlap > 1E-4)
        {
            printf("\n\nERROR : Critical loss of orthogonality ");
            printf("among orbitals. Exiting ...\n\n");
            fclose(t_file);
            fclose(rho_file);
            fclose(orb_file);
            exit(EXIT_FAILURE);
        }

        // If the number of steps done reach record interval = record data
        // compute some observables to print on screen and record solution
        if (l == recInterval)
        {
            E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
            R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
            hermitianEigvalues(Morb,S->rho1,occ);
            norm = carrMod(nc,S->C);
            checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
            checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

            printf("\n%10.6lf  %11.6lf",(i+1)*dt,creal(E));
            printf("    %7.1E    %9.7lf",checkOverlap,norm);
            printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
            printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

            // record 1-body density matrix
            RowMajor(Morb,Morb,S->rho1,rho_vec);
            carr_inline(rho_file,Morb*Morb,rho_vec);
            // record orbitals
            recorb_inline(orb_file,Morb,Mrec,Mpos,S->Omat);
            fprintf(t_file,"\n%.6lf %.10lf %.10lf %d %.10lf %.10lf",
                    (i+1)*dt,XI,XF,Mrec,MC->xi,MC->xf);
            l = 1;
        }
        else { l = l + 1; }

    }

    E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
    R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
    hermitianEigvalues(Morb,S->rho1,occ);
    norm = carrMod(nc,S->C);
    checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
    checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

    printf("\n%10.6lf  %11.6lf",Nsteps*dt,creal(E));
    printf("    %7.1E    %9.7lf",checkOverlap,norm);
    printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
    printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

    sepline();

    free(upper);
    free(lower);
    free(mid);
    free(occ);
    free(rho_vec);

    fclose(t_file);
    fclose(rho_file);
    fclose(orb_file);
}



void realSSFFT(EqDataPkg MC, ManyBodyPkg S, double dt, int Nsteps,
     char prefix [], int recInterval)
{

    int l,
        i,
        j,
        m,
        k,
        nc,
        Npar,
        Mpos,
        Morb,
        Mrec,
        isTrapped;

    MKL_LONG
        p;

    double
        checkOrbNorm,
        checkOverlap,
        norm,
        freq,
        XI,
        XF,
        dx,
        a2,
        R2,
        g;

    double complex
        E,
        a1;

    char
        fname[100];

    FILE
        * t_file,
        * rho_file,
        * orb_file;

    Rarray
        V,
        occ;

    Carray
        rho_vec,
        exp_der;

    DFTI_DESCRIPTOR_HANDLE
        desc;



    isTrapped = 0;
    if (strcmp(MC->Vname,"harmonic") == 0)      isTrapped = 1;
    if (strcmp(MC->Vname,"doublewell") == 0)    isTrapped = 1;
    if (strcmp(MC->Vname,"harmonicgauss") == 0) isTrapped = 1;

    // record interval valid values are > 0
    if (recInterval < 1) recInterval = 1;

    // unpack configurational parameters
    nc = MC->nc;
    Npar = MC->Npar;
    Mpos = MC->Mpos;
    Morb = MC->Morb;
    // unpack equation parameters
    dx = MC->dx;
    a2 = MC->a2;
    a1 = MC->a1;
    g = MC->g;
    V = MC->V;

    // RECORD DOMAIN EXTENSION
    if (isTrapped)
    {
        // Define a number of points to the left and to the right in
        // order to do not interact with the boundariers  throughout
        // all time evolved. This is given by 'k'.
        k = Mpos;
        Mrec = Mpos + 2 * k;
        XI = MC->xi - k*dx;
        XF = MC->xf + k*dx;
    }
    else
    {
        Mrec = Mpos;
        XI = MC->xi;
        XF = MC->xf;
    }

    // natural occupations
    occ = rarrDef(Morb);

    // matrices in row-major form to record data
    rho_vec = carrDef(Morb * Morb);



    // OPEN FILE TO RECORD 1-BODY DENSITY MATRIX
    strcpy(fname,"output/");
    strcat(fname,prefix);
    strcat(fname,"_domain_realtime.dat");
    t_file = fopen(fname, "w");
    if (t_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(t_file, "# time   XI   XF   Ngrid   fig_xi   fig_xf");



    // OPEN FILE TO RECORD 1-BODY DENSITY MATRIX
    strcpy(fname, "output/");
    strcat(fname, prefix);
    strcat(fname, "_rho_realtime.dat");
    rho_file = fopen(fname, "w");
    if (rho_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(rho_file, "# Row-major vector representatio of rho_1\n");



    // OPEN FILE TO RECORD ORBITALS
    strcpy(fname, "output/");
    strcat(fname, prefix);
    strcat(fname, "_orb_realtime.dat");
    orb_file = fopen(fname, "w");
    if (orb_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(orb_file, "# Row-major vector representatio of Orbitals\n");



    // Exponential of derivatives in FFT momentum space. The FFTs ignores
    // the last grid-point assuming periodicity there. Thus the size of
    // functions in FFT must be Mpos - 1
    m = Mpos - 1;
    exp_der = carrDef(m);

    // setup descriptor (MKL implementation of FFT)
    p = DftiCreateDescriptor(&desc,DFTI_DOUBLE, DFTI_COMPLEX,1,m);
    p = DftiSetValue(desc,DFTI_FORWARD_SCALE,1.0 / sqrt(m));
    p = DftiSetValue(desc,DFTI_BACKWARD_SCALE,1.0 / sqrt(m));
    p = DftiCommitDescriptor(desc);

    // Exponential of derivative operator in momentum space
    for (i = 0; i < m; i++)
    {
        if (i <= (m - 1) / 2) { freq = (2 * PI * i) / (m * dx);       }
        else                  { freq = (2 * PI * (i - m)) / (m * dx); }
        // exponential of derivative operators in half time-step
        exp_der[i] = cexp(-0.5 * I * dt * (I * a1 * freq - a2 * freq * freq));
    }



    // Setup one/two-body hamiltonian matrix elements
    SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

    // Setup one/two-body density matrix
    OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);
    RegularizeMat(Morb,Npar*1E-6,S->rho1);



    // Initial observables
    E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
    R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
    hermitianEigvalues(Morb,S->rho1,occ);
    norm = carrMod(nc, S->C);
    checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
    checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

    printf("\n  time         E/Npar      Overlap");
    printf("    Coef-Norm    O-Avg-Norm   sqrt<R^2>   n0");
    sepline();
    printf("%10.6lf  %11.6lf",0.0,creal(E));
    printf("    %7.1E    %9.7lf",checkOverlap,norm);
    printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
    printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

    // Record initial data
    RowMajor(Morb, Morb, S->rho1, rho_vec);
    carr_inline(rho_file, Morb * Morb, rho_vec);
    recorb_inline(orb_file,Morb,Mrec,Mpos,S->Omat);
    fprintf(t_file,"\n%.6lf %.10lf %.10lf %d %.10lf %.10lf",
            0*dt,XI,XF,Mrec,MC->xi,MC->xf);



    // Evolve in fixed basis while the least occupied state is not
    // above a given tolerance
    l = 1;
    j = 0;
    if (((double)occ[0])/occ[Morb-1] < 1.5E-4)
    {
        printf("\n**    Fixed orbital evolution    **");
    }
    while (((double)occ[0])/occ[Morb-1] < 1.5E-4)
    {
        LanczosIntegrator(5,MC,S->Ho,S->Hint,dt,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        hermitianEigvalues(Morb,S->rho1,occ);

        // If the number of steps done reach record interval = record data
        // compute some observables to print on screen and record solution
        if (l == recInterval)
        {
            TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,
                  MC->strideTT,MC->IF,S->C,S->rho2);
            E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
            R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
            hermitianEigvalues(Morb,S->rho1,occ);
            norm = carrMod(nc,S->C);
            checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
            checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

            printf("\n%10.6lf  %11.6lf",(j+1)*dt,creal(E));
            printf("    %7.1E    %9.7lf",checkOverlap,norm);
            printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
            printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

            // record 1-body density matrix
            RowMajor(Morb,Morb,S->rho1,rho_vec);
            carr_inline(rho_file,Morb*Morb,rho_vec);
            // record orbitals
            recorb_inline(orb_file,Morb,Mrec,Mpos,S->Omat);
            fprintf(t_file,"\n%.6lf %.10lf %.10lf %d %.10lf %.10lf",
                    (j+1)*dt,XI,XF,Mrec,MC->xi,MC->xf);
            l = 1;
        }
        else { l = l + 1; }

        j = j + 1;
    }
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);
    if (j > 0) printf("\n**    Variational orbital evolution    **");



    for (i = j; i < Nsteps; i++)
    {

        if (j > 0 && i == j + 100)
        {
            Ortonormalize(Morb,Mpos,dx,S->Omat);
            SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
            SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);
        }

        // HALF STEP THE COEFFICIENTS
        LanczosIntegrator(3,MC,S->Ho,S->Hint,dt/2,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);
        RegularizeMat(Morb,Npar*1E-6,S->rho1);



        // FULL TIME STEP ORBITALS
        linearFFT(Mpos,Morb,&desc,exp_der,S->Omat);
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        realNLTRAP_RK4(MC,S,dt);
        linearFFT(Mpos,Morb,&desc,exp_der,S->Omat);
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);



        // ANOTHER HALF STEP FOR COEFFICIENTS
        LanczosIntegrator(3,MC,S->Ho,S->Hint,dt/2,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);
        RegularizeMat(Morb,Npar*1E-6,S->rho1);



        // Check if the boundaries are still good for trapped systems
        // since the boundaries shall not affect the physics
        if ( (i + 1) % 50 == 0 && isTrapped)
        {

            extentDomain(MC,S);

            Mpos = MC->Mpos;
            V = MC->V;

            SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
            SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);

            // Reconfigure FFT space
            free(exp_der);
            p = DftiFreeDescriptor(&desc);
            m = Mpos - 1;
            exp_der = carrDef(m);

            // setup descriptor (MKL implementation of FFT)
            p = DftiCreateDescriptor(&desc,DFTI_DOUBLE,DFTI_COMPLEX,1,m);
            p = DftiSetValue(desc,DFTI_FORWARD_SCALE,1.0/sqrt(m));
            p = DftiSetValue(desc,DFTI_BACKWARD_SCALE,1.0/sqrt(m));
            p = DftiCommitDescriptor(desc);

            // Exponential of derivative operator in momentum space
            for (k = 0; k < m; k++)
            {
                if (k <= (m - 1) / 2) { freq = (2 * PI * k) / (m * dx);       }
                else                  { freq = (2 * PI * (k - m)) / (m * dx); }
                // exponential of derivative operators in half time-step
                exp_der[k] = cexp(-0.5*I*dt*(I*a1*freq - a2*freq*freq));
            }

        }



        // Check for consistency in orbitals orthogonality
        if (checkOverlap > 5E-5)
        {
            printf("\n\nERROR : Critical loss of orthogonality ");
            printf("among orbitals. Exiting ...\n\n");
            fclose(t_file);
            fclose(rho_file);
            fclose(orb_file);
            exit(EXIT_FAILURE);
        }

        // If the number of steps done reach record interval = record data
        // compute some observables to print on screen and record solution
        if (l == recInterval)
        {
            E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
            R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
            hermitianEigvalues(Morb,S->rho1,occ);
            norm = carrMod(nc,S->C);
            checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
            checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

            printf("\n%10.6lf  %11.6lf",(i+1)*dt,creal(E));
            printf("    %7.1E    %9.7lf",checkOverlap,norm);
            printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
            printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

            // record 1-body density matrix
            RowMajor(Morb, Morb, S->rho1, rho_vec);
            carr_inline(rho_file, Morb * Morb, rho_vec);
            // record orbitals
            recorb_inline(orb_file,Morb,Mrec,Mpos,S->Omat);
            fprintf(t_file,"\n%.6lf %.10lf %.10lf %d %.10lf %.10lf",
                    (i+1)*dt,XI,XF,Mrec,MC->xi,MC->xf);
            l = 1;
        }
        else { l = l + 1; }

    }

    E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
    R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
    hermitianEigvalues(Morb,S->rho1,occ);
    norm = carrMod(nc,S->C);
    checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
    checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

    printf("\n%10.6lf  %11.6lf",Nsteps*dt,creal(E));
    printf("    %7.1E    %9.7lf",checkOverlap,norm);
    printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
    printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

    sepline();

    // FINISH - FREE MEMORY USED
    p = DftiFreeDescriptor(&desc);
    free(occ);
    free(exp_der);
    free(rho_vec);
    fclose(t_file);
    fclose(rho_file);
    fclose(orb_file);
}










/***********************************************************************
 ***********************************************************************
 ******************                                     ****************
 ******************           DVR INTEGRATORS           ****************
 ******************                                     ****************
 ***********************************************************************
 ***********************************************************************/




void SINEDVRRK4(EqDataPkg MC, ManyBodyPkg S, Rarray D2DVR, double dt)
{

    int
        i,
        j,
        k,
        s,
        Morb,
        Mpos;

    double
        g,
        a2,
        dx;

    Rarray
        V;

    double complex
        a1,
        sum;

    Cmatrix
        Ok,
        dOdt,
        Oarg,
        rho1_inv;

    Morb = MC->Morb;
    Mpos = MC->Mpos;
    a2 = MC->a2;
    a1 = MC->a1;
    g = MC->g;
    V = MC->V;
    dx = MC->dx;

    Ok = cmatDef(Morb,Mpos);
    dOdt = cmatDef(Morb,Mpos);
    Oarg = cmatDef(Morb,Mpos);
    rho1_inv = cmatDef(Morb,Morb);

    // 1-BODY MATRIX INVERSION REQUIRED ON NONLINEAR PART
    s = HermitianInv(Morb,S->rho1,rho1_inv);
    if (s != 0)
    {
        printf("\n\n\nFailed on Lapack inversion routine!\n");
        printf("-----------------------------------\n\n");

        printf("Matrix given was :\n");
        cmat_print(Morb,Morb,S->rho1);

        if (s > 0) printf("\nSingular decomposition : %d\n\n", s);
        else       printf("\nInvalid argument given : %d\n\n", s);

        exit(EXIT_FAILURE);
    }

    // COMPUTE K1
    derSINEDVR(MC,S->Omat,dOdt,rho1_inv,S->rho2,D2DVR);
    for (k = 0; k < Morb; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            Ok[k][i] = dOdt[k][i];
            Oarg[k][i] = S->Omat[k][i] + dOdt[k][i] * 0.5 * dt;
        }
    }

    // COMPUTE K2
    derSINEDVR(MC,Oarg,dOdt,rho1_inv,S->rho2,D2DVR);
    for (k = 0; k < Morb; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            Ok[k][i] += 2 * dOdt[k][i];
            Oarg[k][i] = S->Omat[k][i] + dOdt[k][i] * 0.5 * dt;
        }
    }

    // COMPUTE K3
    derSINEDVR(MC,Oarg,dOdt,rho1_inv,S->rho2,D2DVR);
    for (k = 0; k < Morb; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            Ok[k][i] += 2 * dOdt[k][i];
            Oarg[k][i] = S->Omat[k][i] + dOdt[k][i] * dt;
        }
    }

    // COMPUTE K4
    derSINEDVR(MC,Oarg,dOdt,rho1_inv,S->rho2,D2DVR);
    for (k = 0; k < Morb; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            Ok[k][i] += dOdt[k][i];
        }
    }

    // Until now Ok holds the sum K1 + 2 * K2 + 2 * K3 + K4
    // from the Fourth order Runge-Kutta algorithm, update:
    for (k = 0; k < Morb; k++)
    {   // Update Orbitals
        for (j = 0; j < Mpos; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt / 6;
        }
    }

    cmatFree(Morb,Ok);
    cmatFree(Morb,Oarg);
    cmatFree(Morb,dOdt);
    cmatFree(Morb,rho1_inv);
}



void realSINEDVR(EqDataPkg MC, ManyBodyPkg S, double dt, int Nsteps,
     char prefix [], int recInterval)
{

    int l,
        i,
        j,
        k,
        p,
        q,
        i1,
        j1,
        nc,
        Npar,
        Mpos,
        Morb,
        Mrec,
        isTrapped;

    double
        checkOrbNorm,
        checkOverlap,
        norm,
        sum,
        R2,
        XI,
        XF,
        dx,
        a2,
        g,
        L;

    double complex
        E,
        a1;

    char
        fname[100];

    FILE
        * t_file,
        * rho_file,
        * orb_file;

    Rarray
        V,
        occ,
        uDVR,
        D2mat,
        D2DVR;

    Carray
        rho_vec;



    // DECIDE FROM 1-BODY POTENTIAL THE TYPE OF BOUNDARIES
    isTrapped = 0;
    if (strcmp(MC->Vname,"harmonic") == 0)      isTrapped = 1;
    if (strcmp(MC->Vname,"doublewell") == 0)    isTrapped = 1;
    if (strcmp(MC->Vname,"harmonicgauss") == 0) isTrapped = 1;

    // RECORD INTERVAL VALID VALUES ARE > 0
    if (recInterval < 1) recInterval = 1;

    // UNPACK CONFIGURATIONAL PARAMETERS
    nc = MC->nc;
    Npar = MC->Npar;
    Mpos = MC->Mpos;
    Morb = MC->Morb;
    dx = MC->dx;
    // UNPACK EQUATION PARAMETERS AND 1-BODY POTENTIAL
    a2 = MC->a2;
    a1 = MC->a1;
    g = MC->g;
    V = MC->V;

    // LENGTH OF SINE DVR BOX. NEIGHBOORS OUTSIDE THE DOMAIN ARE RIGID WALL
    L = (Mpos + 1) * dx;

    // REQUEST MEMORY ALLOCATION
    occ = rarrDef(Morb);
    rho_vec = carrDef(Morb * Morb);
    D2mat = rarrDef(Mpos);        // second derivative matrix(diagonal)
    D2DVR = rarrDef(Mpos * Mpos); // second derivative matrix in DVR basis
    uDVR = rarrDef(Mpos * Mpos);  // unitary transformation to DVR basis

    // Setup DVR transformation matrix with  eigenvector  organized
    // by columns in row major format. It also setup the derivarive
    // matrices in the sine basis that are known analytically
    for (i = 0; i < Mpos; i++)
    {
        i1 = i + 1;
        for (j = 0; j < Mpos; j++)
        {
            j1 = j + 1;
            uDVR[i*Mpos + j] = sqrt(2.0/(Mpos+1)) * sin(i1*j1*PI/(Mpos+1));
        }
        D2mat[i] = - (i1 * PI / L) * (i1 * PI / L);
    }

    // Transform second order derivative matrix to DVR basis
    // multiplying by the equation coefficient
    for (i = 0; i < Mpos; i++)
    {
        for (j = 0; j < Mpos; j++)
        {
            sum = 0;
            for (k = 0; k < Mpos; k++)
            {
                sum += uDVR[i*Mpos + k] * D2mat[k] * uDVR[k*Mpos + j];
            }
            D2DVR[i*Mpos + j] = a2 * sum;
        }
    }

    // RECORD DOMAIN INFORMATION
    if (isTrapped)
    {
        // Define a number of points to the left and to the right in
        // order to do not interact with the boundariers  throughout
        // all time evolved. This is given by 'k'.
        k = Mpos;
        Mrec = Mpos + 2 * k;
        XI = MC->xi - k*dx;
        XF = MC->xf + k*dx;
    }
    else
    {
        Mrec = Mpos;
        XI = MC->xi;
        XF = MC->xf;
    }

    strcpy(fname,"output/");
    strcat(fname,prefix);
    strcat(fname,"_domain_realtime.dat");
    t_file = fopen(fname, "w");
    if (t_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(t_file, "# time   XI   XF   Ngrid   fig_xi   fig_xf");

    // OPEN FILE TO RECORD 1-BODY DENSITY MATRIX
    strcpy(fname, "output/");
    strcat(fname, prefix);
    strcat(fname, "_rho_realtime.dat");
    rho_file = fopen(fname, "w");
    if (rho_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(rho_file, "# Row-major vector representatio of rho_1\n");

    // OPEN FILE TO RECORD ORBITALS
    strcpy(fname, "output/");
    strcat(fname, prefix);
    strcat(fname, "_orb_realtime.dat");
    orb_file = fopen(fname, "w");
    if (orb_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(orb_file, "# Row-major vector representatio of Orbitals\n");

    // SETUP ONE/TWO-BODY HAMILTONIAN MATRIX ELEMENTS
    SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

    // SETUP ONE/TWO-BODY DENSITY MATRIX
    OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);

    // QUALITY CONTROL PARAMETERS
    E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
    R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
    hermitianEigvalues(Morb,S->rho1,occ);
    norm = carrMod(nc, S->C);
    checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
    checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

    printf("\n  time         E/Npar      Overlap");
    printf("    Coef-Norm    O-Avg-Norm   sqrt<R^2>   n0");
    sepline();
    printf("%10.6lf  %11.6lf",0.0,creal(E));
    printf("    %7.1E    %9.7lf",checkOverlap,norm);
    printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
    printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

    // RECORD INITIAL DATA
    RowMajor(Morb, Morb, S->rho1, rho_vec);
    carr_inline(rho_file, Morb * Morb, rho_vec);
    recorb_inline(orb_file,Morb,Mrec,Mpos,S->Omat);
    fprintf(t_file,"\n%.6lf %.10lf %.10lf %d %.10lf %.10lf",
            0*dt,XI,XF,Mrec,MC->xi,MC->xf);

    // EVOLVE IN FIXED BASIS WHILE THE LEAST OCCUPATION IS SMALL
    l = 1;
    j = 0; // number of steps evolved with fixed orbital basis
    if (((double)occ[0])/occ[Morb-1] < 3E-4)
    {
        printf("\n**    Fixed orbital evolution    **");
    }
    while (((double)occ[0])/occ[Morb-1] < 3E-4)
    {
        LanczosIntegrator(5,MC,S->Ho,S->Hint,dt,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        hermitianEigvalues(Morb,S->rho1,occ);

        // If the number of steps done reach record interval = record data
        // compute some observables to print on screen and record solution
        if (l == recInterval)
        {
            TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,
                  MC->strideTT,MC->IF,S->C,S->rho2);
            E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
            R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
            hermitianEigvalues(Morb,S->rho1,occ);
            norm = carrMod(nc,S->C);
            checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
            checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

            printf("\n%10.6lf  %11.6lf",(j+1)*dt,creal(E));
            printf("    %7.1E    %9.7lf",checkOverlap,norm);
            printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
            printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

            // record 1-body density matrix
            RowMajor(Morb,Morb,S->rho1,rho_vec);
            carr_inline(rho_file,Morb*Morb,rho_vec);
            // record orbitals
            recorb_inline(orb_file,Morb,Mrec,Mpos,S->Omat);
            fprintf(t_file,"\n%.6lf %.10lf %.10lf %d %.10lf %.10lf",
                    (j+1)*dt,XI,XF,Mrec,MC->xi,MC->xf);
            l = 1;
        }
        else { l = l + 1; }

        j = j + 1;
    }
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);
    if (j > 0) printf("\n**    Variational orbital evolution    **");

    // EVOLVE WITH VARIATIONAL ORBITALS
    for (i = j; i < Nsteps; i++)
    {

        if (j > 0 && i == j + 50)
        {
            Ortonormalize(Morb,Mpos,dx,S->Omat);
            SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
            SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);
        }

        // HALF STEP THE COEFFICIENTS
        LanczosIntegrator(3,MC,S->Ho,S->Hint,dt/2,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);
        RegularizeMat(Morb,Npar*1E-6,S->rho1);

        // HALF TIME STEP ORBITALS
        SINEDVRRK4(MC,S,D2DVR,dt/2);
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

        // ANOTHER HALF STEP FOR COEFFICIENTS
        LanczosIntegrator(3,MC,S->Ho,S->Hint,dt/2,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);
        RegularizeMat(Morb,Npar*1E-6,S->rho1);

        // HALF TIME STEP ORBITALS
        SINEDVRRK4(MC,S,D2DVR,dt/2);
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

        // CHECK IF THE BOUNDARIES ARE STILL GOOD FOR TRAPPED SYSTEMS
        if ( (i + 1) % 50 == 0 && isTrapped) extentDomain(MC,S);
        if (Mpos < MC->Mpos)
        {
            Mpos = MC->Mpos;
            V = MC->V;

            SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
            SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);

            free(D2mat);
            free(D2DVR);
            free(uDVR);
            D2mat = rarrDef(Mpos);
            D2DVR = rarrDef(Mpos * Mpos);
            uDVR = rarrDef(Mpos * Mpos);
            // Setup DVR transformation matrix with  eigenvector  organized
            // by columns in row major format. It also setup the derivarive
            // matrices in the sine basis that are known analytically
            for (p = 0; p < Mpos; p++)
            {
                i1 = p + 1;
                for (q = 0; q < Mpos; q++)
                {
                    j1 = q + 1;
                    uDVR[p*Mpos + q] = sqrt(2.0/(Mpos+1))*sin(i1*j1*PI/(Mpos+1));
                }
                D2mat[p] = - (i1 * PI / L) * (i1 * PI / L);
            }

            // Transform second order derivative matrix to DVR basis
            // multiplying by the equation coefficient
            for (p = 0; p < Mpos; p++)
            {
                for (q = 0; q < Mpos; q++)
                {
                    sum = 0;
                    for (k = 0; k < Mpos; k++)
                    {
                        sum += uDVR[p*Mpos + k] * D2mat[k] * uDVR[k*Mpos + q];
                    }
                    D2DVR[p*Mpos + q] = a2 * sum;
                }
            }
        }

        // CHECK FOR CONSISTENCY IN ORBITALS ORTHOGONALITY
        if (checkOverlap > 1E-5)
        {
            printf("\n\nERROR : Critical loss of orthogonality ");
            printf("among orbitals. Exiting ...\n\n");
            fclose(t_file);
            fclose(rho_file);
            fclose(orb_file);
            exit(EXIT_FAILURE);
        }

        // IF THE NUMBER OF STEPS DONE REACH RECORD INTERVAL = RECORD DATA
        // COMPUTE SOME OBSERVABLES TO PRINT ON SCREEN AND RECORD SOLUTION
        if (l == recInterval)
        {
            E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
            R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
            hermitianEigvalues(Morb,S->rho1,occ);
            norm = carrMod(nc,S->C);
            checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
            checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

            printf("\n%10.6lf  %11.6lf",(i+1)*dt,creal(E));
            printf("    %7.1E    %9.7lf",checkOverlap,norm);
            printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
            printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

            // record 1-body density matrix
            RowMajor(Morb,Morb,S->rho1,rho_vec);
            carr_inline(rho_file,Morb*Morb,rho_vec);
            // record orbitals
            recorb_inline(orb_file,Morb,Mrec,Mpos,S->Omat);
            fprintf(t_file,"\n%.6lf %.10lf %.10lf %d %.10lf %.10lf",
                    (i+1)*dt,XI,XF,Mrec,MC->xi,MC->xf);
            l = 1;
        }
        else { l = l + 1; }

    }

    E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
    R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
    hermitianEigvalues(Morb,S->rho1,occ);
    norm = carrMod(nc,S->C);
    checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
    checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

    printf("\n%10.6lf  %11.6lf",Nsteps*dt,creal(E));
    printf("    %7.1E    %9.7lf",checkOverlap,norm);
    printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
    printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

    sepline();
    free(occ);
    free(rho_vec);
    free(uDVR);
    free(D2mat);
    free(D2DVR);

    fclose(t_file);
    fclose(rho_file);
    fclose(orb_file);
}



void EXPDVRRK4(EqDataPkg MC, ManyBodyPkg S, Carray DerDVR, double dt)
{

    int
        i,
        j,
        k,
        s,
        Morb,
        Mpos;

    Cmatrix
        Ok,
        dOdt,
        Oarg,
        rho1_inv;

    Morb = MC->Morb;
    Mpos = MC->Mpos;

    Ok = cmatDef(Morb,Mpos);
    dOdt = cmatDef(Morb,Mpos);
    Oarg = cmatDef(Morb,Mpos);
    rho1_inv = cmatDef(Morb,Morb);

    // INVERSION OF 1-BODY DENSITY MATRIX REQUIRED TO EVALUATE NONLINEAR PART
    s = HermitianInv(Morb,S->rho1,rho1_inv);
    if (s != 0)
    {
        printf("\n\n\nFailed on Lapack inversion routine!\n");
        printf("-----------------------------------\n\n");

        printf("Matrix given was :\n");
        cmat_print(Morb,Morb,S->rho1);

        if (s > 0) printf("\nSingular decomposition : %d\n\n", s);
        else       printf("\nInvalid argument given : %d\n\n", s);

        exit(EXIT_FAILURE);
    }

    // COMPUTE K1
    derEXPDVR(MC,S->Omat,dOdt,rho1_inv,S->rho2,DerDVR);
    for (k = 0; k < Morb; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            Ok[k][i] = dOdt[k][i];
            Oarg[k][i] = S->Omat[k][i] + dOdt[k][i] * 0.5 * dt;
        }
    }

    // COMPUTE K2
    derEXPDVR(MC,Oarg,dOdt,rho1_inv,S->rho2,DerDVR);
    for (k = 0; k < Morb; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            Ok[k][i] += 2 * dOdt[k][i];
            Oarg[k][i] = S->Omat[k][i] + dOdt[k][i] * 0.5 * dt;
        }
    }

    // COMPUTE K3
    derEXPDVR(MC,Oarg,dOdt,rho1_inv,S->rho2,DerDVR);
    for (k = 0; k < Morb; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            Ok[k][i] += 2 * dOdt[k][i];
            Oarg[k][i] = S->Omat[k][i] + dOdt[k][i] * dt;
        }
    }

    // COMPUTE K4
    derEXPDVR(MC,Oarg,dOdt,rho1_inv,S->rho2,DerDVR);
    for (k = 0; k < Morb; k++)
    {
        for (i = 0; i < Mpos; i++)
        {
            Ok[k][i] += dOdt[k][i];
        }
    }

    // Until now Ok holds the sum K1 + 2 * K2 + 2 * K3 + K4
    // from the Fourth order Runge-Kutta algorithm, update:
    for (k = 0; k < Morb; k++)
    {   // Update Orbitals
        for (j = 0; j < Mpos-1; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt / 6;
        }
        S->Omat[k][Mpos-1] = S->Omat[k][0];
    }

    cmatFree(Morb,Ok);
    cmatFree(Morb,Oarg);
    cmatFree(Morb,dOdt);
    cmatFree(Morb,rho1_inv);
}



void realEXPDVR(EqDataPkg MC, ManyBodyPkg S, double dt, int Nsteps,
     char prefix [], int recInterval)
{

    int
        l,
        i,
        j,
        k,
        p,
        q,
        N,
        nc,
        kmom,
        Npar,
        Mpos,
        Morb,
        Mrec,
        isTrapped;

    double
        checkOrbNorm,
        checkOverlap,
        norm,
        R2,
        XI,
        XF,
        dx,
        a2,
        g,
        L;

    double complex
        E,
        a1,
        sum,
        diag;

    char
        fname[100];

    FILE
        * t_file,
        * rho_file,
        * orb_file;

    Rarray
        V,
        occ;

    Carray
        D1DVR,
        D2DVR,
        DerDVR;

    Carray
        rho_vec;



    // FROM 1-BODY POTENTIAL DECIDE THE TYPE OF BOUNDARIES
    isTrapped = 0;
    if (strcmp(MC->Vname,"harmonic") == 0)      isTrapped = 1;
    if (strcmp(MC->Vname,"doublewell") == 0)    isTrapped = 1;
    if (strcmp(MC->Vname,"harmonicgauss") == 0) isTrapped = 1;

    // RECORD INTERVAL VALID VALUES MUST BE POSITIVE INTEGER
    if (recInterval < 1) recInterval = 1;

    // UNPACK DOMAIN PARAMETERS TO LOCAL VARIABLES
    nc = MC->nc;
    Npar = MC->Npar;
    Mpos = MC->Mpos;
    Morb = MC->Morb;
    dx = MC->dx;

    // UNPACK EQUATION COEFFICIENTS AND ONE-BODY POTENTIIAL
    a2 = MC->a2;
    a1 = MC->a1;
    g = MC->g;
    V = MC->V;

    // NUMBER OF DVR POINTS IGNORING LAST GRID POINT CONSIDERED AS
    // THE SAME OF THE FIRST ONE - PERIODIC BOUNDARY
    N = Mpos - 1;

    // LENGTH OF GRID DOMAIN
    L = N * dx;

    // REQUEST MEMORY ALLOCATION
    occ = rarrDef(Morb);
    rho_vec = carrDef(Morb*Morb);
    D2DVR = carrDef(N * N);  // second derivative matrix in DVR basis
    D1DVR = carrDef(N * N);  // unitary transformation to DVR basis
    DerDVR = carrDef(N * N); // add two matrices of derivatives

    // SETUP FIRST ORDER DERIVATIVE MATRIX IN DVR BASIS
    for (i = 0; i < N; i++)
    {
        // USE COMPLEX CONJ. TO COMPUTE UPPER TRIANGULAR PART ONLY j > i
        for (j = i + 1; j < N; j++)
        {
            sum = 0;
            for (k = 0; k < N; k++)
            {
                kmom = k - N/2;
                // NOTE THIS MINUS SIGN - I HAD NO EXPLANATION FOR IT
                diag = - (2*I*PI*kmom/L);
                sum = sum + diag * cexp(2*I*kmom*(j-i)*PI/N) / N;
            }
            D1DVR[i*N + j] = sum;
            D1DVR[j*N + i] = -conj(sum);
        }
        // COMPUTE SEPARATELY THE DIAGONAL
        sum = 0;
        for (k = 0; k < N; k++)
        {
            kmom = k - N/2;
            diag = - (2*I*PI*kmom/L);
            sum = sum + diag / N;
        }
        D1DVR[i*N + i] = sum;
    }

    // SETUP SECOND ORDER DERIVATIVE MATRIX IN DVR BASIS
    // 'mom'entum variables because of exponential basis
    for (i = 0; i < N; i++)
    {
        // USE COMPLEX CONJ. TO COMPUTE UPPER TRIANGULAR PART ONLY j > i
        for (j = i + 1; j < N; j++)
        {
            sum = 0;
            for (k = 0; k < N; k++)
            {
                kmom = k - N/2;
                diag = - (2*PI*kmom/L)*(2*PI*kmom/L);
                sum = sum + diag * cexp(2*I*kmom*(j-i)*PI/N) / N;
            }
            D2DVR[i*N + j] = sum;
            D2DVR[j*N + i] = conj(sum);
        }
        // COMPUTE SEPARATELY THE DIAGONAL
        sum = 0;
        for (k = 0; k < N; k++)
        {
            kmom = k - N/2;
            diag = - (2*PI*kmom/L)*(2*PI*kmom/L);
            sum = sum + diag / N;
        }
        D2DVR[i*N + i] = sum;
    }

    // SETUP MATRIX CORRESPONDING TO DERIVATIVES ON HAMILTONIAN
    // INCLUDING THE EQUATION COEFFICIENTS IN FRONT OF THEM
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            DerDVR[i*N + j] = a2 * D2DVR[i*N + j] + a1 * D1DVR[i*N + j];
        }
    }

    // RECORD DOMAIN EXTENSION
    if (isTrapped)
    {
        // Define a number of points to the left and to the right in
        // order to do not interact with the boundariers  throughout
        // all time evolved for trapped systems.  It is also the max
        // size the domain can be extended during evolution
        k = Mpos;
        Mrec = Mpos + 2 * k;
        XI = MC->xi - k*dx;
        XF = MC->xf + k*dx;
    }
    else
    {
        // Periodic systems
        Mrec = Mpos;
        XI = MC->xi;
        XF = MC->xf;
    }

    // OPEN FILE TO RECORD TIME AND DOMAIN INFO
    strcpy(fname,"output/");
    strcat(fname,prefix);
    strcat(fname,"_domain_realtime.dat");
    t_file = fopen(fname, "w");
    if (t_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(t_file, "# time   XI   XF   Ngrid   fig_xi   fig_xf");

    // OPEN FILE TO RECORD 1-BODY DENSITY MATRIX
    strcpy(fname, "output/");
    strcat(fname, prefix);
    strcat(fname, "_rho_realtime.dat");
    rho_file = fopen(fname, "w");
    if (rho_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(rho_file, "# Row-major vector representatio of rho_1\n");

    // OPEN FILE TO RECORD ORBITALS
    strcpy(fname, "output/");
    strcat(fname, prefix);
    strcat(fname, "_orb_realtime.dat");
    orb_file = fopen(fname, "w");
    if (orb_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(orb_file, "# Row-major vector representatio of Orbitals\n");

    // SETUP ONE/TWO-BODY HAMILTONIAN MATRIX ELEMENTS ON ORBITAL BASIS
    SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

    // SETUP ONE/TWO-BODY DENSITY MATRIX
    OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);

    // TIME MONITORING OBSERVABLES AND ACCURACY MEASURES
    E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
    R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
    hermitianEigvalues(Morb,S->rho1,occ);
    norm = carrMod(nc, S->C);
    checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
    checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

    printf("\n  time         E/Npar      Overlap");
    printf("    Coef-Norm    O-Avg-Norm   sqrt<R^2>   n0");
    sepline();
    printf("%10.6lf  %11.6lf",0.0,creal(E));
    printf("    %7.1E    %9.7lf",checkOverlap,norm);
    printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
    printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

    // RECORD INITIAL DATA
    RowMajor(Morb, Morb, S->rho1, rho_vec);
    carr_inline(rho_file, Morb * Morb, rho_vec);
    recorb_inline(orb_file,Morb,Mrec,Mpos,S->Omat);
    fprintf(t_file,"\n%.6lf %.10lf %.10lf %d %.10lf %.10lf",
            0*dt,XI,XF,Mrec,MC->xi,MC->xf);

    // EVOLVE IN FIXED BASIS WHILE LEAST OCCUPATION IS SMALL
    l = 1;
    j = 0; // number of steps evolved in fixed orbital basis
    if (occ[0]/occ[Morb-1] < 1E-4)
    {
        printf("\n**    Fixed orbital evolution    **");
    }
    while (occ[0]/occ[Morb-1] < 1E-4)
    {
        LanczosIntegrator(5,MC,S->Ho,S->Hint,dt,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        hermitianEigvalues(Morb,S->rho1,occ);

        // IF THE NUMBER OF STEPS DONE REACH RECORD INTERVAL = RECORD DATA
        // COMPUTE SOME OBSERVABLES TO PRINT ON SCREEN AND RECORD SOLUTION
        if (l == recInterval)
        {
            TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,
                  MC->strideTT,MC->IF,S->C,S->rho2);
            E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
            R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
            hermitianEigvalues(Morb,S->rho1,occ);
            norm = carrMod(nc,S->C);
            checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
            checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

            printf("\n%10.6lf  %11.6lf",(j+1)*dt,creal(E));
            printf("    %7.1E    %9.7lf",checkOverlap,norm);
            printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
            printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

            // record 1-body density matrix
            RowMajor(Morb,Morb,S->rho1,rho_vec);
            carr_inline(rho_file,Morb*Morb,rho_vec);
            // record orbitals
            recorb_inline(orb_file,Morb,Mrec,Mpos,S->Omat);
            fprintf(t_file,"\n%.6lf %.10lf %.10lf %d %.10lf %.10lf",
                    (j+1)*dt,XI,XF,Mrec,MC->xi,MC->xf);
            l = 1;
        }
        else { l = l + 1; }

        j = j + 1;
    }
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);
    if (j > 0) printf("\n**    Variational orbital evolution    **");



    // VARIATIONAL ORBITAL EVOLUTION
    for (i = j; i < Nsteps; i++)
    {

        if (j > 0 && i == j + 50)
        {
            // small occupations can make it unstable in the first
            // few steps, thus ortonormalize
            Ortonormalize(Morb,Mpos,dx,S->Omat);
            SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
            SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);
        }

        // HALF STEP THE COEFFICIENTS
        LanczosIntegrator(3,MC,S->Ho,S->Hint,dt/2,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);
        RegularizeMat(Morb,Npar*1E-6,S->rho1);

        // HALF TIME STEP ORBITALS
        EXPDVRRK4(MC,S,DerDVR,dt/2);
        SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
        SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);

        // ANOTHER HALF STEP FOR COEFFICIENTS
        LanczosIntegrator(3,MC,S->Ho,S->Hint,dt/2,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);
        RegularizeMat(Morb,Npar*1E-6,S->rho1);

        // ANOTHER HALF TIME STEP ORBITALS
        EXPDVRRK4(MC,S,DerDVR,dt/2);
        SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
        SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);

        // CHECK IF THE BOUNDARIES ARE STILL GOOD FOR TRAPPED SYSTEMS
        if ((i + 1) % 50 == 0 && isTrapped) extentDomain(MC,S);

        // IF DOMAIN WAS ENLARGED UPDATE VARIABLES
        if (Mpos < MC->Mpos)
        {

            Mpos = MC->Mpos;
            V = MC->V;

            SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
            SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);

            // SETUP DVR MATRICES AS DONE INITIALLY
            N = Mpos - 1;
            L = N * dx;

            free(D2DVR);
            free(D1DVR);
            free(DerDVR);
            D2DVR = carrDef(N * N);
            D1DVR = carrDef(N * N);
            DerDVR = carrDef(N * N);

            for (p = 0; p < N; p++)
            {
                for (q = 0; q < N; q++)
                {
                    sum = 0;
                    for (k = 0; k < N; k++)
                    {
                        kmom = k - N/2;
                        diag = - (2*I*PI*kmom/L);
                        sum = sum + diag * cexp(2*I*kmom*(q-p)*PI/N)/N;
                    }
                    D1DVR[p*N + q] = sum;
                }
            }

            for (p = 0; p < N; p++)
            {
                for (q = 0; q < N; q++)
                {
                    sum = 0;
                    for (k = 0; k < N; k++)
                    {
                        kmom = k - N/2;
                        diag = - (2*PI*kmom/L)*(2*PI*kmom/L);
                        sum = sum + diag * cexp(2*I*kmom*(q-p)*PI/N)/N;
                    }
                    D2DVR[p*N + q] = sum;
                }
            }

            for (p = 0; p < N; p++)
            {
                for (q = 0; q < N; q++)
                {
                    DerDVR[p*N + q] = a2 * D2DVR[p*N+q] + a1 * D1DVR[p*N+q];
                }
            }

            // FINISH RE-ASSEMBLE OF DVR MATRICES
        }

        // CHECK FOR CONSISTENCY IN ORBITALS ORTHOGONALITY
        if (checkOverlap > 1E-5)
        {
            printf("\n\nERROR : Critical loss of orthogonality ");
            printf("among orbitals. Exiting ...\n\n");
            fclose(t_file);
            fclose(rho_file);
            fclose(orb_file);
            exit(EXIT_FAILURE);
        }

        // IF THE NUMBER OF STEPS DONE REACH RECORD INTERVAL = RECORD DATA
        // COMPUTE SOME OBSERVABLES TO PRINT ON SCREEN AND RECORD SOLUTION
        if (l == recInterval)
        {
            E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
            R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
            hermitianEigvalues(Morb,S->rho1,occ);
            norm = carrMod(nc,S->C);
            checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
            checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

            printf("\n%10.6lf  %11.6lf",(i+1)*dt,creal(E));
            printf("    %7.1E    %9.7lf",checkOverlap,norm);
            printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
            printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

            // record 1-body density matrix
            RowMajor(Morb,Morb,S->rho1,rho_vec);
            carr_inline(rho_file,Morb*Morb,rho_vec);
            // record orbitals
            recorb_inline(orb_file,Morb,Mrec,Mpos,S->Omat);
            fprintf(t_file,"\n%.6lf %.10lf %.10lf %d %.10lf %.10lf",
                    (i+1)*dt,XI,XF,Mrec,MC->xi,MC->xf);
            l = 1;
        }
        else { l = l + 1; }

    }

    // FINISHED TIME EVOLUTION

    E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
    R2 = MeanQuadraticR(MC,S->Omat,S->rho1);
    hermitianEigvalues(Morb,S->rho1,occ);
    norm = carrMod(nc,S->C);
    checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
    checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

    printf("\n%10.6lf  %11.6lf",Nsteps*dt,creal(E));
    printf("    %7.1E    %9.7lf",checkOverlap,norm);
    printf("    %9.7lf     %6.4lf",checkOrbNorm,R2);
    printf("    %3.0lf%%",100*occ[Morb-1]/Npar);

    sepline();
    free(occ);
    free(rho_vec);
    free(D2DVR);
    free(D1DVR);
    free(DerDVR);

    fclose(t_file);
    fclose(rho_file);
    fclose(orb_file);
}
