#include "../include/MCTDHB_integrator.h"










double complex nonlinear (int M, int k, int n, double g, Cmatrix Orb,
               Cmatrix Rinv, Carray R2, Cmatrix Ho, Carray Hint )
{

/** For a orbital 'k' computed at discretized position 'n' calculate
  * the right-hand-side part of MCTDHB orbital's equation of  motion
  * that is nonlinear, part because of projections that made the eq.
  * an integral-differential equation, and other part due to contact
  * interactions. Assume that Rinv, R2 are  defined  by  the  set of
  * configuration-state coefficients as the inverse of  one-body and
  * two-body density matrices respectively. Ho and Hint are  assumed
  * to be defined accoding to 'Orb' variable as well.
  *
**/



    int a,
        j,
        s,
        q,
        l,
        M2,
        M3,
        ind;



    double complex
        G,
        X;



    X = 0;
    M2 = M * M;
    M3 = M * M * M;



    for (s = 0; s < M; s++)
    {
        // Subtract one-body projection
        X = X - Ho[s][k] * Orb[s][n];

        for (a = 0; a < M; a++)
        {

            for (q = 0; q < M; q++)
            {
                // Particular case with the two last indices equals
                // to take advantage of the symmetry afterwards

                G = Rinv[k][a] * R2[a + M*s + M2*q + M3*q];

                // Sum interacting part contribution
                X = X + g * G * conj(Orb[s][n]) * Orb[q][n] * Orb[q][n];

                // Subtract interacting projection
                for (j = 0; j < M; j++)
                {
                    ind = j + s * M + q * M2 + q * M3;
                    X = X - G * Orb[j][n] * Hint[ind];
                }

                for (l = q + 1; l < M; l++)
                {
                    G = 2 * Rinv[k][a] * R2[a + M*s + M2*q + M3*l];

                    // Sum interacting part
                    X = X + g * G * conj(Orb[s][n]) * Orb[l][n] * Orb[q][n];

                    // Subtract interacting projection
                    for (j = 0; j < M; j++)
                    {
                        ind = j + s * M + l * M2 + q * M3;
                        X = X - G * Orb[j][n] * Hint[ind];
                    }
                }
            }
        }
    }

    return X;
}










void MCNLTRAP_dOdt(MCTDHBsetup MC, Cmatrix Orb, Cmatrix dOdt,
     Cmatrix Ho, Carray Hint, Cmatrix rho1, Carray rho2)
{

//  Time derivative of the set of orbitals due exclusively to nonlinear
//  part plus the potential. Used for split-step-spectral method  where
//  the derivative and potential part are solved separately
//
//  OUTPUT ARGUMENT : dOdt



    int
        k,
        s,
        j,
        M  = MC->Morb,
        N  = MC->Npar,
        Mpos  = MC->Mpos,
        ** IF = MC->IF,
        ** NCmat = MC->NCmat;



    double
        dx = MC->dx,
        * V = MC->V;



    double complex
        Proj,
        g = MC->inter;



    Cmatrix
        rho_inv = cmatDef(M, M);



    /* Inversion of one-body density matrix
    ====================================================================== */
    s = HermitianInv(M, rho1, rho_inv);

    if (s != 0)
    {
        printf("\n\n\n\n\t\tFailed on Lapack inversion routine!\n");
        printf("\t\t-----------------------------------\n\n");

        printf("\nMatrix given was : \n");
        cmat_print(M, M, rho1);

        if (s > 0) printf("\nSingular decomposition : %d\n\n", s);
        else       printf("\nInvalid argument given : %d\n\n", s);

        exit(EXIT_FAILURE);
    }
    /* =================================================================== */



    // Update k-th orbital at discretized position j
    #pragma omp parallel for private(k, j) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
            dOdt[k][j] = - I * ( V[j] * Orb[k][j] + \
            nonlinear(M, k, j, g, Orb, rho_inv, rho2, Ho, Hint) );
    }



    // Release memory
    cmatFree(M, rho_inv);

}










void MCNL_dOdt (MCTDHBsetup MC, Cmatrix Orb, Cmatrix dOdt,
     Cmatrix Ho, Carray Hint, Cmatrix rho1, Carray rho2)
{

//  Time derivative of the set of orbitals due exclusively  to nonlinear
//  part. Used for split-step-Finite-Differences method where the linear
//  and nonlinear part are solved separately
//
//  OUTPUT ARGUMENT : dOdt



    int
        k,
        s,
        j,
        M = MC->Morb,
        N = MC->Npar,
        Mpos  = MC->Mpos,
        ** IF = MC->IF,
        ** NCmat = MC->NCmat;



    double
        dx = MC->dx;



    double complex
        Proj,
        g = MC->inter;



    Cmatrix
        rho_inv = cmatDef(M, M);



    /* Inversion of one-body density matrix
    ====================================================================== */
    s = HermitianInv(M, rho1, rho_inv);

    if (s != 0)
    {
        printf("\n\n\n\n\t\tFailed on Lapack inversion routine!\n");
        printf("\t\t-----------------------------------\n\n");

        printf("\nMatrix given was : \n");
        cmat_print(M, M, rho1);

        if (s > 0) printf("\nSingular decomposition : %d\n\n", s);
        else       printf("\nInvalid argument given : %d\n\n", s);

        exit(EXIT_FAILURE);
    }
    /* =================================================================== */



    // Update k-th orbital at discretized position j
    #pragma omp parallel for private(k, j) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
            dOdt[k][j] = - I * \
            nonlinear(M, k, j, g, Orb, rho_inv, rho2, Ho, Hint);
    }



    // Release memory
    cmatFree(M, rho_inv);
}










void MC_dCdt (MCTDHBsetup MC, Carray C, Cmatrix Ho, Carray Hint, Carray dCdt)
{

//  Time derivative of coefficients from the expansion in configuration
//  basis. Assume elements of Ho and Hint previuosly setted up with the
//  Single Particle Wave function whose the configurations refers to.
//
//  OUTPUT PARAMETERS : dCdt

    int i;

    applyHconf(MC->Npar, MC->Morb, MC->NCmat, MC->IF, C, Ho, Hint, dCdt);

    for (i = 0; i < MC->nc; i++) dCdt[i] = - I * dCdt[i];
}










int lanczos(MCTDHBsetup MCdata, Cmatrix Ho, Carray Hint,
    int lm, Carray diag, Carray offdiag, Cmatrix lvec)
{

//  Improved lanczos iterations with reorthogonalization for the vector
//  of coefficients of the many-body state represented in configuration
//  basis. Lanczos procedure give a (real)tridiagonal whose the spectra
//  approximate the original one, as better as more iterations are done.
//  It work, however, just with a routine to apply the original  matrix
//  to any vector, what can save memory.
//
//  OUTPUT PARAMETERS :
//      lvec - Lanczos vectors used to convert eigenvectors
//      diag - diagonal elements of tridiagonal symmetric matrix
//      offdiag - symmetric elements of tridiagonal matrix
//
//  RETURN :
//      number of itertion done (just lm if the method does not breakdown)



    int i,
        j,
        k,
        nc = MCdata->nc,
        Npar = MCdata->Npar,
        Morb = MCdata->Morb,
        ** IF = MCdata->IF,
        ** NCmat = MCdata->NCmat;

    double
        tol,
        maxCheck;

    Carray
        out = carrDef(nc),
        ortho = carrDef(lm);



    applyHconf(Npar, Morb, NCmat, IF, lvec[0], Ho, Hint, out);
    diag[0] = carrDot(nc, lvec[0], out);

    for (j = 0; j < nc; j++) out[j] = out[j] - diag[0] * lvec[0][j];



    // Check for a source of breakdown in the algorithm to do not
    // divide by zero. Instead of zero use  a  tolerance (tol) to
    // avoid numerical instability
    maxCheck = 0;
    tol = 1E-15;



    // Core iteration procedure
    for (i = 0; i < lm - 1; i++)
    {
        offdiag[i] = carrMod(nc, out);

        if (maxCheck < creal(offdiag[i])) maxCheck = creal(offdiag[i]);

        // If method break return number of iterations achieved
        if (creal(offdiag[i]) / maxCheck < tol) return (i + 1);

        carrScalarMultiply(nc, out, 1.0 / offdiag[i], lvec[i + 1]);
        applyHconf(Npar, Morb, NCmat, IF, lvec[i + 1], Ho, Hint, out);

        for (j = 0; j < nc; j++)
        {
            out[j] = out[j] - offdiag[i] * lvec[i][j];
        }

        diag[i + 1] = carrDot(nc, lvec[i + 1], out);

        for (j = 0; j < nc; j++)
        {
            out[j] = out[j] - diag[i+1]*lvec[i+1][j];
        }

        // Additional re-orthogonalization procedure
        carrFill(lm, 0, ortho);
        for (k = 0; k < i + 2; k++) ortho[k] += carrDot(nc, lvec[k], out);
        for (j = 0; j < nc; j++)
        {
            for (k = 0; k < i + 2; k++) out[j] -= lvec[k][j] * ortho[k];
        }
    }

    free(ortho);
    free(out);

    return lm;
}










double LanczosGround (int Niter, MCTDHBsetup MC, Cmatrix Orb, Carray C)
{

//  Find the lowest Eigenvalue using Lanczos tridiagonal decomposition
//  for the hamiltonian in configurational space, with orbitals fixed.
//  Use up to Niter(unless the a breakdown occur) in Lanczos method to
//  obtain a basis-fixed ground state approximation  of  the truncated
//  configuration space.
//
//  INPUT/OUTPUT : C (end up as eigenvector approximation)
//
//  RETURN : Lowest eigenvalue found



    int
        i,
        k,
        j,
        predictedIter,
        nc = MC->nc,
        Morb = MC->Morb,
        Mpos = MC->Mpos;



    // variables to call lapack diagonalization routine for tridiagonal
    // symmetric matrix
    // ----------------------------------------------------------------

    double
        sentinel,
        * d = malloc(Niter * sizeof(double)),
        * e = malloc(Niter * sizeof(double)),
        * eigvec = malloc(Niter * Niter * sizeof(double));



    // variables to store lanczos vectors and tridiagonal symmetric matrix
    // -------------------------------------------------------------------

    // Elements of tridiagonal lanczos matrix
    Carray
        diag = carrDef(Niter),
        offdiag = carrDef(Niter),
        Hint = carrDef(Morb * Morb * Morb * Morb);
    // Lanczos Vectors (organize in rows instead of columns rows)
    Cmatrix
        Ho = cmatDef(Morb, Morb),
        lvec = cmatDef(Niter, nc);
    
    
    
    SetupHo(Morb, Mpos, Orb, MC->dx, MC->a2, MC->a1, MC->V, Ho);
    SetupHint(Morb, Mpos, Orb, MC->dx, MC->inter, Hint);



    // Setup values needed to solve the equations for C
    // ------------------------------------------------

    offdiag[Niter-1] = 0;     // Useless
    carrCopy(nc, C, lvec[0]); // Setup initial lanczos vector



    // Call Lanczos what setup tridiagonal matrix and lanczos vectors
    // -----------------------------------------------------------------
    predictedIter = Niter;
    Niter = lanczos(MC, Ho, Hint, Niter, diag, offdiag, lvec);
    if (Niter < predictedIter)
    {
        printf("\n\n\t\tlanczos iterations exit before expected - %d\n\n", Niter);
    }



    // Transfer data to use lapack routine
    // --------------------------------------------------------------
    for (k = 0; k < Niter; k++)
    {
        d[k] = creal(diag[k]);    // Supposed to be real
        e[k] = creal(offdiag[k]); // Supposed to be real
        for (j = 0; j < Niter; j++) eigvec[k * Niter + j] = 0;
    }

    k = LAPACKE_dstev(LAPACK_ROW_MAJOR, 'V', Niter, d, e, eigvec, Niter);
    if (k != 0)
    {
        printf("\n\n\t\tERROR IN DIAGONALIZATION\n\n");
        exit(EXIT_FAILURE);
    }



    sentinel = 1E10;
    // Get Index of smallest eigenvalue, keep it on j
    for (k = 0; k < Niter; k++)
    {
        if (sentinel > d[k]) { sentinel = d[k];   j = k; }
    }



    // Update C with the coefficients of ground state
    for (i = 0; i < nc; i++)
    {
        C[i] = 0;
        for (k = 0; k < Niter; k++) C[i] += lvec[k][i] * eigvec[k * Niter + j];
    }



    free(d);
    free(e);
    free(eigvec);
    free(diag);
    free(offdiag);
    free(Hint);
    cmatFree(Morb, Ho);
    cmatFree(predictedIter, lvec);

    return sentinel;
    
}










void LanczosIntegrator (MCTDHBsetup MC, Cmatrix Orb, Carray C, double complex dt)
{

//  Integrate Coefficients in a imaginary time-step dt using Lanczos
//  iteration to compute approximations to eigenvalues and then use
//  them to exact diagonalize in this subspace
//
//  INPUT/OUTPUT : C (advanced in dt)


    int
        i,
        k,
        j,
        lm,
        predictedIter,
        M = MC->Morb,
        Mpos = MC->Mpos,
        Npar = MC->Npar;

    lm = 4; // Follows the order of Runge-Kutta



    /* variables to call lapack diagonalization routine for tridiagonal
       symmetric matrix
    ---------------------------------------------------------------- */
    double
        * d = malloc(lm * sizeof(double)),
        * e = malloc(lm * sizeof(double)),
        * eigvec = malloc(lm * lm * sizeof(double));
    /* ------------------------------------------------------------- */



    /* variables to store lanczos vectors and matrix iterations
    -------------------------------------------------------- */
    // Lanczos Vectors (organize in rows instead of columns rows)
    Cmatrix lvec = cmatDef(lm, MC->nc);
    // Elements of tridiagonal lanczos matrix
    Carray diag = carrDef(lm);
    Carray offdiag = carrDef(lm);
    // Solve system of ODEs in lanczos vector space
    Carray Clanczos = carrDef(lm);
    Carray aux = carrDef(lm);
    /* ----------------------------------------------------- */



    /* One/Two-body Hamiltonian matrices elements
    ------------------------------------------ */
    Cmatrix  Ho = cmatDef(M, M);
    Carray Hint = carrDef(M * M * M *M);
    SetupHo(M, Mpos, Orb, MC->dx, MC->a2, MC->a1, MC->V, Ho);
    SetupHint(M, Mpos, Orb, MC->dx, MC->inter, Hint);
    /* --------------------------------------- */
    
    
    
    /* ---------------------------------------------
    Setup values needed to solve the equations for C
    ------------------------------------------------ */
    offdiag[lm-1] = 0; // Useless
    // Setup initial lanczos vector
    carrCopy(MC->nc, C, lvec[0]);
    /* --------------------------------------------- */



    /* ================================================================= *
    
            SOLVE ODE FOR COEFFICIENTS USING LANCZOS VECTOR SPACE

     * ================================================================= */



    /* --------------------------------------------------------------
    Call Lanczos what setup tridiagonal symmetric and lanczos vectors
    ----------------------------------------------------------------- */
    predictedIter = lm;
    lm = lanczos(MC, Ho, Hint, lm, diag, offdiag, lvec);
    /* -------------------------------------------------------------- */



    /* --------------------------------------------------------------
    Transfer data to use lapack routine
    ----------------------------------------------------------------- */
    for (k = 0; k < lm; k++)
    {
        d[k] = creal(diag[k]);    // Supposed to be real
        e[k] = creal(offdiag[k]); // Supposed to be real
        for (j = 0; j < lm; j++) eigvec[k * lm + j] = 0;
    }

    k = LAPACKE_dstev(LAPACK_ROW_MAJOR, 'V', lm, d, e, eigvec, lm);
    if (k != 0)
    {
        printf("\n\n\t\tERROR IN DIAGONALIZATION\n\n");
        exit(EXIT_FAILURE);
    }
    /* -------------------------------------------------------------- */



    /* --------------------------------------------------------------
    Solve exactly the equation in lanczos vector space using 
    matrix-eigenvalues to exactly exponentiate
    ----------------------------------------------------------------- */
    // Initial condition in Lanczos vector space
    carrFill(lm, 0, Clanczos); Clanczos[0] = 1.0;

    for (k = 0; k < lm; k++)
    {   // Solve in diagonal basis and for this apply eigvec trasformation
        aux[k] = 0;
        for (j = 0; j < lm; j++) aux[k] += eigvec[j*lm + k] * Clanczos[j];
        aux[k] = aux[k] * cexp(- I * d[k] * dt);
    }

    for (k = 0; k < lm; k++)
    {   // Backward transformation from diagonal matrix
        Clanczos[k] = 0;
        for (j = 0; j < lm; j++) Clanczos[k] += eigvec[k*lm + j] * aux[j];
    }

    for (i = 0; i < MC->nc; i++)
    {   // Matrix multiplication by lanczos vector give the solution
        C[i] = 0;
        for (j = 0; j < lm; j++) C[i] += lvec[j][i] * Clanczos[j];
    }
    /* -------------------------------------------------------------- */



    /* ================================================================= *
    
                                RELEASE MEMORY

     * ================================================================= */

    free(d);
    free(e);
    free(eigvec);
    free(diag);
    free(offdiag);
    free(Clanczos);
    free(aux);

    cmatFree(predictedIter, lvec);

    cmatFree(M, Ho);
    free(Hint);
}










void MC_NLTRAPC_IRK4 (MCTDHBsetup MC, MCTDHBmaster S, double complex dt)
{

//  (M)ulti-(C)onfiguration (N)on (L)inear part + (TRAP) potential
//  and  (C)oefficients  solver  for  (I)maginary  time-step  with
//  (R)unge-(K)utta method of 4-th order : MC_NLTRAPC_IRK4
//
//  INPUT/OUTPUT :
//      C   - End up advanced in dt
//      Orb - End up advanced in dt



    int
        i,
        k,
        j,
        nc,
        Morb,
        Mpos,
        Npar;

    double
        g,
        dx,
        a2,
        * V;

    double complex
        a1;

    Carray
        Ck,
        dCdt,
        Carg;

    Cmatrix
        Ok,
        dOdt,
        Oarg;

    nc = MC->nc;

    Morb = MC->Morb;

    Mpos = MC->Mpos;

    Npar = MC->Npar;

    a1 = MC->a1;

    a2 = MC->a2;

    dx = MC->dx;

    g = MC->inter;

    V = MC->V;

    Carg = carrDef(nc);

    dCdt = carrDef(nc);
    
    Ck = carrDef(nc);
    
    dOdt = cmatDef(Morb,Mpos);

    Oarg = cmatDef(Morb,Mpos);

    Ok = cmatDef(Morb,Mpos);



    // ------------------------------------------------------------------
    // COMPUTE K1
    // ------------------------------------------------------------------

    MCNLTRAP_dOdt(MC, S->Omat, dOdt, S->Ho, S->Hint, S->rho1, S->rho2);
    MC_dCdt(MC, S->C, S->Ho, S->Hint, dCdt);

    for (i = 0; i < nc; i++)
    {   // Add K1 contribution
        Ck[i] = dCdt[i];
        // Prepare next argument to compute K2
        Carg[i] = S->C[i] + dCdt[i] * 0.5 * dt;
    }

    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K1 contribution
            Ok[k][j] = dOdt[k][j];
            // Prepare next argument to compute K2
            Oarg[k][j] = S->Omat[k][j] + dOdt[k][j] * 0.5 * dt;
        }
    }



    // Update hamiltonian matrix elements
    SetupHo(Morb, Mpos, Oarg, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, Oarg, dx, g, S->Hint);
    // Update density matrices
    OBrho(Npar, Morb, MC->NCmat, MC->IF, Carg, S->rho1);
    TBrho(Npar, Morb, MC->NCmat, MC->IF, Carg, S->rho2);



    // ------------------------------------------------------------------
    // COMPUTE K2
    // ------------------------------------------------------------------

    MCNLTRAP_dOdt(MC, Oarg, dOdt, S->Ho, S->Hint, S->rho1, S->rho2);
    MC_dCdt(MC, Carg, S->Ho, S->Hint, dCdt);

    for (i = 0; i < nc; i++)
    {   // Add K2 contribution
        Ck[i] += 2 * dCdt[i];
        // Prepare next argument to compute K3
        Carg[i] = S->C[i] + dCdt[i] * 0.5 * dt;
    }

    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K2 contribution
            Ok[k][j] += 2 * dOdt[k][j];
            // Prepare next argument to compute K3
            Oarg[k][j] = S->Omat[k][j] + dOdt[k][j] * 0.5 * dt;
        }
    }



    SetupHo(Morb, Mpos, Oarg, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, Oarg, dx, g, S->Hint);
    // Update density matrices
    OBrho(Npar, Morb, MC->NCmat, MC->IF, Carg, S->rho1);
    TBrho(Npar, Morb, MC->NCmat, MC->IF, Carg, S->rho2);



    // ------------------------------------------------------------------
    // COMPUTE K3
    // ------------------------------------------------------------------

    MCNLTRAP_dOdt(MC, Oarg, dOdt, S->Ho, S->Hint, S->rho1, S->rho2);
    MC_dCdt(MC, Carg, S->Ho, S->Hint, dCdt);

    for (i = 0; i < nc; i++)
    {   // Add K3 contribution
        Ck[i] += 2 * dCdt[i];
        // Prepare next argument to compute K4
        Carg[i] = S->C[i] + dCdt[i] * dt;
    }

    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K3 contribution
            Ok[k][j] += 2 * dOdt[k][j];
            // Prepare next argument to compute K4
            Oarg[k][j] = S->Omat[k][j] + dOdt[k][j] * dt;
        }
    }



    SetupHo(Morb, Mpos, Oarg, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, Oarg, dx, g, S->Hint);
    // Update density matrices
    OBrho(Npar, Morb, MC->NCmat, MC->IF, Carg, S->rho1);
    TBrho(Npar, Morb, MC->NCmat, MC->IF, Carg, S->rho2);



    // ------------------------------------------------------------------
    // COMPUTE K4
    // ------------------------------------------------------------------

    MCNLTRAP_dOdt(MC, Oarg, dOdt, S->Ho, S->Hint, S->rho1, S->rho2);
    MC_dCdt(MC, Carg, S->Ho, S->Hint, dCdt);

    for (i = 0; i < nc; i++)
    {   // Add K4 contribution
        Ck[i] += dCdt[i];
    }

    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K4 contribution
            Ok[k][j] += dOdt[k][j];
        }
    }

    // Until now Ck and Ok holds the sum K1 + 2 * K2 + 2 * K3 + K4
    // from the Fourth order Runge-Kutta algorithm.  Then update :

    for (i = 0; i < nc; i++)
    {   // Update Coeficients
        S->C[i] = S->C[i] + Ck[i] * dt / 6;
    }

    for (k = 0; k < Morb; k++)
    {   // Update Orbitals
        for (j = 0; j < Mpos; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt / 6;
        }
    }



    free(Ck);
    free(dCdt);
    free(Carg);

    cmatFree(Morb, dOdt);
    cmatFree(Morb, Ok);
    cmatFree(Morb, Oarg);

}









/*
void MC_NL_IRK4 (MCTDHBsetup MC, Cmatrix Orb, Carray C, double complex dt)
{

//  (M)ulti-(C)onfiguration (N)on-(L)inear part solver for  (I)maginary
//  time-step with (R)unge-(K)utta method of 4-th order : MC_NL_IRK4
//
//  INPUT/OUTPUT : Orb - end up advanced in dt



    int
        i,
        k,
        j,
        M = MC->Morb,
        Mpos = MC->Mpos;

    Cmatrix dOdt = cmatDef(M, Mpos);
    Cmatrix Ok = cmatDef(M, Mpos);
    Cmatrix Oarg = cmatDef(M, Mpos);

    Cmatrix  Ho = cmatDef(M, M);
    Carray Hint = carrDef(M * M * M *M);



    SetupHo(M, Mpos, Orb, MC->dx, MC->a2, MC->a1, MC->V, Ho);
    SetupHint(M, Mpos, Orb, MC->dx, MC->inter, Hint);

    // ------------------------------------------------------------------
    // COMPUTE K1
    // ------------------------------------------------------------------

    MCNL_dOdt(MC, C, Orb, dOdt, Ho, Hint);
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K1 contribution
            Ok[k][j] = dOdt[k][j];
            // Prepare next argument to compute K2
            Oarg[k][j] = Orb[k][j] + dOdt[k][j] * 0.5 * dt;
        }
    }



    SetupHo(M, Mpos, Oarg, MC->dx, MC->a2, MC->a1, MC->V, Ho);
    SetupHint(M, Mpos, Oarg, MC->dx, MC->inter, Hint);

    // ------------------------------------------------------------------
    // COMPUTE K2
    // ------------------------------------------------------------------

    MCNL_dOdt(MC, C, Oarg, dOdt, Ho, Hint);
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K2 contribution
            Ok[k][j] += 2 * dOdt[k][j];
            // Prepare next argument to compute K3
            Oarg[k][j] = Orb[k][j] + dOdt[k][j] * 0.5 * dt;
        }
    }



    SetupHo(M, Mpos, Oarg, MC->dx, MC->a2, MC->a1, MC->V, Ho);
    SetupHint(M, Mpos, Oarg, MC->dx, MC->inter, Hint);

    // ------------------------------------------------------------------
    // COMPUTE K3
    // ------------------------------------------------------------------

    MCNL_dOdt(MC, C, Oarg, dOdt, Ho, Hint);
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K3 contribution
            Ok[k][j] += 2 * dOdt[k][j];
            // Prepare next argument to compute K4
            Oarg[k][j] = Orb[k][j] + dOdt[k][j] * dt;
        }
    }



    SetupHo(M, Mpos, Oarg, MC->dx, MC->a2, MC->a1, MC->V, Ho);
    SetupHint(M, Mpos, Oarg, MC->dx, MC->inter, Hint);

    // ------------------------------------------------------------------
    // COMPUTE K4
    // ------------------------------------------------------------------

    MCNL_dOdt(MC, C, Oarg, dOdt, Ho, Hint);
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K4 contribution
            Ok[k][j] += dOdt[k][j];
        }
    }



    // Until now Ok holds the sum K1 + 2 * K2 + 2 * K3 + K4
    // from the Fourth order Runge-Kutta algorithm.  Then :
    for (k = 0; k < M; k++)
    {   // Update Orbitals
        for (j = 0; j < Mpos; j++)
        {
            Orb[k][j] = Orb[k][j] + Ok[k][j] * dt / 6;
        }
    }

    cmatFree(M, dOdt);
    cmatFree(M, Ok);
    cmatFree(M, Oarg);

    cmatFree(M, Ho);
    free(Hint);
}
*/










void MC_NLC_IRK4 (MCTDHBsetup MC, MCTDHBmaster S, double complex dt)
{

//  (M)ulti-(C)onfiguration (N)on (L)inear part and  (C)oefficients
//  solver for (I)maginary time-step with (R)unge-(K)utta method of
//  4-th order : MC_NLC_IRK4
//
//  INPUT/OUTPUT :
//      C   - End up advanced in dt
//      Orb - End up advanced in dt



    int
        i,
        k,
        j,
        nc,
        Morb,
        Mpos,
        Npar;

    double
        g,
        dx,
        a2,
        * V;

    double complex
        a1;

    Carray
        Ck,
        dCdt,
        Carg;

    Cmatrix
        Ok,
        dOdt,
        Oarg;

    nc = MC->nc;

    Morb = MC->Morb;

    Mpos = MC->Mpos;

    Npar = MC->Npar;

    a1 = MC->a1;

    a2 = MC->a2;

    dx = MC->dx;

    g = MC->inter;

    V = MC->V;

    Carg = carrDef(nc);

    dCdt = carrDef(nc);
    
    Ck = carrDef(nc);
    
    dOdt = cmatDef(Morb,Mpos);

    Oarg = cmatDef(Morb,Mpos);

    Ok = cmatDef(Morb,Mpos);



    // ------------------------------------------------------------------
    // COMPUTE K1 and add its contribution in Ck and Ok
    // ------------------------------------------------------------------

    MCNL_dOdt(MC, S->Omat, dOdt, S->Ho, S->Hint, S->rho1, S->rho2);
    MC_dCdt(MC, S->C, S->Ho, S->Hint, dCdt);

    for (i = 0; i < nc; i++)
    {   // Add K1 contribution
        Ck[i] = dCdt[i];
        // Prepare next argument to compute K2
        Carg[i] = S->C[i] + dCdt[i] * 0.5 * dt;
    }

    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K1 contribution
            Ok[k][j] = dOdt[k][j];
            // Prepare next argument to compute K2
            Oarg[k][j] = S->Omat[k][j] + dOdt[k][j] * 0.5 * dt;
        }
    }



    SetupHo(Morb, Mpos, Oarg, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, Oarg, dx, g, S->Hint);
    // Update density matrices
    OBrho(Npar, Morb, MC->NCmat, MC->IF, Carg, S->rho1);
    TBrho(Npar, Morb, MC->NCmat, MC->IF, Carg, S->rho2);



    // ------------------------------------------------------------------
    // COMPUTE K2 and add its contribution in Ck and Ok
    // ------------------------------------------------------------------

    MCNL_dOdt(MC, Oarg, dOdt, S->Ho, S->Hint, S->rho1, S->rho2);
    MC_dCdt(MC, Carg, S->Ho, S->Hint, dCdt);

    for (i = 0; i < nc; i++)
    {   // Add K2 contribution
        Ck[i] += 2 * dCdt[i];
        // Prepare next argument to compute K3
        Carg[i] = S->C[i] + dCdt[i] * 0.5 * dt;
    }

    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K2 contribution
            Ok[k][j] += 2 * dOdt[k][j];
            // Prepare next argument to compute K3
            Oarg[k][j] = S->Omat[k][j] + dOdt[k][j] * 0.5 * dt;
        }
    }



    SetupHo(Morb, Mpos, Oarg, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, Oarg, dx, g, S->Hint);
    // Update density matrices
    OBrho(Npar, Morb, MC->NCmat, MC->IF, Carg, S->rho1);
    TBrho(Npar, Morb, MC->NCmat, MC->IF, Carg, S->rho2);



    // ------------------------------------------------------------------
    // COMPUTE K3 and add its contribution in Ck and Ok
    // ------------------------------------------------------------------

    MCNL_dOdt(MC, Oarg, dOdt, S->Ho, S->Hint, S->rho1, S->rho2);
    MC_dCdt(MC, Carg, S->Ho, S->Hint, dCdt);

    for (i = 0; i < nc; i++)
    {   // Add K3 contribution
        Ck[i] += 2 * dCdt[i];
        // Prepare next argument to compute K4
        Carg[i] = S->C[i] + dCdt[i] * dt;
    }

    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K3 contribution
            Ok[k][j] += 2 * dOdt[k][j];
            // Prepare next argument to compute K4
            Oarg[k][j] = S->Omat[k][j] + dOdt[k][j] * dt;
        }
    }



    SetupHo(Morb, Mpos, Oarg, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, Oarg, dx, g, S->Hint);
    // Update density matrices
    OBrho(Npar, Morb, MC->NCmat, MC->IF, Carg, S->rho1);
    TBrho(Npar, Morb, MC->NCmat, MC->IF, Carg, S->rho2);



    // ------------------------------------------------------------------
    // COMPUTE K4 and add its contribution in Ck and Ok
    // ------------------------------------------------------------------

    MCNL_dOdt(MC, Oarg, dOdt, S->Ho, S->Hint, S->rho1, S->rho2);
    MC_dCdt(MC, Carg, S->Ho, S->Hint, dCdt);

    for (i = 0; i < nc; i++)
    {   // Add K4 contribution
        Ck[i] += dCdt[i];
    }

    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {   // Add K4 contribution
            Ok[k][j] += dOdt[k][j];
        }
    }



    // Until now Ok and Ck holds the sum K1 + 2 * K2 + 2 * K3 + K4
    // from the Fourth order Runge-Kutta algorithm.  Then update :

    for (i = 0; i < nc; i++)
    {   // Update Coeficients
        S->C[i] = S->C[i] + Ck[i] * dt / 6;
    }

    for (k = 0; k < Morb; k++)
    {   // Update Orbitals
        for (j = 0; j < Mpos; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt / 6;
        }
    }

    free(Ck);
    free(dCdt);
    free(Carg);
    
    cmatFree(Morb, dOdt);
    cmatFree(Morb, Ok);
    cmatFree(Morb, Oarg);
}










void MCLP_CNSM (int Mpos, int Morb, CCSmat cnmat, Carray upper,
     Carray lower, Carray mid, Cmatrix Orb)
{

//  (M)ulti-(C)onfiguration  (L)inear  (P)art  solver by
//  (C)rank-(N)icolson with (S)herman-(M)orrison formula
//  to a cyclic-tridiagonal system : MCLP_CNSM
//  ----------------------------------------------------
//  Given a complex matrix with orbitals  organized  in
//  each  row,  Solve  cyclic-tridiagonal  system  that
//  arises from Crank-Nicolson finite difference scheme
//  with the discretization matrix to multiply RHS in a
//  Compressed-Column Storage format
//
//  INPUT/OUTPUT PARAMETER : Orb

    int
        k,
        size = Mpos - 1;

    Carray
        rhs = carrDef(size);

    for (k = 0; k < Morb; k++)
    {   // For each orbital k solve a tridiagonal system obtained by CN
        CCSvec(size, cnmat->vec, cnmat->col, cnmat->m, Orb[k], rhs);
        triCyclicSM(size, upper, lower, mid, rhs, Orb[k]);
    }

    free(rhs);
}










void MCLP_CNLU (int Mpos, int Morb, CCSmat cnmat, Carray upper, Carray lower,
     Carray mid, Cmatrix Orb )
{

//  (M)ulti-(C)onfiguration  (L)inear  (P)art  solver by
//  (C)rank-(N)icolson with (LU) decomposition: MCLPCNLU
//  ----------------------------------------------------
//  Given a complex matrix with orbitals  organized  in
//  each  row,  Solve  cyclic-tridiagonal  system  that
//  arises from Crank-Nicolson finite difference scheme
//  with the discretization matrix to multiply RHS in a
//  Compressed-Column Storage format
//
//  INPUT/OUTPUT PARAMETER : Orb

    int
        k,
        size = Mpos - 1;

    Carray
        rhs;

    #pragma omp parallel for private(k, rhs)
    for (k = 0; k < Morb; k++)
    {   // For each orbital k solve a tridiagonal system obtained by CN
        rhs = carrDef(size);
        CCSvec(size, cnmat->vec, cnmat->col, cnmat->m, Orb[k], rhs);
        triCyclicLU(size, upper, lower, mid, rhs, Orb[k]);
        free(rhs);
    }

}










void MCLP_FFT (int Mpos, int Morb, DFTI_DESCRIPTOR_HANDLE * desc,
     Carray exp_der, Cmatrix Orb)
{

//  (M)ulti-(C)onfiguration (L)inear (P)art solver by
//  (F)ast (F)ourier (T)ransform : MCLPFFT
//  --------------------------------------------------
//  Given a complex matrix with orbitals organized  in
//  each row, apply exponential of derivative operator
//  whose is part of split-step formal solution.
//
//  INPUT/OUTPUT PARAMETER : Orb

    int
        k;

    MKL_LONG
        s;

    Carray
        forward_fft = carrDef(Mpos - 1),
        back_fft    = carrDef(Mpos - 1);

    for (k = 0; k < Morb; k++)
    {
        carrCopy(Mpos - 1, Orb[k], forward_fft);
        s = DftiComputeForward( (*desc), forward_fft );
        // Apply Exp. derivative operator in momentum space
        carrMultiply(Mpos - 1, exp_der, forward_fft, back_fft);
        // Go back to position space
        s = DftiComputeBackward( (*desc), back_fft );
        carrCopy(Mpos - 1, back_fft, Orb[k]);
        // last point assumed as cyclic boundary
        Orb[k][Mpos-1] = Orb[k][0];
    }

    free(forward_fft);
    free(back_fft);
}










    /* =============================================================


                         MAIN INTEGRATOR ROUTINES


       ============================================================= */










int MC_IMAG_RK4_FFTRK4 (MCTDHBsetup MC, MCTDHBmaster S, Carray E,
    Carray virial, double dT, int Nsteps)
{

/** Multi-Configuration Imaginary time propagation
    ==============================================


    Methods
    -------

    Configuration Coefficients Integrator : 4-th order Runge-Kutta

    Orbitals Integrator : Split-Step with FFT(linear)
    and 4-th order Runge-Kutta(nonlinear)


    Description
    -----------

    Evolve half step linear part, then full step nonlinear part together
    with coefficients and another half step linear part
**/



    int i,
        j,
        k,
        m,
        nc = MC->nc,
        Npar = MC->Npar,
        Mpos = MC->Mpos,
        Morb = MC->Morb;

    MKL_LONG
        p;

    double
        R2,
        freq,
        Idt = - dT,
        dx = MC->dx,
        a2 = MC->a2,
        g = MC->inter,
        * V = MC->V;

    double complex
        a1 = MC->a1,
        dt = - I * dT;

    Carray
        exp_der = carrDef(Mpos - 1);





    m = Mpos - 1; // Size of arrays to take the Fourier-Transform

    // setup descriptor (MKL implementation of FFT)
    // -------------------------------------------------------------------
    DFTI_DESCRIPTOR_HANDLE desc;
    p = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, m);
    p = DftiSetValue(desc, DFTI_FORWARD_SCALE, 1.0 / sqrt(m));
    p = DftiSetValue(desc, DFTI_BACKWARD_SCALE, 1.0 / sqrt(m));
    p = DftiCommitDescriptor(desc);
    // -------------------------------------------------------------------



    // Exponential of derivative operator in momentum space
    // -------------------------------------------------------------------
    for (i = 0; i < m; i++)
    {
        if (i <= (m - 1) / 2) { freq = (2 * PI * i) / (m * dx);       }
        else                  { freq = (2 * PI * (i - m)) / (m * dx); }
        // exponential of derivative operators in half time-step
        exp_der[i] = cexp( -0.5 * dT * (I * a1 * freq - a2 * freq * freq) );
    }
    // -------------------------------------------------------------------





    // Setup one/two-body hamiltonian matrix elements
    SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

    // Setup one/two-body density matrix
    OBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho1);
    TBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho2);



    // Store the initial energy
    E[0] = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint);
    virial[0] = Virial(MC, S->Omat, S->rho1, S->rho2);
    R2 = MCMeanQuadraticR(MC, S->Omat, S->rho1);

    printf("\n\n\t Nstep         Energy/particle         Virial");
    printf("               sqrt<R^2>");
    SepLine();
    printf("\n\t%6d       %15.7E", 0, creal(E[0]));
    printf("         %15.7E       %7.4lf", creal(virial[0]), R2);



    for (i = 0; i < Nsteps; i++)
    {

        MCLP_FFT(Mpos, Morb, &desc, exp_der, S->Omat);

        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

        MC_NLTRAPC_IRK4(MC, S, dt);

        MCLP_FFT(Mpos, Morb, &desc, exp_der, S->Omat);



        // Loss of Norm => undefined behavior on orthogonality
        Ortonormalize(Morb, Mpos, dx, S->Omat);

        // Renormalize coeficients
        renormalizeVector(nc, S->C, 1.0);



        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        OBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho1);
        TBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho2);



        if ( i == Nsteps / 2 )
        {
            // After half of time steps that must be evolved do a
            // fixed orbital  basis  diagonalization  to  quicker
            // convergence. Restrict the number of iterations  in
            // lanczos routine to avoid massive memory usage. Try
            // to use 200 iterations unless either it exceeds half
            // of the dimension of configuration space or if it
            // exceeds a memory Threshold.

            if (200 * nc < 5E7)
            {
                if (2 * nc / 3 < 200) k = 2 * nc / 3;
                else                  k = 200;
            }
            else k = 5E7 / nc;

            E[i + 1] = LanczosGround( k, MC, S->Omat, S->C );
            // Renormalize coeficients
            renormalizeVector(nc, S->C, 1.0);

            OBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho1);
            TBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho2);

            SepLine();
            printf("\n\tDiagonalization Done E = %.7E\n", creal(E[i+1]));
            SepLine();
        }



        // Store energy
        E[i + 1] = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint);
        virial[i + 1] = Virial(MC, S->Omat, S->rho1, S->rho2);
        R2 = MCMeanQuadraticR(MC, S->Omat, S->rho1);



        printf("\n\t%6d       %15.7E", i + 1, creal(E[i + 1]));
        printf("         %15.7E       %7.4lf", creal(virial[i+1]), R2);

        j = i - 199;
        if (j > 0 && fabs( creal(E[i+1] - E[j]) / creal(E[j]) ) < 7E-11)
        {

            p = DftiFreeDescriptor(&desc);
            free(exp_der);

            if (200 * nc < 5E7)
            {
                if (2 * nc / 3 < 200) k = 2 * nc / 3;
                else                  k = 200;
            }
            else k = 5E7 / nc;

            E[i + 1] = LanczosGround( k, MC, S->Omat, S->C );
            // Renormalize coeficients
            renormalizeVector(nc, S->C, 1.0);

            SepLine();
            printf("\n\tDiagonalization Done E = %.7E\n", creal(E[i+1]));
            SepLine();

            printf("\nProcess ended before because \n");
            printf("\n\t1. Energy stop decreasing  \n");

            if ( fabs( creal(virial[i+1]) / creal(E[i+1]) ) < 1E-3 )
            {
                printf("\n\t2. Achieved virial accuracy\n\n");
                return i + 1;
            } else
            {
                printf("\n\t2. Not so good virial value ");
                printf("achieved. Try smaller time-step\n\n");
            }

            return i + 1;
        }

    }

    if (200 * nc < 5E7)
    {
        if (2 * nc / 3 < 200) k = 2 * nc / 3;
        else                  k = 200;
    }
    else k = 5E7 / nc;

    E[Nsteps] = LanczosGround( k, MC, S->Omat, S->C );
    // Renormalize coeficients
    renormalizeVector(nc, S->C, 1.0);
    
    SepLine();
    printf("\nProcess ended without achieving");
    printf(" stability and/or accuracy\n\n");
    
    p = DftiFreeDescriptor(&desc);
    free(exp_der);


    return Nsteps + 1;
}










int MC_IMAG_RK4_CNSMRK4 (MCTDHBsetup MC, MCTDHBmaster S, Carray E,
     Carray virial, double dT, int Nsteps, int cyclic)
{

/** Multi-Configuration Imaginary time propagation
    ==============================================


    Methods
    -------

    Configuration Coefficients Integrator : 4-th order Runge-Kutta

    Orbitals Integrator : Split-Step with Crank-Nicolson(linear)
    with Sherman-Morrison and 4-th order  Runge-Kutta(nonlinear)


    Description
    -----------

    Evolve half step linear part, then full step nonlinear part together
    with coefficients and another half step linear part

**/



    int i,
        j,
        k,
        nc = MC->nc,
        Npar = MC->Npar,
        Mpos = MC->Mpos,
        Morb = MC->Morb;

    double
        R2,
        dx = MC->dx,
        a2 = MC->a2,
        g = MC->inter,
        * V = MC->V;

    double complex
        a1 = MC->a1,
        dt = - I * dT;

    Carray
        upper  = carrDef(Mpos - 1),
        lower  = carrDef(Mpos - 1),
        mid    = carrDef(Mpos - 1);

    CCSmat
        cnmat;





    // Setup one/two-body hamiltonian matrix elements
    SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

    // Setup one/two-body density matrix
    OBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho1);
    TBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho2);



    // Store the initial energy
    E[0] = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint);
    virial[0] = Virial(MC, S->Omat, S->rho1, S->rho2);
    R2 = MCMeanQuadraticR(MC, S->Omat, S->rho1);
    
    printf("\n\n\t Nstep         Energy/particle         Virial");
    printf("               sqrt<R^2>");
    SepLine();
    printf("\n\t%6d       %15.7E", 0, creal(E[0]));
    printf("         %15.7E       %7.4lf", creal(virial[0]), R2);



    // Configure the linear system from Crank-Nicolson scheme
    cnmat = CNmat(Mpos, dx, dt/2, a2, a1, g, V, cyclic, upper, lower, mid);



    for (i = 0; i < Nsteps; i++)
    {

        MCLP_CNSM(Mpos, Morb, cnmat, upper, lower, mid, S->Omat);

        // The boundary
        if (cyclic)
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = S->Omat[k][0]; }
        else
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = 0;             }

        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);



        MC_NLC_IRK4(MC, S, dt);



        MCLP_CNSM(Mpos, Morb, cnmat, upper, lower, mid, S->Omat);

        // The boundary
        if (cyclic)
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = S->Omat[k][0]; }
        else
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = 0;             }



        // Renormalize coeficients
        renormalizeVector(nc, S->C, 1.0);



        // Loss of Norm => undefined behavior on orthogonality
        Ortonormalize(Morb, Mpos, dx, S->Omat);



        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        OBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho1);
        TBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho2);



        if ( i == Nsteps / 2 )
        {
            // After half of time steps that must be evolved do a
            // fixed orbital  basis  diagonalization  to  quicker
            // convergence. Restrict the number of iterations  in
            // lanczos routine to avoid massive memory usage. Try
            // to use 200 iterations unless either it exceeds half
            // of the dimension of configuration space or if it
            // exceeds a memory Threshold.

            if (200 * nc < 5E7)
            {
                if (2 * nc / 3 < 200) k = 2 * nc / 3;
                else                  k = 200;
            }
            else k = 5E7 / nc;

            E[i + 1] = LanczosGround( k, MC, S->Omat, S->C );
            // Renormalize coeficients
            renormalizeVector(nc, S->C, 1.0);

            OBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho1);
            TBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho2);

            SepLine();
            printf("\n\tDiagonalization Done E = %.7E\n", creal(E[i+1]));
            SepLine();
        }



        // Store energy
        E[i + 1] = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint);
        virial[i + 1] = Virial(MC, S->Omat, S->rho1, S->rho2);
        R2 = MCMeanQuadraticR(MC, S->Omat, S->rho1);



        printf("\n\t%6d       %15.7E", i + 1, creal(E[i + 1]));
        printf("         %15.7E       %7.4lf", creal(virial[i+1]), R2);

        j = i - 199;
        if (j > 0 && fabs( creal(E[i+1] - E[j]) / creal(E[j]) ) < 7E-11)
        {

            CCSFree(cnmat);
            free(upper);
            free(lower);
            free(mid);

            if (200 * nc < 5E7)
            {
                if (2 * nc / 3 < 200) k = 2 * nc / 3;
                else                  k = 200;
            }
            else k = 5E7 / nc;

            E[i + 1] = LanczosGround( k, MC, S->Omat, S->C );
            // Renormalize coeficients
            renormalizeVector(nc, S->C, 1.0);

            SepLine();
            printf("\n\tDiagonalization Done E = %.7E\n", creal(E[i+1]));
            SepLine();

            printf("\nProcess ended before because \n");
            printf("\n\t1. Energy stop decreasing  \n");

            if ( fabs( creal(virial[i+1]) / creal(E[i+1]) ) < 1E-3 )
            {
                printf("\n\t2. Achieved virial accuracy\n\n");
            } else
            {
                printf("\n\t2. Not so good virial value ");
                printf("achieved. Try smaller time-step\n\n");
            }

            return i + 1;
        }
    }

    if (200 * nc < 5E7)
    {
        if (2 * nc / 3 < 200) k = 2 * nc / 3;
        else                  k = 200;
    }
    else k = 5E7 / nc;

    E[Nsteps] = LanczosGround( k, MC, S->Omat, S->C );
    // Renormalize coeficients
    renormalizeVector(nc, S->C, 1.0);

    SepLine();
    printf("\nProcess ended without achieving");
    printf(" stability and/or accuracy\n\n");

    CCSFree(cnmat);
    free(upper);
    free(lower);
    free(mid);

    return Nsteps + 1;
}










int MC_IMAG_RK4_CNLURK4 (MCTDHBsetup MC, MCTDHBmaster S, Carray E,
    Carray virial, double dT, int Nsteps, int cyclic)
{

/** Multi-Configuration Imaginary time propagation
    ==============================================


    Methods
    -------

    Configuration Coefficients Integrator : 4-th order Runge-Kutta

    Orbitals Integrator : Split-Step with Crank-Nicolson(linear)
    with Sherman-Morrison and 4-th order  Runge-Kutta(nonlinear)


    Description
    -----------

    Evolve half step linear part, then full step nonlinear part together
    with coefficients and another half step linear part */



    int i,
        j,
        k,
        nc = MC->nc,
        Npar = MC->Npar,
        Mpos = MC->Mpos,
        Morb = MC->Morb;

    double
        R2,
        dx = MC->dx,
        a2 = MC->a2,
        g = MC->inter,
        * V = MC->V;

    double complex
        a1 = MC->a1,
        dt = - I * dT;

    Carray
        upper  = carrDef(Mpos - 1),
        lower  = carrDef(Mpos - 1),
        mid    = carrDef(Mpos - 1);

    CCSmat
        cnmat;



    // Setup one/two-body hamiltonian matrix elements
    SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

    // Setup one/two-body density matrix
    OBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho1);
    TBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho2);



    // Store the initial energy
    E[0] = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint);
    virial[0] = Virial(MC, S->Omat, S->rho1, S->rho2);
    R2 = MCMeanQuadraticR(MC, S->Omat, S->rho1);

    printf("\n\n\t Nstep         Energy/particle         Virial");
    printf("               sqrt<R^2>");
    SepLine();
    printf("\n\t%6d       %15.7E", 0, creal(E[0]));
    printf("         %15.7E       %7.4lf", creal(virial[0]), R2);



    // Configure the linear system from Crank-Nicolson scheme
    cnmat = CNmat(Mpos, dx, dt/2, a2, a1, g, V, cyclic, upper, lower, mid);



    for (i = 0; i < Nsteps; i++)
    {

        MCLP_CNLU(Mpos, Morb, cnmat, upper, lower, mid, S->Omat);

        // The boundary
        if (cyclic)
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = S->Omat[k][0]; }
        else
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = 0;             }



        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);



        MC_NLC_IRK4(MC, S, dt);



        MCLP_CNLU(Mpos, Morb, cnmat, upper, lower, mid, S->Omat);

        // The boundary
        if (cyclic)
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = S->Omat[k][0]; }
        else
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = 0;             }



        // Renormalize coeficients
        renormalizeVector(nc, S->C, 1.0);



        // Loss of Norm => undefined behavior on orthogonality
        Ortonormalize(Morb, Mpos, dx, S->Omat);



        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        OBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho1);
        TBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho2);



        if ( i == Nsteps / 2 )
        {
            // After half of time steps that must be evolved do a
            // fixed orbital  basis  diagonalization  to  quicker
            // convergence. Restrict the number of iterations  in
            // lanczos routine to avoid massive memory usage. Try
            // to use 200 iterations unless either it exceeds half
            // of the dimension of configuration space or if it
            // exceeds a memory Threshold.

            if (200 * nc < 5E7)
            {
                if (2 * nc / 3 < 200) k = 2 * nc / 3;
                else                  k = 200;
            }
            else k = 5E7 / nc;

            E[i + 1] = LanczosGround( k, MC, S->Omat,S->C );
            // Renormalize coeficients
            renormalizeVector(nc, S->C, 1.0);
            
            OBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho1);
            TBrho(Npar, Morb, MC->NCmat, MC->IF, S->C, S->rho2);

            SepLine();
            printf("\n\tDiagonalization Done E = %.7E\n", creal(E[i+1]));
            SepLine();
        }



        // Store energy
        E[i + 1] = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint);
        virial[i + 1] = Virial(MC, S->Omat, S->rho1, S->rho2);
        R2 = MCMeanQuadraticR(MC, S->Omat, S->rho1);



        printf("\n\t%6d       %15.7E", i + 1, creal(E[i + 1]));
        printf("         %15.7E       %7.4lf", creal(virial[i+1]), R2);
        
        
        
        j = i - 199;
        if (j > 0 && fabs( creal(E[i+1] - E[j]) / creal(E[j]) ) < 7E-11)
        {

            CCSFree(cnmat);
            free(upper);
            free(lower);
            free(mid);

            if (200 * nc < 5E7)
            {
                if (2 * nc / 3 < 200) k = 2 * nc / 3;
                else                  k = 200;
            }
            else k = 5E7 / nc;

            E[i + 1] = LanczosGround( k, MC, S->Omat, S->C );
            // Renormalize coeficients
            renormalizeVector(nc, S->C, 1.0);

            SepLine();
            printf("\n\tDiagonalization Done E = %.7E\n", creal(E[i+1]));
            SepLine();

            printf("\nProcess ended before because \n");
            printf("\n\t1. Energy stop decreasing  \n");

            if ( fabs( creal(virial[i+1]) / creal(E[i+1]) ) < 1E-3 )
            {
                printf("\n\t2. Achieved virial accuracy\n\n");
            } else
            {
                printf("\n\t2. Not so good virial value ");
                printf("achieved. Try smaller time-step\n\n");
            }
            
            return i + 1;
        }
    }

    if (200 * nc < 5E7)
    {
        if (2 * nc / 3 < 200) k = 2 * nc / 3;
        else                  k = 200;
    }
    else k = 5E7 / nc;

    E[Nsteps] = LanczosGround( k, MC, S->Omat, S->C );
    // Renormalize coeficients
    renormalizeVector(nc, S->C, 1.0);

    SepLine();
    printf("\nProcess ended without achieving");
    printf(" stability and/or accuracy\n\n");

    CCSFree(cnmat);
    free(upper);
    free(lower);
    free(mid);

    return Nsteps + 1;
}
