#include "coeffIntegration.h"

void dCdt (EqDataPkg MC, Carray C, Cmatrix Ho, Carray Hint, Carray der)
{

/** TIME DERIVATIVE OF CONFIGURATIONAL COEFFICIENTS **/

    int 
        i,
        Npar = MC->Npar,
        Morb = MC->Morb;

    Iarray
        IF = MC->IF,
        map1 = MC->Map,
        map12 = MC->MapOT,
        map22 = MC->MapTT,
        s12 = MC->strideOT,
        s22 = MC->strideTT;

    applyHconf(Npar,Morb,map1,map12,map22,s12,s22,IF,C,Ho,Hint,der);

    for (i = 0; i < MC->nc; i++) der[i] = - I * der[i];
}



int lanczos(EqDataPkg MCdata, Cmatrix Ho, Carray Hint,
    int lm, Carray diag, Carray offdiag, Cmatrix lvec)
{

/** LANCZOS TRIDIAGONAL REDUCTION FOR MULTICONFIGURATIONAL
    ======================================================
    Given the routine to apply the many-particle  Hamiltonian  in  the
    configuration basis, build the Lanczos vectors and two vector with
    diagonal and off-diagonal elements of the reduced tridiagonal form

    It is an implementation with full re-orthogonalization, improvement
    done to minimize numerical errors in floating point  arithmetic, to
    avoid loss of orthogonality among eigenvectors For more information
    check out:

    "Lectures on solving large scale eigenvalue problem", Peter Arbenz,
    ETH Zurich, 2006. url : http://people.inf.ethz.ch/arbenz/ewp/

    and other references there mentioned.

    INPUT PARAMETERS :
        lvec[0]  - Contains the initial vector of Lanczos algorithm
        Ho, Hint - additional parameters to apply Hamiltonian
        MCdata   - structure paramenters that setup the configurational space

    OUTPUT PARAMETERS :
        lvec    - Lanczos vectors
        diag    - diagonal elements of tridiagonal reduction
        offdiag - symmetric off-diagonal elements of tridiagonal reduction

    RETURN :
        number of itertion successfully done ( = lm if no breakdown occurs) **/

    int i,
        j,
        k,
        nc = MCdata->nc,
        Npar = MCdata->Npar,
        Morb = MCdata->Morb;

    Iarray
        IF = MCdata->IF,
        map1 = MCdata->Map,
        map12 = MCdata->MapOT,
        map22 = MCdata->MapTT,
        s12 = MCdata->strideOT,
        s22 = MCdata->strideTT;

    double
        tol,
        maxCheck;

    Carray
        out = carrDef(nc),
        ortho = carrDef(lm);



    // Variables to check for a source of breakdown or numerical instability
    maxCheck = 0;
    tol = 1E-14;



    // Compute the first diagonal element of the resulting tridiagonal
    // matrix outside the main loop because there is a different  rule
    // that came up from Modified Gram-Schmidt orthogonalization in
    // Krylov space
    applyHconf(Npar,Morb,map1,map12,map22,s12,s22,IF,lvec[0],Ho,Hint,out);
    diag[0] = carrDot(nc, lvec[0], out);

    for (i = 0; i < lm - 1; i++)
    {

        for (j = 0; j < nc; j++) out[j] = out[j] - diag[i] * lvec[i][j];
        // in the line above 'out' holds a new Lanczos vector but not
        // normalized, just orthogonal to the previous ones in 'exact
        // arithmetic'.



        // Additional re-orthogonalization procedure.  The Lanczos vectors
        // are suppose to form an orthonormal set, and thus when organized
        // in a matrix it forms an unitary  transformation up to numerical
        // precision 'Q'. Thus in addition, we subtract the QQ† applied to
        // the unnormalized new Lanczos vector we get above. Note that here
        // lvec has Lanczos vector organized by rows.
        for (k = 0; k < i + 1; k++) ortho[k] = carrDot(nc, lvec[k], out);

        for (j = 0; j < nc; j++)
        {
            for (k = 0; k < i + 1; k++) out[j] -= lvec[k][j] * ortho[k];
        }

        offdiag[i] = carrMod(nc, out);



        // Check up to numerical precision if it is safe  to  continue
        // This is equivalent to find a null vector and thus the basis
        // of Lanczos vectors from the initial guess given is  said to
        // have an invariant subspace of the operator
        if (maxCheck < creal(offdiag[i])) maxCheck = creal(offdiag[i]);
        if (creal(offdiag[i]) / maxCheck < tol) return (i + 1);



        // Compute new Lanczos vector by normalizing the orthogonal vector
        carrScalarMultiply(nc, out, 1.0 / offdiag[i], lvec[i + 1]);



        // Perform half of the operation to obtain a new diagonal element
        // of the tridiagonal system. See lines 9 and 10 from the ref.[1]
        applyHconf(Npar,Morb,map1,map12,map22,s12,s22,IF,lvec[i+1],Ho,Hint,out);
        for (j = 0; j < nc; j++)
        {
            out[j] = out[j] - offdiag[i] * lvec[i][j];
        }
        diag[i+1] = carrDot(nc, lvec[i + 1], out);
    }

    free(ortho);
    free(out);

    return lm;
}



double LanczosGround(int Niter, EqDataPkg MC, Cmatrix Orb, Carray C)
{

/** GROUND STATE BY APPROXIMATIVE DIAGONALIZATION WITH LANCZOS ITERATIONS
    =====================================================================
    Use the routine implemented above for lanczos tridiagonal reduction  and
    the LAPACK dstev to diagonalize the unerlying tridiagonal system and get
    approximately the low lying(ground state) eigenvalue and eigenvector

    INPUT PARAMETERS :
        Niter - Suggested number of lanczos iterations
        MC - Multiconfigurational data package
        Orb - Fixed orbitals whose the configurational space is built on
        C - input for lanczos (first lanczos vector)

    OUTPUT PARAMETERS :
        C - End up with low lying eigenvector(ground state)

    RETURN :
        Low lying eigenvalue/ground state energy **/

    int
        i,
        k,
        j,
        nc,
        Norb,
        Ngrid,
        predictedIter;

    double
        sentinel,
        * d,
        * e,
        * eigvec;

    Carray
        Hint,
        diag,
        offdiag;

    Cmatrix
        Ho,
        lvec;

    nc = MC->nc;
    Norb = MC->Morb;
    Ngrid = MC->Mpos;

    // variables to call lapack diagonalization routine for tridiagonal
    // real symmetric matrix from Lanczos iterations output
    d = malloc(Niter * sizeof(double));
    e = malloc(Niter * sizeof(double));
    eigvec = malloc(Niter * Niter * sizeof(double));

    // tridiagonal decomposition from Lanczos iterations. They are of
    // but must be real in the end (up to numerical precision)
    diag = carrDef(Niter);
    offdiag = carrDef(Niter);
    // Lanczos Vectors organized in rows of the matrix 'lvec'
    lvec = cmatDef(Niter,nc);

    // 1- and 2-body orbital matrices needed to apply many-body Hamiltonian
    Hint = carrDef(Norb*Norb*Norb*Norb);
    Ho = cmatDef(Norb,Norb);



    SetupHo(Norb,Ngrid,Orb,MC->dx,MC->a2,MC->a1,MC->V,Ho);
    SetupHint(Norb,Ngrid,Orb,MC->dx,MC->g,Hint);



    offdiag[Niter-1] = 0;   // Useless
    carrCopy(nc,C,lvec[0]); // Setup initial lanczos vector



    // Call Lanczos to setup tridiagonal matrix and lanczos vectors
    predictedIter = Niter;
    Niter = lanczos(MC, Ho, Hint, Niter, diag, offdiag, lvec);
    if (Niter < predictedIter)
    {
        printf("\n\nWARNING : ");
        printf("lanczos iterations exit before expected - %d", Niter);
        printf("\n\n");
    }



    // Transfer data to use lapack routine
    for (k = 0; k < Niter; k++)
    {
        if (fabs(cimag(diag[k])) > 1E-10)
        {
            printf("\n\nWARNING : Nonzero imaginary part in Lanczos\n\n");
        }
        d[k] = creal(diag[k]);    // Supposed to be real
        e[k] = creal(offdiag[k]); // Supposed to be real
        for (j = 0; j < Niter; j++) eigvec[k * Niter + j] = 0;
    }

    k = LAPACKE_dstev(LAPACK_ROW_MAJOR, 'V', Niter, d, e, eigvec, Niter);
    if (k != 0)
    {
        printf("\n\nERROR IN DIAGONALIZATION\n\n");
        printf("LAPACK dstev routin returned %d\n\n",k);
        exit(EXIT_FAILURE);
    }



    sentinel = 1E15;
    // Get Index of smallest eigenvalue
    for (k = 0; k < Niter; k++)
    {
        if (sentinel > d[k]) { sentinel = d[k]; j = k; }
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
    cmatFree(Norb,Ho);
    cmatFree(predictedIter, lvec);

    return sentinel;
}



void LanczosIntegrator(int Liter, EqDataPkg MC, Cmatrix Ho, Carray Hint,
                       doublec dt, Carray C)
{

/** MULTICONFIGURATIONAL LINEAR SYSTEM INTEGRATION USING LANCZOS
    ============================================================
    Use lanczos to integrate the linear system of equations of the
    configurational coefficients. For more information about  this
    integrator check out:
    
    "Unitary quantum time evolution by iterative Lanczos recution",
    Tae Jun Park and J.C. Light, J. Chemical Physics 85, 5870, 1986
    DOI 10.1063/1.451548

    INPUT PARAMETERS
        C - initial condition
        Ho - 1-body hamiltonian matrix (coupling to orbitals)
        Hint - 2-body hamiltonian matrix (coupling to orbitals)

    OUTPUT PARAMETERS
        C - End advanced in a time step 'dt' **/

    int
        i,
        k,
        j,
        nc,
        lm;

    double
        sentinel,
        * d,
        * e,
        * eigvec;

    Carray
        aux,
        diag,
        offdiag,
        Clanczos;

    Cmatrix
        lvec;



    nc = MC->nc;

    // variables to call lapack diagonalization routine for tridiagonal
    // real symmetric matrix from Lanczos iterations output
    d = malloc(Liter * sizeof(double));
    e = malloc(Liter * sizeof(double));
    eigvec = malloc(Liter * Liter * sizeof(double));

    // Lanczos Vectors organize in rows of the matrix 'lvec'
    lvec = cmatDef(Liter,nc);
    // Elements of tridiagonal matrix from Lanczos reduction
    diag = carrDef(Liter);
    offdiag = carrDef(Liter);
    // Solve system of ODEs in lanczos vector space of dimension 'lm'
    Clanczos = carrDef(Liter);
    // auxiliar to backward transformation to original space
    aux = carrDef(Liter);



    offdiag[Liter-1] = 0;            // Useless
    carrCopy(nc,C,lvec[0]); // Setup initial lanczos vector



    /* ================================================================= *

            SOLVE ODE FOR COEFFICIENTS USING LANCZOS VECTOR SPACE

     * ================================================================= */



    // Call Lanczos to perform tridiagonal symmetric reduction
    lm = lanczos(MC,Ho,Hint,Liter,diag,offdiag,lvec);
    if (lm < Liter)
    {
        printf("\n\nWARNING : ");
        printf("lanczos iterations exit before expected - %d", lm);
        printf("\n\n");
    }



    // Transfer data to use lapack routine
    for (k = 0; k < lm; k++)
    {
        d[k] = creal(diag[k]);    // Supposed to be real
        e[k] = creal(offdiag[k]); // Supposed to be real
        for (j = 0; j < lm; j++) eigvec[k * lm + j] = 0;
    }

    k = LAPACKE_dstev(LAPACK_ROW_MAJOR,'V',lm,d,e,eigvec,lm);
    if (k != 0)
    {
        printf("\n\nERROR IN DIAGONALIZATION\n\n");
        printf("LAPACK dstev routin returned %d\n\n",k);
        exit(EXIT_FAILURE);
    }



    // Solve exactly the equation in Lanczos vector space. The transformation
    // between the original space and the Lanczos one is given by the Lanczos
    // vectors organize in columns. When we apply such a matrix to 'Clanczos'
    // we need to get just the first Lanczos vector, that is, the coefficient
    // vector in the previous time step we load in Lanczos routine.  In other
    // words our initial condition is what we has in previous time step.
    carrFill(lm,0,Clanczos); Clanczos[0] = 1.0;

    for (k = 0; k < lm; k++)
    {   // Solve in diagonal basis and for this apply eigvec trasformation
        aux[k] = 0;
        for (j = 0; j < lm; j++) aux[k] += eigvec[j*lm + k] * Clanczos[j];
        aux[k] = aux[k] * cexp(- I * d[k] * dt);
    }

    for (k = 0; k < lm; k++)
    {   // Backward transformation from diagonal representation
        Clanczos[k] = 0;
        for (j = 0; j < lm; j++) Clanczos[k] += eigvec[k*lm + j] * aux[j];
    }

    for (i = 0; i < nc; i++)
    {   // Return from Lanczos vector space to configurational
        C[i] = 0;
        for (j = 0; j < lm; j++) C[i] += lvec[j][i] * Clanczos[j];
    }



    free(d);
    free(e);
    free(eigvec);
    free(diag);
    free(offdiag);
    free(Clanczos);
    free(aux);
    cmatFree(Liter, lvec);

}



void coef_RK4(EqDataPkg MC, ManyBodyPkg S, doublec dt)
{

/** ADVANCE TIME STEP IN COEFFICIENTS USING 4th order RUNGE-KUTTA **/

    int
        i,
        nc;

    Carray
        Ck,
        Cder,
        Carg;

    nc = MC->nc;
    Carg = carrDef(nc);
    Cder = carrDef(nc);
    Ck = carrDef(nc);


    // COMPUTE K1
    dCdt(MC,S->C,S->Ho,S->Hint,Cder);
    for (i = 0; i < nc; i++)
    {   // Add K1 contribution
        Ck[i] = Cder[i];
        // Prepare next argument to compute K2
        Carg[i] = S->C[i] + Cder[i] * 0.5 * dt;
    }



    // COMPUTE K2
    dCdt(MC,Carg,S->Ho,S->Hint,Cder);
    for (i = 0; i < nc; i++)
    {   // Add K2 contribution
        Ck[i] += 2 * Cder[i];
        // Prepare next argument to compute K3
        Carg[i] = S->C[i] + Cder[i] * 0.5 * dt;
    }



    // COMPUTE K3
    dCdt(MC,Carg,S->Ho,S->Hint,Cder);
    for (i = 0; i < nc; i++)
    {   // Add K3 contribution
        Ck[i] += 2 * Cder[i];
        // Prepare next argument to compute K4
        Carg[i] = S->C[i] + Cder[i] * dt;
    }



    // COMPUTE K4
    dCdt(MC,Carg,S->Ho,S->Hint,Cder);
    for (i = 0; i < nc; i++)
    {   // Add K4 contribution
        Ck[i] += Cder[i];
    }



    // Until now Ck holds the sum K1 + 2 * K2 + 2 * K3 + K4
    // from the Fourth order Runge-Kutta algorithm. Update:
    for (i = 0; i < nc; i++)
    {   // Update Coeficients
        S->C[i] = S->C[i] + Ck[i] * dt / 6;
    }

    free(Ck);
    free(Cder);
    free(Carg);
    
}
