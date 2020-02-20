#include "integrators.h"



double eigQuality(EqDataPkg MC, Carray C, Cmatrix Ho, Carray Hint, double E0)
{

/** Return the "Max" norm of the difference from applying  the  Hamiltonian
    to the configurational coefficients and to multiply by the  energy, i.e
    || Ĥ\Psi - E0\Psi ||_inf. Since it is hoped we have a eigenstate in the
    configurational basis this must be close to zero.                   **/

    int 
        i,
        nc = MC->nc,
        Npar = MC->Npar,
        Morb = MC->Morb;

    double
        maxDev;

    Iarray
        IF = MC->IF,
        map1 = MC->Map,
        map12 = MC->MapOT,
        map22 = MC->MapTT,
        s12 = MC->strideOT,
        s22 = MC->strideTT;

    Carray
        Cout;

    Cout = carrDef(nc);

    applyHconf(Npar,Morb,map1,map12,map22,s12,s22,IF,C,Ho,Hint,Cout);

    maxDev = 0;
    for (i = 0; i < nc; i++)
    {
        if (cabs(E0 * C[i] - Cout[i]/Npar) > maxDev)
        {
            maxDev = cabs(E0 * C[i] - Cout[i]/Npar);
        }
    }

    free(Cout);

    return maxDev;
}



double overlapFactor(int Morb, int Mpos, double dx, Cmatrix Orb)
{

/** ORTHOGONALITY TEST FOR THE ORBITALS
    ===================================
    In real time, the equations of the MCTDHB conserves norm and orthogonality
    of the orbitals just in exact arithmetic computations.  Depending  on  the
    time integrator used these conservation laws may fail in finite  precision

    This function return the sum of absolute values of the off-diagonal  terms
    of the overlap matrix | <ORB_i,ORB_j> |, that should give us zero      **/

    int
        i,
        k,
        l;

    double
        sum;

    Carray
        prod;

    prod = carrDef(Mpos);

    sum = 0;

    for (k = 0; k < Morb; k++)
    {
        for (l = k + 1; l < Morb; l++)
        {
            for (i = 0; i < Mpos; i++) prod[i] = Orb[k][i] * conj(Orb[l][i]);
            sum = sum + cabs(Csimps(Mpos, prod, dx));
        }
    }

    free(prod);

    return sum / Morb;
}



double avgOrbNorm(int Morb, int Mpos, double dx, Cmatrix Orb)
{

/** NORM TEST FOR THE ORBITALS
    ==========================
    In real time, the equations of the MCTDHB conserves norm and orthogonality
    of the orbitals just in exact arithmetic computations.  Depending  on  the
    time integrator used these conservation laws may fail in finite  precision

    This function return the sum of L^2 norm of orbitals divided by the number
    of orbitals, which need to be close to 1, since each orbital  should  have
    unit L^2 norm.                                                         **/

    int
        i,
        k,
        l;

    double
        sum;

    Rarray
        orbAbs2;

    orbAbs2 = rarrDef(Mpos);

    sum = 0;

    for (k = 0; k < Morb; k++)
    {
        carrAbs2(Mpos,Orb[k],orbAbs2);
        sum = sum + sqrt(Rsimps(Mpos,orbAbs2,dx));
    }

    free(orbAbs2);

    return sum / Morb;
}










/**************************************************************************
 **************************************************************************
 ****************                                          ****************
 ****************     ORBITAL DOMAIN HANDLING ROUTINES     ****************
 ****************                                          ****************
 **************************************************************************
 **************************************************************************/

double borderNorm(int m, int chunkSize, Rarray f, double h)
{

/** COMPUTE THE NORM OF THE FUNCTION LYING NEAR THE BOUNDARY
    ========================================================
    Given a 'chunkSize' number, use this as number of points to be integrated
    in the far left and far right of the domain, summing both parts. This may
    be used to check if the domain should be resized.                     **/

    int
        n,
        i;

    double
        sum;

    sum = 0;
    n = chunkSize;

    if (chunkSize < 3)
    {
        printf("\n\nERROR : chunk size must be greater than 2 ");
        printf("to compute border norm!\n\n");
        exit(EXIT_FAILURE);
    }

    if (chunkSize > m / 4)
    {
        printf("\n\nERROR : chunk size in border norm is too large!\n");
        printf("Exceeded 1/4 the size of domain.\n\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < n - 1; i++) sum = sum + (f[i+1] + f[i]) * 0.5 * h;
    for (i = m - n; i < m - 1; i++) sum = sum + (f[i+1] + f[i]) * 0.5 * h;

    return sum;

}



int NonVanishingId(int n, Carray f, double dx, double tol)
{

/** NORM IN (PROGRESSIVE)CHUNKS OF DOMAIN
    =====================================
    This is an auxiliar function to 'ResizeDomain' below.  For trapped
    systems  where the orbitals can be localized,  starting  from  the
    initial grid point at far left in the domain spatial line, compute
    the norm of a 'chunk' of the function f. If the norm in this chunk
    is less than the 'tol' input parameter, advance a chunk, and  keep
    doing until the norm exceed the 'tol'. This may tell if the domain
    is oversized to represent the function 'f'.
    Return the point that the the norm exceeded 'tol' when a chunk was
    advanced **/

    int
        i,
        chunkSize;

    Rarray
        chunkAbs2;

    if (n / 30 > 3) { chunkSize = n / 30; }
    else            { chunkSize = 3;      }

    i = chunkSize;
    chunkAbs2 = rarrDef(n);
    carrAbs2(n,f,chunkAbs2);

    // while (sqrt(Rsimps(i,chunkAbs2,dx)) < tol) i = i + chunkSize;
    while (sqrt(borderNorm(n,i,chunkAbs2,dx)) < tol) i = i + chunkSize;

    free(chunkAbs2);

    return i - chunkSize;
}



void ResizeDomain(EqDataPkg mc, ManyBodyPkg S)
{

/** SHRINK DOMAIN SIZE IF IT IS TOO LARGE TO REPRESENT ORBITALS
    ===========================================================
    In trapped gas or for attractive interactions, along the imaginary time
    propagation the domain can get too large from what was initially taken.
    In this case this function cut off the useless part of domain,  but  to
    maintain the grid points it interpolate the orbitals to compute them in
    the new grid points. The orbitals in ManyBodyPkg and grid domain params
    are update in EqDataPkg structures.
    
    See also the auxiliar routines NonVanishingId and MeanQuadraticR to  do
    the job. IT WILL ONLY WORK FOR SYMMETRIC DOMAIN [-a,a] for a > 0    **/

    int
        i,
        j,
        Morb,
        Mpos,
        minR2,
        minId;

    double
        R2,
        xi,
        xf,
        dx,
        oldxf,
        oldxi,
        olddx;

    Rarray
        real,
        imag,
        real_intpol,
        imag_intpol,
        oldx,
        x;

    // If the system is not trapped do nothing
    if (strcmp(mc->Vname, "harmonic") != 0) return;

    // Unpack some structure parameters
    Morb = mc->Morb;
    Mpos = mc->Mpos;
    oldxi = mc->xi;
    oldxf = mc->xf;
    olddx = mc->dx;

    x = rarrDef(Mpos);
    oldx = rarrDef(Mpos);
    real = rarrDef(Mpos);
    imag = rarrDef(Mpos);
    real_intpol = rarrDef(Mpos);
    imag_intpol = rarrDef(Mpos);

    rarrFillInc(Mpos,oldxi,olddx,oldx);



    // Method 1 : Use the 'dispersion' of distribution
    R2 = MeanQuadraticR(mc,S->Omat,S->rho1);
    minR2 = 0;
    while ( fabs(oldx[minR2]) > 6.5 * R2 ) minR2 = minR2 + 1;

    // Method 2 : take the norm beginning from the boundary and get the
    // position where it exceed a certain tolerance (last arg below)
    minId = 0;
    for (i = 0; i < Morb; i++)
    {
        j = NonVanishingId(Mpos,S->Omat[i],olddx,2E-7/Morb);
        minId = minId + j;
    }
    minId = minId / Morb;
    printf("\n\nMin. Id. %d\n\n",minId);
    printf("\n\nsqrt(<R^2>). %.2lf\n\n",R2);

    // Take the weighted average
    minId = (3 * minId + minR2) / 4;



    // Check if it is woth to resize the domain comparing the
    // percentual reduction that would be done
    if (100 * abs(oldx[minId] - oldxi) / (oldxf - oldxi) < 6)
    {
        free(x);
        free(oldx);
        free(real);
        free(imag);
        free(real_intpol);
        free(imag_intpol);

        return;
    }

    xi = oldxi + minId * olddx;
    xf = oldxf - minId * olddx;
    dx = (xf - xi) / (Mpos - 1);



    // SETUP new grid points and domain limits
    rarrFillInc(Mpos, xi, dx, x);
    mc->xi = xi;
    mc->xf = xf;
    mc->dx = dx;
    printf("\n");
    sepline();
    printf("\t\t*    Domain resized to [%.2lf,%.2lf]    *",xi,xf);
    sepline();



    // SETUP new one-body potential in discretized positions
    GetPotential(Mpos,mc->Vname,x,mc->V,mc->p[0],mc->p[1],mc->p[2]);



    // INTERPOLATE to compute function in the new shrinked domain
    for (i = 0; i < Morb; i ++)
    {
        // separe real and imaginary part
        carrRealPart(Mpos, S->Omat[i], real);
        carrImagPart(Mpos, S->Omat[i], imag);

        // Interpolate in real and imaginary part
        lagrange(Mpos,5,oldx,real,Mpos,x,real_intpol);
        lagrange(Mpos,5,oldx,imag,Mpos,x,imag_intpol);

        // Update orbital
        for (j = 0; j < Mpos; j ++)
        {
            S->Omat[i][j] = real_intpol[j] + I * imag_intpol[j];
        }
    }

    // FINISH free allocated memory
    free(x);
    free(oldx);
    free(real);
    free(imag);
    free(real_intpol);
    free(imag_intpol);
}



void extentDomain(EqDataPkg mc, ManyBodyPkg S)
{

    int
        i,
        j,
        Morb,
        Mpos,
        minR2,
        minId,
        extra,
        newMpos;

    double
        R2,
        xi,
        xf,
        dx,
        oldxi;

    Rarray
        oldx,
        x;

    Cmatrix
        newOrb;

    // If the system is not trapped do nothing
    if (strcmp(mc->Vname, "harmonic") != 0) return;

    // Unpack some structure parameters
    Morb = mc->Morb;
    Mpos = mc->Mpos;
    dx = mc->dx;
    oldxi = mc->xi;

    oldx = rarrDef(Mpos);
    rarrFillInc(Mpos,oldxi,dx,oldx);



    // Method 1 : Use the 'dispersion' of distribution
    R2 = MeanQuadraticR(mc,S->Omat,S->rho1);
    minR2 = 0;
    while ( fabs(oldx[minR2]) > 6.5 * R2 ) minR2 = minR2 + 1;

    // Method 2 : take the norm beginning from the boundary and get the
    // position where it exceed a certain tolerance (last arg below)
    minId = 0;
    for (i = 0; i < Morb; i++)
    {
        j = NonVanishingId(Mpos,S->Omat[i],dx,5E-7/Morb);
        minId = minId + j;
    }
    minId = minId / Morb;

    // Take the weighted average
    minId = (3 * minId + minR2) / 4;

    // Check if it is woth to resize the domain, i.e, orbitals too
    // close to the boundaries
    if (minId > 10)
    {
        free(oldx);
        return;
    }

    extra = Mpos / 10;
    newMpos = Mpos + 2 * extra;

    // Define new grid points preserving dx
    x = rarrDef(newMpos);
    for (i = 0; i < Mpos; i++) x[i + extra] = oldx[i];
    for (i = extra; i > 0; i--) x[i - 1] = x[i] - dx;
    for (i = Mpos + extra; i < newMpos; i++) x[i] = x[i - 1] + dx;

    // Update grid data on structure
    xi = x[0];
    xf = x[newMpos - 1];
    mc->xi = xi;
    mc->xf = xf;
    mc->Mpos = newMpos;
    printf("\n");
    sepline();
    printf("\t\t*    Domain resized to [%.2lf,%.2lf]    *",xi,xf);
    sepline();

    printf("\n\ndx is the same %.10lf = %.10lf\n\n",(xf-xi)/(newMpos-1),dx);

    // SETUP new one-body potential in discretized positions
    free(mc->V);
    mc->V = rarrDef(newMpos);
    GetPotential(newMpos,mc->Vname,x,mc->V,mc->p[0],mc->p[1],mc->p[2]);

    // Update orbitals
    newOrb = cmatDef(Morb,newMpos);
    for (j = 0; j < Morb; j++)
    {
        for (i = 0; i < Mpos; i++)
        {
            newOrb[j][i + extra] = S->Omat[j][i];
        }
        for (i = extra; i > 0; i--)
        {
            newOrb[j][i - 1] = 0;
        }
        for (i = Mpos + extra; i < newMpos; i++)
        {
            newOrb[j][i] = 0;
        }
    }

    // update on structure
    cmatFree(Morb,S->Omat);
    S->Omat = newOrb;
    S->Mpos = newMpos;



    // FINISH free allocated memory
    free(x);
    free(oldx);
}















/*************************************************************************
 *************************************************************************
 ****************                                         ****************
 ****************      ORBITAL NONLINEAR DERIVATIVES      ****************
 ****************                                         ****************
 *************************************************************************
 *************************************************************************/

doublec nonlinearOrtho (int M, int k, int n, double g, Cmatrix Orb,
        Cmatrix Rinv, Carray R2, Cmatrix Ho, Carray Hint, Cmatrix Ortho )
{

/** NONLINEAR PART OF TIME DERIVATIVE - ORBITAL'S EQUATION
    ======================================================
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



doublec nonlinear (int M, int k, int n, double g, Cmatrix Orb,
        Cmatrix Rinv, Carray R2, Cmatrix Ho, Carray Hint )
{

/** Same stuff from function above without  inverted  overlap matrix for
    imaginary time integration, because the re-orthogonalization is done
    each time by hand. Must do the same  thing  the  function above with
    the overlap matrix considered as the unit                        **/

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
        Ginv,
        X;



    X = 0;
    M2 = M * M;
    M3 = M * M * M;



    for (a = 0; a < M; a++)
    {
        // Subtract one-body projection
        X = X - Ho[a][k] * Orb[a][n];

        for (q = 0; q < M; q++)
        {
            // Particular case with the two last indices equals
            // to take advantage of the symmetry afterwards

            G = Rinv[k][a] * R2[a + M*a + M2*q + M3*q];

            // Sum interacting part contribution
            X = X + g * G * conj(Orb[a][n]) * Orb[q][n] * Orb[q][n];

            // Subtract interacting projection
            for (j = 0; j < M; j++)
            {
                ind = j + a * M + q * M2 + q * M3;
                X = X - G * Orb[j][n] * Hint[ind];
            }

            for (l = q + 1; l < M; l++)
            {
                G = 2 * Rinv[k][a] * R2[a + M*a + M2*q + M3*l];

                // Sum interacting part
                X = X + g * G * conj(Orb[a][n]) * Orb[l][n] * Orb[q][n];

                // Subtract interacting projection
                for (j = 0; j < M; j++)
                {
                    ind = j + a * M + l * M2 + q * M3;
                    X = X - G * Orb[j][n] * Hint[ind];
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
                for (j = 0; j < M; j++)
                {
                    ind = j + s * M + q * M2 + q * M3;
                    X = X - G * Orb[j][n] * Hint[ind];
                    ind = j + a * M + q * M2 + q * M3;
                    X = X - Ginv * Orb[j][n] * Hint[ind];
                }

                for (l = q + 1; l < M; l++)
                {
                    G = 2 * Rinv[k][a] * R2[a + M*s + M2*q + M3*l];
                    Ginv = 2 * Rinv[k][s] * R2[a + M*s + M2*q + M3*l];

                    // Sum interacting part
                    X = X + g * (G*conj(Orb[s][n]) + Ginv*conj(Orb[a][n])) * \
                            Orb[l][n]*Orb[q][n];

                    // Subtract interacting projection
                    for (j = 0; j < M; j++)
                    {
                        ind = j + s * M + l * M2 + q * M3;
                        X = X - G * Orb[j][n] * Hint[ind];
                        ind = j + a * M + l * M2 + q * M3;
                        X = X - Ginv * Orb[j][n] * Hint[ind];
                    }
                }
            }
        }
    }

    return X;
}



void imagNLTRAP_dOdt(EqDataPkg MC, Cmatrix Orb, Cmatrix dOdt,
     Cmatrix Ho, Carray Hint, Cmatrix rho1, Carray rho2)
{

/** TIME DERIVATIVE ACCORDING TO NONLINEAR + LINEAR POTENTIAL
    =========================================================
    Compute essentially the nonlinear part of the equation together with
    the 1-body(trap) potential, multiplied by minus imaginary  unit.  In
    split-step techniques give us the derivative when solving  nonlinear
    part alone. Since it includes the 1-body(trap) potential it  is more
    useful when solving the derivatives part by  spectral  methods.  The
    1-body density matrix inversion is evaluated here

    OUTPUT ARGUMENT : dOdt **/

    int
        k,
        s,
        j,
        M  = MC->Morb,
        Mpos  = MC->Mpos;



    double
        dx = MC->dx,
        * V = MC->V;



    double complex
        Proj,
        g = MC->g;



    Cmatrix
        rho_inv = cmatDef(M,M);



    // Invert the matrix and check if the operation was successfull
    s = HermitianInv(M,rho1,rho_inv);
    if (s != 0)
    {
        printf("\n\n\n\nFailed on Lapack inversion routine!\n");
        printf("-----------------------------------\n\n");

        printf("Matrix given was :\n");
        cmat_print(M,M,rho1);

        if (s > 0) printf("\nSingular decomposition : %d\n\n",s);
        else       printf("\nInvalid argument given : %d\n\n",s);

        exit(EXIT_FAILURE);
    }



    // Update k-th orbital at discretized position j
    #pragma omp parallel for private(k,j) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
            dOdt[k][j] = - I * ( V[j] * Orb[k][j] + \
            nonlinear(M,k,j,g,Orb,rho_inv,rho2,Ho,Hint) );
    }

    cmatFree(M, rho_inv);

}



void imagNL_dOdt (EqDataPkg MC, Cmatrix Orb, Cmatrix dOdt, Cmatrix Ho,
                  Carray Hint, Cmatrix rho1, Carray rho2)
{

/** TIME DERIVATIVE ACCORDING TO NONLINEAR PART
    ===========================================
    Compute essentially the nonlinear part of the equation, multiplied by
    minus imaginary unit. In split-step techniques give us the derivative
    when solving nonlinear part alone. 

    OUTPUT ARGUMENT : dOdt **/

    int
        k,
        s,
        j,
        M = MC->Morb,
        Mpos  = MC->Mpos;



    double
        dx = MC->dx;



    double complex
        Proj,
        g = MC->g;



    Cmatrix
        rho_inv = cmatDef(M, M);



    // Compute inversion and check if it was successfull
    s = HermitianInv(M, rho1, rho_inv);
    if (s != 0)
    {
        printf("\n\n\n\nFailed on Lapack inversion routine!\n");
        printf("-----------------------------------\n\n");

        printf("Matrix given was :\n");
        cmat_print(M,M,rho1);

        if (s > 0) printf("\nSingular decomposition : %d\n\n", s);
        else       printf("\nInvalid argument given : %d\n\n", s);

        exit(EXIT_FAILURE);
    }



    // Update k-th orbital at discretized position j
    #pragma omp parallel for private(k,j) schedule(static)
    for (k = 0; k < M; k++)
    {
        for (j = 0; j < Mpos; j++)
            dOdt[k][j] = - I * \
            nonlinear(M, k, j, g, Orb, rho_inv, rho2, Ho, Hint);
    }

    cmatFree(M, rho_inv);
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















/**************************************************************************
 **************************************************************************
 **************                                            ****************
 **************    COEFFICIENT INTEGRATION/GROUND-STATE    ****************
 **************                                            ****************
 **************************************************************************
 **************************************************************************/

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















/*************************************************************************
 *************************************************************************
 **************                                             **************
 **************    RUNGE-KUTTA INTEGRATOS NONLINEAR PART    **************
 **************                                             **************
 *************************************************************************
 *************************************************************************/

void imagNLTRAP_RK2(EqDataPkg MC, ManyBodyPkg S, doublec dt)
{

/** NonLinear plus TRAP 1-body potential integrator with
    Runge-Kutta of 2nd order for imaginary time
    ====================================================
    Apply 2nd order Runge-Kutta integrator to evolve in time the equation
    corresponding to nonlinear part from split-step with additionally the
    1-body potential to completely separate the derivatives  which  shall
    be evaluated with specral methods. For this purpose 'imagNLTRAP_dOdt'
    is used. The time must be t = - i T.

    OUTPUT :
        S->Orb - Advanced in time step 'dt' using only nonlinear part **/

    int
        i,
        k,
        j,
        Morb,
        Mpos;

    double
        g,
        dx,
        a2,
        * V;

    double complex
        a1;

    Cmatrix
        Ok,
        Oarg;

    Morb = MC->Morb;
    Mpos = MC->Mpos;
    a1 = MC->a1;
    a2 = MC->a2;
    dx = MC->dx;
    g = MC->g;
    V = MC->V;

    Oarg = cmatDef(Morb,Mpos);
    Ok = cmatDef(Morb,Mpos);



    // COMPUTE K1
    imagNLTRAP_dOdt(MC,S->Omat,Ok,S->Ho,S->Hint,S->rho1,S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            // Prepare next argument to compute K2
            Oarg[k][j] = S->Omat[k][j] + Ok[k][j] * 0.5 * dt;
        }
    }

    // Update hamiltonian matrix elements
    SetupHo(Morb,Mpos,Oarg,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,Oarg,dx,g,S->Hint);

    // COMPUTE K2
    imagNLTRAP_dOdt(MC,Oarg,Ok,S->Ho,S->Hint,S->rho1,S->rho2);

    // Runge-Kutta of 2nd order update
    // y(t+dt) = y(t) + dt * ( f(t + dt/2, y(t) + f(t,y(t))*dt/2) )
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt;
        }
    }

    cmatFree(Morb,Ok);
    cmatFree(Morb,Oarg);

}



void imagNL_RK2(EqDataPkg MC, ManyBodyPkg S, doublec dt)
{

/** NonLinear integrator with Runge-Kutta of 2nd order for imaginary time
    =====================================================================
    Apply 2nd order Runge-Kutta integrator to evolve in time the equation
    corresponding to nonlinear part from split-step without including the
    1-body potential which is taken in account on Finite differences. For
    this purpose 'imagNL_dOdt' is used. The time must be t = - i T.

    OUTPUT :
        S->Orb - Advanced in time step 'dt' using only nonlinear part **/

    int
        i,
        k,
        j,
        Morb,
        Mpos;

    double
        g,
        dx,
        a2,
        * V;

    double complex
        a1;

    Cmatrix
        Ok,
        Oarg;

    Morb = MC->Morb;
    Mpos = MC->Mpos;
    a1 = MC->a1;
    a2 = MC->a2;
    dx = MC->dx;
    g = MC->g;
    V = MC->V;
    
    Oarg = cmatDef(Morb,Mpos);
    Ok = cmatDef(Morb,Mpos);



    // COMPUTE K1
    imagNL_dOdt(MC,S->Omat,Ok,S->Ho,S->Hint,S->rho1,S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            // Prepare next argument to compute K2
            Oarg[k][j] = S->Omat[k][j] + Ok[k][j] * 0.5 * dt;
        }
    }

    // Update hamiltonian matrix elements
    SetupHo(Morb,Mpos,Oarg,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,Oarg,dx,g,S->Hint);

    // COMPUTE K2
    imagNL_dOdt(MC,Oarg,Ok,S->Ho,S->Hint,S->rho1,S->rho2);

    // Runge-Kutta of 2nd order update
    // y(t+dt) = y(t) + dt * ( f(t + dt/2, y(t) + f(t,y(t))*dt/2) )
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt;
        }
    }

    cmatFree(Morb,Ok);
    cmatFree(Morb,Oarg);

}



void realNLTRAP_RK2(EqDataPkg MC, ManyBodyPkg S, double dt)
{

/** NonLinear plus TRAP 1-body potential integrator with
    Runge-Kutta of 2nd order for real time
    ====================================================
    Apply 2nd order Runge-Kutta integrator to evolve in time the equation
    corresponding to nonlinear part from split-step with additionally the
    1-body potential to completely separate the derivatives  which  shall
    be evaluated with specral methods. For this purpose 'realNLTRAP_dOdt'
    is used. The time t must be real.

    OUTPUT :
        S->Orb - Advanced in time step 'dt' using only nonlinear part **/

    int
        i,
        k,
        j,
        Morb,
        Mpos;

    double
        g,
        dx,
        a2,
        * V;

    double complex
        a1;

    Cmatrix
        Ok,
        Oarg;

    Morb = MC->Morb;
    Mpos = MC->Mpos;
    a1 = MC->a1;
    a2 = MC->a2;
    dx = MC->dx;
    g = MC->g;
    V = MC->V;

    Oarg = cmatDef(Morb,Mpos);
    Ok = cmatDef(Morb,Mpos);



    // COMPUTE K1
    realNLTRAP_dOdt(MC,S->Omat,Ok,S->Ho,S->Hint,S->rho1,S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            // Prepare next argument to compute K2
            Oarg[k][j] = S->Omat[k][j] + Ok[k][j] * 0.5 * dt;
        }
    }

    // Update hamiltonian matrix elements
    SetupHo(Morb,Mpos,Oarg,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,Oarg,dx,g,S->Hint);

    // COMPUTE K2
    realNLTRAP_dOdt(MC,Oarg,Ok,S->Ho,S->Hint,S->rho1,S->rho2);

    // Runge-Kutta of 2nd order update
    // y(t+dt) = y(t) + dt * ( f(t + dt/2, y(t) + f(t,y(t))*dt/2) )
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt;
        }
    }

    cmatFree(Morb,Ok);
    cmatFree(Morb,Oarg);

}



void realNL_RK2(EqDataPkg MC, ManyBodyPkg S, double dt)
{

/** NonLinear integrator with Runge-Kutta of 2nd order for real time
    ================================================================
    Apply 2nd order Runge-Kutta integrator to evolve in time the equation
    corresponding to nonlinear part from split-step without including the
    1-body potential which is taken in account on Finite differences. For
    this purpose 'realNL_dOdt' is used. The time t must be real.

    OUTPUT :
        S->Orb - Advanced in time step 'dt' using only nonlinear part **/

    int
        i,
        k,
        j,
        Morb,
        Mpos;

    double
        g,
        dx,
        a2,
        * V;

    double complex
        a1;

    Cmatrix
        Ok,
        Oarg;

    Morb = MC->Morb;
    Mpos = MC->Mpos;
    a1 = MC->a1;
    a2 = MC->a2;
    dx = MC->dx;
    g = MC->g;
    V = MC->V;
    
    Oarg = cmatDef(Morb,Mpos);
    Ok = cmatDef(Morb,Mpos);



    // COMPUTE K1
    realNL_dOdt(MC,S->Omat,Ok,S->Ho,S->Hint,S->rho1,S->rho2);
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            // Prepare next argument to compute K2
            Oarg[k][j] = S->Omat[k][j] + Ok[k][j] * 0.5 * dt;
        }
    }

    // Update hamiltonian matrix elements
    SetupHo(Morb,Mpos,Oarg,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,Oarg,dx,g,S->Hint);

    // COMPUTE K2
    realNL_dOdt(MC,Oarg,Ok,S->Ho,S->Hint,S->rho1,S->rho2);

    // Runge-Kutta of 2nd order update
    // y(t+dt) = y(t) + dt * ( f(t + dt/2, y(t) + f(t,y(t))*dt/2) )
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
        {
            S->Omat[k][j] = S->Omat[k][j] + Ok[k][j] * dt;
        }
    }

    cmatFree(Morb,Ok);
    cmatFree(Morb,Oarg);

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















/************************************************************************
 ************************************************************************
 ****************                                        ****************
 ****************     GENERAL INTEGRATOS LINEAR PART     ****************
 ****************                                        ****************
 ************************************************************************
 ************************************************************************/

void LP_CNSM (int Mpos, int Morb, CCSmat cnmat, Carray upper,
     Carray lower, Carray mid, Cmatrix Orb)
{

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



void LP_CNLU (int Mpos, int Morb, CCSmat cnmat, Carray upper, Carray lower,
     Carray mid, Cmatrix Orb )
{

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



void LP_FFT (int Mpos, int Morb, DFTI_DESCRIPTOR_HANDLE * desc,
     Carray exp_der, Cmatrix Orb)
{

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















/***********************************************************************
 ***********************************************************************
 ******************                                    *****************
 ******************       SPLIT-STEP INTEGRATORS       *****************
 ******************                                    *****************
 ***********************************************************************
 ***********************************************************************/

int imagFFT(EqDataPkg MC, ManyBodyPkg S, double dT, int Nsteps, int coefInteg)
{

/** INTEGRATE EQUATIONS IN IMAGINARY TIME
    =====================================
    Integrate the MCTDHB equations in imaginary time to find the ground
    state. The integration is done using split-step technique  for  the
    orbitals, where the nonlinear/potential part is  evolved  with  2nd
    order Runge-Kutta scheme and the linear(derivatives)  with spectral
    method using FFTs. The coefficients may be chosen in job.conf  file
    between Lanczos and 4th order Runge-Kutta.

    The many-body data in 'S' is updated until find the ground state **/

    int i,
        j,
        k,
        m,
        nc,
        Npar,
        Mpos,
        Morb,
        isTrapped;

    MKL_LONG
        p;

    double
        eigFactor,
        freq,
        Idt,
        R2,
        dx,
        a2,
        g;

    Rarray
        V;

    double complex
        vir,
        E,
        prevE,
        a1,
        dt;

    Carray
        exp_der;

    DFTI_DESCRIPTOR_HANDLE
        desc;



    // unpack equation parameters
    V = MC->V;
    g = MC->g;
    dx = MC->dx;
    a2 = MC->a2;
    a1 = MC->a1;

    // pure imag. time for relaxation with dT the step size
    dt = - I * dT;
    // (- i * dt) factor found in exponential of split-step
    Idt = - dT;

    // Unpack domain and configuration parameters
    nc = MC->nc;
    Npar = MC->Npar;
    Mpos = MC->Mpos;
    Morb = MC->Morb;

    // Exponential of derivatives in FFT momentum space. The FFTs ignores
    // the last grid-point assuming periodicity there. Thus the size of
    // functions in FFT must be Mpos - 1
    m = Mpos - 1;
    exp_der = carrDef(m);



    // Check if the system has some trap potential to later verify if
    // the grid domain is not oversized for localized distributions
    isTrapped = strcmp(MC->Vname, "harmonic");



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
        exp_der[i] = cexp( -0.5 * dT * (I * a1 * freq - a2 * freq * freq) );
    }



    // Setup one/two-body hamiltonian matrix elements
    SetupHo(Morb,Mpos,S->Omat,dx,a2,a1,V,S->Ho);
    SetupHint(Morb,Mpos,S->Omat,dx,g,S->Hint);

    // Setup one/two-body density matrix 
    OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);



    // Store the initial energy
    E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
    prevE = E;
    vir = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
    R2 = MeanQuadraticR(MC, S->Omat, S->rho1);
    eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);

    printf("\n\nProgress    E/particle     sqrt<R^2>");
    printf("     |Virial/E|     H[C] - E*C");
    sepline();
    printf("%5.1lf%%     %11.7lf",0.0,creal(E));
    printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
    printf("      %8.5lf",eigFactor);



    for (i = 0; i < Nsteps; i++)
    {

        // INTEGRATE ORBITALS
        // Half-step linear part
        LP_FFT(Mpos, Morb, &desc, exp_der, S->Omat);
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        // Full step nonlinear part
        imagNLTRAP_RK2(MC, S, dt);
        // Half-step linear part again
        LP_FFT(Mpos, Morb, &desc, exp_der, S->Omat);
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);



        // INTEGRATE COEFFICIENTS
        if (coefInteg < 2)
        {
            coef_RK4(MC,S,dt);
        }
        else
        {
            LanczosIntegrator(coefInteg,MC,S->Ho,S->Hint,dt,S->C);
        }



        // Loss of Norm => undefined behavior on orthogonality
        Ortonormalize(Morb, Mpos, dx, S->Omat);
        // Renormalize coeficients
        renormalizeVector(nc, S->C, 1.0);



        // Finish time steo calculations updating matrices
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);



        if ( (i + 1) % (Nsteps/5) == 0 && isTrapped == 0)
        {

            // After some time evolved check if initial domain is suitable
            // for the current working orbitals, to avoid oversized domain,
            // a useless length where the functions are zero anyway

            ResizeDomain(MC,S);

            dx = MC->dx;

            Ortonormalize(Morb, Mpos, dx, S->Omat);

            SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
            SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

            // Reconfigure exponential of derivatives in fourier space
            for (j = 0; j < m; j++)
            {
                if (j <= (m-1)/2) { freq = (2 * PI * j) / (m * dx);       }
                else              { freq = (2 * PI * (j - m)) / (m * dx); }
                // exponential of derivative operators in half time-step
                exp_der[j] = cexp(-0.5*dT * (I*a1*freq - a2*freq*freq));
            }

        }



        E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
        vir = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
        R2 = MeanQuadraticR(MC, S->Omat, S->rho1);
        if ( (i + 1) % (Nsteps / 500) == 0 )
        {
            eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
            // Print in screen them on screen
            printf("\n%5.1lf%%     %11.7lf",(100.0*i)/Nsteps,creal(E));
            printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
            printf("      %8.5lf",eigFactor);
        }



        // At every 200 time-steps performed
        if ( (i + 1) % 200 == 0)
        {
            // Check if the energy stop decreasing and break the process
            if (fabs(creal(E - prevE) / creal(prevE)) < 5E-10)
            {
                p = DftiFreeDescriptor(&desc);
                free(exp_der);

                sepline();

                printf("\nProcess ended before because ");
                printf("energy stop decreasing.\n\n");

                return i + 1;
            }

            prevE = E;
        }

    }



    // PROCESS DID NOT STOP AUTOMATICALLY
    // Decide a number of iterations to apprimorate the ground state
    // using Lanczos. Try to avoid memory breakdown
    if (200 * nc < 5E7)
    {
        if (2 * nc / 3 < 200) k = 2 * nc / 3;
        else                  k = 200;
    }
    else k = 5E7 / nc;

    E = LanczosGround(k,MC,S->Omat,S->C) / Npar;
    // Renormalize coeficients
    renormalizeVector(nc,S->C,1.0);

    sepline();
    printf("\nFinal E/particle = %.7lf\n", creal(E));
    printf("Process did not stop automatically because energy were");
    printf(" varying above accepted tolerance\n\n");

    p = DftiFreeDescriptor(&desc);
    free(exp_der);

    return Nsteps + 1;
}



int imagCNSM(EqDataPkg MC, ManyBodyPkg S, double dT, int Nsteps,
             int coefInteg, int cyclic)
{

/** INTEGRATE EQUATIONS IN IMAGINARY TIME
    =====================================
    Integrate the MCTDHB equations in imaginary time to find the ground
    state. The integration is done using split-step technique  for  the
    orbitals, where the nonlinear/potential part is  evolved  with  2nd
    order Runge-Kutta scheme and the  linear  part  with crank-nicolson
    finite differences scheme. The coefficients may be chosen in job.conf
    file between Lanczos and 4th order Runge-Kutta.

    The many-body data in 'S' is updated until find the ground state or
    reach the requested number of steps                             **/

    int i,
        j,
        k,
        nc = MC->nc,
        Npar = MC->Npar,
        Mpos = MC->Mpos,
        Morb = MC->Morb,
        isTrapped;

    double
        R2,
        eigFactor,
        dx = MC->dx,
        a2 = MC->a2,
        g = MC->g,
        * V = MC->V;

    double complex
        E,
        vir,
        prevE,
        a1 = MC->a1,
        dt = - I * dT;

    Carray
        upper  = carrDef(Mpos - 1),
        lower  = carrDef(Mpos - 1),
        mid    = carrDef(Mpos - 1);

    CCSmat
        cnmat;



    isTrapped = strcmp(MC->Vname, "harmonic");

    // Setup one/two-body hamiltonian matrix elements
    SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

    // Setup one/two-body density matrix
    OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);



    // Store the initial energy
    E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
    prevE = E;
    vir = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
    R2 = MeanQuadraticR(MC, S->Omat, S->rho1);
    eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);

    printf("\n\nProgress    E/particle     sqrt<R^2>");
    printf("     |Virial/E|     H[C] - E*C");
    sepline();
    printf("%5.1lf%%     %11.7lf",0.0,creal(E));
    printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
    printf("      %8.5lf",eigFactor);



    // Configure the linear system from Crank-Nicolson scheme with
    // half time step because of split step
    cnmat = CNmat(Mpos,dx,dt/2,a2,a1,g,V,cyclic,upper,lower,mid);



    for (i = 0; i < Nsteps; i++)
    {

        // PROPAGATE LINEAR PART HALF STEP
        LP_CNSM(Mpos, Morb, cnmat, upper, lower, mid, S->Omat);

        // The boundary
        if (cyclic)
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = S->Omat[k][0]; }
        else
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = 0;             }

        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

        // PROPAGATE NONLINEAR PART AN ENTIRE STEP
        imagNL_RK2(MC, S, dt);

        // PROPAGATE LINEAR PART HALF STEP
        LP_CNSM(Mpos, Morb, cnmat, upper, lower, mid, S->Omat);

        // The boundary
        if (cyclic)
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = S->Omat[k][0]; }
        else
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = 0;             }

        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);



        // PROPAGATE COEFFICIENTS
        // INTEGRATE COEFFICIENTS
        if (coefInteg < 2)
        {
            coef_RK4(MC,S,dt);
        }
        else
        {
            LanczosIntegrator(coefInteg,MC,S->Ho,S->Hint,dt,S->C);
        }



        // Renormalize coeficients
        renormalizeVector(nc, S->C, 1.0);
        // Loss of Norm => undefined behavior on orthogonality
        Ortonormalize(Morb, Mpos, dx, S->Omat);



        // Update quantities that depends on orbitals and coefficients
        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);



        if ( (i + 1) % (Nsteps/5) == 0 && isTrapped == 0)
        {
            // After some time evolved check if initial domain is suitable
            // for the current working orbitals, to avoid oversized domain,
            // a useless length where the functions are zero anyway

            ResizeDomain(MC, S);

            dx = MC->dx;

            Ortonormalize(Morb, Mpos, dx, S->Omat);

            SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
            SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

            CCSFree(cnmat);

            // re-onfigure again the Crank-Nicolson Finite-difference matrix
            cnmat = CNmat(Mpos,dx,dt/2,a2,a1,g,V,cyclic,upper,lower,mid);
        }



        E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
        vir = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
        R2 = MeanQuadraticR(MC, S->Omat, S->rho1);
        if ( (i + 1) % (Nsteps / 500) == 0 )
        {
            eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
            // Print in screen them on screen
            printf("\n%5.1lf%%     %11.7lf",(100.0*i)/Nsteps,creal(E));
            printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
            printf("      %8.5lf",eigFactor);
        }



        // CHECK IF THE ENERGY HAS STABILIZED TO STOP PROCESS
        // At every 200 time-steps performed
        if ( (i + 1) % 200 == 0)
        {
            // Check if the energy stop decreasing and break the process
            if (fabs(creal(E - prevE) / creal(prevE)) < 5E-10)
            {
                CCSFree(cnmat);
                free(upper);
                free(lower);
                free(mid);

                sepline();

                printf("\nProcess ended before because ");
                printf("energy stop decreasing.\n\n");

                return i + 1;
            }

            prevE = E;
        }

    }

    if (200 * nc < 5E7)
    {
        if (2 * nc / 3 < 200) k = 2 * nc / 3;
        else                  k = 200;
    }
    else k = 5E7 / nc;

    E = LanczosGround(k,MC,S->Omat,S->C) / Npar;
    renormalizeVector(nc, S->C, 1.0);

    sepline();
    printf("\nFinal E/particle = %.7lf\n", creal(E));
    printf("Process did not stop automatically because energy were");
    printf(" varying above accepted tolerance\n\n");

    CCSFree(cnmat);
    free(upper);
    free(lower);
    free(mid);

    return Nsteps + 1;
}



int imagCNLU(EqDataPkg MC, ManyBodyPkg S, double dT, int Nsteps,
             int coefInteg, int cyclic)
{

/** INTEGRATE EQUATIONS IN IMAGINARY TIME
    =====================================
    Integrate the MCTDHB equations in imaginary time to find the ground
    state. The integration is done using split-step technique  for  the
    orbitals, where the nonlinear/potential part is  evolved  with  2nd
    order Runge-Kutta scheme and the  linear  part  with crank-nicolson
    finite differences scheme. The coefficients may be chosen in job.conf
    file between Lanczos and 4th order Runge-Kutta.

    The many-body data in 'S' is updated until find the ground state or
    reach the requested number of steps                             **/

    int
        i,
        j,
        k,
        nc = MC->nc,
        Npar = MC->Npar,
        Mpos = MC->Mpos,
        Morb = MC->Morb,
        isTrapped;

    double
        R2,
        eigFactor,
        dx = MC->dx,
        a2 = MC->a2,
        g = MC->g,
        * V = MC->V;

    double complex
        E,
        vir,
        prevE,
        a1 = MC->a1,
        dt = - I * dT;

    Carray
        upper  = carrDef(Mpos - 1),
        lower  = carrDef(Mpos - 1),
        mid    = carrDef(Mpos - 1);

    CCSmat
        cnmat;



    isTrapped = strcmp(MC->Vname, "harmonic");
    // Setup one/two-body hamiltonian matrix elements
    SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
    SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

    // Setup one/two-body density matrix
    OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
    TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
          MC->IF,S->C,S->rho2);



    // Store the initial energy
    E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
    prevE = E;
    vir = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
    R2 = MeanQuadraticR(MC, S->Omat, S->rho1);
    eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);

    printf("\n\nProgress    E/particle     sqrt<R^2>");
    printf("     |Virial/E|     H[C] - E*C");
    sepline();
    printf("%5.1lf%%     %11.7lf",0.0,creal(E));
    printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
    printf("      %8.5lf",eigFactor);



    // Configure the linear system from Crank-Nicolson scheme
    cnmat = CNmat(Mpos, dx, dt/2, a2, a1, g, V, cyclic, upper, lower, mid);



    for (i = 0; i < Nsteps; i++)
    {

        LP_CNLU(Mpos, Morb, cnmat, upper, lower, mid, S->Omat);

        // The boundary
        if (cyclic)
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = S->Omat[k][0]; }
        else
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = 0;             }

        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

        imagNL_RK2(MC, S, dt);

        LP_CNLU(Mpos, Morb, cnmat, upper, lower, mid, S->Omat);

        // The boundary
        if (cyclic)
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = S->Omat[k][0]; }
        else
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = 0;             }

        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);



        // PROPAGATE COEFFICIENTS
        // INTEGRATE COEFFICIENTS
        if (coefInteg < 2)
        {
            coef_RK4(MC,S,dt);
        }
        else
        {
            LanczosIntegrator(coefInteg,MC,S->Ho,S->Hint,dt,S->C);
        }



        // Renormalize coeficients
        renormalizeVector(nc, S->C, 1.0);
        // Loss of Norm => undefined behavior on orthogonality
        Ortonormalize(Morb, Mpos, dx, S->Omat);



        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);



        if ( (i + 1) % (Nsteps/5) == 0 && isTrapped == 0)
        {
            // After some time evolved check if initial domain is suitable
            // for the current working orbitals, to avoid oversized domain
            // a useless length where the functions are zero anyway

            ResizeDomain(MC, S);

            dx = MC->dx;

            // Loss of Norm => undefined behavior on orthogonality
            Ortonormalize(Morb, Mpos, dx, S->Omat);

            SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
            SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

            CCSFree(cnmat);

            // re-onfigure again the Crank-Nicolson Finite-difference matrix
            cnmat = CNmat(Mpos,dx,dt/2,a2,a1,g,V,cyclic,upper,lower,mid);

        }



        E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
        vir = Virial(MC, S->Omat, S->rho1, S->rho2) / Npar;
        R2 = MeanQuadraticR(MC, S->Omat, S->rho1);
        if ( (i + 1) % (Nsteps / 500) == 0 )
        {
            eigFactor = eigQuality(MC,S->C,S->Ho,S->Hint,E);
            // Print in screen them on screen
            printf("\n%5.1lf%%     %11.7lf",(100.0*i)/Nsteps,creal(E));
            printf("     %7.4lf       %9.6lf",R2,cabs(vir/E));
            printf("      %8.5lf",eigFactor);
        }



        // CHECK IF THE ENERGY HAS STABILIZED TO STOP PROCESS
        // At every 200 time-steps performed
        if ( (i + 1) % 200 == 0)
        {
            // Check if the energy stop decreasing and break the process
            if (fabs(creal(E - prevE) / creal(prevE)) < 5E-10)
            {
                CCSFree(cnmat);
                free(upper);
                free(lower);
                free(mid);

                sepline();

                printf("\nProcess ended before because ");
                printf("energy stop decreasing.\n\n");

                return i + 1;
            }

            prevE = E;
        }
    }

    if (200 * nc < 5E7)
    {
        if (2 * nc / 3 < 200) k = 2 * nc / 3;
        else                  k = 200;
    }
    else k = 5E7 / nc;

    E = LanczosGround(k,MC,S->Omat,S->C) / Npar;
    // Renormalize coeficients
    renormalizeVector(nc, S->C, 1.0);

    sepline();
    printf("\nFinal E/particle = %.7lf\n", creal(E));
    printf("Process did not stop automatically because energy were");
    printf(" varying above accepted tolerance\n\n");

    CCSFree(cnmat);
    free(upper);
    free(lower);
    free(mid);

    return Nsteps + 1;
}



void realCNSM(EqDataPkg MC, ManyBodyPkg S, double dt, int Nsteps, int cyclic,
     char prefix [], int recInterval)
{

    int l,
        i,
        j,
        k,
        nc,
        Npar,
        Mpos,
        Morb;

    double
        checkOrbNorm,
        checkOverlap,
        norm,
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
        V;

    Carray
        rho_vec,
        orb_vec,
        upper,
        lower,
        mid;

    CCSmat
        cnmat;



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

    rho_vec = carrDef(Morb * Morb);
    orb_vec = carrDef(Morb * Mpos);
    // Crank-Nicolson tridiagonal system
    upper = carrDef(Mpos - 1);
    lower = carrDef(Mpos - 1);
    mid = carrDef(Mpos - 1);



    // OPEN FILE TO RECORD 1-BODY DENSITY MATRIX
    strcpy(fname,"output/");
    strcat(fname,prefix);
    strcat(fname,"_t_realtime.dat");

    t_file = fopen(fname, "w");
    if (t_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(t_file, "# time instants solution is recorded");



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

    RegularizeMat(Morb,Npar*1E-7,S->rho1);



    // initial energy
    E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
    norm = carrMod(nc, S->C);
    checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
    checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

    printf("\n  time         E/Npar      Overlap");
    printf("     Coef-Norm      Orb-Avg-Norm");
    sepline();
    printf("%10.6lf  %11.6lf",0.0,creal(E));
    printf("    %8.2E    %9.7lf",checkOverlap,norm);
    printf("      %9.7lf",checkOrbNorm);

    // record initial data
    RowMajor(Morb, Morb, S->rho1, rho_vec);
    RowMajor(Morb, Mpos, S->Omat, orb_vec);
    carr_inline(rho_file, Morb * Morb, rho_vec);
    carr_inline(orb_file, Morb * Mpos, orb_vec);
    fprintf(t_file,"\n%.6lf",0*dt);



    // Configure the linear system from Crank-Nicolson scheme with half
    // time step from split-step approach
    cnmat = CNmat(Mpos, dx, dt/2, a2, a1, g, V, cyclic, upper, lower, mid);



    l = 1;
    for (i = 0; i < Nsteps; i++)
    {

        // HALF STEP THE COEFFICIENTS
        LanczosIntegrator(3,MC,S->Ho,S->Hint,dt/2,S->C);

        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);
        RegularizeMat(Morb,Npar*1E-7,S->rho1);



        // FULL TIME STEP ORBITALS
        LP_CNSM(Mpos,Morb,cnmat,upper,lower,mid,S->Omat);

        if (cyclic)
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = S->Omat[k][0]; }
        else
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = 0;             }

        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

        realNL_RK4(MC,S,dt);

        LP_CNSM(Mpos, Morb, cnmat, upper, lower, mid, S->Omat);

        if (cyclic)
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = S->Omat[k][0]; }
        else
        { for (k = 0; k < Morb; k++) S->Omat[k][Mpos-1] = 0;             }

        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);



        // ANOTHER HALF STEP FOR COEFFICIENTS
        LanczosIntegrator(3,MC,S->Ho,S->Hint,dt/2,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);
        RegularizeMat(Morb,Npar*1E-7,S->rho1);



        E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
        norm = carrMod(nc,S->C);
        checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
        checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

        // If the number of steps done reach record interval, record data
        if (l == recInterval)
        {
            printf("\n%10.6lf  %11.6lf",(i+1)*dt,creal(E));
            printf("    %8.2E    %9.7lf",checkOverlap,norm);
            printf("      %9.7lf",checkOrbNorm);
            RowMajor(Morb, Morb, S->rho1, rho_vec);
            RowMajor(Morb, Mpos, S->Omat, orb_vec);
            carr_inline(rho_file, Morb * Morb, rho_vec);
            carr_inline(orb_file, Morb * Mpos, orb_vec);
            fprintf(t_file,"\n%.6lf",(i+1)*dt);
            l = 1;
        }
        else { l = l + 1; }

    }

    E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
    norm = carrMod(nc,S->C);
    checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
    checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

    printf("\n%10.6lf  %11.6lf",Nsteps*dt,creal(E));
    printf("    %8.2E    %9.7lf",checkOverlap,norm);
    printf("      %9.7lf",checkOrbNorm);
    // RowMajor(Morb, Morb, S->rho1, rho_vec);
    // RowMajor(Morb, Mpos, S->Omat, orb_vec);
    // carr_inline(rho_file, Morb * Morb, rho_vec);
    // carr_inline(orb_file, Morb * Mpos, orb_vec);
    // fprintf(t_file,"\n%.6lf",Nsteps*dt);

    sepline();

    CCSFree(cnmat);
    free(upper);
    free(lower);
    free(mid);
    free(rho_vec);
    free(orb_vec);

    fclose(t_file);
    fclose(rho_file);
    fclose(orb_file);
}



void realFFT(EqDataPkg MC, ManyBodyPkg S, double dt, int Nsteps,
     char prefix [], int recInterval)
{

    int l,
        i,
        m,
        k,
        nc,
        Npar,
        Mpos,
        Morb,
        isTrapped;

    MKL_LONG
        p;

    double
        checkOrbNorm,
        checkOverlap,
        norm,
        freq,
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
        V;

    Carray
        rho_vec,
        orb_vec,
        exp_der;

    DFTI_DESCRIPTOR_HANDLE
        desc;



    isTrapped = strcmp(MC->Vname, "harmonic");

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

    // matrices in row-major form to record data
    rho_vec = carrDef(Morb * Morb);
    orb_vec = carrDef(Morb * Mpos);



    // OPEN FILE TO RECORD 1-BODY DENSITY MATRIX
    strcpy(fname,"output/");
    strcat(fname,prefix);
    strcat(fname,"_t_realtime.dat");

    t_file = fopen(fname, "w");
    if (t_file == NULL)
    {
        printf("\n\nERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }
    fprintf(t_file, "# time instants solution is recorded");



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

    RegularizeMat(Morb,Npar*1E-7,S->rho1);



    // initial energy
    E = Energy(Morb, S->rho1, S->rho2, S->Ho, S->Hint) / Npar;
    norm = carrMod(nc, S->C);
    checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
    checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

    printf("\n  time         E/Npar      Overlap");
    printf("     Coef-Norm      Orb-Avg-Norm");
    sepline();
    printf("%10.6lf  %11.6lf",0.0,creal(E));
    printf("    %8.2E    %9.7lf",checkOverlap,norm);
    printf("      %9.7lf",checkOrbNorm);

    // record initial data
    RowMajor(Morb, Morb, S->rho1, rho_vec);
    RowMajor(Morb, Mpos, S->Omat, orb_vec);
    carr_inline(rho_file, Morb * Morb, rho_vec);
    carr_inline(orb_file, Morb * Mpos, orb_vec);
    fprintf(t_file,"\n%.6lf",0*dt);



    l = 1;
    for (i = 0; i < Nsteps; i++)
    {

        // HALF STEP THE COEFFICIENTS
        LanczosIntegrator(3,MC,S->Ho,S->Hint,dt/2,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);
        RegularizeMat(Morb,Npar*1E-7,S->rho1);



        // FULL TIME STEP ORBITALS
        LP_FFT(Mpos,Morb,&desc,exp_der,S->Omat);

        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);

        realNLTRAP_RK4(MC,S,dt);

        LP_FFT(Mpos,Morb,&desc,exp_der,S->Omat);

        SetupHo(Morb, Mpos, S->Omat, dx, a2, a1, V, S->Ho);
        SetupHint(Morb, Mpos, S->Omat, dx, g, S->Hint);



        // ANOTHER HALF STEP FOR COEFFICIENTS
        LanczosIntegrator(3,MC,S->Ho,S->Hint,dt/2,S->C);
        OBrho(Npar,Morb,MC->Map,MC->IF,S->C,S->rho1);
        TBrho(Npar,Morb,MC->Map,MC->MapOT,MC->MapTT,MC->strideOT,MC->strideTT,
              MC->IF,S->C,S->rho2);
        RegularizeMat(Morb,Npar*1E-7,S->rho1);



        // Check if the boundaries are still good
        if ( (i + 1) % 20 == 0 && isTrapped == 0)
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

            free(orb_vec);
            orb_vec = carrDef(Morb * Mpos);

        }



        E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
        norm = carrMod(nc,S->C);
        checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
        checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

        if (checkOverlap > 1E-5)
        {
            printf("\n\nERROR : Critical loss of orthogonality ");
            printf("among orbitals. Exiting ...\n\n");
            fclose(t_file);
            fclose(rho_file);
            fclose(orb_file);
            exit(EXIT_FAILURE);
        }

        // If the number of steps done reach record interval, record data
        if (l == recInterval)
        {
            printf("\n%10.6lf  %11.6lf",(i+1)*dt,creal(E));
            printf("    %8.2E    %9.7lf",checkOverlap,norm);
            printf("      %9.7lf",checkOrbNorm);
            RowMajor(Morb, Morb, S->rho1, rho_vec);
            RowMajor(Morb, Mpos, S->Omat, orb_vec);
            carr_inline(rho_file, Morb * Morb, rho_vec);
            carr_inline(orb_file, Morb * Mpos, orb_vec);
            fprintf(t_file,"\n%.6lf",(i+1)*dt);
            l = 1;
        }
        else { l = l + 1; }

    }

    E = Energy(Morb,S->rho1,S->rho2,S->Ho,S->Hint) / Npar;
    norm = carrMod(nc,S->C);
    checkOverlap = overlapFactor(Morb,Mpos,dx,S->Omat);
    checkOrbNorm = avgOrbNorm(Morb,Mpos,dx,S->Omat);

    printf("\n%10.6lf  %11.6lf",Nsteps*dt,creal(E));
    printf("    %8.2E    %9.7lf",checkOverlap,norm);
    printf("      %9.7lf",checkOrbNorm);
    // RowMajor(Morb, Morb, S->rho1, rho_vec);
    // RowMajor(Morb, Mpos, S->Omat, orb_vec);
    // carr_inline(rho_file, Morb * Morb, rho_vec);
    // carr_inline(orb_file, Morb * Mpos, orb_vec);
    // fprintf(t_file,"\n%.6lf",Nsteps*dt);

    sepline();

    free(exp_der);
    free(rho_vec);
    free(orb_vec);

    p = DftiFreeDescriptor(&desc);

    fclose(t_file);
    fclose(rho_file);
    fclose(orb_file);
}
