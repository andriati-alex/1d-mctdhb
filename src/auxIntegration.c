#include "auxIntegration.h"

void recorb_inline(FILE * f, int Morb, int Mrec, int Mpos, Cmatrix Omat)
{

/** WRITE ALL ORBITALS IN FILE LINE
    ===============================
    Concatenate all orbitals that corresponds to rows in 'Omat' and
    write data in file 'f' current line. There is a 'safety' border
    since the gas density can expand during the dynamics. The total
    domain with the safety border must be  larger  than the current
    solution domain and all extra points are recorded as zero   **/

    int
        extra,
        i,
        k,
        j;

    double
        real,
        imag;

    if (f == NULL)
    {
        printf("\n\n\nERROR: NULL file in carr_inline routine");
        printf(" in module src/inout.c\n\n");
        exit(EXIT_FAILURE);
    }

    // extra number of points to the left and right of solution
    extra = (Mrec - Mpos) / 2;

    // domain of the solution shall not exceed recording domain
    if (Mpos > Mrec)
    {
        printf("\n\nERROR: exceed max domain boundaries in dynamics\n\n");
        exit(EXIT_FAILURE);
    }

    for (k = 0; k < Morb; k++)
    {
        for (i = 0; i < Mrec; i++)
        {
            j = i - extra;
            if (j < 0 || j > Mpos - 1)
            {
                real = 0;
                imag = 0;
            }
            else
            {
                real = creal(Omat[k][j]);
                imag = cimag(Omat[k][j]);
            }
            if (imag >= 0) fprintf(f,"(%.15E+%.15Ej) ",real,imag);
            else           fprintf(f,"(%.15E%.15Ej) ",real,imag);
        }
    }

    fprintf(f,"\n");
}



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
    of the overlap matrix | <ORB_i,ORB_j> | divide by then umber of orbitals */

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

    if (chunkSize < 2)
    {
        printf("\n\nERROR : chunk size must be greater than 1 ");
        printf("to compute border norm!\n\n");
        exit(EXIT_FAILURE);
    }

    if (chunkSize > m / 2)
    {
        printf("\n\nERROR : chunk size in border norm is too large!\n");
        printf("Exceeded 1/2 the size of domain.\n\n");
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

    if (n / 50 > 2) { chunkSize = n / 50; }
    else            { chunkSize = 2;      }

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
    while ( fabs(oldx[minR2]) > 7 * R2 ) minR2 = minR2 + 1;

    // Method 2 : take the norm beginning from the boundary and get the
    // position where it exceed a certain tolerance (last arg below)
    minId = 0;
    for (i = 0; i < Morb; i++)
    {
        j = NonVanishingId(Mpos,S->Omat[i],olddx,5E-8/Morb);
        minId = minId + j;
    }
    minId = minId / Morb;

    // Take the weighted average of the two methods
    minId = (minId + minR2) / 2;

    // Check if it is woth to resize the domain comparing the
    // percentual reduction that would be done
    if (100 * abs(oldx[minId] - oldxi) / (oldxf - oldxi) < 5)
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
    printf("Useless border from |x| = %.2lf",oldxf-minId*olddx);
    printf("    *    Domain resized to [%.2lf,%.2lf]",xi,xf);
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
        j = NonVanishingId(Mpos,S->Omat[i],dx,1E-6/Morb);
        minId = minId + j;
    }
    minId = minId / Morb;

    // Take the weighted average
    minId = (minId + 2 * minR2) / 3;

    // Check if it is woth to resize the domain, i.e, orbitals too
    // close to the boundaries
    if (minId > Mpos/20)
    {
        free(oldx);
        return;
    }

    extra = Mpos / 15;
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
