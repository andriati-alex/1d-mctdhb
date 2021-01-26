#include "calculus.h"

double complex Csimps(int n, Carray f, double h)
{

/** COMPUTE NUMERICALLY THE INTEGRAL OF FUNCTION USING SIMPSON (complex) **/

    int
        i;
    double complex
        sum;

    sum = 0;

    if (n < 3)
    {
        printf("\n\nERROR : less than 3 point to integrate by simps !\n\n");
        exit(EXIT_FAILURE);
    }

    if (n % 2 == 0)
    {
    //  Case the number of points is even then must integrate the last
    //  chunk using simpson's 3/8 rule to maintain accuracy
        for (i = 0; i < (n - 4); i = i + 2)
        {
            sum = sum + f[i] + 4 * f[i + 1] + f[i + 2];
        }
        sum = sum * h / 3; // End 3-point simpsons intervals
        sum = sum + (f[n-4] + 3 * (f[n-3] + f[n-2]) + f[n-1]) * 3 * h / 8;

    }
    else
    {
        for (i = 0; i < n - 2; i = i + 2)
        {
            sum = sum + f[i] + 4 * f[i + 1] + f[i + 2];
        }
        sum = sum * h / 3; // End 3-point simpsons intervals
    }

    return sum;
}



double Rsimps(int n, Rarray f, double h)
{

/** COMPUTE NUMERICALLY THE INTEGRAL OF FUNCTION USING SIMPSON (real) **/

    int
        i;
    double
        sum;

    sum = 0;

    if (n < 3)
    {
        printf("\n\nERROR : less than 3 point to integrate by simps !\n\n");
        exit(EXIT_FAILURE);
    }

    if (n % 2 == 0)
    {
    //  Case the number of points is even then must integrate the last
    //  chunk using simpson's 3/8 rule to maintain accuracy
        for (i = 0; i < (n - 4); i = i + 2)
        {
            sum = sum + f[i] + 4 * f[i + 1] + f[i + 2];
        }
        sum = sum * h / 3; // End 3-point simpsons intervals
        sum = sum + (f[n-4] + 3 * (f[n-3] + f[n-2]) + f[n-1]) * 3 * h / 8;

    }
    else
    {
        for (i = 0; i < n - 2; i = i + 2)
        {
            sum = sum + f[i] + 4 * f[i + 1] + f[i + 2];
        }
        sum = sum * h / 3; // End 3-point simpsons intervals
    }

    return sum;
}



void renormalize(int n, Carray f, double dx, double norm)
{

/** Given a function f in discretized positions in a domain with n
    points and spacing dx among them, multiply by a factor so that
    change to 'norm' the L2 norm. **/

    int
        i;
    double
        renorm;
    Rarray
        ModSquared;

    ModSquared = rarrDef(n);
    carrAbs2(n,f,ModSquared);

    renorm = norm*sqrt(1.0/Rsimps(n,ModSquared,dx));
    for (i = 0; i < n; i++) f[i] = f[i]*renorm;

    free(ModSquared);
}



double complex innerL2(int n, Carray fstar, Carray f, double h)
{

/** Inner product according to L2-norm of functions discretized
    in a 1D-grid with 'n' points and spacing 'h' among them **/

    int
        i;
    double complex
        overlap;
    Carray
        integ;

    integ = carrDef(n);

    for (i = 0; i < n; i++) integ[i] = conj(fstar[i]) * f[i];

    overlap = Csimps(n,integ,h);
    free(integ);
    return overlap;
}



void Ortonormalize(int Nfun, int Npts, double dx, Cmatrix F)
{

/** Given a set of functions F[k][:], 0 <= k < Nfun, where column index
    enumerate the grid points, orthonomalize the set using Gram-Schimdt **/

    int
        i,
        j,
        k;
    Carray
        integ;

    integ = carrDef(Npts);
    renormalize(Npts,F[0],dx,1.0); // must initiate with normalized function

    for (i = 1; i < Nfun; i++)
    {
        for (j = 0; j < i; j++)
        {
            // The projection are integrals of the product below
            for (k = 0; k < Npts; k++) integ[k] = conj(F[j][k]) * F[i][k];
            // Iterative Gram-Schmidt (see wikipedia) is
            // different to improve  numerical stability
            for (k = 0; k < Npts; k++)
            {
                F[i][k] = F[i][k] - Csimps(Npts,integ,dx)*F[j][k];
            }
        }
        // normalized to unit the new vector
        renormalize(Npts,F[i],dx,1.0);
    }

    free(integ);
}



void dxFFT(int Npts, Carray f, double dx, Carray dfdx)
{

/** Compute derivative of a function in Npts grid points with
    periodic boundary conditions,  that  is  f[n - 1] = f[0].
    Use Fast Fouriers Transforms(FFT) in this implementation
    Output parameter : dfdx
    Poor performance compared to finite-difference method. **/

    int
        i,
        N;
    double
        Ndx,
        freq;
    MKL_LONG
        s;      // returned status of called MKL FFT functions
    DFTI_DESCRIPTOR_HANDLE
        desc;   // descriptor with grid and nomalization info

    N = Npts-1; // Assume the connection f[n-1] = f[0] at the boundary
    Ndx = N*dx; // total domain length

    carrCopy(N,f,dfdx); // Copy to execute in-place computation.

    s = DftiCreateDescriptor(&desc,DFTI_DOUBLE,DFTI_COMPLEX,1,N);
    s = DftiSetValue(desc,DFTI_FORWARD_SCALE,1.0/sqrt((double)N));
    s = DftiSetValue(desc,DFTI_BACKWARD_SCALE,1.0/sqrt((double)N));
    // s = DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    s = DftiCommitDescriptor(desc);

    s = DftiComputeForward(desc,dfdx);

    for (i = 0; i < N; i++)
    {
        if (i <= (N-1)/2) freq = (2*PI*i)/Ndx;
        else              freq = (2*PI*(i-N))/Ndx;
        dfdx[i] = dfdx[i]*freq*I;
    }

    s = DftiComputeBackward(desc,dfdx);
    s = DftiFreeDescriptor(&desc);

    dfdx[N] = dfdx[0]; // boundary point
}



void d2xFFT(int Npts, Carray f, double dx, Carray dfdx)
{

/** Compute 2nd derivative of 'f' given in 'Npts' grid points
    With Fast Fouriers Transforms(FFT)
    Output parameter : dfdx        **/

    int
        i,
        N;
    double
        Ndx,
        freq;
    MKL_LONG
        s;      // returned status of called MKL FFT functions
    DFTI_DESCRIPTOR_HANDLE
        desc;   // descriptor with grid and nomalization info

    N = Npts-1; // Assumes the connection f[n-1] = f[0] at the boundary
    Ndx = N*dx; // total domain length

    carrCopy(N,f,dfdx); // Copy to execute in-place computation.

    s = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX,1,N);
    s = DftiSetValue(desc,DFTI_FORWARD_SCALE,1.0/sqrt((double)N));
    s = DftiSetValue(desc,DFTI_BACKWARD_SCALE,1.0/sqrt((double)N));
    // s = DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    s = DftiCommitDescriptor(desc);

    s = DftiComputeForward(desc,dfdx);  // FORWARD FFT

    for (i = 0; i < N; i++)
    {
        if (i <= (N-1)/2) freq = (2*PI*i)/Ndx;
        else              freq = (2*PI*(i-N))/Ndx;
        dfdx[i] = dfdx[i] * (-freq*freq);
    }

    s = DftiComputeBackward(desc,dfdx); // BACKWARD FFT
    s = DftiFreeDescriptor(&desc);

    dfdx[N] = dfdx[0]; // boundary point
}



void dxFD(int Npts, Carray f, double dx, Carray dfdx)
{

/** Compute derivative of function 'f' in 'Npts' grid points with
    spacing 'dx' considering  periodic boundary conditions,  that
    is f[n-1] = f[0],  with 4th-order Finite-Differences accuracy
    Output parameter : dfdx                                   **/

    int
        n,
        i;
    double
        r;

    n = Npts;            // make the life easier
    r = 1.0 / (12 * dx); // ratio for a fourth-order scheme

    // COMPUTE USING PERIODIC BOUNDARY CONDITIONS

    dfdx[0]   = ( f[n-3] - f[2] + 8 * (f[1] - f[n-2]) ) * r;
    dfdx[1]   = ( f[n-2] - f[3] + 8 * (f[2] - f[0]) )   * r;
    dfdx[n-2] = ( f[n-4] - f[1] + 8 * (f[0] - f[n-3]) ) * r;
    dfdx[n-1] = dfdx[0]; // assume last point as the boundary

    for (i = 2; i < n - 2; i++)
    {
        dfdx[i] = ( f[i-2] - f[i+2] + 8 * (f[i+1] - f[i-1]) ) * r;
    }

}



void d2xFD(int Npts, Carray f, double dx, Carray dfdx)
{

/** Compute 2nd order derivative of function 'f' in 'Npts' grid
    points  with  spacing  'dx'  considering  periodic boundary
    conditions, that is f[n-1] = f[0],  with 2th-order accuracy
    Output parameter : dfdx                                 **/

    int
        i;

    dfdx[0] = (f[1] - 2*f[0] + f[Npts-2])/dx/dx;
    dfdx[Npts-1] = dfdx[0];

    for (i = 1; i < Npts-1; i++) dfdx[i] = (f[i+1] - 2*f[i] + f[i-1])/dx/dx;
}
