#include "calculus.h"





double complex Ctrapezium(int n, Carray f, double h)
{
    int
        i;

    double complex
        sum;

    sum = 0;
    for (i = 0; i < n - 1; i++) sum = sum + (f[i+1] + f[i]) * 0.5 * h;

    return sum;
}





double Rtrapezium(int n, Rarray f, double h)
{
    int
        i;

    double
        sum;

    sum = 0;
    for (i = 0; i < n - 1; i++) sum = sum + (f[i+1] + f[i]) * 0.5 * h;

    return sum;
}





double complex Csimps(int n, Carray f, double h)
{

    int
        i;

    double complex
        sum;

    sum = 0;

    if (n < 3)
    {
        printf("\n\n\tERROR : less than 3 point to integrate by simps !\n\n");
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

    int
        i;

    double
        sum;

    sum = 0;

    if (n < 3)
    {
        printf("\n\n\tERROR : less than 3 point to integrate by simps !\n\n");
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
  * points and spacing dx among them, multiply by a factor so that
  * change to 'norm' the L2 norm. **/

    int
        i;

    double
        renorm;

    Rarray
        ModSquared;

    ModSquared = rarrDef(n);

    carrAbs2(n, f, ModSquared);

    renorm = norm * sqrt(1.0 / Rsimps(n, ModSquared, dx));
    for (i = 0; i < n; i++) f[i] = f[i] * renorm;

    free(ModSquared);
}





void Ortonormalize(int Mfun, int Mpos, double dx, Cmatrix F)
{

/** Given F[k][:] with the k-th funciion of the basis, that is
  * columns enumerate discretized positions and lines the func
  * orthonomalize the set using Gram-Schimdt method **/

    int
        i,
        j,
        k;

    Carray
        toInt = carrDef(Mpos);

    renormalize(Mpos,F[0],dx,1.0);

    for (i = 1; i < Mfun; i++)
    {

        for (j = 0; j < i; j++)
        {
            // The projection are integrals of the product below

            for (k = 0; k < Mpos; k++) toInt[k] = conj(F[j][k]) * F[i][k];

            // Iterative Gram-Schmidt (see wikipedia)

            for (k = 0; k < Mpos; k++)
                F[i][k] = F[i][k] - Csimps(Mpos,toInt,dx) * F[j][k];
        }

        // normalized to unit the new vector

        renormalize(Mpos, F[i], dx, 1.0);
    }

    free(toInt);
}





void dxFFT(int n, Carray f, double dx, Carray dfdx)
{

/** Compute derivative of a function in n discretized positions with
  * periodic boundary conditions,  that is f[n - 1] = f[0]. Use Fast
  * Fouriers Transforms(FFT) to do the job
  *
  * Output parameter : dfdx
  *
  * Poor performance compared to finite-difference method. **/

    int
        i,
        N;

    double
        Ndx,
        freq;

    MKL_LONG s; // status of called MKL FFT functions

    DFTI_DESCRIPTOR_HANDLE desc;

    N = n - 1; // Assumes the connection f[n-1] = f[0] at the boundary

    Ndx = N * dx; // total domain length

    carrCopy(N, f, dfdx); // Copy to execute in-place computation.

    s = DftiCreateDescriptor(&desc, DFTI_DOUBLE, DFTI_COMPLEX, 1, N);
    s = DftiSetValue(desc, DFTI_FORWARD_SCALE, 1.0 / sqrt((double) N));
    s = DftiSetValue(desc, DFTI_BACKWARD_SCALE, 1.0 / sqrt((double) N));
    // s = DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    s = DftiCommitDescriptor(desc);

    s = DftiComputeForward(desc, dfdx);

    for (i = 0; i < N; i++) {
        if (i <= (N - 1) / 2) { freq = (2 * PI * i) / Ndx;       }
        else                  { freq = (2 * PI * (i - N)) / Ndx; }
        dfdx[i] = dfdx[i] * freq * I;
    }

    s = DftiComputeBackward(desc, dfdx);

    s = DftiFreeDescriptor(&desc);

    dfdx[N] = dfdx[0]; // boundary point
}





void dxFD(int n, Carray f, double dx, Carray dfdx)
{

/** Compute derivative of a function in n discretized positions with
  * periodic boundary conditions, that is f[n-1] = f[0]. Use Finite-
  * Differences(FD) to do the job
  *
  * Output parameter : dfdx
  *
**/

    int
        i;

    double
        r;

    r = 1.0 / (12 * dx); // ratio for a fourth-order scheme

    // compute using periodic boundary conditions

    dfdx[0]   = ( f[n-3] - f[2] + 8 * (f[1] - f[n-2]) ) * r;

    dfdx[1]   = ( f[n-2] - f[3] + 8 * (f[2] - f[0]) )   * r;

    dfdx[n-2] = ( f[n-4] - f[1] + 8 * (f[0] - f[n-3]) ) * r;

    dfdx[n-1] = dfdx[0]; // assume last point as the boundary

    for (i = 2; i < n - 2; i++)
    {
        dfdx[i] = ( f[i-2] - f[i+2] + 8 * (f[i+1] - f[i-1]) ) * r;
    }

}
