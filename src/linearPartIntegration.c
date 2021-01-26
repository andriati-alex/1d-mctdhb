#include "linearPartIntegration.h"

void FDrhs(EqDataPkg MC, doublec dt, Carray f, Carray out, int isTrapped)
{
    int
        i,
        N;

    double
        a2,
        dx;

    double complex
        a1,
        mid,
        upper,
        lower;

    Rarray
        V;

    N = MC->Mpos;
    a2 = MC->a2;
    a1 = MC->a1;
    dx = MC->dx;
    V = MC->V;

    if (isTrapped)
    {
        mid = I - a2*dt/dx/dx + dt*V[0]/2;
        upper = a2*dt/dx/dx/2 + a1*dt/dx/4;
        out[0] = mid * f[0] + upper * f[1];
    }
    else
    {
        mid = I - a2*dt/dx/dx + dt*V[0]/2;
        upper = a2*dt/dx/dx/2 + a1*dt/dx/4;
        lower = a2*dt/dx/dx/2 - a1*dt/dx/4;
        out[0] = mid * f[0] + upper * f[1] + lower * f[N-2];
    }

    for (i = 1; i < N-1; i++)
    {
        mid = I - a2*dt/dx/dx + dt*V[i]/2;
        upper = a2*dt/dx/dx/2 + a1*dt/dx/4;
        lower = a2*dt/dx/dx/2 - a1*dt/dx/4;
        out[i] = mid * f[i] + upper * f[i+1] + lower * f[i-1];
    }

    if (isTrapped)
    {
        mid = I - a2*dt/dx/dx + dt*V[N-1]/2;
        lower = a2*dt/dx/dx/2 - a1*dt/dx/4;
        out[N-1] = mid * f[N-1] + lower * f[N-2];
    }
    else
    {
        mid = I - a2*dt/dx/dx + dt*V[N-2]/2;
        upper = a2*dt/dx/dx/2 + a1*dt/dx/4;
        lower = a2*dt/dx/dx/2 - a1*dt/dx/4;
        out[N-2] = mid * f[N-2] + upper * f[0] + lower * f[N-3];
    }

}



void linearCN(EqDataPkg MC, Carray upper, Carray lower, Carray mid,
             Cmatrix Orb, int isTrapped, doublec dt)
{

    int
        k,
        size;

    Carray
        rhs;

    rhs = carrDef(MC->Mpos);
    if (isTrapped) size = MC->Mpos;
    else           size = MC->Mpos - 1;

    // FOR EACH ORBITAL SOLVE THE CRANK-NICOLSON TRI-DIAGONAL SYSTEM
    for (k = 0; k < MC->Morb; k++)
    {
        FDrhs(MC,dt,Orb[k],rhs,isTrapped);
        if (isTrapped)
        {
            triDiag(size,upper,lower,mid,rhs,Orb[k]);
        }
        else
        {
            // triCyclicLU(size,upper,lower,mid,rhs,Orb[k]);
            triCyclicSM(size,upper,lower,mid,rhs,Orb[k]);
            Orb[k][size] = Orb[k][0];
        }
    }

    free(rhs);
}



void linearFFT(int Mpos, int Morb, DFTI_DESCRIPTOR_HANDLE * desc,
               Carray exp_der, Cmatrix Orb)
{

    int
        k;

    MKL_LONG
        s;

    Carray
        forward_fft,
        back_fft;

    forward_fft = carrDef(Mpos - 1);
    back_fft = carrDef(Mpos - 1);

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
