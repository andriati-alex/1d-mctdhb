#include "observables.h"





void SetupHo (int Morb, int Mpos, Cmatrix Omat, double dx, double a2,
     doublec a1, Rarray V, Cmatrix Ho)
{

/** Configure one-body hamiltonian matrix elements in a chosen orbital basis
  *
  * Output parameter : Ho
  *
  * REMIND FOR DIRAC DELTA BARRIER POTENTIAL
  *
  * The potential part V must then be integrated by trapezium/rectangle
  * rule. Therefore erase V[i] part contribution in the looping. Change
  * the following line that define a matrix element Ho[i][j] :
  *
  * part = Csimps(M,toInt,dx);
  * for (k = 0; k < M; k++) part += dx * V[k] * conj(Omat[i][k]) * Omat[j][k];
  *
  * This shall work for a numerical implementation like 1 / dx
  *
  ***************************************************************************/


    int i,
        j,
        k;

    double complex
        part;

    Carray
        ddxi  = carrDef(Mpos),
        ddxj  = carrDef(Mpos),
        toInt = carrDef(Mpos);



    for (i = 0; i < Morb; i++)
    {

        dxFD(Mpos,Omat[i],dx,ddxi);

        for (j = i + 1; j < Morb; j++)
        {

            dxFD(Mpos,Omat[j],dx,ddxj);

            for (k = 0; k < Mpos; k++)
            {
                part = - a2 * conj(ddxi[k]) * ddxj[k];
                part = part + a1 * conj(Omat[i][k]) * ddxj[k];
                part = part + V[k] * conj(Omat[i][k]) * Omat[j][k];
                toInt[k] = part;
            }

            part = Csimps(Mpos,toInt,dx);
            Ho[i][j] = part;
            Ho[j][i] = conj(part);

        }

        for (k = 0; k < Mpos; k++)
        {
            part = - a2 * conj(ddxi[k]) * ddxi[k];
            part = part + a1 * conj(Omat[i][k]) * ddxi[k];
            part = part + V[k] * conj(Omat[i][k]) * Omat[i][k];
            toInt[k] = part;
        }

        part = Csimps(Mpos,toInt,dx);
        Ho[i][i] = creal(part);
    }

    free(ddxi); free(ddxj); free(toInt);

}





void SetupHint (int Morb, int Mpos, Cmatrix Omat, double dx, double g,
     Carray Hint)
{

/** Configure two-body hamiltonian matrix elements in a chosen orbital basis
  * for contact interactions
  *
  * Output parameter : Hint
  *
  *************************************************************************/

    int i,
        k,
        s,
        q,
        l,
        M,
        M2,
        M3;

    double complex
        Integral;

    Carray
        toInt;

    M  = Morb;
    M2 = M * M;
    M3 = M * M2;

    toInt = carrDef(Mpos);

    for (k = 0; k < Morb; k++)
    {

        for (i = 0; i < Mpos; i++)
        {
            toInt[i] = conj(Omat[k][i]*Omat[k][i]) * Omat[k][i]*Omat[k][i];
        }

        Hint[k * (1 + M + M2 + M3)] = g * Csimps(Mpos,toInt,dx);

        for (s = k + 1; s < Morb; s++)
        {

            for (i = 0; i < Mpos; i++)
            {
                toInt[i] = conj(Omat[k][i]*Omat[s][i]) * Omat[k][i]*Omat[k][i];
            }

            Integral = g * Csimps(Mpos,toInt,dx);

            Hint[k + s * M + k * M2 + k * M3] = Integral;
            Hint[s + k * M + k * M2 + k * M3] = Integral;
            Hint[k + k * M + k * M2 + s * M3] = conj(Integral);
            Hint[k + k * M + s * M2 + k * M3] = conj(Integral);

            for (i = 0; i < Mpos; i++)
            {
                toInt[i] = conj(Omat[s][i]*Omat[k][i]) * Omat[s][i]*Omat[s][i];
            }

            Integral = g * Csimps(Mpos,toInt,dx);

            Hint[s + k * M + s * M2 + s * M3] = Integral;
            Hint[k + s * M + s * M2 + s * M3] = Integral;
            Hint[s + s * M + s * M2 + k * M3] = conj(Integral);
            Hint[s + s * M + k * M2 + s * M3] = conj(Integral);

            for (i = 0; i < Mpos; i++)
            {
                toInt[i] = conj(Omat[k][i]*Omat[s][i]) * Omat[s][i]*Omat[k][i];
            }

            Integral = g * Csimps(Mpos,toInt,dx);

            Hint[k + s * M + s * M2 + k * M3] = Integral;
            Hint[s + k * M + s * M2 + k * M3] = Integral;
            Hint[s + k * M + k * M2 + s * M3] = Integral;
            Hint[k + s * M + k * M2 + s * M3] = Integral;

            for (i = 0; i < Mpos; i++)
            {
                toInt[i] = conj(Omat[k][i]*Omat[k][i]) * Omat[s][i]*Omat[s][i];
            }

            Integral = g * Csimps(Mpos,toInt,dx);

            Hint[k + k * M + s * M2 + s * M3] = Integral;
            Hint[s + s * M + k * M2 + k * M3] = conj(Integral);

            for (q = s + 1; q < Morb; q++)
            {

                for (i = 0; i < Mpos; i++)
                {
                    toInt[i] = conj(Omat[k][i]*Omat[s][i]) * \
                               Omat[q][i]*Omat[k][i];
                }

                Integral = g * Csimps(Mpos,toInt,dx);

                Hint[k + s * M + q * M2 + k * M3] = Integral;
                Hint[k + s * M + k * M2 + q * M3] = Integral;
                Hint[s + k * M + k * M2 + q * M3] = Integral;
                Hint[s + k * M + q * M2 + k * M3] = Integral;

                Hint[k + q * M + s * M2 + k * M3] = conj(Integral);
                Hint[k + q * M + k * M2 + s * M3] = conj(Integral);
                Hint[q + k * M + k * M2 + s * M3] = conj(Integral);
                Hint[q + k * M + s * M2 + k * M3] = conj(Integral);

                for (i = 0; i < Mpos; i++)
                {
                    toInt[i] = conj(Omat[s][i]*Omat[k][i]) * \
                               Omat[q][i]*Omat[s][i];
                }

                Integral = g * Csimps(Mpos,toInt,dx);

                Hint[s + k * M + q * M2 + s * M3] = Integral;
                Hint[k + s * M + q * M2 + s * M3] = Integral;
                Hint[k + s * M + s * M2 + q * M3] = Integral;
                Hint[s + k * M + s * M2 + q * M3] = Integral;

                Hint[s + q * M + k * M2 + s * M3] = conj(Integral);
                Hint[s + q * M + s * M2 + k * M3] = conj(Integral);
                Hint[q + s * M + s * M2 + k * M3] = conj(Integral);
                Hint[q + s * M + k * M2 + s * M3] = conj(Integral);

                for (i = 0; i < Mpos; i++)
                {
                    toInt[i] = conj(Omat[q][i]*Omat[s][i]) * \
                               Omat[k][i]*Omat[q][i];
                }

                Integral = g * Csimps(Mpos,toInt,dx);

                Hint[q + s * M + k * M2 + q * M3] = Integral;
                Hint[q + s * M + q * M2 + k * M3] = Integral;
                Hint[s + q * M + q * M2 + k * M3] = Integral;
                Hint[s + q * M + k * M2 + q * M3] = Integral;

                Hint[k + q * M + s * M2 + q * M3] = conj(Integral);
                Hint[k + q * M + q * M2 + s * M3] = conj(Integral);
                Hint[q + k * M + s * M2 + q * M3] = conj(Integral);
                Hint[q + k * M + q * M2 + s * M3] = conj(Integral);

                for (i = 0; i < Mpos; i++)
                {
                    toInt[i] = conj(Omat[k][i]*Omat[k][i]) * \
                               Omat[q][i]*Omat[s][i];
                }

                Integral = g * Csimps(Mpos,toInt,dx);

                Hint[k + k * M + q * M2 + s * M3] = Integral;
                Hint[k + k * M + s * M2 + q * M3] = Integral;
                Hint[q + s * M + k * M2 + k * M3] = conj(Integral);
                Hint[s + q * M + k * M2 + k * M3] = conj(Integral);

                for (i = 0; i < Mpos; i++)
                {
                    toInt[i] = conj(Omat[s][i]*Omat[s][i]) * \
                               Omat[k][i]*Omat[q][i];
                }

                Integral = g * Csimps(Mpos,toInt,dx);

                Hint[s + s * M + k * M2 + q * M3] = Integral;
                Hint[s + s * M + q * M2 + k * M3] = Integral;
                Hint[k + q * M + s * M2 + s * M3] = conj(Integral);
                Hint[q + k * M + s * M2 + s * M3] = conj(Integral);

                for (i = 0; i < Mpos; i++)
                {
                    toInt[i] = conj(Omat[q][i]*Omat[q][i]) * \
                               Omat[k][i]*Omat[s][i];
                }

                Integral = g * Csimps(Mpos,toInt,dx);

                Hint[q + q * M + k * M2 + s * M3] = Integral;
                Hint[q + q * M + s * M2 + k * M3] = Integral;
                Hint[k + s * M + q * M2 + q * M3] = conj(Integral);
                Hint[s + k * M + q * M2 + q * M3] = conj(Integral);

                for (l = q + 1; l < Morb; l++)
                {

                    for (i = 0; i < Mpos; i++)
                    {
                        toInt[i] = conj(Omat[k][i] * Omat[s][i]) * \
                                   Omat[q][i] * Omat[l][i];
                    }

                    Integral = g * Csimps(Mpos,toInt,dx);

                    Hint[k + s * M + q * M2 + l * M3] = Integral;
                    Hint[k + s * M + l * M2 + q * M3] = Integral;
                    Hint[s + k * M + q * M2 + l * M3] = Integral;
                    Hint[s + k * M + l * M2 + q * M3] = Integral;

                    Hint[q + l * M + k * M2 + s * M3] = conj(Integral);
                    Hint[l + q * M + k * M2 + s * M3] = conj(Integral);
                    Hint[l + q * M + s * M2 + k * M3] = conj(Integral);
                    Hint[q + l * M + s * M2 + k * M3] = conj(Integral);

                    for (i = 0; i < Mpos; i++)
                    {
                        toInt[i] = conj(Omat[k][i] * Omat[q][i]) * \
                                   Omat[s][i] * Omat[l][i];
                    }

                    Integral = g * Csimps(Mpos,toInt,dx);

                    Hint[k + q * M + s * M2 + l * M3] = Integral;
                    Hint[k + q * M + l * M2 + s * M3] = Integral;
                    Hint[q + k * M + s * M2 + l * M3] = Integral;
                    Hint[q + k * M + l * M2 + s * M3] = Integral;

                    Hint[s + l * M + k * M2 + q * M3] = conj(Integral);
                    Hint[s + l * M + q * M2 + k * M3] = conj(Integral);
                    Hint[l + s * M + q * M2 + k * M3] = conj(Integral);
                    Hint[l + s * M + k * M2 + q * M3] = conj(Integral);

                    for (i = 0; i < Mpos; i++)
                    {
                        toInt[i] = conj(Omat[k][i] * Omat[l][i]) * \
                                   Omat[s][i] * Omat[q][i];
                    }

                    Integral = g * Csimps(Mpos,toInt,dx);

                    Hint[k + l * M + s * M2 + q * M3] = Integral;
                    Hint[k + l * M + q * M2 + s * M3] = Integral;
                    Hint[l + k * M + s * M2 + q * M3] = Integral;
                    Hint[l + k * M + q * M2 + s * M3] = Integral;

                    Hint[s + q * M + k * M2 + l * M3] = conj(Integral);
                    Hint[s + q * M + l * M2 + k * M3] = conj(Integral);
                    Hint[q + s * M + l * M2 + k * M3] = conj(Integral);
                    Hint[q + s * M + k * M2 + l * M3] = conj(Integral);

                }
            }
        }
    }

    free(toInt);

}





doublec Energy (int Morb, Cmatrix rho1, Carray rho2, Cmatrix Ho, Carray Hint)
{

/** Compute and return the energy given the one/two-body density
  * matrices needed for any observable and the  hamiltonian  one
  * and two-body matrices elements to be contracted in sums
  *
  *************************************************************/

    int
        i,
        j,
        k,
        l,
        s,
        q;

    double complex
        z,
        w;

    z = 0;
    w = 0;

    for (k = 0; k < Morb; k++)
    {
        for (l = 0; l < Morb; l++)
        {

            w = w + rho1[k][l] * Ho[k][l];

            for (s = 0; s < Morb; s++)
            {
                for (q = 0; q < Morb; q++)
                {
                    i = k + l * Morb + s * Morb*Morb + q * Morb*Morb*Morb;
                    j = k + l * Morb + q * Morb*Morb + s * Morb*Morb*Morb;
                    z = z + rho2[i] * Hint[j];
                }
            }
        }
    }

    return (w + z / 2);

}





doublec KinectE (int Morb, int Mpos, Cmatrix Omat, double dx, double a2,
        Cmatrix rho)
{

/** Return mean value of second order derivative term in Hamiltonian **/

    int i,
        j,
        k;

    double complex
        r;

    Carray
        ddxi  = carrDef(Mpos),
        ddxj  = carrDef(Mpos),
        toInt = carrDef(Mpos);



    carrFill(Mpos, 0, toInt);

    for (i = 0; i < Morb; i++)
    {

        dxFD(Mpos, Omat[i], dx, ddxi);

        for (j = 0; j < Morb; j++)
        {

            r = rho[i][j];
            dxFD(Mpos, Omat[j], dx, ddxj);

            for (k = 0; k < Mpos; k++)
            {
                toInt[k] = toInt[k] - a2 * r * conj(ddxi[k]) * ddxj[k];
            }
        }
    }

    r = Csimps(Mpos, toInt, dx);

    free(ddxi); free(ddxj); free(toInt);

    return r;
}





doublec PotentialE (int Morb, int Mpos, Cmatrix Omat, double dx, Rarray V,
        Cmatrix rho)
{

    int i,
        j,
        k;

    double complex
        r;

    Carray
        toInt = carrDef(Mpos);



    carrFill(Mpos, 0, toInt);

    for (i = 0; i < Morb; i++)
    {
        for (j = 0; j < Morb; j++)
        {

            r = rho[i][j];

            for (k = 0; k < Mpos; k++)
                toInt[k] = toInt[k] + r * V[k] * conj(Omat[i][k]) * Omat[j][k];
        }
    }

    r = Csimps(Mpos, toInt, dx);

    free(toInt);

    return r;
}





doublec TwoBodyE(int Morb, int Mpos, Cmatrix Omat, double dx, double g,
        Carray rho)
{

    int i,
        k,
        s,
        q,
        l,
        M = Morb,
        M2 = Morb * Morb,
        M3 = Morb * Morb * Morb;

    double complex
        r;

    Carray
        toInt = carrDef(Mpos);



    carrFill(Mpos, 0, toInt);

    for (k = 0; k < Morb; k++)
    {

        for (s = 0; s < Morb; s++)
        {

            for (q = 0; q < Morb; q++)
            {

                for (l = 0; l < Morb; l++)
                {

                    r = rho[k + s * M + q * M2 + l * M3];

                    for (i = 0; i < Mpos; i++)
                    {
                        toInt[i] += r * conj(Omat[k][i] * Omat[s][i]) * \
                        Omat[l][i] * Omat[q][i];
                    }

                }
            }
        }
    }

    r = g * Csimps(Mpos, toInt, dx) / 2;

    free(toInt);

    return r;
}



doublec Virial(EqDataPkg mc, Cmatrix Orb, Cmatrix rho1, Carray rho2)
{

    int
        Npar = mc->Npar,
        Morb = mc->Morb,
        Mpos = mc->Mpos;

    double
        dx = mc->dx,
        g  = mc->g,
        a2 = mc->a2,
        *V = mc->V;

    double complex
        kinect,
        potential,
        interacting,
        a1 = mc->a1;



    kinect = KinectE(Morb, Mpos, Orb, dx, a2, rho1);
    potential = PotentialE(Morb, Mpos, Orb, dx, V, rho1);
    interacting = TwoBodyE(Morb, Mpos, Orb, dx, g, rho2);

    return (2 * potential - 2 * kinect - interacting);

}



double complex SquaredRampl(int n, Carray f, Carray g, double xi, double dx)
{

/** Compute < f | r^2 | g > where r denotes the position operator
    This is an auxiliar function to the MeanQuadraticR        **/

    int
        i;

    double
        x;

    double complex
        r2;

    Carray
        integ;

    integ = carrDef(n);

    x = xi; // Initiate at boundary of domain
    for (i = 0; i < n; i ++)
    {
        integ[i] = conj(f[i]) * g[i] * x * x;
        x = x + dx; // update to next grid point
    }

    r2 = Csimps(n,integ,dx);

    free(integ);

    return r2;
}



double MeanQuadraticR(EqDataPkg mc, Cmatrix Orb, Cmatrix rho1)
{

/** COMPUTE THE MEAN QUADRATIC POSITION OF THE MANY-BODY STATE **/

    int
        i,
        j,
        Npar,
        Norb,
        Ngrid;

    double complex
        R2amp,
        R2;

    double
        xi,
        dx;

    Ngrid = mc->Mpos;
    Npar = mc->Npar;
    Norb = mc->Morb;
    dx = mc->dx;
    xi = mc->xi;

    R2 = 0;

    for (i = 0; i < Norb; i++)
    {
        R2 = R2 + rho1[i][i] * SquaredRampl(Ngrid,Orb[i],Orb[i],xi,dx);
        for (j = i + 1; j < Norb; j++)
        {
            R2amp = rho1[i][j] * SquaredRampl(Ngrid,Orb[i],Orb[j],xi,dx);
            // use hermitian properties to sweep only j > i because for
            // j < i the number is the complex conjugate.
            R2 = R2 + R2amp + conj(R2amp);
        }
    }

    return sqrt(creal(R2) / Npar);
}
