#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "array.h"
#include <omp.h>

#define PI 3.14159265359



Carray carrDef(int n)
{
    double complex * ptr;

    ptr = (double complex * ) malloc( n * sizeof(double complex) );

    if (ptr == NULL)
    {
        printf("\n\n\n\tMEMORY ERROR : malloc fail for complex\n\n");
        exit(EXIT_FAILURE);
    }

    return ptr;
}

Cmatrix cmatDef(int m, int n)
{

/** Complex matrix of m rows and n columns **/

    int i;

    double complex ** ptr;

    ptr = (double complex ** ) malloc( m * sizeof(double complex *) );

    if (ptr == NULL)
    {
        printf("\n\n\n\tMEMORY ERROR : malloc fail for (complex *)\n\n");
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < m; i++) ptr[i] = carrDef(n);

    return ptr;
}



void cprint(double complex z)
{

/** printf complex number in coordinates with 2 decimal digits**/

    printf("(%9.2E,%9.2E )", creal(z), cimag(z));
}





void carr_txt(char fname [], int M, Carray v)
{

/** Record a array of complex elements in a text file in a
  * suitable format to import as numpy array with python.
**/

    int
        j;

    double
        real,
        imag;

    FILE
        * data_file = fopen(fname, "w");



    if (data_file == NULL)
    {
        printf("\n\n\n\tERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }

    for (j = 0; j < M - 1; j ++)
    {

        real = creal(v[j]);
        imag = cimag(v[j]);

        if (imag >= 0) fprintf(data_file, "(%.15E+%.15Ej)", real, imag);
        else           fprintf(data_file, "(%.15E%.15Ej)", real, imag);

        fprintf(data_file, "\n");
    }

    real = creal(v[M-1]);
    imag = cimag(v[M-1]);

    if (imag >= 0) fprintf(data_file, "(%.15E+%.15Ej)", real, imag);
    else           fprintf(data_file, "(%.15E%.15Ej)", real, imag);

    fclose(data_file);
}





void cmat_txt (char fname [], int m, int n, Cmatrix A)
{

    int
        i,
        j;

    double
        real,
        imag;

    FILE
        * f = fopen(fname, "w");



    if (f == NULL)
    {   // impossible to open file with the given name
        printf("\n\n\n\tERROR: impossible to open file %s\n\n", fname);
        exit(EXIT_FAILURE);
    }



    for (i = 0; i < m - 1; i++)
    {

        for (j = 0; j < n; j++)
        {

            real = creal(A[i][j]);
            imag = cimag(A[i][j]);

            if (imag >= 0) fprintf(f, "(%.15E+%.15Ej) ", real, imag);
            else           fprintf(f, "(%.15E%.15Ej) ", real, imag);
        }

        fprintf(f, "\n");
    }

    for (j = 0; j < n; j++)
    {

        real = creal(A[m-1][j]);
        imag = cimag(A[m-1][j]);

        if (imag >= 0) fprintf(f, "(%.15E+%.15Ej) ", real, imag);
        else           fprintf(f, "(%.15E%.15Ej) ", real, imag);
    }

    fclose(f);
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





void SetupHint_X (int Morb, int Mpos, Cmatrix Omat, double dx, double g,
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
        M3,
        ik,
        iq,
        is,
        il;

    double complex
        Integral;

    Carray
        orb,
        toInt;

    M  = Morb;
    M2 = M * M;
    M3 = M * M2;

#pragma omp parallel private(i,k,s,q,l,ik,iq,is,il,Integral,orb,toInt) \
    firstprivate(g,dx)

    {

    toInt = carrDef(Mpos);

    orb = carrDef(Morb*Mpos);

    for (k = 0; k < Morb; k++)
    {
        for (i = 0; i < Mpos; i++) orb[k * Mpos + i] = Omat[k][i];
    }

#pragma omp for schedule(static)
    for (k = 0; k < Morb; k++)
    {

        ik = k * Mpos;

        for (i = 0; i < Mpos; i++)
        {
            toInt[i] = conj(orb[ik+i]*orb[ik+i]) * orb[ik+i]*orb[ik+i];
        }

        Hint[k * (1 + M + M2 + M3)] = g * Csimps(Mpos,toInt,dx);

        for (s = k + 1; s < Morb; s++)
        {

            is = s * Mpos;

            for (i = 0; i < Mpos; i++)
            {
                toInt[i] = conj(orb[ik+i]*orb[is+i]) * orb[ik+i]*orb[ik+i];
            }

            Integral = g * Csimps(Mpos,toInt,dx);

            Hint[k + s * M + k * M2 + k * M3] = Integral;
            Hint[s + k * M + k * M2 + k * M3] = Integral;
            Hint[k + k * M + k * M2 + s * M3] = conj(Integral);
            Hint[k + k * M + s * M2 + k * M3] = conj(Integral);

            for (i = 0; i < Mpos; i++)
            {
                toInt[i] = conj(orb[is+i]*orb[ik+i]) * orb[is+i]*orb[is+i];
            }

            Integral = g * Csimps(Mpos,toInt,dx);

            Hint[s + k * M + s * M2 + s * M3] = Integral;
            Hint[k + s * M + s * M2 + s * M3] = Integral;
            Hint[s + s * M + s * M2 + k * M3] = conj(Integral);
            Hint[s + s * M + k * M2 + s * M3] = conj(Integral);

            for (i = 0; i < Mpos; i++)
            {
                toInt[i] = conj(orb[ik+i]*orb[is+i]) * orb[is+i]*orb[ik+i];
            }

            Integral = g * Csimps(Mpos,toInt,dx);

            Hint[k + s * M + s * M2 + k * M3] = Integral;
            Hint[s + k * M + s * M2 + k * M3] = Integral;
            Hint[s + k * M + k * M2 + s * M3] = Integral;
            Hint[k + s * M + k * M2 + s * M3] = Integral;

            for (i = 0; i < Mpos; i++)
            {
                toInt[i] = conj(orb[ik+i]*orb[ik+i]) * orb[is+i]*orb[is+i];
            }

            Integral = g * Csimps(Mpos,toInt,dx);

            Hint[k + k * M + s * M2 + s * M3] = Integral;
            Hint[s + s * M + k * M2 + k * M3] = conj(Integral);

            for (q = s + 1; q < Morb; q++)
            {

                iq = q * Mpos;

                for (i = 0; i < Mpos; i++)
                {
                    toInt[i] = conj(orb[ik+i]*orb[is+i]) * \
                               orb[iq+i]*orb[ik+i];
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
                    toInt[i] = conj(orb[is+i]*orb[ik+i]) * \
                               orb[iq+i]*orb[is+i];
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
                    toInt[i] = conj(orb[iq+i]*orb[is+i]) * \
                               orb[ik+i]*orb[iq+i];
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
                    toInt[i] = conj(orb[ik+i]*orb[ik+i]) * \
                               orb[iq+i]*orb[is+i];
                }

                Integral = g * Csimps(Mpos,toInt,dx);

                Hint[k + k * M + q * M2 + s * M3] = Integral;
                Hint[k + k * M + s * M2 + q * M3] = Integral;
                Hint[q + s * M + k * M2 + k * M3] = conj(Integral);
                Hint[s + q * M + k * M2 + k * M3] = conj(Integral);

                for (i = 0; i < Mpos; i++)
                {
                    toInt[i] = conj(orb[is+i]*orb[is+i]) * \
                               orb[ik+i]*orb[iq+i];
                }

                Integral = g * Csimps(Mpos,toInt,dx);

                Hint[s + s * M + k * M2 + q * M3] = Integral;
                Hint[s + s * M + q * M2 + k * M3] = Integral;
                Hint[k + q * M + s * M2 + s * M3] = conj(Integral);
                Hint[q + k * M + s * M2 + s * M3] = conj(Integral);

                for (i = 0; i < Mpos; i++)
                {
                    toInt[i] = conj(orb[iq+i]*orb[iq+i]) * \
                               orb[ik+i]*orb[is+i];
                }

                Integral = g * Csimps(Mpos,toInt,dx);

                Hint[q + q * M + k * M2 + s * M3] = Integral;
                Hint[q + q * M + s * M2 + k * M3] = Integral;
                Hint[k + s * M + q * M2 + q * M3] = conj(Integral);
                Hint[s + k * M + q * M2 + q * M3] = conj(Integral);

                for (l = q + 1; l < Morb; l++)
                {

                    il = l * Mpos;

                    for (i = 0; i < Mpos; i++)
                    {
                        toInt[i] = conj(orb[ik+i] * orb[is+i]) * \
                                   orb[iq+i] * orb[il+i];
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
                        toInt[i] = conj(orb[ik+i] * orb[iq+i]) * \
                                   orb[is+i] * orb[il+i];
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
                        toInt[i] = conj(orb[ik+i] * orb[il+i]) * \
                                   orb[is+i] * orb[iq+i];
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

    free(orb);

    }

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





doublec nonlinear (int M, int k, int n, double g, Cmatrix Orb,
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










int main(int argc, char * argv[])
{

    omp_set_num_threads(omp_get_max_threads() / 2);


    int
        i,
        j,
        k,
        Morb,
        Mpos;

    double
        start,
        time_used;

    Carray
        Hint_X,
        rho2,
        Hint;

    Cmatrix
        Ho,
        dOdt,
        rho_inv,
        orb;

    if (argc != 3)
    {
        printf("\n\nERROR: Need two integer numbers from command line ");
        printf("the first number of grid points and second the number of ");
        printf("orbitals.\n\n");
        exit(EXIT_FAILURE);
    }

    sscanf(argv[1],"%d",&Mpos);
    sscanf(argv[2],"%d",&Morb);

    Hint = (doublec * ) malloc(Morb*Morb*Morb*Morb*sizeof(doublec));
    Hint_X = (doublec * ) malloc(Morb*Morb*Morb*Morb*sizeof(doublec));
    rho2 = (doublec * ) malloc(Morb*Morb*Morb*Morb*sizeof(doublec));

    for (i = 0; i < Morb * Morb * Morb * Morb; i++) rho2[i] = 1.3;

    orb = cmatDef(Morb,Mpos);
    dOdt = cmatDef(Morb,Mpos);
    Ho = cmatDef(Morb,Morb);
    rho_inv = cmatDef(Morb,Morb);

    for (i = 0; i < Morb; i++)
    {
        for (j = 0; j < Mpos; j++)
        {
            orb[i][j] = ((i * j) % 8) * sin( (PI / Mpos) * j) - \
                        I * ((i + j) % 5) * cos(6 * (PI / Mpos) * j);
        }

        for (j = 0; j < Morb; j++)
        {
            Ho[i][j] = 1.0;
            rho_inv[i][j] = -0.9;
        }
    }


    start = omp_get_wtime();
    for (i = 0; i < 50; i++) SetupHint_X(Morb,Mpos,orb,0.02,13.764539,Hint_X);
    time_used = (double) (omp_get_wtime() - start) / 50;
    printf("\n\nTime to setup Hint_X : %.1lfms", time_used * 1000);

    start = omp_get_wtime();
    for (i = 0; i < 50; i++) SetupHint(Morb,Mpos,orb,0.02,13.764539,Hint);
    time_used = (double) (omp_get_wtime() - start) / 50;
    printf("\n\nTime to setup Hint : %.1lfms", time_used * 1000);

    start = omp_get_wtime();
    #pragma omp parallel for private(k, j) schedule(static)
    for (k = 0; k < Morb; k++)
    {
        for (j = 0; j < Mpos; j++)
            dOdt[k][j] = - I * \
            nonlinear(Morb, k, j, 12.4534, orb, rho_inv, rho2, Ho, Hint);
    }
    time_used = (double) (omp_get_wtime() - start);
    printf("\n\nTime to compute nonlinear : %.1lfms", time_used * 1000);

    carr_txt("Hint_X.dat",Morb*Morb*Morb*Morb,Hint_X);
    carr_txt("Hint.dat",Morb*Morb*Morb*Morb,Hint);

    free(Hint);
    free(Hint_X);

    for(i = 0; i < Morb; i++) free(orb[i]);
    free(orb);

    printf("\n\n");
    return 0;
}
