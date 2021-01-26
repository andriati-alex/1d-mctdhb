#include "linear_potential.h"



void harmonic(int M, Rarray x, Rarray V, double omega)
{
    for (int i = 0; i < M; i ++) V[i] = 0.5 * omega * omega * x[i] * x[i];
}



void doublewell(int M, Rarray x, Rarray V, double a, double b)
{
    double
        a2,
        b2,
        a4,
        x2,
        x4;

    a2 = a * a;
    b2 = b * b;
    a4 = a2 * a2;

    for (int i = 0; i < M; i ++)
    {
        x2 = x[i] * x[i];
        x4 = x2 * x2;
        V[i] = b2*x4/4.0 - a2*x2/2.0 + a4/(4*b2);
    }
}



void harmonicgauss(int M, Rarray x, Rarray V, double a, double b, double c)
{
    double
        x2;

    for (int i = 0; i < M; i ++)
    {
        x2 = x[i] * x[i];
        V[i] = a * a * x2 / 2 + b * b * exp(- x2 / (2 * c * c));
    }
}



void deltabarrier(int M, Rarray x, Rarray V, double height)
{
    rarrFill(M, 0, V);
    V[M / 2] = height / (x[1] - x[0]);
}



void barrier(int M, Rarray x, Rarray V, double height, double T)
{
    int i, j;

    if (T < x[1] - x[0])
    {
        printf("\n\n\nERROR : linear potential barrier requires a ");
        printf("width greater than spatial grid step size dx.\n\n");
        exit(EXIT_FAILURE);
    }

    rarrFill(M, 0, V);

    for (i = 0; i < M; i++)
    {
        if( fabs(x[i]) < T / 2 ) break;
    }

    for (j = i; j < M; j++)
    {
        if ( x[j] > T / 2 ) break;
        V[j] = height * cos(x[j] * PI / T) * cos(x[j] * PI / T);
    }
}



void opticallattice(int M, Rarray x, Rarray V, double V0, double k)
{
    int
        i;
    double
        L;

    L = x[M-1] - x[0];  // get length of domain grid
    for (i = 0; i < M; i++)
    {
        V[i] = V0*sin(PI*k*x[i]/L)*sin(PI*k*x[i]/L);
    }
}



void GetPotential(int M, char name [], Rarray x, Rarray V,
     double p1, double p2, double p3)
{

    if (strcmp(name, "harmonic") == 0)
    {
        harmonic(M, x, V, p1);
        return;
    }

    if (strcmp(name, "doublewell") == 0)
    {
        doublewell(M, x, V, p1, p2);
        return;
    }

    if (strcmp(name, "harmonicgauss") == 0)
    {
        harmonicgauss(M, x, V, p1, p2, p3);
        return;
    }
    
    if (strcmp(name, "deltabarrier") == 0)
    {
        deltabarrier(M, x, V, p1);
        return;
    }

    if (strcmp(name, "barrier") == 0)
    {
        barrier(M, x, V, p1, p2);
        return;
    }

    if (strcmp(name, "opticallattice") == 0)
    {
        opticallattice(M,x,V,p1,p2);
        return;
    }

    if (strcmp(name, "zero") == 0)
    {
        rarrFill(M, 0, V);
        return;
    }

    printf("\n\n\nERROR: Potential '%s' not implemented\n\n", name);
    exit(EXIT_FAILURE);

}
