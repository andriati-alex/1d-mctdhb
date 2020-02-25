#include "linear_potential.h"



void harmonic(int M, Rarray x, Rarray V, double omega)
{
    for (int i = 0; i < M; i ++) V[i] = 0.5 * omega * omega * x[i] * x[i];
}



void doublewell(int M, Rarray x, Rarray V, double a, double b)
{
    double
        x2,
        x4;

    for (int i = 0; i < M; i ++)
    {
        x2 = x[i] * x[i];
        x4 = x2 * x2;
        V[i] = 2 * (b * b * x4 / 4.0 - a * a * x2 / 2.0);
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

    if (strcmp(name, "zero") == 0)
    {
        rarrFill(M, 0, V);
        return;
    }

    printf("\n\n\nERROR: Potential '%s' not implemented\n\n", name);
    exit(EXIT_FAILURE);

}
