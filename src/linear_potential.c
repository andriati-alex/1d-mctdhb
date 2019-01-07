#include "../include/linear_potential.h"

void harmonic(int M, Rarray x, Rarray V, double omega)
{
    for (int i = 0; i < M; i ++) V[i] = 0.5 * omega * omega * x[i] * x[i];
}

void deltabarrier(int M, Rarray x, Rarray V, double height)
{
    rarrFill(M, 0, V);
    V[M / 2] = height / (x[1] - x[0]);
}

void GetPotential(int M, char name [], Rarray x, Rarray V,
     double p1, double p2, double p3)
{
    if (strcmp(name, "harmonic") == 0)
    {
        harmonic(M, x, V, p1);
        return;
    }
    
    if (strcmp(name, "deltabarrier") == 0)
    {
        deltabarrier(M, x, V, p1);
        return;
    }

    if (strcmp(name, "zero") == 0)
    {
        rarrFill(M, 0, V);
        return;
    }

    printf("\n\n\n\nERROR: Potential '%s' not implemented\n\n", name);
    exit(EXIT_FAILURE);
}
