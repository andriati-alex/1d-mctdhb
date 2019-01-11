#include "interpolation.h"

void lagrange(int n, int chunk, double xs[], double ys[], int nx,
     double x[], double y [])
{

/** Given a set of data points (xs[i],ys[i]) with i = 0 .. n - 1, compute the
  * interpolation funcion using chunks of fixed size evaluated at points x[j]
  * with j = 0 .. nx - 1, constrained to x[0] > xs[0] and x[nx-1] < xs[n-1].
  *
  * Output Parameter : y[j] = f_pol(x[j])
  * -------------------------------------------------------------------------
**/

    int
        i,
        j,
        k,
        l;

    double
        prod,
        sum;



    if (x[nx-1] > xs[n-1])
    {
        printf("\n\n\t\tERROR : Invalid point to interpolate detected ! ");
        printf("x = %.2lf exceeded data domain limit %.2lf",x[nx-1],xs[n-1]);
        printf("\n\n");
        exit(EXIT_FAILURE);
    }



    i = 0;

    for (j = 0; j < nx; j ++)
    {

        while (x[j] > xs[i]) i = i + 1;

        if (i > n - chunk / 2)
        {
            printf("\n\nWARNING : Initial chunk resized to do not get");
            printf(" segmentation fault in Interpolation.\n\n");
            chunk = (n - i) * 2;
        }

        sum = 0; // sum terms in LaGrange expansion

// After end while loop we have xs[i] > x[j] then for even 'chunk' the
// priority is to go backwards using the middle  point  as xs[i]. Then
// from lagrange formula it is done a displacement,  the initial point
// x1 becomes x[i - (1 + chunk)/2]

        for (k = i - (1+chunk)/2; k < i + chunk/2; k++)
        {

            prod = 1;

            for (l = i - (1+chunk)/2; l < i + chunk/2; l++)
            {
                if (l == k) continue;
                prod = prod * (x[j] - xs[l]) / (xs[k] - xs[l]);
            }

            sum = sum + prod * ys[k];
        }

        y[j] = sum;

    }
}
