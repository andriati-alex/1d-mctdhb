#include "interpolation.h"

void lagrange(int n, int chunk, double xs[], double ys[], int nx,
     double x[], double y[])
{

/** Given a set of data points (xs[i],ys[i]) with i = 0 .. n - 1, that is
    related to some not known function  (ys[i] = f(xs[i])),  use  them to
    compute (approximately) the function in other set of data points x[k]
    k = 0 .. nx - 1,  which must lay within the given data domain,  where
    y[k] = f(x[k]). The user suggest how many data points must be used to
    interpolate the function at each new point x[k]. This routine however
    may reduce the 'chunk' of data points if it exceed the boundaries  of
    the domain.

    INPUT PARAMETERS
    ================
        chunk - number of points used to interpolate at each x[k]
        xs - grid points of the data domain
        ys - function values at grid points
        x - New points where the function need to be approximated

    OUPUT PARAMETERS
    ================
        y - function computed at the grid points x
 **/

    int
        i,
        j,
        k,
        l,
        initialChunk;

    double
        prod,
        sum;



    // The interpolated function domain must lay within the data
    // domain, otherwise it would required extrapolation instead
    if (x[nx-1] > xs[n-1])
    {
        printf("\n\nERROR : Invalid point to interpolate detected ! ");
        printf("x = %.2lf exceeded data domain upper bound %.2lf",
                x[nx-1],xs[n-1]);
        printf("\n\n");
        exit(EXIT_FAILURE);
    }

    if (x[0] < xs[0])
    {
        printf("\n\nERROR : Invalid point to interpolate detected ! ");
        printf("x = %.2lf exceeded data domain lower bound %.2lf",
                x[nx-1],xs[n-1]);
        printf("\n\n");
        exit(EXIT_FAILURE);
    }

    i = 0;

    for (j = 0; j < nx; j ++)
    {
        // x[j] is the points where we want to interpolate the function
        // Thus it is fixed in what follows. First Search  the  nearest
        // data point to x[j] and uses the neighboors to interpolate
        while (x[j] > xs[i]) i = i + 1;

        // Resize the chunk if the number of points used in interpolation
        // exceeds the boundaries of data points
        initialChunk = chunk;
        if ( i + (1 + chunk) / 2 > n || i < chunk / 2 )
        {
            printf("\n\nWARNING : Initial chunk resized to do not get");
            printf(" segmentation fault in Interpolation.\n\n");
        }
        while ( i + (1 + chunk) / 2 > n || i < chunk / 2 )
        {
            chunk = chunk - 1;
        }

        sum = 0; // sum product-terms in LaGrange expansion

        // Start LaGrange interpolation
        for (k = i - chunk/2; k < i + (1+chunk)/2; k++)
        {

            prod = 1;

            for (l = i - chunk/2; l < i + (1+chunk)/2; l++)
            {
                if (l == k) continue;
                prod = prod * (x[j] - xs[l]) / (xs[k] - xs[l]);
            }

            sum = sum + prod * ys[k];
        }

        y[j] = sum;

        // recover the suggested chunk by the user
        chunk = initialChunk;
    }
}
