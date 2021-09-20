#include "linalg/tridiagonal_solver.h"
#include "assistant/arrays_definition.h"
#include "linalg/basic_linalg.h"
#include <stdlib.h>

void
solve_cplx_tridiag(
    uint32_t n, Carray upper, Carray lower, Carray mid, Carray rhs, Carray ans)
{
    uint32_t i, k;
    dcomplex new_rhs1, new_rhs2;
    Carray   u, l, z;
    u = get_dcomplex_array(n);
    l = get_dcomplex_array(n - 1);
    z = get_dcomplex_array(n);

    if (cabs(mid[0]) == 0)
    {
        // In this case there is a system reduction
        // where we solve for [x1  x3  x4 ... xn]
        // what is equivalent to adjust the two first
        // equations and starts counters from 1

        new_rhs1 = rhs[1] - mid[1] * rhs[0] / upper[0];
        new_rhs2 = rhs[2] - lower[1] * rhs[0] / upper[0];

        // u and z factor initizlization with the changed system
        u[1] = lower[0];
        z[1] = new_rhs1;

        // One iteration need to be performed out because
        // the change of values in lower and RHS

        l[1] = 0;
        u[2] = mid[2] - l[1] * upper[1];
        z[2] = new_rhs2 - l[1] * z[1];

        for (i = 2; i < n - 1; i++)
        {
            k = i + 1;
            l[i] = lower[i] / u[i];
            u[k] = mid[k] - l[i] * upper[i];
            z[k] = rhs[k] - l[i] * z[i];
        }

        ans[n - 1] = z[n - 1] / u[n - 1];

        for (i = 2; i <= n - 1; i++)
        {
            k = n - i;
            ans[k] = (z[k] - upper[k] * ans[k + 1]) / u[k];
        }

        // Obtained order ans[0..n] = [nan  x1  x3  x4 .. xn]
        // Organize ans[0..n] = [x1  x2  x3  .. xn]
        ans[0] = ans[1];
        ans[1] = rhs[0] / upper[0];

        // Free local alilocated memory
        free(u);
        free(l);
        free(z);

        return;
    }

    u[0] = mid[0];
    z[0] = rhs[0];

    for (i = 0; i < n - 1; i++)
    {
        k = i + 1;
        l[i] = lower[i] / u[i];
        u[k] = mid[k] - l[i] * upper[i];
        z[k] = rhs[k] - l[i] * z[i];
    }

    ans[n - 1] = z[n - 1] / u[n - 1];

    for (i = 2; i <= n; i++)
    {
        k = n - i;
        ans[k] = (z[k] - upper[k] * ans[k + 1]) / u[k];
    }

    // Free local allocated memory
    free(u);
    free(l);
    free(z);
}

void
solve_cplx_cyclic_tridiag_lu(
    uint32_t n, Carray upper, Carray lower, Carray mid, Carray rhs, Carray ans)
{
    uint32_t i, k;
    Carray   u, l, g, h, z;

    // Modified L.U decomposition requires two new vectors g and h
    // Additional line in L denoted by g vector
    // Additional column in U defined by h vector
    u = get_dcomplex_array(n);
    l = get_dcomplex_array(n - 2);
    g = get_dcomplex_array(n - 1);
    h = get_dcomplex_array(n - 1);
    z = get_dcomplex_array(n);

    u[0] = mid[0];
    z[0] = rhs[0];

    /****** extras steps ******/
    g[0] = lower[n - 1] / mid[0];
    h[0] = upper[n - 1];
    /**************************/

    for (i = 0; i < n - 2; i++)
    {
        k = i + 1;
        l[i] = lower[i] / u[i];
        u[k] = mid[k] - l[i] * upper[i];
        z[k] = rhs[k] - l[i] * z[i];

        /*********** extras steps ***********/
        h[k] = (-1) * l[i] * h[i];
        g[k] = (-1) * upper[i] * g[i] / u[k];
        /************************************/
    }

    // little correction due to last terms - L . U = A
    h[n - 2] = h[n - 2] + upper[n - 2];
    g[n - 2] = g[n - 2] + lower[n - 2] / u[n - 2];

    // Change from simple tridiagonal by unconjugate product of arrays
    z[n - 1] = rhs[n - 1] - unconj_carrDot(n - 1, g, z);
    u[n - 1] = mid[n - 1] - unconj_carrDot(n - 1, g, h);

    ans[n - 1] = z[n - 1] / u[n - 1];

    // store the last column multiplication in U
    carrScalarMultiply(n - 1, h, ans[n - 1], h);

    // Additionaly subtract h compared to tridiagonal
    ans[n - 2] = (z[n - 2] - h[n - 2]) / u[n - 2];

    for (i = 3; i <= n; i++)
    {
        k = n - i;
        ans[k] = (z[k] - h[k] - upper[k] * ans[k + 1]) / u[k];
    }

    // Free local allocated memory
    free(u);
    free(l);
    free(z);
    free(h);
    free(g);
}

void
solve_cplx_cyclic_tridiag_sm(
    uint32_t n, Carray upper, Carray lower, Carray mid, Carray rhs, Carray ans)
{
    dcomplex factor, recover1, recoverN;
    Carray   x, w, U, V;

    x = get_dcomplex_array(n);
    w = get_dcomplex_array(n);
    U = get_dcomplex_array(n);
    V = get_dcomplex_array(n);

    recover1 = mid[0];
    recoverN = mid[n - 1];

    carrFill(n, 0, U);
    carrFill(n, 0, V);

    // Choice of 'gamma' factor
    if (cabs(mid[0]) == 0)
    {
        factor = upper[0];
        mid[0] = -factor;
    } else
    {
        factor = mid[0];
        mid[0] = 0;
    }

    U[0] = factor;
    V[0] = 1;
    U[n - 1] = lower[n - 1];
    V[n - 1] = upper[n - 1] / factor;

    // Adjust last main diagonal element(required by the algorithm)
    mid[n - 1] = mid[n - 1] - upper[n - 1] * lower[n - 1] / factor;

    solve_cplx_tridiag(n, upper, lower, mid, rhs, x);
    solve_cplx_tridiag(n, upper, lower, mid, U, w);

    factor = unconj_carrDot(n, V, x) / (1.0 + unconj_carrDot(n, V, w));

    carrUpdate(n, x, (-1) * factor, w, ans);

    mid[0] = recover1;
    mid[n - 1] = recoverN;

    // Free local allocated memory
    free(x);
    free(w);
    free(U);
    free(V);
}
