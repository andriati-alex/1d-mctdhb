#include "iterative_solver.h"



int CCG(int n, Cmatrix A, Carray b, Carray x, double eps, int maxiter)
{

    double complex
        a,
        beta;
    
    int
        l; // Interation counter - return for convergence statistics

    Carray
        r,
        d,
        aux,
        prev_x,
        prev_r;

    l = 0;

    r = carrDef(n);      // residue
    d = carrDef(n);      // Direction
    aux = carrDef(n);    // to store some intermediate algebra
    prev_x = carrDef(n);
    prev_r = carrDef(n); // to compute scalars need 2 residues

    // Initiate variables
    cmatvec(n, n, A, x, aux);
    carrSub(n, b, aux, r);
    carrCopy(n, r, d);

    while (carrMod(n, r) > eps)
    {
        // Matrix-Vector multiply
        cmatvec(n, n, A, d, aux);
        a = carrMod2(n, r) / carrDot(n, d, aux);
        // Update residue
        carrCopy(n, r, prev_r);
        carrUpdate(n, prev_r, -a, aux, r);
        // Update solution
        carrCopy(n, x, prev_x);
        carrUpdate(n, prev_x, a, d, x);
        beta = carrMod2(n, r) / carrMod2(n, prev_r);
        // Update direction
        carrScalarMultiply(n, d, beta, aux);
        carrAdd(n, r, aux, d);

        l = l + 1;

        if (l == maxiter) break;
    }

    if (l == maxiter)
    {
        printf("\n\n\n\tWARNING: Exceeded max number of iterations");
        printf(" in Conjugate-Gradients method.\n");
        printf("\tResidual error achieved : %.2E\n\n", carrMod(n,r));
    }

    // Free function allocated memory
    free(r);
    free(d);
    free(aux);
    free(prev_x);
    free(prev_r);

    return l;
}





int preCCG(int n, Cmatrix A, Carray b, Carray x, double eps, int maxiter,
    Cmatrix M)
{

    double complex
        a,
        beta;

    int
        l; // Interation counter - return for convergence statistics

    Carray
        r,
        d,
        aux,
        prev_x,
        prev_r,
        M_r;



    l = 0;

    r = carrDef(n);      // residue
    d = carrDef(n);      // Direction
    aux = carrDef(n);   // to store some algebra
    prev_x = carrDef(n);
    prev_r = carrDef(n); // to compute scalars need 2 residues
    M_r = carrDef(n);    // Pre-conditioner applied to residue

    cmatvec(n, n, A, x, aux);
    carrSub(n, b, aux, r);
    cmatvec(n, n, M, r, d);
    carrCopy(n, d, M_r);

    while (carrMod(n, r) > eps)
    {
        // Matrix-Vector multiply
        cmatvec(n, n, A, d, aux);
        a = carrDot(n, r, M_r) / carrDot(n, d, aux);
        // Update residue
        carrCopy(n, r, prev_r);
        carrUpdate(n, prev_r, -a, aux, r);
        // Update solution
        carrCopy(n, x, prev_x);
        carrUpdate(n, prev_x, a, d, x);
        // compute beta
        carrCopy(n, M_r, aux);
        cmatvec(n, n, M, r, M_r);
        beta = carrDot(n, r, M_r) / carrDot(n, prev_r, aux);
        // Update direction
        carrScalarMultiply(n, d, beta, aux);
        carrAdd(n, M_r, aux, d);

        l = l + 1;
        
        if (l == maxiter) break;
    }

    if (l == maxiter)
    {
        printf("\n\n\n\tWARNING: Exceeded max number of iterations");
        printf(" in Conjugate-Gradients method.\n");
        printf("\tResidual error achieved : %.2E\n\n", carrMod(n,r));
    }

    // Free function allocated memory
    free(r);
    free(d);
    free(aux);
    free(prev_x);
    free(prev_r);
    free(M_r);

    return l;
}





int tripreCCG(int n, Cmatrix A, Carray b, Carray x, double eps, int maxiter,
    Carray upper, Carray lower, Carray mid)
{

    double complex
        a,
        beta;

    int
        l; // Interation counter - return for convergence statistics

    Carray
        r,
        d,
        aux,
        prev_x,
        prev_r,
        M_r;



    l = 0;

    r = carrDef(n);      // residue
    d = carrDef(n);      // Direction
    aux = carrDef(n);    // to store some algebra
    prev_x = carrDef(n);
    prev_r = carrDef(n); // to compute scalars need 2 residues
    M_r = carrDef(n);    // Pre-conditioner applied to residue

    cmatvec(n, n, A, x, aux);
    carrSub(n, b, aux, r);
    triDiag(n, upper, lower, mid, r, d);
    carrCopy(n, d, M_r);

    while (carrMod(n, r) > eps)
    {
        // Matrix-vector multiply
        cmatvec(n, n, A, d, aux);
        a = carrDot(n, r, M_r) / carrDot(n, d, aux);
        // Update residue
        carrCopy(n, r, prev_r);
        carrUpdate(n, prev_r, -a, aux, r);
        // Update solution
        carrCopy(n, x, prev_x);
        carrUpdate(n, prev_x, a, d, x);
        // compute beta
        carrCopy(n, M_r, aux);
        triDiag(n, upper, lower, mid, r, M_r);
        beta = carrDot(n, r, M_r) / carrDot(n, prev_r, aux);
        // update direction
        carrScalarMultiply(n, d, beta, aux);
        carrAdd(n, M_r, aux, d);

        l = l + 1;

        if (l == maxiter) break;
    }

    if (l == maxiter)
    {
        printf("\n\n\n\tWARNING: Exceeded max number of iterations");
        printf(" in Conjugate-Gradients method.\n");
        printf("\tResidual error achieved : %.2E\n\n", carrMod(n,r));
    }

    // Free function allocated memory
    free(r);
    free(d);
    free(aux);
    free(prev_x);
    free(prev_r);
    free(M_r);

    return l;
}





int CCSCCG(int n, CCSmat A, Carray b, Carray x, double eps, int maxiter)
{

    double complex
        a,
        beta;

    int
        l; // Interation counter - return for convergence statistics

    Carray
        r,
        d,
        aux,
        prev_x,
        prev_r;

    r      = carrDef(n); // residue
    d      = carrDef(n); // Direction
    aux    = carrDef(n); // store intermediate steps
    prev_x = carrDef(n);
    prev_r = carrDef(n);

    CCSvec(n, A->vec, A->col, A->m, x, aux);
    carrSub(n, b, aux, r);
    carrCopy(n, r, d);

    while (carrMod(n, r) > eps)
    {
        // Matrix-vector multiply
        CCSvec(n, A->vec, A->col, A->m, d, aux);
        a = carrMod2(n, r) / carrDot(n, d, aux);
        // Update residue
        carrCopy(n, r, prev_r);
        carrUpdate(n, prev_r, -a, aux, r);
        // Update solution
        carrCopy(n, x, prev_x);
        carrUpdate(n, prev_x, a, d, x);
        // Beta and Update direction
        beta = carrMod2(n, r) / carrMod2(n, prev_r);
        carrScalarMultiply(n, d, beta, aux);
        carrAdd(n, r, aux, d);

        l = l + 1;

        if (l == maxiter) break;
    }

    if (l == maxiter)
    {
        printf("\n\n\n\tWARNING: Exceeded max number of iterations");
        printf(" in Conjugate-Gradients method.\n");
        printf("\tResidual error achieved : %.2E\n\n", carrMod(n,r));
    }

    // Free function allocated memory
    free(r);
    free(d);
    free(aux);
    free(prev_x);
    free(prev_r);

    return l;
}





int preCCSCCG(int n, CCSmat A, Carray b, Carray x, double eps, int maxiter,
    Cmatrix M)
{

    double complex
        a,
        beta;

    int
        l; // Interation counter - return for convergence statistics

    Carray
        r,
        d,
        aux,
        prev_x,
        prev_r,
        M_r;



    l = 0;

    r = carrDef(n);      // residue
    d = carrDef(n);      // Direction
    aux = carrDef(n);    // to store some algebra
    prev_x = carrDef(n);
    prev_r = carrDef(n); // to compute scalars need 2 residues
    M_r = carrDef(n);    // Pre-conditioner applied to residue



    CCSvec(n, A->vec, A->col, A->m, x, aux);
    carrSub(n, b, aux, r);
    cmatvec(n, n, M, r, d);
    carrCopy(n, d, M_r);

    while (carrMod(n, r) > eps)
    {
        // Matrix-vector multiply
        CCSvec(n, A->vec, A->col, A->m, d, aux);
        // scalar
        a = carrDot(n, r, M_r) / carrDot(n, d, aux);
        // Update Residue
        carrCopy(n, r, prev_r);
        carrUpdate(n, prev_r, -a, aux, r);
        // Update solution
        carrCopy(n, x, prev_x);
        carrUpdate(n, prev_x, a, d, x);
        // Process to compute beta
        carrCopy(n, M_r, aux);
        cmatvec(n, n, M, r, M_r);
        beta = carrDot(n, r, M_r) / carrDot(n, prev_r, aux);
        // Update Direction
        carrScalarMultiply(n, d, beta, aux);
        carrAdd(n, M_r, aux, d);

        l = l + 1;

        if (l == maxiter) break;
    }

    if (l == maxiter)
    {
        printf("\n\n\n\tWARNING: Exceeded max number of iterations");
        printf(" in Conjugate-Gradients method.\n");
        printf("\tResidual error achieved : %.2E\n\n", carrMod(n,r));
    }

    // Free function allocated memory
    free(r);
    free(d);
    free(aux);
    free(prev_x);
    free(prev_r);
    free(M_r);

    return l;
}





int tripreCCSCCG(int n, CCSmat A, Carray b, Carray x, double eps, int maxiter,
    Carray upper, Carray lower, Carray mid)
{

    double complex
        a,
        beta;

    int
        l; // Interation counter - return for convergence statistics

    Carray
        r,
        d,
        aux,
        prev_x,
        prev_r,
        M_r;



    l = 0;

    r = carrDef(n);      // residue
    d = carrDef(n);      // Direction
    aux = carrDef(n);    // to store some algebra
    prev_x = carrDef(n);
    prev_r = carrDef(n); // to compute scalars need 2 residues
    M_r = carrDef(n);    // Pre-conditioner applied to residue

    CCSvec(n, A->vec, A->col, A->m, x, aux);
    carrSub(n, b, aux, r);
    triDiag(n, upper, lower, mid, r, d);
    carrCopy(n, d, M_r);

    while (carrMod(n, r) > eps)
    {
        CCSvec(n, A->vec, A->col, A->m, d, aux); // matrix-vector mult
        a = carrDot(n, r, M_r) / carrDot(n, d, aux);
        // Update residue
        carrCopy(n, r, prev_r);
        carrUpdate(n, prev_r, (-1) * a, aux, r);
        // Update solution
        carrCopy(n, x, prev_x);
        carrUpdate(n, prev_x, a, d, x);
        carrCopy(n, M_r, aux);                  // aux get M-1 . r
        triDiag(n, upper, lower, mid, r, M_r);  // Store for the next loop
        beta = carrDot(n, r, M_r) / carrDot(n, prev_r, aux);
        carrScalarMultiply(n, d, beta, aux);
        carrAdd(n, M_r, aux, d); // Update direction

        l = l + 1; // Update iteration counter

        if (l == maxiter) break;
    }

    if (l == maxiter)
    {
        printf("\n\n\n\twarning: exceeded max number of iterations");
        printf(" in conjugate-gradients method.\n");
        printf("\tresidual error achieved : %.2E\n\n", carrMod(n,r));
    }

    // Free function local memory
    free(r);
    free(d);
    free(aux);
    free(prev_x);
    free(prev_r);
    free(M_r);

    return l;
}
