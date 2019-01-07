#ifndef _iterative_solver_h
#define _iterative_solver_h

#ifdef _OPENMP
    #include <omp.h>
#endif



#include "matrix_operations.h"
#include "tridiagonal_solver.h"



int CCG(int n, Cmatrix A, Carray b, Carray x, double eps, int maxiter);
/* Solve (complex)linear system by Conjugate-Gradient method
 * *********************************************************
 *
 * n is the size of the system
 * A symmetric matrix (self-adjoint)
 * A . x = b (RHS)
 * x initial guess ends up with solution
 * eps the tolerated residual error
 * ---------------------------------------------------------
 *
 * RETURN the number of iterations to converge
 *
 * *********************************************************/





int preCCG(int n, Cmatrix A, Carray b, Carray x, double eps,
    int maxiter, Cmatrix M);
/* Pre-Conditioned Conjugate-Gradient Method
 * *****************************************
 *
 * takes an extra(last) argument that is 
 * properly the inversion of some Matrix M
 * similar to A, but with known inversion.
 * Here the method needs an extra matrix 
 * vector multiplication but can iterate 
 * less times in return.
 *
 * *****************************************/





int tripreCCG(int n, Cmatrix A, Carray b, Carray x, double eps, 
    int maxiter, Carray upper, Carray lower, Carray mid);
/* Instead of take the inversion of M, takes M itself as tridiagonal mat
 * *********************************************************************
 *
 * As the inversion is not given, in the loop it solves tridiagonal system
 *
 * *********************************************************************/



int CCSCCG(int n, CCSmat A, Carray b, Carray x, double eps, int maxiter);
/* Take A as Compressed Column Storaged(CCS) matrix */



int preCCSCCG(int n, CCSmat A, Carray b, Carray x, double eps,
    int maxiter, Cmatrix M);
/* As in preCCG use matrix vector multiplication by pre-conditioning */



int tripreCCSCCG(int n, CCSmat A, Carray b, Carray x, double eps, 
    int maxiter, Carray upper, Carray lower, Carray mid);
/* Solve tridiagonal system to apply pre-conditioning */



#endif
