#ifndef _tridiagonal_solver_h
#define _tridiagonal_solver_h

#include "array_memory.h"
#include "array_operations.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

double cond(int n, Carray upper, Carray lower, Carray mid);
double errBack(int n, Carray upper, Carray lower, Carray mid);
/* Compute condition number u
 * **************************
 *
 * See BUENO, M. Isabel & DOPICO, FROILÁN M. for the meaning of condition
 * numbers in "Stability and sensitivity of tridiagonal LU factorization"
 *
 * A good condition number is < 1E10 (assure at least 6 digits precision).
 *
 * The Backward error should be of 10E-15
 *
 * **********************************************************************/





double complex checkLU(int n, Carray upper, Carray lower, Carray mid);
/* Check if the tridiagonal system has L.U. factorization
 * ******************************************************
 *
 * n is the size of the system
 * upper is upper diagonal and has size (n-1)
 * lower is lower diagonal and has size (n-1)
 * mid is the main diagonal of size (n)
 *
 * The process ends up with the determinant, otherwise it
 * gives zero(a source of breakdown to LU factorization)
 *
 * ******************************************************/





void triDiag(int n, Carray upper, Carray lower, Carray mid,
     Carray RHS, Carray ans);
/* Solve Tri-diagonal linear system using A = L . U factorization
 * **************************************************************
 *
 * n is the size of the system
 * upper is upper diagonal and has size (n-1)
 * lower is lower diagonal and has size (n-1)
 * mid is the main diagonal of size (n)
 * RHS (Right Hand side) vector of A . x = RHS
 * ans ends up with solution
 *
 * **************************************************************/





void triCyclicLU(int n, Carray upper, Carray lower, Carray mid,
     Carray RHS, Carray ans);
/* Solve Cyclic Tri-diagonal linear system using Modified LU decomposition
 * ***********************************************************************
 *
 * upper is upper diagonal and has size n (last element is the top coupling)
 * lower is lower diagonal and has size n (''   ''   is the bottom coupling)
 * mid is the main diagonal of size n
 * RHS (Right Hand side) vector of A . x = RHS
 * n is the size of the system
 * ans ends up with solution
 * -----------------------------------------------------------------------
 *
 * Implementarion follows (Thiab R.)Taha's paper
 * "Solution of Periodic Tridiagonal Linear Systems on a Hypercube"
 * Method fails if first main diagonal element is zero.
 *
 * ************************************************************************/





void triCyclicSM(int n, Carray upper, Carray lower, Carray mid,
     Carray RHS, Carray ans);
/* Solve Cyclic Tri-diagonal linear system using Sherman-Morrison
 * **************************************************************
 *
 * upper is upper diagonal and has size n (last element is the top coupling)
 * lower is lower diagonal and has size n (''   ''   is the bottom coupling)
 * mid is the main diagonal of size n
 * RHS (Right Hand side) vector of A . x = RHS
 * n is the size of the system
 * ans ends up with solution
 * --------------------------------------------------------------
 *
 * Implementation follows numerical recipes, pag ~ 70
 *
 * ***************************************************************/


#endif
