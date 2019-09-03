#ifndef _matrix_operations_h
#define _matrix_operations_h

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "array_memory.h"
#include "array_operations.h"



/*          ***********************************************          */
/*                   SETUP VALUES IN MATRIX ENTRIES                  */
/*          ***********************************************          */



void cmatFill(int m, int n, double complex z, Cmatrix M);
/* Fill all matrix with a constant value
 * *************************************
 *
 *  m the number of lines
 *  n the number of columns
 *  z the value to fill the entire matrix
 *
 * *************************************/



void cmatFillDK(int n, int k, Carray z, Cmatrix M);
/* Fill k Diagonal of square matrix of n by n elements with an array z
 * *******************************************************************
 *
 * z must contain n - k elements
 * k = 0 is the main diagonal
 * k > 0 upper diagonals
 * k < 0 lower diagonals
 *
 * *******************************************************************/



void cmatFillTri(int n, Carray upper, Carray mid, Carray lower, Cmatrix M);
/*   Fill a matrix with just the tridiagonals entries, rest with zeros   */



void setValueCCS(int n, int i, int j, int col, double complex z, CCSmat M);
/*        Set a value in the i row and original column number col        */



CCSmat tri2CCS(int n, Carray upper, Carray lower, Carray mid);
/* Fill a Sparse Matrix in Compressed Column Storage format from a tridiagonal
 * ***************************************************************************
 *
 * n the dimension of the matrix. Returns a pointer to CCS struct
 *
 *****************************************************************************/



CCSmat cyclic2CCS(int n, Carray upper, Carray lower, Carray mid);
/*      Fill in CCS format given tridiagonal cyclic system      */



void RowMajor(int m, int n, Cmatrix M, Carray v);
/* Store Matrix M(m x n) in a vector v using Row Major scheme */



CCSmat CNmat(int M, double dx, doublec dt, double a2, doublec a1, double inter,
       Rarray V, int cyclic, Carray upper, Carray lower, Carray mid);





/*          **********************************************          */
/*          MATRIX-VECTOR AND MATRIX-MATRIX MULTIPLICATION          */
/*          **********************************************          */





void cmatvec(int m, int n, Cmatrix M, Carray v, Carray ans);
/* General Matrix Vector multiplication: M . v = ans
 * *************************************************
 *
 *  m number of lines of M
 *  n number of columns of M and components of v
 *
 * *************************************************/



void cmatmat(int m, int n, int l, Cmatrix M, Cmatrix A, Cmatrix ans);
/* General Matrix Matrix multiplication: M . A = ans
 * *************************************************
 *
 *  M has m(lines) by n(columns)
 *  A has n(lines) by l(columns)
 *  ans has m(lines) by l(columns)
 *
 * *************************************************/



void CCSvec(int n, Carray vals, int * cols, int m, Carray vec, Carray ans);
/* Matrix(in CCS format) vector multiplication
 * 
 * Given CCSmat A the arguments taken are
 *
 * vals = A->vec
 * cols = A->col
 * m    = A->m
 *
 * *******************************************/



int HermitianInv(int M, Cmatrix A, Cmatrix A_inv);
/* Invert an hermitian matrix */

int HermitianEig(int n, Cmatrix A, Cmatrix eigvec, Rarray eigvals);
void RegularizeMat(int n, double x, Cmatrix A);


#endif
