#ifndef _calculus_h
#define _calculus_h

#include "memoryHandling.h"
#include "array_operations.h"





/* DIFFERENTIATION AND INTEGRATION ROUTINES
 * ------------------------------------------------------------------------
 *
 * Input parameters
 *
 *      size_of_f is the number of discretized points the domain is divided
 *
 *      f is the vector with function values at discretized points
 *
 *      dx is the size of grid step
 *
 * Output parameter:
 *
 *      dfdx for functions that compute derivative
 *
 * ------------------------------------------------------------------------ */

double complex Csimps(int size_of_f, Carray f, double dx);

double Rsimps(int size_of_f, Rarray f, double dx);

void dxFFT(int size_of_f, Carray f, double dx, Carray dfdx);

void d2xFFT(int size_of_f, Carray f, double dx, Carray dfdx);

void dxFD(int size_of_f, Carray f, double dx, Carray dfdx);





/* NORMALIZATION AND ORTHOGONALIZATION
 * ------------------------------------------------------------------------
 *
 * Input parameters
 *
 *      size_of_f is the number of discretized points the domain is divided
 *
 *      dx is the size of grid step
 *
 *      norm is the new norm demanded
 *
 * Output parameter:
 *
 *      f ended up multiplied by a scaling factor to normalize to 'norm'
 *
 *      F matrix contains functions represented by rows where columns are
 *      the discretized positions. End up with rows(functions) orthogonal
 *      to each other and with unit norm.
 *
 * ------------------------------------------------------------------------ */

void renormalize(int size_of_f, Carray f, double dx, double norm);

double complex innerL2(int n, Carray fstar, Carray f, double h);

void Ortonormalize(int N_rows, int N_cols, double dx, Cmatrix F);

#endif
