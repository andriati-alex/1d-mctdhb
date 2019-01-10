#ifndef _calculus_h
#define _calculus_h

#include <mkl.h>
#include <mkl_dfti.h>
#include <math.h>

#include "array_memory.h"
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

double complex Ctrapezium(int n, Carray f, double dx);

double Rtrapezium(int n, Rarray f, double dx);

void dxFFT(int size_of_f, Carray f, double dx, Carray dfdx);

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

void Ortonormalize(int N_rows, int N_cols, double dx, Cmatrix F);





double complex SquaredRampl(int n, Carray f, Carray g, double dx);
/** Quantum Mechanical Matrix element of r^2 operator < f | r^2 | g > **/



int NonVanishingId(int n, Carray f, double dx, double tol);

#endif
