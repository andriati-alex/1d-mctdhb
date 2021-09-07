#ifndef CALCULUS_H
#define CALCULUS_H

#include "mctdhb_types.h"

/** \brief Numerical integration using Simpson's rule */
dcomplex
Csimps(uint32_t grid_size, double dx, Carray f);

/** \brief Numerical integration using Simpson's rule */
double
Rsimps(uint32_t grid_size, double dx, Rarray f);

/** \brief Compute integral of initial and final chunk grid points */
double
real_border_integral(int grid_size, int chunk, Rarray f, double h);

/** \brief Compute integral of initial and final chunk grid points */
dcomplex
cplx_border_integral(int grid_size, int chunk, Carray f, double h);

void
dxFFT(uint32_t grid_size, double dx, Carray f, Carray dfdx);

void
d2xFFT(uint32_t grid_size, double dx, Carray f, Carray dfdx);

void
dxFD(uint32_t grid_size, double dx, Carray f, Carray dfdx);

void
set_fft_freq(int32_t fft_size, double h, Rarray freq);

/** \brief Renormalize a function according to L2 <.,.>
 *
 * \param[in] grid_size number of points to represent functions
 * \param[in] dx        grid step size
 * \param[in] new_norm  new desired norm for the input function
 * \param[in/out] f     input function which is rescaled to have `new_norm`
 */
void
renormalize(uint32_t grid_size, double dx, double new_norm, Carray f);

/** \brief Return <f1, f2> with default L2 <.,.> for complex functions */
dcomplex
scalar_product(uint32_t grid_size, double dx, Carray f1, Carray f2);

/** \brief Return sqrt(<f, f>) with default L2 <.,.> for complex functions */
double
cplx_function_norm(uint32_t n, double h, Carray f);

/** \brief Return sqrt(<f, f>) with default L2 <.,.> for real functions */
double
real_function_norm(uint32_t n, double h, Rarray f);

/** \brief Orthonormalize a set of functions
 * 
 * Apply (Modified)Gram-Schmidt algorithm using L2 <.,.> for complex functions
 *
 * \param[in] grid_size number of points to represent functions
 * \param[in] dx        grid step size
 * \param[in] nfun      number of functions to orthogonalize
 * \param[in/out] fun   A set of arbitrary functions(not even normalized)
 *                      given in rows of a matrix. The first row `fun[0]`
 *                      is also the first function to start the algorithm
 *                      `fun` must have `nfun` rows and `grid_size` cols.
 */
void
orthonormalize(uint32_t grid_size, double dx, uint32_t nfun, Cmatrix fun);

#endif
