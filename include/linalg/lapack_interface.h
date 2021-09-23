/** \file lapack_interface.h
 *
 * \author Alex Andriati - andriat@if.usp.br
 * \date September/2021
 * \brief Interface to call Lapack(E) routines with project data types
 */

#ifndef LAPACK_INTERFACE_H
#define LAPACK_INTERFACE_H

#include "mctdhb_types.h"

/** \brief Interface to compute hermitian matrix inversion using \c zhesv */
int cmat_hermitian_inversion(int rows, Cmatrix mat, Cmatrix mat_inv);

/** \brief Interface to compute full hermitian eig problem with \c zheev */
int cmat_hermitian_eig(int rows, Cmatrix mat, Cmatrix eigvec, Rarray eigvals);

/** \brief Interface to compute only eigvalues with \c zheev */
void cmat_hermitian_eigenvalues(int rows, Cmatrix mat, Rarray eigvals);

/** \brief Matrix regularization to compute inverse
 *
 * A matrix with one or more zero eigenvalues cannot be inverted. In some
 * cases a small perturbation value can be added to the matrix in diagonal
 * form to make it invertible, which is done here as
 *
 * \code A = A + x exp(- A / x) \endcode
 *
 * where \c x must be a small parameter compared to largest eigenvalues
 *
 * \param[in] rows    number of rows (also columns) in square matrix
 * \param[in] x       regularization small parameter
 * \param[in/out] mat input matrix which ends regularized
 */
void cmat_regularization(int rows, double x, Cmatrix mat);

#endif
