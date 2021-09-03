#ifndef LAPACK_INTERFACE_H
#define LAPACK_INTERFACE_H

#include "mctdhb_types.h"

int cmat_hermitian_inversion(int rows, Cmatrix mat, Cmatrix mat_inv);

int cmat_hermitian_eig(int rows, Cmatrix A, Cmatrix eigvec, Rarray eigvals);

void cmat_hermitian_eigenvalues(int rows, Cmatrix A, Rarray eigvals);

void cmat_regularization(int rows, double x, Cmatrix A);

#endif
