/** \file basic_linalg.h
 *
 * \author Alex Andriati - andriati@if.usp.br
 * \date September/2021
 * \brief Module with basic vector operatios
 */

#ifndef BASIC_LINALG_H
#define BASIC_LINALG_H

#include "mctdhb_types.h"

void carrFill(uint32_t n, dcomplex z, Carray v);

void rarrFill(uint32_t n, double x, Rarray v);

void rarrFillInc(uint32_t n, double x0, double dx, Rarray v);

void carrCopy(uint32_t n, Carray from, Carray to);

void rarrCopy(uint32_t n, Rarray from, Rarray to);

void MKL2Carray(uint32_t n, MKLCarray a, Carray b);

void Carray2MKL(uint32_t n, Carray b, MKLCarray a);

void carrRealPart(uint32_t n, Carray v, Rarray vreal);

void carrImagPart(uint32_t n, Carray v, Rarray vimag);

void carrConj(uint32_t n, Carray v, Carray v_conj);

void carrAdd(uint32_t n, Carray v1, Carray v2, Carray v);

void rarrAdd(uint32_t n, Rarray v1, Rarray v2, Rarray v);

void carrSub(uint32_t n, Carray v1, Carray v2, Carray v);

void rarrSub(uint32_t n, Rarray v1, Rarray v2, Rarray v);

void carrMultiply(uint32_t n, Carray v1, Carray v2, Carray v);

void rarrMultiply(uint32_t n, Rarray v1, Rarray v2, Rarray v);

void carrScalarMultiply(uint32_t n, Carray v, dcomplex z, Carray ans);

void rarrScalarMultiply(uint32_t n, Rarray v, double z, Rarray ans);

void carrScalarAdd(uint32_t n, Carray v, dcomplex z, Carray ans);

void rarrScalarAdd(uint32_t n, Rarray v, double z, Rarray ans);

void carrDiv(uint32_t n, Carray v1, Carray v2, Carray v);

void rarrDiv(uint32_t n, Rarray v1, Rarray v2, Rarray v);

void carrUpdate(uint32_t n, Carray v1, dcomplex z, Carray v2, Carray v);

void rcarrUpdate(uint32_t n, Carray v1, dcomplex z, Rarray v2, Carray v);

void rarrUpdate(uint32_t n, Rarray v1, double z, Rarray v2, Rarray v);

void carrAbs(uint32_t n, Carray v, Rarray vabs);

void rarrAbs(uint32_t n, Rarray v, Rarray vabs);

void carrAbs2(uint32_t n, Carray v, Rarray vabs);

void rarrAbs2(uint32_t n, Rarray v, Rarray vabs);

void renormalizeVector(uint32_t n, Carray v, double norm);

dcomplex carrDot(uint32_t n, Carray v1, Carray v2);

dcomplex unconj_carrDot(uint32_t n, Carray v1, Carray v2);

double rarrDot(uint32_t n, Rarray v1, Rarray v2);

double carrMod(uint32_t n, Carray v);

double carrMod2(uint32_t n, Carray v);

dcomplex carrReduction(uint32_t n, Carray v);

double rarrReduction(uint32_t n, Rarray v);

void
cplx_matrix_set_from_rowmajor(
    uint32_t nrows, uint32_t ncols, Carray vec, Cmatrix mat);

void
cplx_rowmajor_set_from_matrix(
    uint32_t nrows, uint32_t ncols, Cmatrix mat, Carray vec);

void
cmat_times_vec(
    uint32_t rows, uint32_t cols, Cmatrix mat, Carray vec, Carray res);

void
carr_rowmajor_times_vec(
    uint32_t rows, uint32_t cols, Carray mat, Carray vec, Carray res);

void
cmat_times_mat(
    uint32_t rows_left,
    uint32_t rows_right,
    uint32_t cols_right,
    Cmatrix  mleft,
    Cmatrix  mright,
    Cmatrix  res);

#endif
