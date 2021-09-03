#include "linalg/basic_linalg.h"
#include <math.h>

void
carrFill(uint32_t n, dcomplex z, Carray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = z;
}

void
rarrFill(uint32_t n, double x, Rarray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = x;
}

void
rarrFillInc(uint32_t n, double x0, double dx, Rarray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = x0 + i * dx;
}

void
carrCopy(uint32_t n, Carray from, Carray to)
{
    for (uint32_t i = 0; i < n; i++) to[i] = from[i];
}

void
rarrCopy(uint32_t n, Rarray from, Rarray to)
{
    for (uint32_t i = 0; i < n; i++) to[i] = from[i];
}

void
MKL2Carray(uint32_t n, MKLCarray a, Carray b)
{
    for (uint32_t i = 0; i < n; i++) b[i] = a[i].real + I * a[i].imag;
}

void
Carray2MKL(uint32_t n, Carray b, MKLCarray a)
{
    for (uint32_t i = 0; i < n; i++)
    {
        a[i].real = creal(b[i]);
        a[i].imag = cimag(b[i]);
    }
}

void
carrRealPart(uint32_t n, Carray v, Rarray vreal)
{
    for (uint32_t i = 0; i < n; i++) vreal[i] = creal(v[i]);
}

void
carrImagPart(uint32_t n, Carray v, Rarray vimag)
{
    for (uint32_t i = 0; i < n; i++) vimag[i] = cimag(v[i]);
}

void
carrConj(uint32_t n, Carray v, Carray v_conj)
{
    for (uint32_t i = 0; i < n; i++) v_conj[i] = conj(v[i]);
}

void
carrAdd(uint32_t n, Carray v1, Carray v2, Carray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = v1[i] + v2[i];
}

void
rarrAdd(uint32_t n, Rarray v1, Rarray v2, Rarray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = v1[i] + v2[i];
}

void
carrSub(uint32_t n, Carray v1, Carray v2, Carray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = v1[i] - v2[i];
}

void
rarrSub(uint32_t n, Rarray v1, Rarray v2, Rarray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = v1[i] - v2[i];
}

void
carrMultiply(uint32_t n, Carray v1, Carray v2, Carray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = v1[i] * v2[i];
}

void
rarrMultiply(uint32_t n, Rarray v1, Rarray v2, Rarray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = v1[i] * v2[i];
}

void
carrScalarMultiply(uint32_t n, Carray v, dcomplex z, Carray ans)
{
    for (uint32_t i = 0; i < n; i++) ans[i] = v[i] * z;
}

void
rarrScalarMultiply(uint32_t n, Rarray v, double z, Rarray ans)
{
    for (uint32_t i = 0; i < n; i++) ans[i] = v[i] * z;
}

void
carrScalarAdd(uint32_t n, Carray v, dcomplex z, Carray ans)
{
    for (uint32_t i = 0; i < n; i++) ans[i] = v[i] + z;
}

void
rarrScalarAdd(uint32_t n, Rarray v, double z, Rarray ans)
{
    for (uint32_t i = 0; i < n; i++) ans[i] = v[i] + z;
}

void
carrDiv(uint32_t n, Carray v1, Carray v2, Carray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = v1[i] / v2[i];
}

void
rarrDiv(uint32_t n, Rarray v1, Rarray v2, Rarray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = v1[i] / v2[i];
}

void
carrUpdate(uint32_t n, Carray v1, dcomplex z, Carray v2, Carray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = v1[i] + z * v2[i];
}

void
rcarrUpdate(uint32_t n, Carray v1, dcomplex z, Rarray v2, Carray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = v1[i] + z * v2[i];
}

void
rarrUpdate(uint32_t n, Rarray v1, double z, Rarray v2, Rarray v)
{
    for (uint32_t i = 0; i < n; i++) v[i] = v1[i] + z * v2[i];
}

void
carrAbs(uint32_t n, Carray v, Rarray vabs)
{
    for (uint32_t i = 0; i < n; i++) vabs[i] = cabs(v[i]);
}

void
rarrAbs(uint32_t n, Rarray v, Rarray vabs)
{
    for (uint32_t i = 0; i < n; i++) vabs[i] = fabs(v[i]);
}

void
rarrAbs2(uint32_t n, Rarray v, Rarray vabs)
{
    for (uint32_t i = 0; i < n; i++) vabs[i] = v[i] * v[i];
}

void
carrAbs2(uint32_t n, Carray v, Rarray vabs)
{
    for (uint32_t i = 0; i < n; i++)
    {
        vabs[i] = creal(v[i]) * creal(v[i]) + cimag(v[i]) * cimag(v[i]);
    }
}

void
renormalizeVector(uint32_t n, Carray v, double norm)
{
    double renorm = norm / carrMod(n, v);
    for (uint32_t i = 0; i < n; i++) v[i] = v[i] * renorm;
}

dcomplex
carrDot(uint32_t n, Carray v1, Carray v2)
{
    dcomplex z = 0;
    for (uint32_t i = 0; i < n; i++) z = z + conj(v1[i]) * v2[i];
    return z;
}

dcomplex
unconj_carrDot(uint32_t n, Carray v1, Carray v2)
{
    dcomplex z = 0;
    for (uint32_t i = 0; i < n; i++) z = z + v1[i] * v2[i];
    return z;
}

double
rarrDot(uint32_t n, Rarray v1, Rarray v2)
{
    double z = 0;
    for (uint32_t i = 0; i < n; i++) z = z + v1[i] * v2[i];
    return z;
}

double
carrMod(uint32_t n, Carray v)
{
    double mod = 0;
    for (uint32_t i = 0; i < n; i++)
    {
        mod = mod + creal(v[i]) * creal(v[i]) + cimag(v[i]) * cimag(v[i]);
    }
    return sqrt(mod);
}

double
carrMod2(uint32_t n, Carray v)
{
    double mod = 0;
    for (uint32_t i = 0; i < n; i++)
    {
        mod = mod + creal(v[i]) * creal(v[i]) + cimag(v[i]) * cimag(v[i]);
    }
    return mod;
}

dcomplex
carrReduction(uint32_t n, Carray v)
{
    dcomplex red = 0;
    for (uint32_t i = 0; i < n; i++) red = red + v[i];
    return red;
}

double
rarrReduction(uint32_t n, Rarray v)
{
    double red = 0;
    for (uint32_t i = 0; i < n; i++) red = red + v[i];
    return red;
}

void
cplx_matrix_set_from_rowmajor(
    uint32_t nrows, uint32_t ncols, Carray vec, Cmatrix mat)
{
    for (uint32_t i = 0; i < nrows; i++)
    {
        for (uint32_t j = 0; j < ncols; j++) mat[i][j] = vec[i * ncols + j];
    }
}

void
cplx_rowmajor_set_from_matrix(
    uint32_t nrows, uint32_t ncols, Cmatrix mat, Carray vec)
{
    for (uint32_t i = 0; i < nrows; i++)
    {
        for (uint32_t j = 0; j < ncols; j++) vec[i * ncols + j] = mat[i][j];
    }
}

void
cmat_times_vec(
    uint32_t rows, uint32_t cols, Cmatrix mat, Carray vec, Carray res)
{

    uint32_t i, j;
    dcomplex summ;
    for (i = 0; i < rows; i++)
    {
        summ = 0;
        for (j = 0; j < cols; j++) summ = summ + mat[i][j] * vec[j];
        res[i] = summ;
    }
}

void
carr_rowmajor_times_vec(
    uint32_t rows, uint32_t cols, Carray mat, Carray vec, Carray res)
{
    uint32_t i, j, stride;
    dcomplex summ;
    for (i = 0; i < rows; i++)
    {
        stride = i * cols;
        summ = 0;
        for (j = 0; j < cols; j++) summ = summ + mat[stride + j] * vec[j];
        res[i] = summ;
    }
}

void
cmat_times_mat(
    uint32_t rows_left,
    uint32_t rows_right,
    uint32_t cols_right,
    Cmatrix  mleft,
    Cmatrix  mright,
    Cmatrix  res)
{
    uint32_t i, j, k;
    dcomplex summ;
    for (i = 0; i < rows_left; i++)
    {
        for (j = 0; j < cols_right; j++)
        {
            summ = 0;
            for (k = 0; k < rows_right; k++)
                summ = summ + mleft[i][k] * mright[k][j];
            res[i][j] = summ;
        }
    }
}
