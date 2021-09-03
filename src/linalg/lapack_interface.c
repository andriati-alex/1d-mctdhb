#include "assistant/arrays_definition.h"
#include "linalg/lapack_interface.h"
#include "mctdhb_types.h"
#include "mkl_lapacke.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int
cmat_hermitian_inversion(int rows, Cmatrix mat, Cmatrix mat_inv)
{

    /** Use Lapack routine to solve systems of equations with the
     * right-hand-side being identity matrix  to get the inverse **/

    int       i, j, l;
    Iarray    ipiv;
    MKLCarray ArrayForm, Id;

    ipiv = get_int_array(rows);
    ArrayForm = get_mklcomplex16_array(rows * rows);
    Id = get_mklcomplex16_array(rows * rows);

    for (i = 0; i < rows; i++)
    {
        // Setup (L)ower triangular part as a Row-Major-Array to use lapack
        ArrayForm[i * rows + i].real = creal(mat[i][i]);
        ArrayForm[i * rows + i].imag = 0;
        Id[i * rows + i].real = 1;
        Id[i * rows + i].imag = 0;

        for (j = 0; j < i; j++)
        {
            ArrayForm[i * rows + j].real = creal(mat[i][j]);
            ArrayForm[i * rows + j].imag = cimag(mat[i][j]);
            ArrayForm[j * rows + i].real = 0; // symbolic values
            ArrayForm[j * rows + i].imag = 0; // for upper triangular part
            Id[i * rows + j].real = 0;
            Id[i * rows + j].imag = 0;
            Id[j * rows + i].real = 0;
            Id[j * rows + i].imag = 0;
        }
    }

    l = LAPACKE_zhesv(
        LAPACK_ROW_MAJOR, 'L', rows, rows, ArrayForm, rows, ipiv, Id, rows);

    // Result transcription to original double pointer format
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < rows; j++)
        {
            mat_inv[i][j] = Id[i * rows + j].real + I * Id[i * rows + j].imag;
        }
    }

    free(ipiv);
    free(Id);
    free(ArrayForm);

    return l;
}

int
cmat_hermitian_eig(int rows, Cmatrix A, Cmatrix eigvec, Rarray eigvals)
{

    int       i, j, ldz, check;
    MKLCarray Arow;

    ldz = rows;
    Arow = get_mklcomplex16_array(rows * rows);

    // transcription to mkl row major matrix (UPPER PART ONLY => Hermitian)
    for (i = 0; i < rows; i++)
    {
        Arow[i * rows + i].real = creal(A[i][i]);
        Arow[i * rows + i].imag = 0;

        for (j = i + 1; j < rows; j++)
        {
            Arow[i * rows + j].real = creal(A[i][j]);
            Arow[i * rows + j].imag = cimag(A[i][j]);
        }
    }

    check = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', rows, Arow, ldz, eigvals);

    // transcription eigenvectors to default double pointer complex datatype
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < rows; j++)
        {
            eigvec[i][j] =
                Arow[i * rows + j].real + I * Arow[i * rows + j].imag;
        }
    }
    free(Arow);

    return check;
}

void
cmat_hermitian_eigenvalues(int rows, Cmatrix A, Rarray eigvals)
{

    int       i, j, ldz, check;
    MKLCarray Arow;

    ldz = rows;
    Arow = get_mklcomplex16_array(rows * rows);

    // transcription to mkl row major matrix (UPPER PART ONLY => Hermitian)
    for (i = 0; i < rows; i++)
    {
        Arow[i * rows + i].real = creal(A[i][i]);
        Arow[i * rows + i].imag = 0;

        for (j = i + 1; j < rows; j++)
        {
            Arow[i * rows + j].real = creal(A[i][j]);
            Arow[i * rows + j].imag = cimag(A[i][j]);
        }
    }

    check = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'N', 'U', rows, Arow, ldz, eigvals);
    free(Arow);
}

void
cmat_regularization(int rows, double x, Cmatrix A)
{

    /** For a positive definite matrix (only positive eigenvalues) regularize
     * small eigenvalues (near numerical precision) in order to maintain the
     * existence of inverse matrix.   It  impose  a  minimum value x for all
     * eigenvalues whereas maintain the larger ones unaffected.          **/

    int      i, j, k, check;
    dcomplex z;
    Cmatrix  eigvec;
    Rarray   eigvals;

    eigvec = get_dcomplex_matrix(rows, rows);
    eigvals = get_double_array(rows);

    check = cmat_hermitian_eig(rows, A, eigvec, eigvals);

    if (check != 0)
    {
        printf("\n\nERROR IN DIAGONALIZATION\n\n");
        printf("In Lapack call of zheev function it returned %d\n\n", check);
        exit(EXIT_FAILURE);
    }

    // Evaluate two matrix multiplications to return to original basis
    // after exponential done in diagonal  basis.  The  transformation
    // matrices are given by the eigenvector.

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < rows; j++)
        {
            z = 0;
            for (k = 0; k < rows; k++)
            {
                z += eigvec[i][k] * (x * exp(-eigvals[k] / x)) *
                     conj(eigvec[j][k]);
            }
            A[i][j] = A[i][j] + z;
        }
    }
    destroy_dcomplex_matrix(rows, eigvec);
    free(eigvals);
}
